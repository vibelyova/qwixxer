//! Training for the pairwise differential network: per-opponent TD(λ) chains,
//! board-swap sample doubling, decoupled μ/σ loss, pure self-play loop.
//!
//! Only compiled with the `dqn` feature, mirroring `dqn::train`.

use crate::bot::{self, DNA};
use crate::dqn::pair::{
    pair_batch_forward, pair_features, PairModel, PairModelConfig, PairStrategy, BOARD_FEATURES, PAIR_FEATURES,
};
use crate::dqn::{MyBackend, LOG_VAR_MAX, LOG_VAR_MIN, TRAIN_SEED};
use crate::state::{Mark, State};
use crate::strategy::Strategy;
use burn::{
    backend::Autodiff,
    data::{
        dataloader::{batcher::Batcher, DataLoaderBuilder},
        dataset::InMemDataset,
    },
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{metric::LossMetric, InferenceStep, Learner, RegressionOutput, SupervisedTraining, TrainOutput, TrainStep},
};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rayon::prelude::*;
use std::sync::Arc;

type MyAutodiffBackend = Autodiff<MyBackend>;

const LAMBDA: f32 = 0.8;

/// Absolute-space TD(λ) targets over one diff chain.
///
/// `mus[t]` is the model's *future-space* μ_diff at step t (used for the
/// bootstrap at t+1), `cdiffs[t]` the current point differential. Returns
/// `G_t` in absolute-diff space:
///
/// ```text
/// G_{n−1} = final_diff
/// G_t     = (1−λ)·(μ_{t+1} + cdiff_{t+1}) + λ·G_{t+1}
/// ```
fn td_diff_targets(mus: &[f32], cdiffs: &[f32], final_diff: f32, lambda: f32) -> Vec<f32> {
    let n = cdiffs.len();
    let mut g = vec![0.0f32; n];
    g[n - 1] = final_diff;
    for t in (0..n - 1).rev() {
        g[t] = (1.0 - lambda) * (mus[t + 1] + cdiffs[t + 1]) + lambda * g[t + 1];
    }
    g
}

// ---- Training samples / loss / batcher ----

/// serde adapter for `[f32; PAIR_FEATURES]` (length 45 exceeds serde's built-in
/// array impls). Round-trips through a `Vec<f32>`.
mod pair_features_serde {
    use super::PAIR_FEATURES;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(arr: &[f32; PAIR_FEATURES], s: S) -> Result<S::Ok, S::Error> {
        arr.as_slice().serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[f32; PAIR_FEATURES], D::Error> {
        let v = Vec::<f32>::deserialize(d)?;
        <[f32; PAIR_FEATURES]>::try_from(v.as_slice()).map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct PairSample {
    // serde's built-in array impls stop at length 32; PAIR_FEATURES is 45, so
    // (de)serialize via a slice helper. (train.rs's TrainingSample escapes this
    // because NUM_FEATURES = 25.)
    #[serde(with = "pair_features_serde")]
    pub features: [f32; PAIR_FEATURES],
    /// TD(λ) target in future-diff space: `G_t − current_diff_t`.
    pub value: f32,
    /// Actual final diff in future space: `final_diff − current_diff_t`.
    /// σ's residual reference (unbiased, bypasses TD smoothing).
    pub final_diff: f32,
}

impl<B: Backend> PairModel<B> {
    /// Decoupled μ/σ loss on the differential — same recipe as the old net:
    /// μ: MSE toward the TD(λ) target; σ: Gaussian NLL against the actual
    /// final-diff residual with μ detached.
    pub fn forward_step(&self, batch: PairBatch<B>) -> RegressionOutput<B> {
        let output = self.forward(batch.inputs);
        let mean = output.clone().narrow(1, 0, 1);
        let log_var = output.narrow(1, 1, 1).clamp(LOG_VAR_MIN, LOG_VAR_MAX);

        let targets = batch.targets.clone().unsqueeze_dim(1);
        let final_diffs = batch.final_diffs.clone().unsqueeze_dim(1);

        let mu_residual = targets.clone() - mean.clone();
        let mu_loss = (mu_residual.clone() * mu_residual).mean();

        let mean_detached = mean.clone().detach();
        let sigma_residual = final_diffs - mean_detached;
        let sigma_sq = sigma_residual.clone() * sigma_residual;
        let inv_var = log_var.clone().neg().exp();
        let sigma_nll = log_var + sigma_sq * inv_var;
        let sigma_loss = sigma_nll.mean().mul_scalar(0.5);

        let loss = mu_loss + sigma_loss;

        RegressionOutput {
            loss,
            output: mean,
            targets,
        }
    }
}

impl<B: AutodiffBackend> TrainStep for PairModel<B> {
    type Input = PairBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, batch: PairBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_step(batch);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for PairModel<B> {
    type Input = PairBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, batch: PairBatch<B>) -> RegressionOutput<B> {
        self.forward_step(batch)
    }
}

#[derive(Clone)]
pub struct PairBatcher<B: Backend> {
    _phantom: std::marker::PhantomData<B>,
}

#[derive(Clone, Debug)]
pub struct PairBatch<B: Backend> {
    pub inputs: Tensor<B, 2>,
    pub targets: Tensor<B, 1>,
    pub final_diffs: Tensor<B, 1>,
}

/// Color-permutation augmentation applied identically to BOTH board blocks.
/// Per-row groups within each block: progress [0–3], marks [4–7],
/// locked [8–11], wprob [12–15]; pair-level features 40–44 are untouched
/// (they are row-permutation invariant).
fn permute_colors(f: &mut [f32; PAIR_FEATURES], swap_ry: bool, swap_gb: bool, swap_pairs: bool) {
    for block in [0, BOARD_FEATURES] {
        if swap_ry {
            for base in [0, 4, 8, 12] {
                f.swap(block + base, block + base + 1);
            }
        }
        if swap_gb {
            for base in [2, 6, 10, 14] {
                f.swap(block + base, block + base + 1);
            }
        }
        if swap_pairs {
            for base in [0, 4, 8, 12] {
                f.swap(block + base, block + base + 2);
                f.swap(block + base + 1, block + base + 3);
            }
        }
    }
}

impl<B: Backend> Batcher<B, PairSample, PairBatch<B>> for PairBatcher<B> {
    fn batch(&self, items: Vec<PairSample>, device: &B::Device) -> PairBatch<B> {
        let batch_size = items.len();
        let batch_seed = TRAIN_SEED
            .wrapping_add(items[0].value.to_bits() as u64)
            .wrapping_add(batch_size as u64);
        let mut rng = SmallRng::seed_from_u64(batch_seed);

        let inputs: Vec<f32> = items
            .iter()
            .flat_map(|s| {
                let mut f = s.features;
                permute_colors(&mut f, rng.gen(), rng.gen(), rng.gen());
                f
            })
            .collect();
        let targets: Vec<f32> = items.iter().map(|s| s.value).collect();
        let final_diffs: Vec<f32> = items.iter().map(|s| s.final_diff).collect();

        let inputs = Tensor::<B, 1>::from_floats(inputs.as_slice(), device).reshape([batch_size, PAIR_FEATURES]);
        let targets = Tensor::<B, 1>::from_floats(targets.as_slice(), device);
        let final_diffs = Tensor::<B, 1>::from_floats(final_diffs.as_slice(), device);

        PairBatch {
            inputs,
            targets,
            final_diffs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn td_diff_targets_hand_computed() {
        // 3 steps, λ = 0.8.
        let mus = [1.0, 2.0, 3.0];
        let cdiffs = [0.0, 5.0, 10.0];
        let g = td_diff_targets(&mus, &cdiffs, 20.0, 0.8);
        // G2 = 20
        // G1 = 0.2·(μ2 + cd2) + 0.8·G2 = 0.2·13 + 16    = 18.6
        // G0 = 0.2·(μ1 + cd1) + 0.8·G1 = 0.2·7 + 14.88  = 16.28
        assert!((g[2] - 20.0).abs() < 1e-5);
        assert!((g[1] - 18.6).abs() < 1e-5);
        assert!((g[0] - 16.28).abs() < 1e-5);
    }

    #[test]
    fn permute_colors_swaps_both_blocks_and_preserves_pair_features() {
        // Distinct values everywhere so swaps are observable.
        let mut f = [0.0f32; PAIR_FEATURES];
        for (i, v) in f.iter_mut().enumerate() {
            *v = i as f32;
        }
        let orig = f;

        permute_colors(&mut f, true, false, false); // red <-> yellow
        for block in [0, BOARD_FEATURES] {
            for base in [0usize, 4, 8, 12] {
                assert_eq!(f[block + base], orig[block + base + 1]);
                assert_eq!(f[block + base + 1], orig[block + base]);
            }
            // green/blue untouched
            for base in [2usize, 6, 10, 14] {
                assert_eq!(f[block + base], orig[block + base]);
            }
            // per-board aggregates untouched
            assert_eq!(f[block + 16..block + 20], orig[block + 16..block + 20]);
        }
        // Pair-level untouched.
        assert_eq!(f[40..45], orig[40..45]);

        // Applying the same swap again restores the original (involution).
        permute_colors(&mut f, true, false, false);
        assert_eq!(f, orig);
    }

    #[test]
    fn permute_colors_covers_gb_and_asc_desc_paths() {
        // Distinct values everywhere so swaps are observable.
        let mut f = [0.0f32; PAIR_FEATURES];
        for (i, v) in f.iter_mut().enumerate() {
            *v = i as f32;
        }
        let orig = f;

        // green <-> blue
        permute_colors(&mut f, false, true, false);
        for block in [0, BOARD_FEATURES] {
            for base in [2usize, 6, 10, 14] {
                assert_eq!(f[block + base], orig[block + base + 1]);
                assert_eq!(f[block + base + 1], orig[block + base]);
            }
            // red/yellow untouched
            for base in [0usize, 4, 8, 12] {
                assert_eq!(f[block + base], orig[block + base]);
            }
            // per-board aggregates untouched
            assert_eq!(f[block + 16..block + 20], orig[block + 16..block + 20]);
        }
        // Pair-level untouched.
        assert_eq!(f[40..45], orig[40..45]);
        // Involution.
        permute_colors(&mut f, false, true, false);
        assert_eq!(f, orig);

        // asc <-> desc pairs (two swaps per base).
        f = orig;
        permute_colors(&mut f, false, false, true);
        for block in [0, BOARD_FEATURES] {
            for base in [0usize, 4, 8, 12] {
                assert_eq!(f[block + base], orig[block + base + 2]);
                assert_eq!(f[block + base + 1], orig[block + base + 3]);
                assert_eq!(f[block + base + 2], orig[block + base]);
                assert_eq!(f[block + base + 3], orig[block + base + 1]);
            }
            // per-board aggregates untouched
            assert_eq!(f[block + 16..block + 20], orig[block + 16..block + 20]);
        }
        // Pair-level untouched.
        assert_eq!(f[40..45], orig[40..45]);
        // Involution.
        permute_colors(&mut f, false, false, true);
        assert_eq!(f, orig);
    }
}
