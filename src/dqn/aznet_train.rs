//! Training for the aznet value net: 2-player-only plain self-play, reusing
//! the pair net's TD(λ) diff targets, decoupled μ/σ loss, and board-swap
//! doubling. No search, no distillation. Only compiled with the `dqn` feature.

use crate::dqn::aznet::{az_batch_forward, az_features, AzBatch, AzModel, AzModelConfig, AzStrategy, AZ_FEATURES, BOARD_RAW, ROW_BLOCK};
use crate::dqn::pair_train::td_diff_targets;
use crate::dqn::{MyBackend, TRAIN_SEED};
use crate::state::{Mark, State};
use crate::strategy::{active_phase1_impl, active_phase2_impl, passive_phase1_impl, Strategy};
use burn::{
    backend::Autodiff,
    data::{dataloader::batcher::Batcher, dataset::InMemDataset, dataloader::DataLoaderBuilder},
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    train::{metric::LossMetric, Learner, SupervisedTraining},
};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::sync::Arc;

type MyAutodiffBackend = Autodiff<MyBackend>;

const LAMBDA: f32 = 0.8;

/// serde adapter for `[f32; AZ_FEATURES]` (length 233 exceeds serde's array
/// impls). Round-trips through a `Vec<f32>`.
mod az_features_serde {
    use super::AZ_FEATURES;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(arr: &[f32; AZ_FEATURES], s: S) -> Result<S::Ok, S::Error> {
        arr.as_slice().serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[f32; AZ_FEATURES], D::Error> {
        let v = Vec::<f32>::deserialize(d)?;
        <[f32; AZ_FEATURES]>::try_from(v.as_slice()).map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct AzSample {
    #[serde(with = "az_features_serde")]
    pub features: [f32; AZ_FEATURES],
    pub value: f32,
    pub final_diff: f32,
}

/// Swap the two 28-float row blocks at offsets `a` and `b`.
fn swap_block(f: &mut [f32; AZ_FEATURES], a: usize, b: usize) {
    for k in 0..ROW_BLOCK {
        f.swap(a + k, b + k);
    }
}

/// Color-permutation augmentation: permute the four 28-dim row blocks within
/// BOTH boards identically. Strikes and cdiff are row-permutation invariant.
pub(crate) fn permute_rows(f: &mut [f32; AZ_FEATURES], swap_ry: bool, swap_gb: bool, swap_pairs: bool) {
    for board in [0usize, BOARD_RAW] {
        let blk = |r: usize| board + r * ROW_BLOCK;
        if swap_ry {
            swap_block(f, blk(0), blk(1));
        }
        if swap_gb {
            swap_block(f, blk(2), blk(3));
        }
        if swap_pairs {
            swap_block(f, blk(0), blk(2));
            swap_block(f, blk(1), blk(3));
        }
    }
}

#[derive(Clone)]
pub struct AzBatcher<B: Backend> {
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> Batcher<B, AzSample, AzBatch<B>> for AzBatcher<B> {
    fn batch(&self, items: Vec<AzSample>, device: &B::Device) -> AzBatch<B> {
        let batch_size = items.len();
        let batch_seed = TRAIN_SEED
            .wrapping_add(items[0].value.to_bits() as u64)
            .wrapping_add(batch_size as u64);
        let mut rng = SmallRng::seed_from_u64(batch_seed);

        let inputs: Vec<f32> = items
            .iter()
            .flat_map(|s| {
                let mut f = s.features;
                permute_rows(&mut f, rng.gen(), rng.gen(), rng.gen());
                f
            })
            .collect();
        let targets: Vec<f32> = items.iter().map(|s| s.value).collect();
        let final_diffs: Vec<f32> = items.iter().map(|s| s.final_diff).collect();

        let inputs = Tensor::<B, 1>::from_floats(inputs.as_slice(), device).reshape([batch_size, AZ_FEATURES]);
        let targets = Tensor::<B, 1>::from_floats(targets.as_slice(), device);
        let final_diffs = Tensor::<B, 1>::from_floats(final_diffs.as_slice(), device);

        AzBatch { inputs, targets, final_diffs }
    }
}

/// One recorded decision: our post-decision state + the opponent's state.
type Snapshot = (State, Vec<State>);

/// Build training samples from one player's 2-player trajectory: a single
/// TD(λ) chain, every sample emitted in both board orders (swap doubling) with
/// negated targets. μ for the bootstrap comes from a burn forward over the
/// chain's features.
fn build_az_samples(
    model: &AzModel<MyBackend>,
    device: &burn::backend::ndarray::NdArrayDevice,
    snapshots: &[Snapshot],
    our_final: f32,
    opp_final: f32,
) -> Vec<AzSample> {
    let mut samples = Vec::new();
    if snapshots.is_empty() {
        return samples;
    }
    let final_diff = our_final - opp_final;

    let feats: Vec<[f32; AZ_FEATURES]> = snapshots
        .iter()
        .map(|(our, opps)| az_features(our, &opps[0]))
        .collect();
    let cdiffs: Vec<f32> = snapshots
        .iter()
        .map(|(our, opps)| (our.count_points() - opps[0].count_points()) as f32)
        .collect();
    let mus: Vec<f32> = az_batch_forward(model, device, &feats).into_iter().map(|(m, _)| m).collect();
    let g = td_diff_targets(&mus, &cdiffs, final_diff, LAMBDA);

    for (t, (our, opps)) in snapshots.iter().enumerate() {
        let value = g[t] - cdiffs[t];
        let fdiff = final_diff - cdiffs[t];
        samples.push(AzSample { features: feats[t], value, final_diff: fdiff });
        samples.push(AzSample {
            features: az_features(&opps[0], our),
            value: -value,
            final_diff: -fdiff,
        });
    }
    samples
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permute_rows_swaps_both_boards_and_is_involution() {
        let mut f = [0.0f32; AZ_FEATURES];
        for (i, v) in f.iter_mut().enumerate() {
            *v = i as f32;
        }
        let orig = f;

        // red <-> yellow: blocks 0 and 1 in both boards.
        permute_rows(&mut f, true, false, false);
        for board in [0usize, BOARD_RAW] {
            for k in 0..ROW_BLOCK {
                assert_eq!(f[board + k], orig[board + ROW_BLOCK + k]);
                assert_eq!(f[board + ROW_BLOCK + k], orig[board + k]);
            }
            // green/blue untouched.
            for k in 0..ROW_BLOCK {
                assert_eq!(f[board + 2 * ROW_BLOCK + k], orig[board + 2 * ROW_BLOCK + k]);
            }
        }
        // cdiff untouched.
        assert_eq!(f[2 * BOARD_RAW], orig[2 * BOARD_RAW]);
        // involution.
        permute_rows(&mut f, true, false, false);
        assert_eq!(f, orig);

        // pair swap: {0<->2, 1<->3}, involution.
        permute_rows(&mut f, false, false, true);
        for board in [0usize, BOARD_RAW] {
            for k in 0..ROW_BLOCK {
                assert_eq!(f[board + k], orig[board + 2 * ROW_BLOCK + k]);
                assert_eq!(f[board + ROW_BLOCK + k], orig[board + 3 * ROW_BLOCK + k]);
            }
        }
        permute_rows(&mut f, false, false, true);
        assert_eq!(f, orig);

        // green <-> blue: blocks 2 and 3 in both boards, involution.
        f = orig;
        permute_rows(&mut f, false, true, false);
        for board in [0usize, BOARD_RAW] {
            for k in 0..ROW_BLOCK {
                assert_eq!(f[board + 2 * ROW_BLOCK + k], orig[board + 3 * ROW_BLOCK + k]);
                assert_eq!(f[board + 3 * ROW_BLOCK + k], orig[board + 2 * ROW_BLOCK + k]);
            }
            // red/yellow untouched.
            for k in 0..ROW_BLOCK {
                assert_eq!(f[board + k], orig[board + k]);
                assert_eq!(f[board + ROW_BLOCK + k], orig[board + ROW_BLOCK + k]);
            }
        }
        // strike one-hots untouched by an actual permutation.
        assert_eq!(f[112..116], orig[112..116]);
        assert_eq!(f[228..232], orig[228..232]);
        permute_rows(&mut f, false, true, false);
        assert_eq!(f, orig);
    }

    #[test]
    fn build_az_samples_emits_negated_swapped_pairs() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let model = AzModelConfig::new().init::<MyBackend>(&device);

        // 2-step 1v1 trajectory.
        let mut our1 = State::default();
        our1.apply_mark(Mark { row: 0, number: 5 });
        let opp1 = State::default();
        let mut our2 = our1;
        our2.apply_mark(Mark { row: 0, number: 7 });
        let mut opp2 = State::default();
        opp2.apply_mark(Mark { row: 2, number: 10 });

        let snapshots = vec![(our1, vec![opp1]), (our2, vec![opp2])];
        let samples = build_az_samples(&model, &device, &snapshots, 30.0, 20.0);

        // 2 steps × 2 orders.
        assert_eq!(samples.len(), 4);
        for pair in samples.chunks(2) {
            let (fwd, swp) = (&pair[0], &pair[1]);
            assert_eq!(swp.value, -fwd.value);
            assert_eq!(swp.final_diff, -fwd.final_diff);
            assert_eq!(fwd.features[0..BOARD_RAW], swp.features[BOARD_RAW..2 * BOARD_RAW]);
            assert_eq!(fwd.features[BOARD_RAW..2 * BOARD_RAW], swp.features[0..BOARD_RAW]);
        }
        // Last forward sample: G_{n-1} = final_diff = 10; cdiff at t=1:
        // our 3 pts (marks 5,7 -> 2 marks = 3) − opp 1 pt (1 mark) = 2.
        // value = final_diff − cdiff = 10 − 2 = 8.
        let last_fwd = &samples[2];
        assert!((last_fwd.value - 8.0).abs() < 1e-5);
        assert!((last_fwd.final_diff - 8.0).abs() < 1e-5);
    }
}
