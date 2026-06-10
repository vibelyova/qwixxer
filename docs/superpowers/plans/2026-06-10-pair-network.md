# Pairwise Differential Network ("pair" bot) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new bot type `pair` whose value network sees *both* boards and predicts the score-differential distribution (μ_diff, σ_diff) in future space, coexisting with the current DQN.

**Architecture:** Concatenated joint MLP `45 → 128 → 64 → 2` over `[our board 20 | opp board 20 | pair-level 5]`. TD(λ=0.8) chains per (player, opponent) pair with board-swap samples materialized at construction; decoupled μ/σ loss; ranking by `(current_diff + μ)/σ` vs the leading opponent. Spec: `docs/superpowers/specs/2026-06-10-pair-network-design.md`.

**Tech Stack:** Rust, burn 0.20 (NdArray backend), rayon. New files mirror the existing `dqn/mod.rs` + `dqn/train.rs` split.

---

## File structure

```
Create: src/dqn/pair.rs        — features, PairModel, PairStrategy (Bot impl). Inference-only, ungated.
Create: src/dqn/pair_train.rs  — PairSample, batcher, TD targets, RecordingPair, self-play loop. #[cfg(feature = "dqn")].
Modify: src/dqn/mod.rs         — register the two modules (2 lines).
Modify: src/main.rs            — BotType::Pair, StrategyTemplates, PairTrain subcommand.
Modify: docs/ARCHITECTURE.md, README.md — short "pair bot" notes.
```

Conventions used throughout: run `cargo fmt` before every commit. All test commands assume default features (`dqn` + `parallel` enabled).

---

### Task 1: Features (`board_features`, `pair_features`)

**Files:**
- Create: `src/dqn/pair.rs`
- Modify: `src/dqn/mod.rs` (register module)

- [ ] **Step 1: Create `src/dqn/pair.rs` with feature constants, functions, and tests**

```rust
//! Pairwise differential network ("pair" bot): joint two-board evaluation.
//!
//! The model sees both players' boards and predicts the distribution of the
//! *future* score differential `final_diff − current_diff` as `(μ, log σ²)`.
//! Move selection ranks candidates by `(current_diff + μ) / σ`, monotone in
//! the Gaussian P(win) against the leading opponent.
//!
//! Design: docs/superpowers/specs/2026-06-10-pair-network-design.md

use super::{aggregate_weighted_probability, lockable_rows, row_progress, total_progress, MyBackend, LOG_VAR_MAX, LOG_VAR_MIN};
use crate::state::State;
use crate::strategy::Bot;
use burn::{
    nn::{Linear, LinearConfig, Relu},
    prelude::*,
    record::CompactRecorder,
};
use std::sync::Arc;

/// Per-board feature block size.
pub const BOARD_FEATURES: usize = 20;
/// Full pair input: two board blocks + 5 pair-level features.
pub const PAIR_FEATURES: usize = 45;

/// Per-board feature block. Same layout as the old net's per-board features,
/// plus per-board aggregates, and with the lock-rule fix: a free pointer
/// sitting on the row's terminal number (12/2) cannot be marked while the row
/// has <5 marks, so it contributes 0 to the weighted probability.
pub fn board_features(state: &State) -> [f32; BOARD_FEATURES] {
    let totals = state.row_totals();
    let frees = state.row_free_values();
    let locked = state.locked();

    let mut f = [0.0f32; BOARD_FEATURES];
    for i in 0..4 {
        f[i] = row_progress(frees[i], i < 2);
        f[4 + i] = totals[i] as f32 / 11.0;
        f[8 + i] = if locked[i] { 1.0 } else { 0.0 };
        f[12 + i] = match frees[i] {
            Some(fr) if fr == State::row_terminal(i) && totals[i] < 5 => 0.0,
            Some(fr) => {
                let ways = 6.0 - (7.0f32 - fr as f32).abs();
                (ways / 6.0) * (totals[i] as f32 + 1.0) / 11.0
            }
            None => 0.0,
        };
    }
    f[16] = state.strikes as f32 / 3.0;
    f[17] = state.blanks() as f32 / 40.0;
    f[18] = aggregate_weighted_probability(state);
    f[19] = lockable_rows(state) as f32 / 4.0;
    f
}

/// Full pair input. `paired` must be one of `all_opps` (the opponent we are
/// being compared against); `all_opps` are ALL opponents of the `our` player —
/// the pair-level features 42–44 are computed over all of them (uniform
/// semantics in 1v1 and multiplayer; redundant with the opp block in 1v1).
///
/// Layout: `[our 20 | paired 20 | cdiff/100, num_opps/4, max opp progress,
/// max opp strikes/3, opp lockable-rows sum/8]`.
pub fn pair_features(our: &State, paired: &State, all_opps: &[State]) -> [f32; PAIR_FEATURES] {
    let mut f = [0.0f32; PAIR_FEATURES];
    f[..BOARD_FEATURES].copy_from_slice(&board_features(our));
    f[BOARD_FEATURES..2 * BOARD_FEATURES].copy_from_slice(&board_features(paired));

    let cdiff = (our.count_points() - paired.count_points()) as f32;
    f[40] = (cdiff / 100.0).clamp(-1.0, 1.0);
    f[41] = all_opps.len() as f32 / 4.0;
    f[42] = all_opps.iter().map(total_progress).fold(0.0, f32::max);
    f[43] = all_opps.iter().map(|s| s.strikes).max().unwrap_or(0) as f32 / 3.0;
    f[44] = all_opps.iter().map(lockable_rows).sum::<u8>() as f32 / 8.0;
    f
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::Mark;

    #[test]
    fn wprob_zero_when_free_pointer_on_unlockable_terminal() {
        // Red (ascending): mark 2, 3, 11 -> free = 12 (terminal) with only 3 marks.
        let mut s = State::default();
        for n in [2u8, 3, 11] {
            s.apply_mark(Mark { row: 0, number: n });
        }
        assert_eq!(s.row_free_values()[0], Some(12));
        assert_eq!(s.row_totals()[0], 3);
        let f = board_features(&s);
        assert_eq!(f[12], 0.0, "terminal with <5 marks must contribute 0 ways");

        // Two more marks (4, 5) -> 5 marks, terminal now lockable: nonzero.
        for n in [4u8, 5] {
            s.apply_mark(Mark { row: 0, number: n });
        }
        let f = board_features(&s);
        assert!(f[12] > 0.0);
    }

    #[test]
    fn pair_features_fresh_1v1_golden() {
        let a = State::default();
        let b = State::default();
        let f = pair_features(&a, &b, &[b]);

        // Both board blocks identical for fresh boards.
        assert_eq!(f[..BOARD_FEATURES], f[BOARD_FEATURES..2 * BOARD_FEATURES]);
        // Fresh rows: progress 0, marks 0, not locked.
        for i in 0..4 {
            assert_eq!(f[i], 0.0);
            assert_eq!(f[4 + i], 0.0);
            assert_eq!(f[8 + i], 0.0);
        }
        // Fresh free pointers sit on 2 (asc) / 12 (desc): 1 way each ->
        // (1/6) * (0+1)/11 = 1/66.
        for i in 0..4 {
            assert!((f[12 + i] - 1.0 / 66.0).abs() < 1e-6);
        }
        assert_eq!(f[16], 0.0); // strikes
        assert_eq!(f[17], 0.0); // blanks
        assert!(f[18] > 0.0); // aggregate wprob
        assert_eq!(f[19], 0.0); // lockable rows
        // Pair-level: equal scores, 1 opponent, fresh opponent stats.
        assert_eq!(f[40], 0.0);
        assert_eq!(f[41], 0.25);
        assert_eq!(f[42], 0.0);
        assert_eq!(f[43], 0.0);
        assert_eq!(f[44], 0.0);
    }

    #[test]
    fn pair_features_swap_relationship() {
        // Asymmetric position: A has marks + a strike, B is fresh.
        let mut a = State::default();
        for n in [2u8, 3, 5] {
            a.apply_mark(Mark { row: 0, number: n });
        }
        a.apply_strike();
        let b = State::default();

        let ab = pair_features(&a, &b, &[b]);
        let ba = pair_features(&b, &a, &[a]);

        // Board blocks exchange.
        assert_eq!(ab[..BOARD_FEATURES], ba[BOARD_FEATURES..2 * BOARD_FEATURES]);
        assert_eq!(ab[BOARD_FEATURES..2 * BOARD_FEATURES], ba[..BOARD_FEATURES]);
        // current_diff negates.
        assert!((ab[40] + ba[40]).abs() < 1e-6);
        assert!(ab[40] != 0.0);
        // Opponent summaries describe the respective opponent.
        assert_eq!(ab[42], 0.0); // A's opponent (B) is fresh
        assert!((ba[42] - total_progress(&a)).abs() < 1e-6);
        assert!((ba[43] - 1.0 / 3.0).abs() < 1e-6); // A has 1 strike
    }
}
```

- [ ] **Step 2: Register the module in `src/dqn/mod.rs`**

Below the existing `#[cfg(feature = "dqn")] pub mod train;` add:

```rust
pub mod pair;
```

- [ ] **Step 3: Run the tests**

Run: `cargo test --lib dqn::pair`
Expected: 3 tests pass. Unused-import warnings (`Linear`, `CompactRecorder`, `Arc`, `Bot`, …) are expected — Task 2 consumes those imports; do not remove them.

- [ ] **Step 4: Commit**

```bash
cargo fmt && git add src/dqn/pair.rs src/dqn/mod.rs && git commit -m "feat(pair): two-board feature extraction with lock-rule fix"
```

---

### Task 2: PairModel + PairStrategy (Bot impl)

**Files:**
- Modify: `src/dqn/pair.rs`

- [ ] **Step 1: Append model, batched forward, and strategy to `src/dqn/pair.rs`**

```rust
// ---- Model ----

#[derive(Module, Debug)]
pub struct PairModel<B: Backend> {
    layer1: Linear<B>,
    layer2: Linear<B>,
    output_mean: Linear<B>,
    output_log_var: Linear<B>,
    activation: Relu,
}

#[derive(Config, Debug)]
pub struct PairModelConfig {
    #[config(default = 128)]
    pub hidden1: usize,
    #[config(default = 64)]
    pub hidden2: usize,
}

impl PairModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PairModel<B> {
        PairModel {
            layer1: LinearConfig::new(PAIR_FEATURES, self.hidden1).with_bias(true).init(device),
            layer2: LinearConfig::new(self.hidden1, self.hidden2).with_bias(true).init(device),
            output_mean: LinearConfig::new(self.hidden2, 1).with_bias(true).init(device),
            output_log_var: LinearConfig::new(self.hidden2, 1).with_bias(true).init(device),
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> PairModel<B> {
    /// Forward pass returning `[batch, 2]`: col 0 = μ_diff (future-space),
    /// col 1 = raw `log σ²_diff` (clamped by callers / the loss).
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.activation.forward(self.layer1.forward(input));
        let x = self.activation.forward(self.layer2.forward(x));
        let mean = self.output_mean.forward(x.clone());
        let log_var = self.output_log_var.forward(x);
        Tensor::cat(vec![mean, log_var], 1)
    }
}

/// Run the model on a batch of pair-feature vectors in one forward pass.
/// Returns `(μ, log σ²)` per input row.
pub fn pair_batch_forward(
    model: &PairModel<MyBackend>,
    device: &burn::backend::ndarray::NdArrayDevice,
    features_list: &[[f32; PAIR_FEATURES]],
) -> Vec<(f32, f32)> {
    if features_list.is_empty() {
        return Vec::new();
    }
    let n = features_list.len();
    let flat: Vec<f32> = features_list.iter().flat_map(|f| f.iter().copied()).collect();
    let input = Tensor::<MyBackend, 1>::from_floats(flat.as_slice(), device).reshape([n, PAIR_FEATURES]);
    let output = model.forward(input);
    let values = output.into_data().to_vec::<f32>().unwrap();
    (0..n).map(|i| (values[2 * i], values[2 * i + 1])).collect()
}

// ---- Strategy ----

pub struct PairStrategy {
    pub model: Arc<PairModel<MyBackend>>,
    pub device: burn::backend::ndarray::NdArrayDevice,
}

impl std::fmt::Debug for PairStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "PairStrategy")
    }
}

impl PairStrategy {
    pub fn load(artifact_dir: &str) -> Self {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let model = PairModelConfig::new()
            .init::<MyBackend>(&device)
            .load_file(format!("{artifact_dir}/model"), &CompactRecorder::new(), &device)
            .expect("Failed to load pair model");
        PairStrategy { model: Arc::new(model), device }
    }

    pub fn from_model(model: PairModel<MyBackend>, device: burn::backend::ndarray::NdArrayDevice) -> Self {
        PairStrategy { model: Arc::new(model), device }
    }

    pub fn from_shared(model: Arc<PairModel<MyBackend>>, device: burn::backend::ndarray::NdArrayDevice) -> Self {
        PairStrategy { model, device }
    }
}

impl Bot for PairStrategy {
    fn evaluate(&self, our_state: &State, opp_states: &[State]) -> f32 {
        self.evaluate_batch(&[*our_state], opp_states)[0]
    }

    fn evaluate_batch(&self, candidates: &[State], opp_states: &[State]) -> Vec<f32> {
        // Solo fallback: rank against an empty default board. Solo play is
        // officially unsupported for the pair bot (the old DQN covers it).
        let default_opps;
        let opp_states = if opp_states.is_empty() {
            default_opps = [State::default()];
            &default_opps[..]
        } else {
            opp_states
        };
        let leader = opp_states.iter().max_by_key(|s| s.count_points()).unwrap();
        let leader_points = leader.count_points();

        let feats: Vec<[f32; PAIR_FEATURES]> =
            candidates.iter().map(|c| pair_features(c, leader, opp_states)).collect();
        pair_batch_forward(&self.model, &self.device, &feats)
            .into_iter()
            .zip(candidates)
            .map(|((mu, log_var), cand)| {
                let cdiff = (cand.count_points() - leader_points) as f32;
                let sigma = (0.5 * log_var.clamp(LOG_VAR_MIN, LOG_VAR_MAX)).exp();
                (cdiff + mu) / sigma
            })
            .collect()
    }
}
```

- [ ] **Step 2: Add tests to the `tests` module in `src/dqn/pair.rs`**

```rust
    #[test]
    fn evaluate_batch_shape_and_determinism() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let bot = PairStrategy::from_model(PairModelConfig::new().init::<MyBackend>(&device), device);

        let mut cand = State::default();
        cand.apply_mark(Mark { row: 1, number: 5 });
        let candidates = [State::default(), cand];
        let opps = [State::default()];

        let v1 = bot.evaluate_batch(&candidates, &opps);
        let v2 = bot.evaluate_batch(&candidates, &opps);
        assert_eq!(v1.len(), 2);
        assert_eq!(v1, v2);
        assert!(v1.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn evaluate_handles_empty_opponents() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let bot = PairStrategy::from_model(PairModelConfig::new().init::<MyBackend>(&device), device);
        let v = bot.evaluate(&State::default(), &[]);
        assert!(v.is_finite());
    }
```

- [ ] **Step 3: Run the tests**

Run: `cargo test --lib dqn::pair`
Expected: 5 tests pass.

- [ ] **Step 4: Commit**

```bash
cargo fmt && git add src/dqn/pair.rs && git commit -m "feat(pair): PairModel and PairStrategy with diff-based P(win) ranking"
```

---

### Task 3: TD diff targets (pure function)

**Files:**
- Create: `src/dqn/pair_train.rs`
- Modify: `src/dqn/mod.rs` (register module)

- [ ] **Step 1: Create `src/dqn/pair_train.rs` with the target math and its test**

```rust
//! Training for the pairwise differential network: per-opponent TD(λ) chains,
//! board-swap sample doubling, decoupled μ/σ loss, pure self-play loop.
//!
//! Only compiled with the `dqn` feature, mirroring `dqn::train`.

use crate::bot::{self, DNA};
use crate::dqn::pair::{pair_batch_forward, pair_features, PairModel, PairModelConfig, PairStrategy, BOARD_FEATURES, PAIR_FEATURES};
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
}
```

- [ ] **Step 2: Register in `src/dqn/mod.rs`**

Next to the existing train registration:

```rust
#[cfg(feature = "dqn")]
pub mod pair_train;
```

- [ ] **Step 3: Run the test**

Run: `cargo test --lib dqn::pair_train`
Expected: 1 test passes (many unused-import warnings — later tasks consume them).

- [ ] **Step 4: Commit**

```bash
cargo fmt && git add src/dqn/pair_train.rs src/dqn/mod.rs && git commit -m "feat(pair): TD(lambda) targets over diff chains"
```

---

### Task 4: PairSample, color-permutation batcher, training step

**Files:**
- Modify: `src/dqn/pair_train.rs`

- [ ] **Step 1: Append sample type, loss, and batcher**

```rust
// ---- Training samples / loss / batcher ----

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct PairSample {
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

        RegressionOutput { loss, output: mean, targets }
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

        PairBatch { inputs, targets, final_diffs }
    }
}
```

- [ ] **Step 2: Add the permutation test to the `tests` module**

```rust
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
```

- [ ] **Step 3: Run the tests**

Run: `cargo test --lib dqn::pair_train`
Expected: 2 tests pass.

- [ ] **Step 4: Commit**

```bash
cargo fmt && git add src/dqn/pair_train.rs && git commit -m "feat(pair): sample type, decoupled diff loss, dual-block color augmentation"
```

---

### Task 5: RecordingPair + chain/sample construction with swap doubling

**Files:**
- Modify: `src/dqn/pair_train.rs`

- [ ] **Step 1: Append the recording strategy and sample builder**

```rust
// ---- Self-play recording ----

/// One recorded decision: our post-decision state + all opponents' states at
/// that moment (turn-ordered relative to us, constant order per game).
type Snapshot = (State, Vec<State>);

/// Pair-bot wrapper used during self-play training. ε-greedy on active
/// decisions, greedy on passive; records snapshots for chain building.
/// Recording cadence: active turns once after phase 2 (post-turn state);
/// passive turns after every real decision — including skips, which the old
/// recorder dropped (skip afterstates are evaluated at inference, so they
/// belong in the training distribution).
struct RecordingPair {
    bot: PairStrategy,
    epsilon: f32,
    rng: SmallRng,
    recorded: std::rc::Rc<std::cell::RefCell<Vec<Snapshot>>>,
}

impl std::fmt::Debug for RecordingPair {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "RecordingPair")
    }
}

impl Strategy for RecordingPair {
    fn active_phase1(&mut self, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
        if self.rng.gen::<f32>() < self.epsilon {
            let white_marks = state.generate_white_moves(dice[0] + dice[1]);
            if white_marks.is_empty() {
                return None;
            }
            let idx = self.rng.gen_range(0..=white_marks.len());
            if idx < white_marks.len() {
                Some(white_marks[idx])
            } else {
                None
            }
        } else {
            crate::strategy::active_phase1_impl(&self.bot, state, opp_states, dice)
        }
    }

    fn active_phase2(&mut self, state: &State, opp_states: &[State], dice: [u8; 6], has_marked: bool) -> Option<Mark> {
        let marks = state.generate_color_moves(dice);
        let no_mark_state = if has_marked {
            *state
        } else {
            let mut s = *state;
            s.apply_strike();
            s
        };

        if marks.is_empty() {
            self.recorded.borrow_mut().push((no_mark_state, opp_states.to_vec()));
            return None;
        }

        let mark = if self.rng.gen::<f32>() < self.epsilon {
            let idx = self.rng.gen_range(0..=marks.len());
            if idx < marks.len() {
                Some(marks[idx])
            } else {
                None
            }
        } else {
            crate::strategy::active_phase2_impl(&self.bot, state, opp_states, dice, has_marked)
        };

        let chosen_state = match mark {
            Some(m) => {
                let mut s = *state;
                s.apply_mark(m);
                s
            }
            None => no_mark_state,
        };
        self.recorded.borrow_mut().push((chosen_state, opp_states.to_vec()));
        mark
    }

    fn passive_phase1(
        &mut self,
        state: &State,
        opp_states: &[State],
        dice: [u8; 6],
        _active_player: usize,
    ) -> Option<Mark> {
        let marks = state.generate_white_moves(dice[0] + dice[1]);
        if marks.is_empty() {
            return None;
        }

        let mark = crate::strategy::passive_phase1_impl(&self.bot, state, opp_states, dice);

        let post = match mark {
            Some(m) => {
                let mut s = *state;
                s.apply_mark(m);
                s
            }
            None => *state, // record skips too
        };
        self.recorded.borrow_mut().push((post, opp_states.to_vec()));
        mark
    }
}

/// Build training samples from one player's trajectory: one TD(λ) chain per
/// opponent, every sample emitted in both board orders (swap doubling) with
/// pair-level features recomputed exactly from the swapped perspective and
/// targets negated.
fn build_pair_samples(
    model: &PairModel<MyBackend>,
    device: &burn::backend::ndarray::NdArrayDevice,
    snapshots: &[Snapshot],
    our_final: f32,
    opp_finals: &[f32],
) -> Vec<PairSample> {
    let mut samples = Vec::new();
    if snapshots.is_empty() {
        return samples;
    }
    let num_opps = snapshots[0].1.len();
    debug_assert_eq!(num_opps, opp_finals.len());

    for k in 0..num_opps {
        let final_diff = our_final - opp_finals[k];
        let feats: Vec<[f32; PAIR_FEATURES]> = snapshots
            .iter()
            .map(|(our, opps)| pair_features(our, &opps[k], opps))
            .collect();
        let cdiffs: Vec<f32> = snapshots
            .iter()
            .map(|(our, opps)| (our.count_points() - opps[k].count_points()) as f32)
            .collect();
        let mus: Vec<f32> = pair_batch_forward(model, device, &feats).into_iter().map(|(m, _)| m).collect();
        let g = td_diff_targets(&mus, &cdiffs, final_diff, LAMBDA);

        for (t, (our, opps)) in snapshots.iter().enumerate() {
            let value = g[t] - cdiffs[t];
            let fdiff = final_diff - cdiffs[t];
            samples.push(PairSample { features: feats[t], value, final_diff: fdiff });

            // Swapped sample: the paired opponent's perspective. Their
            // opponents are us plus the remaining opponents.
            let mut swapped_opps: Vec<State> = Vec::with_capacity(num_opps);
            swapped_opps.push(*our);
            swapped_opps.extend(opps.iter().enumerate().filter(|(j, _)| *j != k).map(|(_, s)| *s));
            samples.push(PairSample {
                features: pair_features(&opps[k], our, &swapped_opps),
                value: -value,
                final_diff: -fdiff,
            });
        }
    }
    samples
}
```

- [ ] **Step 2: Add the swap-doubling test to the `tests` module**

```rust
    #[test]
    fn build_pair_samples_emits_negated_swapped_samples() {
        use crate::dqn::pair::BOARD_FEATURES;
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let model = PairModelConfig::new().init::<MyBackend>(&device);

        // 2-step 1v1 trajectory with asymmetric boards.
        let mut our1 = State::default();
        our1.apply_mark(crate::state::Mark { row: 0, number: 5 });
        let opp1 = State::default();
        let mut our2 = our1;
        our2.apply_mark(crate::state::Mark { row: 0, number: 7 });
        let mut opp2 = State::default();
        opp2.apply_mark(crate::state::Mark { row: 2, number: 10 });

        let snapshots = vec![(our1, vec![opp1]), (our2, vec![opp2])];
        let samples = build_pair_samples(&model, &device, &snapshots, 30.0, &[20.0]);

        // 2 steps × 1 opponent × 2 orders.
        assert_eq!(samples.len(), 4);
        for pair in samples.chunks(2) {
            let (fwd, swp) = (&pair[0], &pair[1]);
            assert_eq!(swp.value, -fwd.value);
            assert_eq!(swp.final_diff, -fwd.final_diff);
            // Board blocks exchanged.
            assert_eq!(fwd.features[..BOARD_FEATURES], swp.features[BOARD_FEATURES..2 * BOARD_FEATURES]);
            assert_eq!(fwd.features[BOARD_FEATURES..2 * BOARD_FEATURES], swp.features[..BOARD_FEATURES]);
            // cdiff input negated (within clamp range here).
            assert!((fwd.features[40] + swp.features[40]).abs() < 1e-6);
        }
        // Last forward sample's targets: G_{n−1} = final_diff = 10;
        // cdiff at t=1: our 3 pts (2 marks) − opp 1 pt (1 mark) = 2.
        let last_fwd = &samples[2];
        assert!((last_fwd.value - 8.0).abs() < 1e-5);
        assert!((last_fwd.final_diff - 8.0).abs() < 1e-5);
    }
```

- [ ] **Step 3: Run the tests**

Run: `cargo test --lib dqn::pair_train`
Expected: 3 tests pass.

- [ ] **Step 4: Commit**

```bash
cargo fmt && git add src/dqn/pair_train.rs && git commit -m "feat(pair): recording strategy and per-opponent chains with swap doubling"
```

---

### Task 6: Self-play training loop + benchmark

**Files:**
- Modify: `src/dqn/pair_train.rs`

- [ ] **Step 1: Append game runner, benchmark, training loop**

```rust
// ---- Self-play training loop ----

/// Play one pure-self-play training game; every player is a recording pair
/// bot (player 0 explores with ε, the rest are greedy). Returns all samples
/// plus player 0's final score.
fn play_training_game(
    model: &PairModel<MyBackend>,
    device: &burn::backend::ndarray::NdArrayDevice,
    num_opponents: usize,
    epsilon: f32,
    seed: u64,
) -> (Vec<PairSample>, f32) {
    use crate::game::{Game, Player};

    let n = num_opponents + 1;
    let mut buffers = Vec::with_capacity(n);
    let mut players: Vec<Player> = Vec::with_capacity(n);
    for i in 0..n {
        let buf = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        buffers.push(std::rc::Rc::clone(&buf));
        players.push(Player::new(
            Box::new(RecordingPair {
                bot: PairStrategy::from_model(model.clone(), device.clone()),
                epsilon: if i == 0 { epsilon } else { 0.0 },
                rng: SmallRng::seed_from_u64(seed.wrapping_add(100 + i as u64)),
                recorded: buf,
            }),
            Box::new(SmallRng::seed_from_u64(seed.wrapping_add(i as u64))),
        ));
    }

    let mut game = Game::new(players);
    game.play();

    let finals: Vec<f32> = game.players.iter().map(|p| p.state.count_points() as f32).collect();

    let mut all_samples = Vec::new();
    for (i, buf) in buffers.iter().enumerate() {
        let snapshots = std::mem::take(&mut *buf.borrow_mut());
        // Opponent k of player i is player (i + 1 + k) % n — matches the
        // turn-ordered opp_states the game loop passes to strategies.
        let opp_finals: Vec<f32> = (1..n).map(|off| finals[(i + off) % n]).collect();
        all_samples.extend(build_pair_samples(model, device, &snapshots, finals[i], &opp_finals));
    }

    (all_samples, finals[0])
}

/// Same fixed-game-set paired benchmark as `train::benchmark_vs_ga`, for the
/// pair bot. Seat-swapped pairs share per-seat dice streams.
const BENCH_SEED: u64 = 0xB54C;

fn benchmark_vs_ga(artifact_dir: &str, champion: &DNA, num_games: usize) -> f64 {
    use crate::game::{Game, Player};

    let wins: u32 = (0..num_games)
        .into_par_iter()
        .map_init(
            || PairStrategy::load(artifact_dir),
            |template, i| {
                let bot = PairStrategy::from_shared(template.model.clone(), template.device.clone());
                let pair = (i / 2) as u64;
                let rotation = i % 2;
                let seat_dice =
                    |seat: u64| Box::new(SmallRng::seed_from_u64(BENCH_SEED.wrapping_add(pair * 2 + seat)));
                let players: Vec<Player> = if rotation == 0 {
                    vec![
                        Player::new(Box::new(bot), seat_dice(0)),
                        Player::new(Box::new(champion.clone()), seat_dice(1)),
                    ]
                } else {
                    vec![
                        Player::new(Box::new(champion.clone()), seat_dice(0)),
                        Player::new(Box::new(bot), seat_dice(1)),
                    ]
                };
                let mut game = Game::new(players);
                game.play();
                let scores: Vec<isize> = game.players.iter().map(|p| p.state.count_points()).collect();
                let idx = rotation;
                if scores[idx] > scores[1 - idx] {
                    1u32
                } else {
                    0u32
                }
            },
        )
        .sum();
    wins as f64 / num_games as f64
}

pub fn self_play_train(
    artifact_dir: &str,
    num_iterations: usize,
    games_per_iteration: usize,
    epochs_per_iteration: usize,
    bench_games: usize,
    checkpoints: bool,
    start_iteration: usize,
) {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    MyBackend::seed(&device, TRAIN_SEED);
    std::fs::create_dir_all(artifact_dir).ok();
    let buffer_iterations = 3;
    let mut replay_buffer: std::collections::VecDeque<Vec<PairSample>> = std::collections::VecDeque::new();

    let scores_log_path = format!("{artifact_dir}/training_scores.csv");
    if start_iteration == 0 {
        std::fs::write(&scores_log_path, "iteration,avg_score,winrate\n").ok();
    }

    let genes = Arc::new(bot::default_genes());
    let champion = DNA::load_weights("champion.txt", genes).expect("No champion.txt");
    let start_time = std::time::Instant::now();

    let mut iteration_stats: Vec<(usize, f32, Option<f64>)> = Vec::new();

    let mut model: PairModel<MyBackend> = PairModelConfig::new()
        .init::<MyBackend>(&device)
        .load_file(format!("{artifact_dir}/model"), &CompactRecorder::new(), &device)
        .unwrap_or_else(|_| {
            println!("  No pretrained pair model, starting fresh");
            PairModelConfig::new().init::<MyBackend>(&device)
        });

    for iteration in 0..num_iterations {
        let global_iter = start_iteration + iteration;
        let epsilon = (0.2 * (0.95f32).powi(global_iter as i32)).max(0.07);
        println!("\n=== Pair iteration {} (epsilon={epsilon:.3}) ===", global_iter + 1);

        let games_each = games_per_iteration / 3;
        // Pure self-play thirds: 1v1, 3p, 4p.
        let game_configs: Vec<usize> = [1usize, 2, 3]
            .iter()
            .flat_map(|&num_opps| std::iter::repeat(num_opps).take(games_each))
            .collect();

        let models: Vec<PairModel<MyBackend>> = (0..game_configs.len()).map(|_| model.clone()).collect();

        let game_results: Vec<(Vec<PairSample>, f32)> = game_configs
            .into_par_iter()
            .zip(models.into_par_iter())
            .enumerate()
            .map(|(game_idx, (num_opps, thread_model))| {
                let seed = TRAIN_SEED.wrapping_add((iteration * games_per_iteration + game_idx) as u64);
                play_training_game(&thread_model, &device, num_opps, epsilon, seed)
            })
            .collect();

        let game_scores: Vec<f32> = game_results.iter().map(|(_, score)| *score).collect();
        let new_samples: Vec<PairSample> = game_results.into_iter().flat_map(|(s, _)| s).collect();
        let avg_score = if game_scores.is_empty() {
            0.0
        } else {
            game_scores.iter().sum::<f32>() / game_scores.len() as f32
        };

        replay_buffer.push_back(new_samples);
        if replay_buffer.len() > buffer_iterations {
            replay_buffer.pop_front();
        }

        let all_samples: Vec<PairSample> = replay_buffer.iter().flatten().copied().collect();
        println!(
            "  Generated {} new samples (avg score: {avg_score:.1}), replay buffer: {} total",
            replay_buffer.back().unwrap().len(),
            all_samples.len()
        );

        model = train_with_epochs(all_samples, artifact_dir, epochs_per_iteration, 4e-4);

        if checkpoints {
            let src = format!("{artifact_dir}/model.mpk");
            let dst = format!("{artifact_dir}/iter-{}.mpk", global_iter + 1);
            if let Err(e) = std::fs::copy(&src, &dst) {
                eprintln!("  Failed to save iter-{} checkpoint: {e}", global_iter + 1);
            }
        }

        let elapsed = start_time.elapsed().as_secs();
        let (mins, secs) = (elapsed / 60, elapsed % 60);

        use std::io::Write;
        let winrate_opt = if bench_games > 0 {
            let winrate = benchmark_vs_ga(artifact_dir, &champion, bench_games);
            println!(
                "  Iteration {:>3}: avg score {:.1}, winrate {:.1}%, elapsed {}m{}s",
                global_iter + 1,
                avg_score,
                winrate * 100.0,
                mins,
                secs,
            );
            if let Ok(mut f) = std::fs::OpenOptions::new().append(true).open(&scores_log_path) {
                writeln!(f, "{},{avg_score:.2},{:.2}", global_iter + 1, winrate * 100.0).ok();
            }
            Some(winrate)
        } else {
            println!(
                "  Iteration {:>3}: avg score {:.1}, elapsed {}m{}s",
                global_iter + 1,
                avg_score,
                mins,
                secs,
            );
            if let Ok(mut f) = std::fs::OpenOptions::new().append(true).open(&scores_log_path) {
                writeln!(f, "{},{avg_score:.2}", global_iter + 1).ok();
            }
            None
        };

        iteration_stats.push((global_iter + 1, avg_score, winrate_opt));
    }

    println!("\nPair self-play training complete. Model saved to {artifact_dir}/model");

    if bench_games > 0 {
        let best_wr = iteration_stats
            .iter()
            .filter_map(|&(i, s, w)| w.map(|w| (i, s, w)))
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
        let best_score = iteration_stats.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        if let Some((i, s, w)) = best_wr {
            println!("  Best winrate:    iter {i:>3}: {:.1}% (avg score {s:.1})", w * 100.0);
        }
        if let Some(&(i, s, w)) = best_score {
            let wr_str = w.map(|w| format!("{:.1}%", w * 100.0)).unwrap_or_else(|| "-".into());
            println!("  Best avg score:  iter {i:>3}: {s:.1} (winrate {wr_str})");
        }
    }
}

fn train_with_epochs(samples: Vec<PairSample>, artifact_dir: &str, num_epochs: usize, lr: f64) -> PairModel<MyBackend> {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;

    let split = (samples.len() * 9) / 10;
    let train_data = InMemDataset::new(samples[..split].to_vec());
    let valid_data = InMemDataset::new(samples[split..].to_vec());

    let model: PairModel<MyAutodiffBackend> = PairModelConfig::new()
        .init::<MyAutodiffBackend>(&device)
        .load_file(format!("{artifact_dir}/model"), &CompactRecorder::new(), &device)
        .unwrap_or_else(|_| PairModelConfig::new().init::<MyAutodiffBackend>(&device));

    let batcher_train = PairBatcher::<MyAutodiffBackend> {
        _phantom: std::marker::PhantomData,
    };
    let batcher_valid = PairBatcher::<MyBackend> {
        _phantom: std::marker::PhantomData,
    };

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(1024)
        .shuffle(TRAIN_SEED)
        .build(train_data);

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(1024)
        .shuffle(TRAIN_SEED)
        .build(valid_data);

    let ckpt_dir = format!("{artifact_dir}/ckpt");
    std::fs::remove_dir_all(&ckpt_dir).ok();
    std::fs::create_dir_all(&ckpt_dir).ok();

    let training = SupervisedTraining::new(&ckpt_dir, dataloader_train, dataloader_valid)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .num_epochs(num_epochs)
        .summary();

    let result = training.launch(Learner::new(model, AdamConfig::new().init(), lr));

    result
        .model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Failed to save pair model");

    std::fs::remove_dir_all(&ckpt_dir).ok();

    PairModelConfig::new()
        .init::<MyBackend>(&device)
        .load_file(format!("{artifact_dir}/model"), &CompactRecorder::new(), &device)
        .expect("Failed to reload pair model for inference")
}
```

- [ ] **Step 2: Verify it compiles and existing tests pass**

Run: `cargo test --lib dqn::pair`
Expected: all pair + pair_train tests pass; no warnings about unused imports remain.

- [ ] **Step 3: Commit**

```bash
cargo fmt && git add src/dqn/pair_train.rs && git commit -m "feat(pair): pure self-play training loop with fixed-game-set benchmark"
```

---

### Task 7: CLI integration

**Files:**
- Modify: `src/main.rs`

- [ ] **Step 1: Add `Pair` to `BotType` and its `Display`**

In the `BotType` enum add `Pair,` after `Dqn,`. In the `Display` impl add:

```rust
            BotType::Pair => write!(f, "PAIR"),
```

- [ ] **Step 2: Wire `make_strategy` and `StrategyTemplates`**

In `make_strategy`, add an arm:

```rust
        BotType::Pair => Box::new(dqn::pair::PairStrategy::load("pair_model")),
```

In `StrategyTemplates`, add a field and handling:

```rust
struct StrategyTemplates {
    dqn: Option<dqn::DqnStrategy>,
    pair: Option<dqn::pair::PairStrategy>,
    champion: Option<bot::DNA>,
}
```

In `StrategyTemplates::new`, add before the struct literal:

```rust
        let needs_pair = bots.iter().any(|b| matches!(b, BotType::Pair));
```

and in the literal:

```rust
            pair: if needs_pair {
                Some(dqn::pair::PairStrategy::load("pair_model"))
            } else {
                None
            },
```

In `StrategyTemplates::create`, add an arm:

```rust
            BotType::Pair => {
                let t = self.pair.as_ref().unwrap();
                Box::new(dqn::pair::PairStrategy::from_shared(t.model.clone(), t.device.clone()))
            }
```

- [ ] **Step 3: Add the `PairTrain` subcommand**

In `Commands`, after `DqnSelfplay { ... }`:

```rust
    /// Pair-network self-play reinforcement learning
    #[cfg(feature = "dqn")]
    PairTrain {
        /// Number of iterations
        #[arg(short, long, default_value = "40")]
        iterations: usize,
        /// Games per iteration
        #[arg(short, long, default_value = "20000")]
        games: usize,
        /// Benchmark games per iteration (0 to disable)
        #[arg(short, long, default_value = "0")]
        bench: usize,
        /// Save per-iteration checkpoints as iter-N.mpk
        #[arg(short, long)]
        checkpoints: bool,
        /// Starting iteration offset (for epsilon schedule when resuming)
        #[arg(long, default_value = "0")]
        start_iteration: usize,
    },
```

(Note: `start_iteration` uses only a long flag — `-s` would collide with nothing here, but DqnSelfplay uses `-s`; keep long-only for clarity.)

In `main()`'s match:

```rust
        #[cfg(feature = "dqn")]
        Some(Commands::PairTrain {
            iterations,
            games,
            bench,
            checkpoints,
            start_iteration,
        }) => dqn::pair_train::self_play_train("pair_model", iterations, games, 10, bench, checkpoints, start_iteration),
```

- [ ] **Step 4: Verify compile + full test suite**

Run: `cargo test 2>&1 | grep "test result"`
Expected: all suites green (41 lib + 2 bin + new pair tests).

- [ ] **Step 5: Commit**

```bash
cargo fmt && git add src/main.rs && git commit -m "feat(pair): CLI — pair bot type and pair-train subcommand"
```

---

### Task 8: End-to-end smoke, docs, baseline run

**Files:**
- Modify: `docs/ARCHITECTURE.md`, `README.md`

- [ ] **Step 1: Tiny end-to-end training smoke test**

Run: `cargo run --release -- pair-train -i 1 -g 60 -b 200`
Expected: completes without panic; prints sample counts (roughly: 60 games × ~20–40 decisions × chains × 2 swap ≈ tens of thousands), one training pass, a winrate line (near-random model: anywhere 20–50%), and `pair_model/model.mpk` exists afterward.

- [ ] **Step 2: Bench smoke on the untrained model**

Run: `cargo run --release -- bench pair ga -n 1000`
Expected: completes; PAIR winrate is poor (untrained) but games are legal and the paired CI prints.

Run: `cargo run --release -- bench pair dqn -n 1000`
Expected: completes (exercises both nets in one process).

- [ ] **Step 3: Docs**

In `README.md`, add to the Strategies table:

```markdown
| **Pair** | Joint two-board net ranking by score-differential P(win) (see docs/superpowers/specs/2026-06-10-pair-network-design.md) | experimental |
```

and to the Usage section:

```bash
cargo run --release -- pair-train -i 40 -b 100000 -c   # Train the pair network
cargo run --release -- bench pair dqn -n 100000        # Head-to-head vs old DQN
```

In `docs/ARCHITECTURE.md`, after the DQN section add:

```markdown
### Pair Network (`dqn/pair.rs`, `dqn/pair_train.rs`)

Experimental successor to the DQN: a joint two-board MLP (45 -> 128 -> 64 -> 2)
that predicts the distribution of the *future score differential*
`final_diff - current_diff` as `(mu, log sigma^2)`. Inputs are two 20-feature
board blocks (per-row progress/marks/locked/weighted-prob with the lock-rule
fix, strikes, blanks, aggregate wprob, lockable rows) plus 5 pair-level
features (current diff, opponent count, all-opponent summaries). Candidates
are ranked by `(current_diff + mu) / sigma` against the leading opponent.
Training is pure self-play with per-opponent TD(lambda=0.8) chains; every
sample is also emitted board-swapped with negated targets (pairwise
consistency); color permutations are applied in the batcher to both blocks.
Passive skips are recorded (the old recorder dropped them). Model dir:
`pair_model/`. Design doc: `docs/superpowers/specs/2026-06-10-pair-network-design.md`.
```

- [ ] **Step 4: Commit**

```bash
cargo fmt && git add README.md docs/ARCHITECTURE.md && git commit -m "docs: pair network usage and architecture notes"
```

- [ ] **Step 5: Real training run (long-running; run detached / let the user decide timing)**

Run: `cargo run --release -- pair-train -i 40 -b 100000 -c`
Expected: ~similar wall-clock per iteration to `dqn-selfplay` (sample volume is higher: per-opponent chains × swap doubling; if iterations are unacceptably slow, reduce to `-b 50000` — pairing keeps iteration comparisons exact either way).

- [ ] **Step 6: Acceptance benchmarks (after training)**

```bash
cargo run --release -- bench pair ga -n 100000          # target: >= 59.10%
cargo run --release -- bench pair dqn -n 100000         # target: > 50%
```

Record results in `docs/EXPERIMENTS.md` (new "Phase 12: Pairwise differential network" section) with the chosen checkpoint, and commit `pair_model/model.mpk` if accepted.

---

## Execution notes

- Tasks 1–6 are purely additive; nothing existing changes behavior. Task 7 touches `main.rs` only additively. The old DQN, its model, and the web crate are untouched.
- `dqn::pair` is ungated (inference-only, like the rest of `dqn/mod.rs`); `dqn::pair_train` is `#[cfg(feature = "dqn")]`. The web crate doesn't reference either, so it keeps compiling.
- `load_from_bytes` (wasm embedding) is deliberately omitted — YAGNI until the pair bot replaces the DQN (small amendment to the spec's "mirrors DqnStrategy" list).
- Checkpoint selection after training: use best-winrate iteration (per-iteration benchmarks share a fixed game set, so the curve is luck-free); confirm the chosen checkpoint on a *different* bench seed before committing it, to avoid selection overfitting to the fixed set.
