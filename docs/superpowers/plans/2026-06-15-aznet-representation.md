# Arm A — `aznet` Shared-Encoder One-Hot Value Net Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new afterstate value net (`aznet`) that encodes each board with one-hot crossing-order per-row blocks through a shared row encoder, drop it into the existing pair-train pipeline (unchanged TD/loss/swap/ranker/search/bench), and train it 2-player-only plain — to test whether the richer representation beats the current hand-scalar pair net.

**Architecture:** Two new files, `src/dqn/aznet.rs` (features + `AzModel` + `AzStrategy`, implementing `Bot` + `WinProb`) and `src/dqn/aznet_train.rs` (sample type, batcher, self-play loop), mirroring `pair.rs`/`pair_train.rs`. The diff-space afterstate contract, TD(λ) targets, decoupled μ/σ loss, board-swap doubling, `(cdiff+μ)/σ` ranking, `SearchBot`, and the paired-CRN bench are all reused. Inference goes through burn (no hand-rolled kernel). `pair.rs` is untouched; `pair_train.rs` only gets a `pub(crate)` promotion.

**Tech Stack:** Rust, `burn` (NdArray backend, autodiff for training), `rayon`, `clap`. Design spec: `docs/superpowers/specs/2026-06-15-aznet-representation-design.md`.

**Conventions for every task:**
- Run a single test with: `cargo test --features dqn <test_name> -- --exact --nocapture` (or drop `--exact` to match a prefix).
- Run a file's module tests: `cargo test --features dqn dqn::aznet:: ` / `dqn::aznet_train::`.
- The crate's default features include `dqn` + `parallel`; `cargo test` alone works, but pass `--features dqn` explicitly so training code is always compiled.
- Commit after each task with the message shown in its final step.

---

## Reference: layout constants (used throughout)

```
ROW_BLOCK   = 28     per-row block
BOARD_RAW   = 116    4*28 rows + 4 strikes
AZ_FEATURES = 233    2*116 boards + 1 cdiff

per-row block (28), indices within a block:
  [ 0..13)  count one-hot          (total ∈ 0..=12, index = min(total,12))
  [13..24)  free-pointer slot one-hot (slot 0..10; ALL-ZERO if locked)
  [24]      is_locked
  [25]      is_lockable  (free == Some(terminal) ∧ total ≥ 5)
  [26]      wprob scalar
  [27]      blanks scalar ((free_slot − count).max(0) / 10; 0 if locked)

flat AZ_FEATURES (233):
  [  0..112)  own 4 row blocks
  [112..116)  own strikes one-hot (index = min(strikes,3))
  [116..228)  opp 4 row blocks
  [228..232)  opp strikes one-hot
  [232]       cdiff/100 clamped to [-1,1]
```

Row→direction: rows 0,1 ascending (R,Y); rows 2,3 descending (G,B). Free-slot:
ascending `slot = free − 2`; descending `slot = 12 − free`.

---

## Task 1: Feature extraction (`aznet.rs` — constants + `az_row_block`/`az_board`/`az_features`)

**Files:**
- Create: `src/dqn/aznet.rs`
- Modify: `src/dqn/mod.rs` (add `pub mod aznet;` after the `pair` decl on line 24)

- [ ] **Step 1: Register the module**

In `src/dqn/mod.rs`, directly below `pub mod pair;` (line 24), add:

```rust
pub mod aznet;
```

- [ ] **Step 2: Write the failing feature tests**

Create `src/dqn/aznet.rs` with the imports, constants, function stubs, and the test module below. The stubs let the file compile so the tests can run and **fail on assertions**.

```rust
//! "aznet" value net: AlphaZero-style one-hot crossing-order representation
//! run through a shared per-row encoder. Afterstate value net like the pair
//! net (diff-space μ/σ heads, `(cdiff+μ)/σ` ranking), differing only in the
//! board encoding. 2-player only. Design:
//! docs/superpowers/specs/2026-06-15-aznet-representation-design.md

use crate::state::State;

/// Per-row one-hot block width.
pub const ROW_BLOCK: usize = 28;
/// Per-board raw width: 4 row blocks + one-hot strikes.
pub const BOARD_RAW: usize = 4 * ROW_BLOCK + 4; // 116
/// Full input: two boards + cdiff.
pub const AZ_FEATURES: usize = 2 * BOARD_RAW + 1; // 233

/// One-hot crossing-order block for row `i` of `state` (afterstate; no roll).
pub fn az_row_block(state: &State, i: usize) -> [f32; ROW_BLOCK] {
    let total = state.row_totals()[i];
    let free = state.row_free_values()[i];
    let ascending = i < 2;
    let terminal = State::row_terminal(i);
    let mut b = [0.0f32; ROW_BLOCK];

    // count one-hot [0..13)
    b[(total as usize).min(12)] = 1.0;

    // free-pointer slot one-hot [13..24); all-zero when locked (free == None)
    if let Some(fr) = free {
        let slot = if ascending { fr as usize - 2 } else { 12 - fr as usize };
        b[13 + slot] = 1.0;
    }

    // is_locked [24]
    b[24] = if free.is_none() { 1.0 } else { 0.0 };

    // is_lockable [25]: free pointer on the terminal AND >= 5 marks
    b[25] = if free == Some(terminal) && total >= 5 { 1.0 } else { 0.0 };

    // wprob scalar [26] (mirror of pair::board_features f[12+i])
    b[26] = match free {
        Some(fr) if fr == terminal && total < 5 => 0.0,
        Some(fr) => {
            let ways = 6.0 - (7.0f32 - fr as f32).abs();
            (ways / 6.0) * (total as f32 + 1.0) / 11.0
        }
        None => 0.0,
    };

    // blanks scalar [27]: cells skipped left of the pointer; 0 if locked
    b[27] = match free {
        Some(fr) => {
            let slot = if ascending { fr as f32 - 2.0 } else { 12.0 - fr as f32 };
            (slot - total as f32).max(0.0) / 10.0
        }
        None => 0.0,
    };

    b
}

/// Per-board block: 4 row blocks followed by one-hot strikes (index clamped to
/// 3 — a terminal 4th-strike afterstate maps to the strikes=3 slot; harmless,
/// terminal states train on the realized outcome).
pub fn az_board(state: &State) -> [f32; BOARD_RAW] {
    let mut b = [0.0f32; BOARD_RAW];
    for i in 0..4 {
        b[i * ROW_BLOCK..(i + 1) * ROW_BLOCK].copy_from_slice(&az_row_block(state, i));
    }
    b[4 * ROW_BLOCK + (state.strikes as usize).min(3)] = 1.0;
    b
}

/// Full 2-player input: own board, opponent board, clamped cdiff.
pub fn az_features(our: &State, opp: &State) -> [f32; AZ_FEATURES] {
    let mut f = [0.0f32; AZ_FEATURES];
    f[0..BOARD_RAW].copy_from_slice(&az_board(our));
    f[BOARD_RAW..2 * BOARD_RAW].copy_from_slice(&az_board(opp));
    let cdiff = (our.count_points() - opp.count_points()) as f32;
    f[2 * BOARD_RAW] = (cdiff / 100.0).clamp(-1.0, 1.0);
    f
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::Mark;

    // Block-local indices for readability.
    const FREE0: usize = 13; // first free-slot index
    const LOCKED: usize = 24;
    const LOCKABLE: usize = 25;
    const WPROB: usize = 26;
    const BLANKS: usize = 27;

    #[test]
    fn fresh_board_block() {
        let s = State::default();
        for i in 0..4 {
            let b = az_row_block(&s, i);
            assert_eq!(b[0], 1.0, "count one-hot at 0");
            assert_eq!(b[FREE0], 1.0, "free slot 0 set");
            assert_eq!(b[LOCKED], 0.0);
            assert_eq!(b[LOCKABLE], 0.0);
            assert!((b[WPROB] - 1.0 / 66.0).abs() < 1e-6, "fresh wprob = 1/66");
            assert_eq!(b[BLANKS], 0.0);
            // exactly two bits set (count + free slot)
            assert_eq!(b.iter().filter(|&&x| x == 1.0).count(), 2);
        }
    }

    #[test]
    fn frontier_one_hot_ascending() {
        let mut s = State::default();
        for n in [2u8, 3, 5] {
            s.apply_mark(Mark { row: 0, number: n });
        }
        // free = 6 -> slot 4; count = 3.
        let b = az_row_block(&s, 0);
        assert_eq!(b[3], 1.0, "count index 3");
        assert_eq!(b[FREE0 + 4], 1.0, "free slot 4");
        assert!((b[BLANKS] - 0.1).abs() < 1e-6, "blanks (4-3)/10 = 0.1");
    }

    #[test]
    fn frontier_one_hot_descending() {
        let mut s = State::default();
        for n in [12u8, 11, 9] {
            s.apply_mark(Mark { row: 2, number: n });
        }
        // descending free = 8 -> slot 12-8 = 4; count = 3.
        let b = az_row_block(&s, 2);
        assert_eq!(b[3], 1.0, "count index 3");
        assert_eq!(b[FREE0 + 4], 1.0, "free slot 4 (12-8)");
    }

    #[test]
    fn locked_row_block() {
        // R 2..=6 (5 marks) then mark 12 -> safe lock; total = 7.
        let mut s = State::default();
        for n in 2..=6 {
            s.apply_mark(Mark { row: 0, number: n });
        }
        s.apply_mark(Mark { row: 0, number: 12 });
        assert_eq!(s.row_free_values()[0], None, "row locked");
        let b = az_row_block(&s, 0);
        assert_eq!(b[7], 1.0, "count index = post-lock total 7");
        assert_eq!(b[LOCKED], 1.0);
        // free-slot one-hot all zero
        assert!(b[FREE0..FREE0 + 11].iter().all(|&x| x == 0.0), "no free slot when locked");
        assert_eq!(b[LOCKABLE], 0.0);
        assert_eq!(b[BLANKS], 0.0);
    }

    #[test]
    fn count_one_hot_reaches_twelve() {
        // Mark 2..=11 (10 marks, free=12) then 12 -> lock; total = 12.
        let mut s = State::default();
        for n in 2..=11 {
            s.apply_mark(Mark { row: 0, number: n });
        }
        s.apply_mark(Mark { row: 0, number: 12 });
        let b = az_row_block(&s, 0);
        assert_eq!(b[12], 1.0, "count index 12 (lock bonus) must exist");
    }

    #[test]
    fn is_lockable_set_on_terminal_with_five() {
        // R 2..=6 (5 marks) without the lock: free should sit on 12 only after
        // reaching it; build a row whose free is on the terminal with >=5 marks.
        let mut s = State::default();
        for n in [2u8, 3, 4, 5, 11] {
            s.apply_mark(Mark { row: 0, number: n });
        }
        // free = 12 (terminal), total = 5 -> lockable.
        assert_eq!(s.row_free_values()[0], Some(12));
        let b = az_row_block(&s, 0);
        assert_eq!(b[LOCKABLE], 1.0, "free on terminal + 5 marks => lockable");
    }

    #[test]
    fn is_lockable_present_on_both_boards() {
        let mut lockable = State::default();
        for n in [2u8, 3, 4, 5, 11] {
            lockable.apply_mark(Mark { row: 0, number: n });
        }
        let fresh = State::default();
        // our = lockable, opp = fresh: own block 0 lockable bit set.
        let f = az_features(&lockable, &fresh);
        assert_eq!(f[25], 1.0, "own row 0 lockable");
        // swap: opp = lockable -> the opponent's block 0 lockable bit set too.
        let f2 = az_features(&fresh, &lockable);
        assert_eq!(f2[BOARD_RAW + 25], 1.0, "opponent row 0 lockable (not zeroed)");
    }

    #[test]
    fn swap_relationship() {
        let mut a = State::default();
        for n in [2u8, 3, 5] {
            a.apply_mark(Mark { row: 0, number: n });
        }
        a.apply_strike();
        let b = State::default();

        let ab = az_features(&a, &b);
        let ba = az_features(&b, &a);
        // Board halves exchange.
        assert_eq!(ab[0..BOARD_RAW], ba[BOARD_RAW..2 * BOARD_RAW]);
        assert_eq!(ab[BOARD_RAW..2 * BOARD_RAW], ba[0..BOARD_RAW]);
        // cdiff negates and is nonzero.
        assert!((ab[2 * BOARD_RAW] + ba[2 * BOARD_RAW]).abs() < 1e-6);
        assert!(ab[2 * BOARD_RAW] != 0.0);
    }
}
```

- [ ] **Step 3: Run the tests to verify they pass**

The stubs above are the real implementation (written inline so the file compiles). Run:

```
cargo test --features dqn dqn::aznet::tests -- --nocapture
```

Expected: all 8 tests **PASS**. (If any fail, the bug is in the listed function, not the test — fix the function.)

- [ ] **Step 4: Confirm the crate still builds**

```
cargo build --features dqn
```

Expected: clean build (one new module, no other code references it yet).

- [ ] **Step 5: Commit**

```bash
git add src/dqn/aznet.rs src/dqn/mod.rs
git commit -m "feat(aznet): one-hot crossing-order feature extraction"
```

---

## Task 2: `AzModel` architecture + batch forward

**Files:**
- Modify: `src/dqn/aznet.rs` (add model + config + `az_batch_forward`)

- [ ] **Step 1: Write the failing model tests**

Append to the `tests` module in `src/dqn/aznet.rs`:

```rust
    #[test]
    fn batch_forward_shape_and_determinism() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let model = AzModelConfig::new().init::<crate::dqn::MyBackend>(&device);

        let mut a = State::default();
        a.apply_mark(Mark { row: 0, number: 4 });
        let b = State::default();
        let rows = vec![az_features(&a, &b), az_features(&b, &a)];

        let v1 = az_batch_forward(&model, &device, &rows);
        let v2 = az_batch_forward(&model, &device, &rows);
        assert_eq!(v1.len(), 2);
        assert!(v1.iter().all(|(m, l)| m.is_finite() && l.is_finite()));
        assert_eq!(v1, v2, "forward is deterministic");
    }

    #[test]
    fn batch_forward_empty_is_empty() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let model = AzModelConfig::new().init::<crate::dqn::MyBackend>(&device);
        assert!(az_batch_forward(&model, &device, &[]).is_empty());
    }
```

- [ ] **Step 2: Run to verify they fail**

```
cargo test --features dqn dqn::aznet::tests::batch_forward -- --nocapture
```

Expected: **compile error** — `AzModel`, `AzModelConfig`, `az_batch_forward` undefined.

- [ ] **Step 3: Implement the model**

Add these imports to the top of `src/dqn/aznet.rs` (below the existing `use crate::state::State;`):

```rust
use crate::dqn::MyBackend;
use burn::{
    nn::{Linear, LinearConfig, Relu},
    prelude::*,
};
```

Add the encoder-output constant near the layout constants:

```rust
/// Per-row embedding width out of the shared encoder. Fixed (the trunk-input
/// width and the forward reshape depend on it); `AzModelConfig::init` asserts
/// the config matches. Tune by editing this constant, mirroring pair.rs's
/// fixed `D_H1`/`D_H2`.
pub const ENC_OUT: usize = 16;
```

Add the model, config, forward, and batch forward (place after `az_features`,
before the `tests` module):

```rust
#[derive(Module, Debug)]
pub struct AzModel<B: Backend> {
    pub enc1: Linear<B>,        // ROW_BLOCK -> encoder_hidden
    pub enc2: Linear<B>,        // encoder_hidden -> ENC_OUT
    pub trunk1: Linear<B>,      // 8*ENC_OUT + 9 -> trunk1
    pub trunk2: Linear<B>,      // trunk1 -> trunk2
    pub output_mean: Linear<B>, // trunk2 -> 1
    pub output_log_var: Linear<B>,
    activation: Relu,
}

#[derive(Config, Debug)]
pub struct AzModelConfig {
    #[config(default = 32)]
    pub encoder_hidden: usize,
    #[config(default = 16)]
    pub encoder_out: usize,
    #[config(default = 128)]
    pub trunk1: usize,
    #[config(default = 64)]
    pub trunk2: usize,
}

impl AzModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> AzModel<B> {
        assert_eq!(self.encoder_out, ENC_OUT, "ENC_OUT const must match config");
        let trunk_in = 8 * ENC_OUT + 9; // 8 row embeds + 8 strike one-hots + cdiff
        AzModel {
            enc1: LinearConfig::new(ROW_BLOCK, self.encoder_hidden).init(device),
            enc2: LinearConfig::new(self.encoder_hidden, self.encoder_out).init(device),
            trunk1: LinearConfig::new(trunk_in, self.trunk1).init(device),
            trunk2: LinearConfig::new(self.trunk1, self.trunk2).init(device),
            output_mean: LinearConfig::new(self.trunk2, 1).init(device),
            output_log_var: LinearConfig::new(self.trunk2, 1).init(device),
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> AzModel<B> {
    /// Apply the shared encoder to one board's 4 row blocks. `rows` is
    /// `[n, 4*ROW_BLOCK]`; returns `[n, 4*ENC_OUT]`.
    fn encode_rows(&self, rows: Tensor<B, 2>, n: usize) -> Tensor<B, 2> {
        let x = rows.reshape([n * 4, ROW_BLOCK]);
        let x = self.activation.forward(self.enc1.forward(x));
        let x = self.activation.forward(self.enc2.forward(x));
        x.reshape([n, 4 * ENC_OUT])
    }

    /// Forward over a `[n, AZ_FEATURES]` batch -> `[n, 2]` (μ_diff, log σ²_diff).
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let n = input.shape().dims[0]; // repo idiom (see pair.rs extract_linear)
        let own_rows = input.clone().narrow(1, 0, 4 * ROW_BLOCK);
        let own_strikes = input.clone().narrow(1, 4 * ROW_BLOCK, 4);
        let opp_rows = input.clone().narrow(1, BOARD_RAW, 4 * ROW_BLOCK);
        let opp_strikes = input.clone().narrow(1, BOARD_RAW + 4 * ROW_BLOCK, 4);
        let cdiff = input.narrow(1, 2 * BOARD_RAW, 1);

        let own = self.encode_rows(own_rows, n);
        let opp = self.encode_rows(opp_rows, n);

        let trunk_in = Tensor::cat(vec![own, opp, own_strikes, opp_strikes, cdiff], 1);
        let x = self.activation.forward(self.trunk1.forward(trunk_in));
        let x = self.activation.forward(self.trunk2.forward(x));
        let mean = self.output_mean.forward(x.clone());
        let log_var = self.output_log_var.forward(x);
        Tensor::cat(vec![mean, log_var], 1)
    }
}

/// Run `model` over a batch of feature vectors in one forward pass.
/// Returns `(μ, log σ²)` per row.
pub fn az_batch_forward(
    model: &AzModel<MyBackend>,
    device: &burn::backend::ndarray::NdArrayDevice,
    features_list: &[[f32; AZ_FEATURES]],
) -> Vec<(f32, f32)> {
    if features_list.is_empty() {
        return Vec::new();
    }
    let n = features_list.len();
    let flat: Vec<f32> = features_list.iter().flat_map(|f| f.iter().copied()).collect();
    let input = Tensor::<MyBackend, 1>::from_floats(flat.as_slice(), device).reshape([n, AZ_FEATURES]);
    let output = model.forward(input);
    let values = output.into_data().to_vec::<f32>().unwrap();
    (0..n).map(|i| (values[2 * i], values[2 * i + 1])).collect()
}
```

- [ ] **Step 4: Run to verify they pass**

```
cargo test --features dqn dqn::aznet::tests -- --nocapture
```

Expected: all Task-1 and Task-2 tests **PASS**.

- [ ] **Step 5: Commit**

```bash
git add src/dqn/aznet.rs
git commit -m "feat(aznet): shared-encoder AzModel + batch forward"
```

---

## Task 3: Decoupled μ/σ loss (`forward_step`, `TrainStep`, `InferenceStep`)

**Files:**
- Modify: `src/dqn/aznet.rs`

This mirrors `PairModel::forward_step` (`src/dqn/pair_train.rs:127-153`) and its
`TrainStep`/`InferenceStep` impls (lines 155-172), but lives on `AzModel`. The
batch type `AzBatch` is defined here and reused by the trainer in Task 6.

- [ ] **Step 1: Write the failing loss test**

Append to the `tests` module in `src/dqn/aznet.rs`:

```rust
    #[test]
    fn forward_step_loss_finite_and_mean_matches_head() {
        use crate::dqn::aznet::AzBatch;
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let model = AzModelConfig::new().init::<crate::dqn::MyBackend>(&device);

        let a = State::default();
        let b = State::default();
        let feats = [az_features(&a, &b), az_features(&b, &a)];
        let flat: Vec<f32> = feats.iter().flat_map(|f| f.iter().copied()).collect();
        let inputs = burn::tensor::Tensor::<crate::dqn::MyBackend, 1>::from_floats(flat.as_slice(), &device)
            .reshape([2, AZ_FEATURES]);
        let targets = burn::tensor::Tensor::<crate::dqn::MyBackend, 1>::from_floats([3.0f32, -3.0].as_slice(), &device);
        let final_diffs =
            burn::tensor::Tensor::<crate::dqn::MyBackend, 1>::from_floats([5.0f32, -5.0].as_slice(), &device);

        let batch = AzBatch {
            inputs,
            targets,
            final_diffs,
        };
        let out = model.forward_step(batch);
        let loss = out.loss.into_data().to_vec::<f32>().unwrap()[0];
        assert!(loss.is_finite(), "loss finite");
        assert!(loss >= 0.0, "loss non-negative");
    }
```

- [ ] **Step 2: Run to verify it fails**

```
cargo test --features dqn dqn::aznet::tests::forward_step_loss -- --nocapture
```

Expected: **compile error** — `AzBatch`/`forward_step` undefined.

- [ ] **Step 3: Implement the batch type, loss, and train/inference steps**

Add to the imports block in `src/dqn/aznet.rs`:

```rust
use crate::dqn::{LOG_VAR_MAX, LOG_VAR_MIN};
use burn::{
    tensor::backend::AutodiffBackend,
    train::{InferenceStep, RegressionOutput, TrainOutput, TrainStep},
};
```

Add (after `az_batch_forward`, before `tests`):

```rust
#[derive(Clone, Debug)]
pub struct AzBatch<B: Backend> {
    pub inputs: Tensor<B, 2>,
    pub targets: Tensor<B, 1>,
    pub final_diffs: Tensor<B, 1>,
}

impl<B: Backend> AzModel<B> {
    /// Decoupled μ/σ loss on the differential — identical recipe to the pair
    /// net: μ = MSE toward the TD(λ) target; σ = Gaussian NLL against the
    /// actual final-diff residual with μ detached.
    pub fn forward_step(&self, batch: AzBatch<B>) -> RegressionOutput<B> {
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

impl<B: AutodiffBackend> TrainStep for AzModel<B> {
    type Input = AzBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, batch: AzBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_step(batch);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for AzModel<B> {
    type Input = AzBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, batch: AzBatch<B>) -> RegressionOutput<B> {
        self.forward_step(batch)
    }
}
```

NOTE: this is the **associated-type** form, matching `pair_train.rs:155-172`
verbatim (this repo's burn version). If a compile error reports a trait-param
mismatch, re-check against `pair_train.rs` and copy its exact shape, swapping
`PairModel`/`PairBatch` → `AzModel`/`AzBatch`.

- [ ] **Step 4: Run to verify it passes**

```
cargo test --features dqn dqn::aznet::tests::forward_step_loss -- --nocapture
```

Expected: **PASS**. Then run the whole module: `cargo test --features dqn dqn::aznet::`.

- [ ] **Step 5: Commit**

```bash
git add src/dqn/aznet.rs
git commit -m "feat(aznet): decoupled mu/sigma loss + train/inference steps"
```

---

## Task 4: `AzStrategy` (`Bot` + `WinProb`)

**Files:**
- Modify: `src/dqn/aznet.rs`

Mirrors `PairStrategy` (`src/dqn/pair.rs:262-364`) but holds only `Arc<AzModel>`
(no `ManualPairNet`) and forwards through `az_batch_forward`. 2-player: the
"leader" is the single opponent.

- [ ] **Step 1: Write the failing strategy tests**

Append to the `tests` module:

```rust
    #[test]
    fn evaluate_batch_finite_and_deterministic() {
        use crate::strategy::Bot;
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let bot = AzStrategy::from_model(AzModelConfig::new().init::<crate::dqn::MyBackend>(&device), device);

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
    fn win_prob_is_probability_and_monotone() {
        use crate::strategy::search::WinProb;
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let bot = AzStrategy::from_model(AzModelConfig::new().init::<crate::dqn::MyBackend>(&device), device);

        let behind = State::default();
        let mut ahead = State::default();
        for n in 2..=8 {
            ahead.apply_mark(Mark { row: 0, number: n });
        }
        let opp = State::default();
        let p = bot.win_prob_multi(&[(&behind, &[opp]), (&ahead, &[opp])]);
        assert_eq!(p.len(), 2);
        assert!(p.iter().all(|x| (0.0..=1.0).contains(x)));
        assert!(p[1] >= p[0], "a points lead must not be rated worse");
    }
```

- [ ] **Step 2: Run to verify they fail**

```
cargo test --features dqn dqn::aznet::tests::evaluate_batch_finite -- --nocapture
```

Expected: **compile error** — `AzStrategy` undefined.

- [ ] **Step 3: Implement `AzStrategy`**

Add to imports:

```rust
use crate::strategy::Bot;
use burn::record::CompactRecorder;
use std::sync::Arc;
```

Add (after the train/inference step impls, before `tests`):

```rust
pub struct AzStrategy {
    pub model: Arc<AzModel<MyBackend>>,
    pub device: burn::backend::ndarray::NdArrayDevice,
}

impl std::fmt::Debug for AzStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "AzStrategy")
    }
}

impl AzStrategy {
    pub fn load(artifact_dir: &str) -> Self {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let model = AzModelConfig::new()
            .init::<MyBackend>(&device)
            .load_file(format!("{artifact_dir}/model"), &CompactRecorder::new(), &device)
            .expect("Failed to load aznet model");
        Self::from_shared(Arc::new(model), device)
    }

    pub fn from_model(model: AzModel<MyBackend>, device: burn::backend::ndarray::NdArrayDevice) -> Self {
        Self::from_shared(Arc::new(model), device)
    }

    pub fn from_shared(model: Arc<AzModel<MyBackend>>, device: burn::backend::ndarray::NdArrayDevice) -> Self {
        AzStrategy { model, device }
    }
}

impl Bot for AzStrategy {
    fn evaluate(&self, our_state: &State, opp_states: &[State]) -> f32 {
        self.evaluate_batch(&[*our_state], opp_states)[0]
    }

    fn evaluate_batch(&self, candidates: &[State], opp_states: &[State]) -> Vec<f32> {
        self.evaluate_batch_multi(&[(candidates, opp_states)]).pop().unwrap()
    }

    fn evaluate_batch_multi(&self, groups: &[(&[State], &[State])]) -> Vec<Vec<f32>> {
        // 2-player net: rank against the leading opponent; extra opponents
        // (multiplayer) are ignored. Solo falls back to a fresh board.
        let default_opps = [State::default()];
        let mut leaders: Vec<isize> = Vec::with_capacity(groups.len());
        let mut feats: Vec<[f32; AZ_FEATURES]> = Vec::new();
        for (candidates, opp_states) in groups {
            let opps: &[State] = if opp_states.is_empty() { &default_opps } else { opp_states };
            let leader = opps.iter().max_by_key(|s| s.count_points()).unwrap();
            for c in *candidates {
                feats.push(az_features(c, leader));
            }
            leaders.push(leader.count_points());
        }
        let values = az_batch_forward(&self.model, &self.device, &feats);

        let mut out = Vec::with_capacity(groups.len());
        let mut idx = 0;
        for ((candidates, _), leader_points) in groups.iter().zip(leaders) {
            let group = candidates
                .iter()
                .map(|cand| {
                    let (mu, log_var) = values[idx];
                    idx += 1;
                    let cdiff = (cand.count_points() - leader_points) as f32;
                    let sigma = (0.5 * log_var.clamp(LOG_VAR_MIN, LOG_VAR_MAX)).exp();
                    (cdiff + mu) / sigma
                })
                .collect();
            out.push(group);
        }
        out
    }
}

impl crate::strategy::search::WinProb for AzStrategy {
    fn win_prob_multi(&self, groups: &[(&State, &[State])]) -> Vec<f32> {
        let default_opps = [State::default()];
        let mut feats = Vec::with_capacity(groups.len());
        let mut cdiffs = Vec::with_capacity(groups.len());
        for (our, opps) in groups {
            let opps: &[State] = if opps.is_empty() { &default_opps } else { opps };
            let leader = opps.iter().max_by_key(|s| s.count_points()).unwrap();
            feats.push(az_features(our, leader));
            cdiffs.push((our.count_points() - leader.count_points()) as f32);
        }
        az_batch_forward(&self.model, &self.device, &feats)
            .into_iter()
            .zip(cdiffs)
            .map(|((mu, log_var), cdiff)| {
                let sigma = (0.5 * log_var.clamp(LOG_VAR_MIN, LOG_VAR_MAX)).exp();
                crate::strategy::search::phi((cdiff + mu) / sigma)
            })
            .collect()
    }
}
```

- [ ] **Step 4: Run to verify they pass**

```
cargo test --features dqn dqn::aznet:: -- --nocapture
```

Expected: every `aznet` test **PASSES**.

- [ ] **Step 5: Commit**

```bash
git add src/dqn/aznet.rs
git commit -m "feat(aznet): AzStrategy implementing Bot + WinProb"
```

---

## Task 5: Promote `td_diff_targets` to `pub(crate)`

**Files:**
- Modify: `src/dqn/pair_train.rs:81`

- [ ] **Step 1: Change the visibility**

In `src/dqn/pair_train.rs`, change line 81 from:

```rust
fn td_diff_targets(mus: &[f32], cdiffs: &[f32], final_diff: f32, lambda: f32) -> Vec<f32> {
```

to:

```rust
pub(crate) fn td_diff_targets(mus: &[f32], cdiffs: &[f32], final_diff: f32, lambda: f32) -> Vec<f32> {
```

- [ ] **Step 2: Verify the existing test still passes**

```
cargo test --features dqn dqn::pair_train::tests::td_diff_targets_hand_computed -- --nocapture
```

Expected: **PASS** (behavior unchanged; only visibility widened).

- [ ] **Step 3: Commit**

```bash
git add src/dqn/pair_train.rs
git commit -m "refactor(pair_train): expose td_diff_targets pub(crate) for reuse"
```

---

## Task 6: `aznet_train.rs` — `AzSample`, `AzBatcher`, `permute_rows`

**Files:**
- Create: `src/dqn/aznet_train.rs`
- Modify: `src/dqn/mod.rs` (add `#[cfg(feature = "dqn")] pub mod aznet_train;`)

- [ ] **Step 1: Register the module**

In `src/dqn/mod.rs`, below `pub mod pair_train;` (line 22), add:

```rust
#[cfg(feature = "dqn")]
pub mod aznet_train;
```

- [ ] **Step 2: Write the failing batcher/sample tests**

Create `src/dqn/aznet_train.rs` with the header, the sample + serde + batcher +
`permute_rows`, and the test module:

```rust
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
    }
}
```

- [ ] **Step 3: Run to verify it passes**

```
cargo test --features dqn dqn::aznet_train::tests::permute_rows -- --nocapture
```

Expected: **PASS**. (The file imports several not-yet-used items — expect
`unused import` warnings only; they are consumed by Tasks 7–9.) If unused-import
warnings are treated as errors in this repo, temporarily prefix the unused
imports — but check first: this repo does not `deny(warnings)`, so leave them.

- [ ] **Step 4: Commit**

```bash
git add src/dqn/aznet_train.rs src/dqn/mod.rs
git commit -m "feat(aznet): training sample type, serde, color-permutation batcher"
```

---

## Task 7: `build_az_samples` (TD chain + swap doubling)

**Files:**
- Modify: `src/dqn/aznet_train.rs`

2-player specialization of `build_pair_samples` (`pair_train.rs:935-983`): one
TD(λ) chain against the single opponent, every sample emitted in both board
orders with negated targets.

- [ ] **Step 1: Write the failing test**

Append to the `tests` module:

```rust
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
```

- [ ] **Step 2: Run to verify it fails**

```
cargo test --features dqn dqn::aznet_train::tests::build_az_samples -- --nocapture
```

Expected: **compile error** — `build_az_samples` undefined.

- [ ] **Step 3: Implement `build_az_samples`**

Add (after `AzBatcher`'s impl, before `tests`):

```rust
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
```

- [ ] **Step 4: Run to verify it passes**

```
cargo test --features dqn dqn::aznet_train::tests::build_az_samples -- --nocapture
```

Expected: **PASS**.

- [ ] **Step 5: Commit**

```bash
git add src/dqn/aznet_train.rs
git commit -m "feat(aznet): build_az_samples — TD chain + swap doubling"
```

---

## Task 8: `RecordingAz` + `play_training_game`

**Files:**
- Modify: `src/dqn/aznet_train.rs`

`RecordingAz` is the static-policy subset of `RecordingPair`
(`pair_train.rs:323-646`): ε-greedy active, greedy passive, records afterstates
(active turns once post-phase-2; passive turns including skips). No distill, no
search, no `epsilon_lock`. `play_training_game` mirrors `pair_train.rs:990-1067`
but is always 1v1, static policy, no distill.

- [ ] **Step 1: Write the failing determinism test**

Append to the `tests` module:

```rust
    #[test]
    fn play_training_game_is_deterministic() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let model = AzModelConfig::new().init::<MyBackend>(&device);

        let run = || play_training_game(&model, &device, 0.1, 4242);
        let (s1, f1) = run();
        let (s2, f2) = run();
        assert_eq!(f1, f2);
        assert_eq!(s1.len(), s2.len());
        for (a, b) in s1.iter().zip(&s2) {
            assert_eq!(a.features, b.features);
            assert_eq!(a.value, b.value);
            assert_eq!(a.final_diff, b.final_diff);
        }
        assert!(!s1.is_empty(), "a full game must produce samples");
    }
```

- [ ] **Step 2: Run to verify it fails**

```
cargo test --features dqn dqn::aznet_train::tests::play_training_game_is_deterministic -- --nocapture
```

Expected: **compile error** — `play_training_game` undefined.

- [ ] **Step 3: Implement `RecordingAz` and `play_training_game`**

Add `use std::cell::RefCell; use std::rc::Rc;` to the imports, then add (after
`build_az_samples`, before `tests`):

```rust
/// Self-play recorder: ε-greedy on active decisions, greedy on passive;
/// records afterstates for chain building. The static-policy subset of the
/// pair net's `RecordingPair` (no search, no distill).
struct RecordingAz {
    bot: AzStrategy,
    epsilon: f32,
    rng: SmallRng,
    recorded: Rc<RefCell<Vec<Snapshot>>>,
}

impl std::fmt::Debug for RecordingAz {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "RecordingAz")
    }
}

impl Strategy for RecordingAz {
    fn active_phase1(&mut self, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
        // ε-uniform: random white mark or skip (recording happens post-phase-2).
        if self.rng.gen::<f32>() < self.epsilon {
            let white_marks = state.generate_white_moves(dice[0] + dice[1]);
            if white_marks.is_empty() {
                return None;
            }
            let idx = self.rng.gen_range(0..=white_marks.len());
            return (idx < white_marks.len()).then(|| white_marks[idx]);
        }
        active_phase1_impl(&self.bot, state, opp_states, dice)
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
            (idx < marks.len()).then(|| marks[idx])
        } else {
            active_phase2_impl(&self.bot, state, opp_states, dice, has_marked)
        };

        let chosen = match mark {
            Some(m) => {
                let mut s = *state;
                s.apply_mark(m);
                s
            }
            None => no_mark_state,
        };
        self.recorded.borrow_mut().push((chosen, opp_states.to_vec()));
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
        let mark = passive_phase1_impl(&self.bot, state, opp_states, dice);
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

/// Play one 1v1 plain self-play game; both players are recording aznet bots
/// (player 0 explores with ε, player 1 greedy). Returns all samples (both
/// players' trajectories, swap-doubled) plus player 0's final score.
fn play_training_game(
    model: &AzModel<MyBackend>,
    device: &burn::backend::ndarray::NdArrayDevice,
    epsilon: f32,
    seed: u64,
) -> (Vec<AzSample>, f32) {
    use crate::game::{Game, Player};

    let mut buffers = Vec::with_capacity(2);
    let mut players: Vec<Player> = Vec::with_capacity(2);
    for i in 0..2usize {
        let bot = AzStrategy::from_model(model.clone(), device.clone());
        let buf = Rc::new(RefCell::new(Vec::new()));
        buffers.push(Rc::clone(&buf));
        players.push(Player::new(
            Box::new(RecordingAz {
                bot,
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
        let opp_final = finals[(i + 1) % 2];
        all_samples.extend(build_az_samples(model, device, &snapshots, finals[i], opp_final));
    }
    (all_samples, finals[0])
}
```

- [ ] **Step 4: Run to verify it passes**

```
cargo test --features dqn dqn::aznet_train::tests::play_training_game_is_deterministic -- --nocapture
```

Expected: **PASS**.

- [ ] **Step 5: Commit**

```bash
git add src/dqn/aznet_train.rs
git commit -m "feat(aznet): RecordingAz recorder + 1v1 self-play game"
```

---

## Task 9: Self-play loop (`benchmark_vs_ga`, `train_with_epochs`, `self_play_train`)

**Files:**
- Modify: `src/dqn/aznet_train.rs`

`benchmark_vs_ga` mirrors `pair_train.rs:1071-1109` (swap `PairStrategy`→
`AzStrategy`). `train_with_epochs` mirrors `pair_train.rs:1282-1334` (swap
`PairModelConfig`/`PairBatcher`→`AzModelConfig`/`AzBatcher`). `self_play_train`
mirrors `pair_train.rs:1111-1280` but: always 1v1, no search/distill, simpler
per-iteration prints.

- [ ] **Step 1: Write the failing smoke test**

Append to the `tests` module:

```rust
    #[test]
    fn self_play_train_smoke_writes_model() {
        let dir = std::env::temp_dir().join(format!("aznet_smoke_{}", std::process::id()));
        let dir_s = dir.to_str().unwrap();
        std::fs::remove_dir_all(&dir).ok();

        // 1 iteration, 6 games, 1 epoch, no bench. Must complete and write a
        // loadable model.
        self_play_train(dir_s, 1, 6, 1, 0, false, 0);
        assert!(dir.join("model.mpk").exists(), "model artifact written");

        // The written model loads and evaluates.
        let bot = AzStrategy::load(dir_s);
        let v = crate::strategy::Bot::evaluate(&bot, &State::default(), &[State::default()]);
        assert!(v.is_finite());

        std::fs::remove_dir_all(&dir).ok();
    }
```

- [ ] **Step 2: Run to verify it fails**

```
cargo test --features dqn dqn::aznet_train::tests::self_play_train_smoke -- --nocapture
```

Expected: **compile error** — `self_play_train` undefined.

- [ ] **Step 3: Implement the loop**

Add to imports: `use crate::bot::{self, DNA}; use rayon::prelude::*;`. Add (after
`play_training_game`, before `tests`):

```rust
/// Same fixed-game-set paired benchmark as the pair bot, for aznet.
const BENCH_SEED: u64 = 0xB54C;

fn benchmark_vs_ga(artifact_dir: &str, champion: &DNA, num_games: usize) -> f64 {
    use crate::game::{Game, Player};

    let wins: u32 = (0..num_games)
        .into_par_iter()
        .map_init(
            || AzStrategy::load(artifact_dir),
            |template, i| {
                let bot = AzStrategy::from_shared(template.model.clone(), template.device.clone());
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
                (scores[idx] > scores[1 - idx]) as u32
            },
        )
        .sum();
    wins as f64 / num_games as f64
}

fn train_with_epochs(samples: Vec<AzSample>, artifact_dir: &str, num_epochs: usize, lr: f64) -> AzModel<MyBackend> {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;

    let split = (samples.len() * 9) / 10;
    let train_data = InMemDataset::new(samples[..split].to_vec());
    let valid_data = InMemDataset::new(samples[split..].to_vec());

    let model: AzModel<MyAutodiffBackend> = AzModelConfig::new()
        .init::<MyAutodiffBackend>(&device)
        .load_file(format!("{artifact_dir}/model"), &CompactRecorder::new(), &device)
        .unwrap_or_else(|_| AzModelConfig::new().init::<MyAutodiffBackend>(&device));

    let batcher_train = AzBatcher::<MyAutodiffBackend> { _phantom: std::marker::PhantomData };
    let batcher_valid = AzBatcher::<MyBackend> { _phantom: std::marker::PhantomData };

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
        .expect("Failed to save aznet model");
    std::fs::remove_dir_all(&ckpt_dir).ok();

    AzModelConfig::new()
        .init::<MyBackend>(&device)
        .load_file(format!("{artifact_dir}/model"), &CompactRecorder::new(), &device)
        .expect("Failed to reload aznet model for inference")
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
    let mut replay_buffer: std::collections::VecDeque<Vec<AzSample>> = std::collections::VecDeque::new();

    let scores_log_path = format!("{artifact_dir}/training_scores.csv");
    if start_iteration == 0 || !std::path::Path::new(&scores_log_path).exists() {
        std::fs::write(&scores_log_path, "iteration,avg_score,winrate\n").ok();
    }

    let genes = Arc::new(bot::default_genes());
    let champion = DNA::load_weights("champion.txt", genes).expect("No champion.txt");
    let start_time = std::time::Instant::now();

    let mut model: AzModel<MyBackend> = AzModelConfig::new()
        .init::<MyBackend>(&device)
        .load_file(format!("{artifact_dir}/model"), &CompactRecorder::new(), &device)
        .unwrap_or_else(|_| {
            println!("  No pretrained aznet model, starting fresh");
            AzModelConfig::new().init::<MyBackend>(&device)
        });

    for iteration in 0..num_iterations {
        let global_iter = start_iteration + iteration;
        let epsilon = (0.2 * (0.95f32).powi(global_iter as i32)).max(0.07);
        println!("\n=== Aznet iteration {} (epsilon={epsilon:.3}) ===", global_iter + 1);

        let models: Vec<AzModel<MyBackend>> = (0..games_per_iteration).map(|_| model.clone()).collect();
        let game_results: Vec<(Vec<AzSample>, f32)> = (0..games_per_iteration)
            .into_par_iter()
            .zip(models.into_par_iter())
            .map(|(game_idx, thread_model)| {
                let seed = TRAIN_SEED.wrapping_add((iteration * games_per_iteration + game_idx) as u64);
                play_training_game(&thread_model, &device, epsilon, seed)
            })
            .collect();

        let game_scores: Vec<f32> = game_results.iter().map(|(_, s)| *s).collect();
        let new_samples: Vec<AzSample> = game_results.into_iter().flat_map(|(s, _)| s).collect();
        let avg_score = if game_scores.is_empty() {
            0.0
        } else {
            game_scores.iter().sum::<f32>() / game_scores.len() as f32
        };

        replay_buffer.push_back(new_samples);
        if replay_buffer.len() > buffer_iterations {
            replay_buffer.pop_front();
        }
        let all_samples: Vec<AzSample> = replay_buffer.iter().flatten().copied().collect();
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
        use std::io::Write;
        if bench_games > 0 {
            let winrate = benchmark_vs_ga(artifact_dir, &champion, bench_games);
            println!(
                "  Iteration {:>3}: avg score {:.1}, winrate {:.1}%, elapsed {}m{}s",
                global_iter + 1,
                avg_score,
                winrate * 100.0,
                elapsed / 60,
                elapsed % 60,
            );
            if let Ok(mut f) = std::fs::OpenOptions::new().append(true).create(true).open(&scores_log_path) {
                writeln!(f, "{},{avg_score:.2},{:.2}", global_iter + 1, winrate * 100.0).ok();
            }
        } else {
            println!("  Iteration {:>3}: avg score {:.1}, elapsed {}m{}s", global_iter + 1, avg_score, elapsed / 60, elapsed % 60);
            if let Ok(mut f) = std::fs::OpenOptions::new().append(true).create(true).open(&scores_log_path) {
                writeln!(f, "{},{avg_score:.2}", global_iter + 1).ok();
            }
        }
    }

    println!("\nAznet self-play training complete. Model saved to {artifact_dir}/model");
}
```

- [ ] **Step 4: Run to verify it passes**

```
cargo test --features dqn dqn::aznet_train::tests::self_play_train_smoke -- --nocapture
```

Expected: **PASS** (a few seconds; the smoke run trains 1 epoch on ~6 games).

- [ ] **Step 5: Run the full test suite**

```
cargo test --features dqn
```

Expected: all tests pass (existing `pair`/`pair_train`/`dqn` suites + new `aznet`
suites). Confirm no existing test regressed.

- [ ] **Step 6: Commit**

```bash
git add src/dqn/aznet_train.rs
git commit -m "feat(aznet): 2-player plain self-play loop + paired bench"
```

---

## Task 10: CLI wiring (`bench`/`play` bot types + `aznet-train` command)

**Files:**
- Modify: `src/main.rs`

- [ ] **Step 1: Add the `BotType` variants and `Display`**

In `src/main.rs`, the `enum BotType` (line 16) — add `Aznet` and `AznetSearch`
after `PairSearch`:

```rust
enum BotType {
    Ga,
    Dqn,
    Pair,
    PairSearch,
    Aznet,
    AznetSearch,
    Mcts,
    Opportunist,
    Conservative,
    Random,
}
```

In the `Display` impl, add the two match arms after the `PairSearch` arm:

```rust
            BotType::Aznet => write!(f, "AZNET"),
            BotType::AznetSearch => write!(f, "AZNET-SEARCH"),
```

- [ ] **Step 2: Wire `make_strategy`**

In `make_strategy` (line 42), add after the `PairSearch` arm:

```rust
        BotType::Aznet => Box::new(dqn::aznet::AzStrategy::load("aznet_model")),
        BotType::AznetSearch => {
            Box::new(strategy::search::SearchBot::new(dqn::aznet::AzStrategy::load("aznet_model")))
        }
```

- [ ] **Step 3: Wire `StrategyTemplates`**

In `struct StrategyTemplates` (line 66), add a field:

```rust
    aznet: Option<dqn::aznet::AzStrategy>,
```

In `StrategyTemplates::new`, add the `needs_aznet` detection and field init
(alongside `needs_pair`):

```rust
        let needs_aznet = bots.iter().any(|b| matches!(b, BotType::Aznet | BotType::AznetSearch));
```

```rust
            aznet: if needs_aznet {
                Some(dqn::aznet::AzStrategy::load("aznet_model"))
            } else {
                None
            },
```

In `StrategyTemplates::create`, add after the `PairSearch` arm:

```rust
            BotType::Aznet => {
                let t = self.aznet.as_ref().unwrap();
                Box::new(dqn::aznet::AzStrategy::from_shared(t.model.clone(), t.device.clone()))
            }
            BotType::AznetSearch => {
                let t = self.aznet.as_ref().unwrap();
                Box::new(strategy::search::SearchBot::new(dqn::aznet::AzStrategy::from_shared(
                    t.model.clone(),
                    t.device.clone(),
                )))
            }
```

- [ ] **Step 4: Add the `AznetTrain` command**

In `enum Commands`, after the `PairTrain { ... }` variant (ends line 230), add:

```rust
    /// Aznet (shared-encoder one-hot) self-play RL — 2-player only, plain
    #[cfg(feature = "dqn")]
    AznetTrain {
        /// Number of iterations
        #[arg(short, long, default_value = "40")]
        iterations: usize,
        /// Games per iteration
        #[arg(short, long, default_value = "20000")]
        games: usize,
        /// Training epochs per iteration over the replay buffer
        #[arg(short, long, default_value = "10")]
        epochs: usize,
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

- [ ] **Step 5: Add the dispatch arm**

In the `match` in `main()`, after the `PairTrain { ... } => ...` arm (ends line
691), add:

```rust
        #[cfg(feature = "dqn")]
        Some(Commands::AznetTrain {
            iterations,
            games,
            epochs,
            bench,
            checkpoints,
            start_iteration,
        }) => dqn::aznet_train::self_play_train("aznet_model", iterations, games, epochs, bench, checkpoints, start_iteration),
```

- [ ] **Step 6: Build and smoke-verify the CLI end-to-end**

```
cargo build --release --features dqn
```

Expected: clean build. Then a tiny end-to-end run (trains a throwaway model, then
benches it so the new bot types are exercised):

```
cargo run --release --features dqn -- aznet-train -i 1 -g 200 -e 1
cargo run --release --features dqn -- bench aznet ga -n 200
cargo run --release --features dqn -- bench aznet-search ga -n 50
```

Expected: training prints one iteration and "complete"; both benches print a
win-rate line without panicking. (Win rate will be near-random — the model is
untrained; this only verifies wiring.)

- [ ] **Step 7: Commit**

```bash
git add src/main.rs
git commit -m "feat(aznet): CLI bot types (aznet, aznet-search) + aznet-train command"
```

---

## Task 11: Pre-registration stub in `EXPERIMENTS.md`

**Files:**
- Modify: `docs/EXPERIMENTS.md` (append a new Phase 20 section)

- [ ] **Step 1: Append the Phase 20 pre-registration**

Add to the end of `docs/EXPERIMENTS.md`:

```markdown

## Phase 20: AZ Representation (aznet) — shared-encoder one-hot value net

**Status: implemented; run pending.** Spec:
`docs/superpowers/specs/2026-06-15-aznet-representation-design.md`.

Tests AlphaZero's *representation* lever in the existing pair-train harness: a
new afterstate value net (`aznet`) encoding each board as one-hot crossing-order
per-row blocks (count 0..=12, free-pointer slot, is_locked, is_lockable on both
boards, wprob + blanks scalars) through a shared `f_row` 28→32→16 encoder, then
a 128→64 trunk with μ/σ diff-space heads (~27.5k params). Everything else is the
pair net's pipeline unchanged: TD(λ=0.8) diff targets, decoupled μ/σ loss,
board-swap doubling, `(cdiff+μ)/σ` ranking, `SearchBot`, paired-CRN bench.
2-player-only plain self-play; burn-only inference.

### Run recipe

```bash
rm -f aznet_model/iter-*.mpk
# Match the pair net's plain recipe; 200k per-iteration bench for checkpoint selection.
cargo run --release -- aznet-train -i 40 -g 20000 -e 3 -b 200000 -c
```

### Pre-registration

- **Control:** recorded plain pair numbers (~59.2% static / ~60.1% search;
  weaker isolation — recorded run used 1v1/3p/4p thirds, not 2p-only).
- **Primary:** selected checkpoint's win rate vs GA @ 1M, seed 42, static AND
  search (`bench ga aznet -n 1000000`, `bench ga aznet-search -n 1000000`).
- **PAY (≥ +0.3pp, CI-separated):** build the hand-rolled inference kernel, run
  a distillation leg, proceed to Arm B (directly-learned win head).
- **KILL (< +0.1pp):** representation is not the lever; stop before Arm B.
- **Capacity-control (only if PAY):** re-run old repr ~27k OR new repr ~14k to
  disentangle representation from the ~2× capacity.
- **Guard:** checkpoint selection on the per-iteration 200k static curve;
  90/10 valid loss watched for overfit (reduce epochs before resizing).

### Results

_(to be filled after the run)_
```

- [ ] **Step 2: Commit**

```bash
git add docs/EXPERIMENTS.md
git commit -m "docs: Phase 20 (aznet representation) pre-registration stub"
```

---

## Final verification (after all tasks)

- [ ] `cargo test --features dqn` — full suite green (existing + new).
- [ ] `cargo build --release --features dqn` — clean release build.
- [ ] The three CLI smoke commands in Task 10 Step 6 run without panic.

**Web build is out of scope for this task** — do not run or worry about
`--no-default-features --features burn`. `aznet.rs` may freely use `burn::train`
types; no `#[cfg]`-gating of the loss/step block is required.
