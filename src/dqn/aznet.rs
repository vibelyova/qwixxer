//! "aznet" value net: AlphaZero-style one-hot crossing-order representation
//! run through a shared per-row encoder. Afterstate value net like the pair
//! net (diff-space μ/σ heads, `(cdiff+μ)/σ` ranking), differing only in the
//! board encoding. 2-player only. Design:
//! docs/superpowers/specs/2026-06-15-aznet-representation-design.md

use crate::state::State;

use crate::dqn::MyBackend;
use crate::dqn::{LOG_VAR_MAX, LOG_VAR_MIN};
use crate::strategy::Bot;
use burn::record::CompactRecorder;
use burn::{
    nn::{Linear, LinearConfig, Relu},
    prelude::*,
    tensor::backend::AutodiffBackend,
    train::{InferenceStep, RegressionOutput, TrainOutput, TrainStep},
};
use std::sync::Arc;

/// Per-row one-hot block width.
pub const ROW_BLOCK: usize = 28;
/// Per-board raw width: 4 row blocks + one-hot strikes.
pub const BOARD_RAW: usize = 4 * ROW_BLOCK + 4; // 116
/// Full input: two boards + cdiff.
pub const AZ_FEATURES: usize = 2 * BOARD_RAW + 1; // 233

/// Per-row embedding width out of the shared encoder. Fixed (the trunk-input
/// width and the forward reshape depend on it); `AzModelConfig::init` asserts
/// the config matches. Tune by editing this constant, mirroring pair.rs's
/// fixed `D_H1`/`D_H2`.
pub const ENC_OUT: usize = 16;

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
        // 2-player net: same leading-opponent reduction as evaluate_batch_multi.
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
        assert!((b[BLANKS] - 0.1).abs() < 1e-6, "descending blanks (4-3)/10 = 0.1");
    }

    #[test]
    fn wprob_zero_on_unlockable_terminal() {
        // R marks 2,3,11 -> free = 12 (terminal) with only 3 marks: cannot lock,
        // so wprob must be 0.
        let mut s = State::default();
        for n in [2u8, 3, 11] {
            s.apply_mark(Mark { row: 0, number: n });
        }
        assert_eq!(s.row_free_values()[0], Some(12));
        assert_eq!(s.row_totals()[0], 3);
        let b = az_row_block(&s, 0);
        assert_eq!(b[WPROB], 0.0, "unlockable terminal => wprob 0");
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

    #[test]
    fn non_default_encoder_hidden_still_forwards() {
        // encoder_hidden is freely tunable (only encoder_out is load-bearing for
        // the reshape/trunk-input coupling). A non-default hidden width must
        // still build and forward.
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let model = AzModelConfig::new()
            .with_encoder_hidden(8)
            .init::<crate::dqn::MyBackend>(&device);
        let a = State::default();
        let b = State::default();
        let out = az_batch_forward(&model, &device, &[az_features(&a, &b)]);
        assert_eq!(out.len(), 1);
        assert!(out[0].0.is_finite() && out[0].1.is_finite());
    }

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

        // AzBatch takes ownership of `inputs`; keep a clone to reconstruct the
        // direct forward for the mean-matches-head check below.
        let inputs2 = inputs.clone();
        let batch = AzBatch {
            inputs,
            targets,
            final_diffs,
        };
        let out = model.forward_step(batch);
        let loss = out.loss.into_data().to_vec::<f32>().unwrap()[0];
        assert!(loss.is_finite(), "loss finite");
        assert!(loss >= 0.0, "loss non-negative");

        // (1) out.output must equal the μ column (narrow(1, 0, 1)) of a direct
        // forward over the same inputs. Pins the `output: mean` contract.
        let mu_head = model.forward(inputs2.clone()).narrow(1, 0, 1);
        let got = out.output.clone().into_data().to_vec::<f32>().unwrap();
        let want = mu_head.clone().into_data().to_vec::<f32>().unwrap();
        assert_eq!(got.len(), want.len(), "output / mu-head length match");
        for (g, w) in got.iter().zip(want.iter()) {
            assert!(
                (g - w).abs() <= 1e-6,
                "out.output tracks mu head: {g} vs {w}"
            );
        }

        // (2) The MSE term responds to target accuracy: feeding targets equal to
        // the model's own μ prediction (perfectly predicted mean) must yield a
        // finite loss strictly below the original loss, which used mismatched
        // targets 3.0/-3.0. We set final_diffs = μ too so the σ residual is
        // also small. The μ column is [2,1]; flatten to a length-2 vector.
        let mu_vec = mu_head.reshape([2]);
        let batch_perfect = AzBatch {
            inputs: inputs2,
            targets: mu_vec.clone(),
            final_diffs: mu_vec,
        };
        let out_perfect = model.forward_step(batch_perfect);
        let loss_perfect = out_perfect.loss.into_data().to_vec::<f32>().unwrap()[0];
        assert!(loss_perfect.is_finite(), "perfect-mu loss finite");
        assert!(
            loss_perfect < loss,
            "better mu prediction lowers loss: {loss_perfect} < {loss}"
        );
    }

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

    #[test]
    fn evaluate_batch_multi_matches_per_group_calls() {
        use crate::strategy::Bot;
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let bot = AzStrategy::from_model(AzModelConfig::new().init::<crate::dqn::MyBackend>(&device), device);

        let mut a = State::default();
        a.apply_mark(Mark { row: 0, number: 4 });
        let mut b = State::default();
        b.apply_mark(Mark { row: 2, number: 9 });
        let mut opp = State::default();
        opp.apply_mark(Mark { row: 1, number: 6 });

        let g1_c = [State::default(), a];
        let g1_o = [opp];
        let g2_c = [b];
        let g2_o = [State::default(), a]; // different leader situation

        let multi = bot.evaluate_batch_multi(&[(&g1_c[..], &g1_o[..]), (&g2_c[..], &g2_o[..])]);
        let solo1 = bot.evaluate_batch(&g1_c, &g1_o);
        let solo2 = bot.evaluate_batch(&g2_c, &g2_o);
        assert_eq!(multi.len(), 2);
        assert_eq!(multi[0], solo1);
        assert_eq!(multi[1], solo2);
    }

    #[test]
    fn evaluate_handles_empty_opponents() {
        use crate::strategy::Bot;
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let bot = AzStrategy::from_model(AzModelConfig::new().init::<crate::dqn::MyBackend>(&device), device);
        let v = bot.evaluate(&State::default(), &[]);
        assert!(v.is_finite());
    }
}
