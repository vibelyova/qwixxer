//! Pairwise differential network ("pair" bot): joint two-board evaluation.
//!
//! The model sees both players' boards and predicts the distribution of the
//! *future* score differential `final_diff − current_diff` as `(μ, log σ²)`.
//! Move selection ranks candidates by `(current_diff + μ) / σ`, monotone in
//! the Gaussian P(win) against the leading opponent.
//!
//! Design: docs/superpowers/specs/2026-06-10-pair-network-design.md

use super::{
    aggregate_weighted_probability, lockable_rows, row_progress, total_progress, MyBackend, LOG_VAR_MAX, LOG_VAR_MIN,
};
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
    pair_features_cached(our, &PairedContext::new(paired, all_opps))
}

/// The candidate-independent part of a pair-feature row: the paired
/// opponent's board block and the pair-level features 41–44. In a
/// multi-candidate group these are constant, so callers ranking many
/// candidates against one opponent set compute this once.
pub struct PairedContext {
    paired_points: isize,
    opp_block: [f32; BOARD_FEATURES],
    pair_level: [f32; 4], // features 41..=44
}

impl PairedContext {
    pub fn new(paired: &State, all_opps: &[State]) -> Self {
        // Invariant: `paired` must be one of `all_opps`.
        debug_assert!(all_opps.contains(paired));
        PairedContext {
            paired_points: paired.count_points(),
            opp_block: board_features(paired),
            pair_level: [
                all_opps.len() as f32 / 4.0,
                all_opps.iter().map(total_progress).fold(0.0, f32::max),
                all_opps.iter().map(|s| s.strikes).max().unwrap_or(0) as f32 / 3.0,
                all_opps.iter().map(lockable_rows).sum::<u8>() as f32 / 8.0,
            ],
        }
    }

    pub fn paired_points(&self) -> isize {
        self.paired_points
    }
}

/// `pair_features` with the candidate-independent part precomputed.
pub fn pair_features_cached(our: &State, ctx: &PairedContext) -> [f32; PAIR_FEATURES] {
    let mut f = [0.0f32; PAIR_FEATURES];
    f[..BOARD_FEATURES].copy_from_slice(&board_features(our));
    f[BOARD_FEATURES..2 * BOARD_FEATURES].copy_from_slice(&ctx.opp_block);

    let cdiff = (our.count_points() - ctx.paired_points) as f32;
    f[40] = (cdiff / 100.0).clamp(-1.0, 1.0);
    f[41..].copy_from_slice(&ctx.pair_level);
    f
}

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
            layer1: LinearConfig::new(PAIR_FEATURES, self.hidden1)
                .with_bias(true)
                .init(device),
            layer2: LinearConfig::new(self.hidden1, self.hidden2)
                .with_bias(true)
                .init(device),
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
        PairStrategy {
            model: Arc::new(model),
            device,
        }
    }

    pub fn from_model(model: PairModel<MyBackend>, device: burn::backend::ndarray::NdArrayDevice) -> Self {
        PairStrategy {
            model: Arc::new(model),
            device,
        }
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
        self.evaluate_batch_multi(&[(candidates, opp_states)]).pop().unwrap()
    }

    fn evaluate_batch_multi(&self, groups: &[(&[State], &[State])]) -> Vec<Vec<f32>> {
        let default_opps = [State::default()];
        let mut leaders: Vec<isize> = Vec::with_capacity(groups.len());
        let mut feats: Vec<[f32; PAIR_FEATURES]> = Vec::new();
        for (candidates, opp_states) in groups {
            // Solo fallback: rank against an empty default board. Solo play is
            // officially unsupported for the pair bot (the old DQN covers it).
            let opps: &[State] = if opp_states.is_empty() {
                &default_opps
            } else {
                opp_states
            };
            let leader = opps.iter().max_by_key(|s| s.count_points()).unwrap();
            let ctx = PairedContext::new(leader, opps);
            for c in *candidates {
                feats.push(pair_features_cached(c, &ctx));
            }
            leaders.push(ctx.paired_points());
        }
        let values = pair_batch_forward(&self.model, &self.device, &feats);

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

impl crate::strategy::search::WinProb for PairStrategy {
    fn win_prob_multi(&self, groups: &[(&State, &[State])]) -> Vec<f32> {
        let default_opps = [State::default()];
        let mut feats = Vec::with_capacity(groups.len());
        let mut cdiffs = Vec::with_capacity(groups.len());
        for (our, opps) in groups {
            let opps: &[State] = if opps.is_empty() { &default_opps } else { opps };
            let leader = opps.iter().max_by_key(|s| s.count_points()).unwrap();
            feats.push(pair_features(our, leader, opps));
            cdiffs.push((our.count_points() - leader.count_points()) as f32);
        }
        pair_batch_forward(&self.model, &self.device, &feats)
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

        // A reachable state with 5 marks whose free pointer rests on the
        // terminal (2,3,4,5,11 -> free = 12, total = 5): terminal now lockable,
        // so it contributes nonzero ways. (We rebuild because once the free
        // pointer reaches 12 the earlier numbers can no longer be marked.)
        let mut s = State::default();
        for n in [2u8, 3, 4, 5, 11] {
            s.apply_mark(Mark { row: 0, number: n });
        }
        assert_eq!(s.row_free_values()[0], Some(12));
        assert_eq!(s.row_totals()[0], 5);
        let f = board_features(&s);
        assert!(f[12] > 0.0);

        // Row 2 (descending): descending rows mark high->low, so mark
        // 12, 11, 3 in that order -> free = 2 (terminal) with only 3 marks.
        let mut s = State::default();
        for n in [12u8, 11, 3] {
            s.apply_mark(Mark { row: 2, number: n });
        }
        assert_eq!(s.row_free_values()[2], Some(2));
        assert_eq!(s.row_totals()[2], 3);
        let f = board_features(&s);
        assert_eq!(f[14], 0.0, "descending terminal with <5 marks must contribute 0 ways");
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
    fn evaluate_batch_multi_matches_per_group_calls() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let bot = PairStrategy::from_model(PairModelConfig::new().init::<MyBackend>(&device), device);

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
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let bot = PairStrategy::from_model(PairModelConfig::new().init::<MyBackend>(&device), device);
        let v = bot.evaluate(&State::default(), &[]);
        assert!(v.is_finite());
    }

    #[test]
    fn win_prob_multi_is_probability_and_monotone_in_score() {
        use crate::strategy::search::WinProb;
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let bot = PairStrategy::from_model(PairModelConfig::new().init::<MyBackend>(&device), device);

        // Same opponent; in the second group we are far ahead on points.
        let behind = State::default();
        let mut ahead = State::default();
        for n in 2..=8 {
            ahead.apply_mark(Mark { row: 0, number: n });
        }
        let opp = State::default();

        let p = bot.win_prob_multi(&[(&behind, &[opp]), (&ahead, &[opp])]);
        assert_eq!(p.len(), 2);
        assert!(p.iter().all(|x| (0.0..=1.0).contains(x)));
        // A 28-point lead with the same opponent must not be rated worse.
        assert!(p[1] >= p[0]);
    }
}
