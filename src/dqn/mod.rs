//! DQN value network inference.
//!
//! Defines the model architecture, feature extraction, and the inference-side
//! [`DqnStrategy`]. Training code lives in the [`train`] submodule (behind the
//! `dqn` feature).
//!
//! The model predicts a Gaussian over the final score: a mean `μ(s)` and a
//! raw `log_var(s) = log σ²(s)` per state. Move selection ranks candidates by
//! P(win) under a Normal approximation:
//!
//! ```text
//! P(win | m) ≈ Φ((μ_m - c) / sqrt(σ²_m + σ²_opp))
//! ```
//!
//! where `c` is the leading opponent's (current) score and we approximate
//! `σ²_opp` with the candidate's own variance (see [`win_rank_score`]).

#[cfg(feature = "dqn")]
pub mod train;

use crate::state::{Move, State};
use crate::strategy::Strategy;
use burn::{
    backend::ndarray::NdArray,
    nn::{Linear, LinearConfig, Relu},
    prelude::*,
    record::{HalfPrecisionSettings, NamedMpkBytesRecorder, Recorder},
};

/// Default backend for native and wasm inference.
pub type MyBackend = NdArray;

/// Number of input features for the state representation.
pub const NUM_FEATURES: usize = 25;

/// Global seed for reproducible training. Used by `dqn::train`.
pub const TRAIN_SEED: u64 = 42;

/// Clamp bounds for `log_var`. Keeps `var ∈ [e^-5, e^10] ≈ [0.0067, 22 026]`,
/// covering all plausible score variances without allowing numerical blowup.
pub const LOG_VAR_MIN: f32 = -5.0;
pub const LOG_VAR_MAX: f32 = 10.0;

/// Context about opponents, updated via `observe_opponents`. All fields are
/// pre-computed aggregates over opponents only; `state_features` combines
/// them with our own state to derive the global "max player X" features.
#[derive(Clone, Debug, Default)]
pub struct OpponentContext {
    pub num_opponents: u8,
    pub max_opponent_strikes: u8,
    pub score_gap_to_leader: isize, // positive = we're ahead
    pub max_opp_locks: u8,
    /// Max over opponents of (sum of 4 row-progresses / 4). Range [0, 1].
    pub max_opp_total_progress: f32,
    /// Count of opponent rows that are lockable (≥5 marks, not locked).
    pub opp_lockable_rows: u8,
}

// ---- Helpers used by both `state_features` and `observe_opponents` ----

fn row_progress(free: Option<u8>, ascending: bool) -> f32 {
    match free {
        Some(f) if ascending => (f as f32 - 2.0) / 10.0,
        Some(f) => (12.0 - f as f32) / 10.0,
        None => 1.0,
    }
}

/// Sum of the 4 row-progresses divided by 4 — range [0, 1]. Grows as the
/// player's free pointers advance; reaches 1.0 only when all 4 rows are locked.
pub fn total_progress(state: &State) -> f32 {
    let frees = state.row_free_values();
    let sum: f32 = (0..4).map(|i| row_progress(frees[i], i < 2)).sum();
    sum / 4.0
}

/// Count of rows that could be locked on a lucky next roll: non-locked with ≥5 marks.
pub fn lockable_rows(state: &State) -> u8 {
    let totals = state.row_totals();
    let locked = state.locked();
    (0..4).filter(|&i| !locked[i] && totals[i] >= 5).count() as u8
}

/// Compute the `OpponentContext` for a player whose score is `our_score` and
/// whose first opponent is `first_opp` plus any `extra_opps`. Used by
/// `observe_opponents` (viewing opponents from the DqnStrategy's perspective)
/// and by `leader_prediction` (viewing opponents from the leader's perspective).
pub fn build_opponent_context_for(
    our_score: isize,
    first_opp: &State,
    extra_opps: &[State],
) -> OpponentContext {
    let num_opponents = 1 + extra_opps.len() as u8;
    let mut max_opp_strikes = first_opp.strikes;
    let mut max_opp_score = first_opp.count_points();
    let mut max_opp_locks = first_opp.count_locked();
    let mut max_opp_total_progress = total_progress(first_opp);
    let mut opp_lockable_rows_sum = lockable_rows(first_opp);

    for s in extra_opps {
        max_opp_strikes = max_opp_strikes.max(s.strikes);
        max_opp_score = max_opp_score.max(s.count_points());
        max_opp_locks = max_opp_locks.max(s.count_locked());
        let tp = total_progress(s);
        if tp > max_opp_total_progress {
            max_opp_total_progress = tp;
        }
        opp_lockable_rows_sum += lockable_rows(s);
    }

    OpponentContext {
        num_opponents,
        max_opponent_strikes: max_opp_strikes,
        score_gap_to_leader: our_score - max_opp_score,
        max_opp_locks,
        max_opp_total_progress,
        opp_lockable_rows: opp_lockable_rows_sum,
    }
}

/// Aggregate weighted-probability: for each possible white-dice sum, weight by
/// P(rolling that sum) and the max (total+1)/11 across rows that could use it.
/// Avoids the per-row double-counting when multiple rows share a free pointer.
pub fn aggregate_weighted_probability(state: &State) -> f32 {
    let frees = state.row_free_values();
    let totals = state.row_totals();
    let locked = state.locked();

    let mut total = 0.0f32;
    for s in 2u8..=12 {
        let ways = 6 - ((s as i32) - 7).abs();
        let p_s = ways as f32 / 36.0;

        let mut best_value = 0.0f32;
        for i in 0..4 {
            if locked[i] {
                continue;
            }
            let Some(f) = frees[i] else { continue };
            let ascending = i < 2;
            let can_mark = match (ascending, s) {
                (true, 12) if totals[i] < 5 => false,
                (false, 2) if totals[i] < 5 => false,
                (true, _) => f <= s,
                (false, _) => f >= s,
            };
            if !can_mark {
                continue;
            }
            let value = (totals[i] as f32 + 1.0) / 11.0;
            if value > best_value {
                best_value = value;
            }
        }
        total += p_s * best_value;
    }
    total
}

/// Extract features from a game state as a fixed-size float array.
pub fn state_features(state: &State, ctx: &OpponentContext) -> [f32; NUM_FEATURES] {
    let totals = state.row_totals();
    let frees = state.row_free_values();
    let locked = state.locked();

    let mut features = [0.0f32; NUM_FEATURES];
    for i in 0..4 {
        // Row progress (0=start, 1=done). Locked = 1.0 (completed).
        features[i] = row_progress(frees[i], i < 2);
        // Row mark count normalized
        features[4 + i] = totals[i] as f32 / 11.0;
        // Row locked
        features[8 + i] = if locked[i] { 1.0 } else { 0.0 };
        // Per-row weighted probability: P(free) * (total+1), normalized
        features[12 + i] = match frees[i] {
            Some(f) => {
                let ways = 6.0 - (7.0f32 - f as f32).abs();
                (ways / 6.0) * (totals[i] as f32 + 1.0) / 11.0
            }
            None => 0.0,
        };
    }
    // Our own stats. Max in-play strikes is 3 (4 ends the game).
    features[16] = state.strikes as f32 / 3.0;
    features[17] = state.blanks() as f32 / 40.0;

    // Opponent / global context.
    features[18] = ctx.num_opponents as f32 / 4.0;
    // Score-gap clamped to [-1, 1] at ±100 points.
    features[19] = (ctx.score_gap_to_leader as f32 / 100.0).clamp(-1.0, 1.0);

    // Max-across-all-players (game-end proximity signals).
    // Normalized by the maximum in-play value (game ends beyond that):
    //   strikes: max in-play = 3 (4 ends the game).
    //   locks:   max in-play = 1 (2 ends the game) — so the feature is ∈ {0, 1}.
    let max_player_strikes = state.strikes.max(ctx.max_opponent_strikes);
    features[20] = max_player_strikes as f32 / 3.0;
    let max_player_locks = state.count_locked().max(ctx.max_opp_locks);
    features[21] = (max_player_locks as f32).min(1.0);

    // Game-duration proxy: leading player's row-progress sum (normalized).
    features[22] = total_progress(state).max(ctx.max_opp_total_progress);

    // Aggregate probability over usable dice sums.
    features[23] = aggregate_weighted_probability(state);

    // Count of rows ready to lock across all players. Not divided by num_players
    // on purpose — multi-player games genuinely have more threat.
    let our_lockable = lockable_rows(state);
    features[24] = (our_lockable + ctx.opp_lockable_rows) as f32 / 8.0;

    features
}

/// Score a move candidate by P(win) under a Gaussian approximation with the
/// max-opponent as the comparison target. Monotonic in
/// `Φ((μ_us − μ_opp) / sqrt(σ²_us + σ²_opp))`, so argmax of this function equals
/// argmax of P(win).
///
/// Requires evaluating V on the leading opponent's actual state — the caller
/// supplies `(opp_mean, opp_log_var)` from that evaluation.
pub fn win_rank_score(
    us_mean: f32,
    us_log_var: f32,
    opp_mean: f32,
    opp_log_var: f32,
) -> f32 {
    let us_lv = us_log_var.clamp(LOG_VAR_MIN, LOG_VAR_MAX);
    let opp_lv = opp_log_var.clamp(LOG_VAR_MIN, LOG_VAR_MAX);
    let total_var = us_lv.exp() + opp_lv.exp();
    (us_mean - opp_mean) / total_var.sqrt()
}

// ---- Model ----

#[derive(Module, Debug)]
pub struct QwixxModel<B: Backend> {
    layer1: Linear<B>,
    layer2: Linear<B>,
    output_mean: Linear<B>,
    output_log_var: Linear<B>,
    activation: Relu,
}

#[derive(Config, Debug)]
pub struct QwixxModelConfig {
    #[config(default = 128)]
    pub hidden1: usize,
    #[config(default = 64)]
    pub hidden2: usize,
}

impl QwixxModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> QwixxModel<B> {
        QwixxModel {
            layer1: LinearConfig::new(NUM_FEATURES, self.hidden1).with_bias(true).init(device),
            layer2: LinearConfig::new(self.hidden1, self.hidden2).with_bias(true).init(device),
            output_mean: LinearConfig::new(self.hidden2, 1).with_bias(true).init(device),
            output_log_var: LinearConfig::new(self.hidden2, 1).with_bias(true).init(device),
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> QwixxModel<B> {
    /// Forward pass returning `[batch, 2]`: column 0 is predicted mean of the
    /// final score, column 1 is raw `log σ²`. Training clamps `log_var` in the
    /// loss; inference callers clamp via [`win_rank_score`].
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.activation.forward(self.layer1.forward(input));
        let x = self.activation.forward(self.layer2.forward(x));
        let mean = self.output_mean.forward(x.clone());
        let log_var = self.output_log_var.forward(x);
        Tensor::cat(vec![mean, log_var], 1)
    }

    /// Evaluate a single state, returning `(mean, log_var)`.
    pub fn evaluate_state(&self, features: &[f32; NUM_FEATURES], device: &B::Device) -> (f32, f32) {
        let input = Tensor::<B, 1>::from_floats(features.as_slice(), device)
            .reshape([1, NUM_FEATURES]);
        let output = self.forward(input);
        let v = output.into_data().to_vec::<f32>().unwrap();
        (v[0], v[1])
    }
}

// ---- Strategy using trained model ----

pub struct DqnStrategy {
    model: QwixxModel<MyBackend>,
    device: burn::backend::ndarray::NdArrayDevice,
    context: OpponentContext,
    /// State of the current leading opponent (max score). Captured by
    /// `observe_opponents`; used in ranking to evaluate V on the leader's
    /// actual state so we have both μ_opp and σ²_opp.
    leader_state: Option<State>,
    /// States of non-leader opponents, needed to build the leader's own
    /// `OpponentContext` (their gap, strikes, locks, etc.) on demand.
    non_leader_states: Vec<State>,
    /// Cache stores `(mean, log_var)` pairs keyed by feature bits.
    cache: std::collections::HashMap<[u32; NUM_FEATURES], (f32, f32)>,
}

impl std::fmt::Debug for DqnStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "DqnStrategy")
    }
}

impl DqnStrategy {
    pub fn load(artifact_dir: &str) -> Self {
        use burn::record::CompactRecorder;
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let model = QwixxModelConfig::new()
            .init::<MyBackend>(&device)
            .load_file(format!("{artifact_dir}/model"), &CompactRecorder::new(), &device)
            .expect("Failed to load model");
        DqnStrategy {
            model,
            device,
            context: OpponentContext::default(),
            leader_state: None,
            non_leader_states: Vec::new(),
            cache: std::collections::HashMap::new(),
        }
    }

    /// Load model from embedded bytes (works in WASM and native).
    pub fn load_from_bytes(model_bytes: &[u8]) -> Self {
        use burn::module::Module;
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let recorder = NamedMpkBytesRecorder::<HalfPrecisionSettings>::new();
        let record: <QwixxModel<MyBackend> as Module<MyBackend>>::Record = recorder
            .load(model_bytes.to_vec(), &device)
            .expect("Failed to load model from bytes");
        let model = QwixxModelConfig::new()
            .init::<MyBackend>(&device)
            .load_record(record);
        DqnStrategy {
            model,
            device,
            context: OpponentContext::default(),
            leader_state: None,
            non_leader_states: Vec::new(),
            cache: std::collections::HashMap::new(),
        }
    }

    /// Evaluate a state with a custom opponent context (for state explorer).
    /// Returns `(mean, log_var)`.
    pub fn evaluate_with_context(&self, state: &State, ctx: &OpponentContext) -> (f32, f32) {
        let features = state_features(state, ctx);
        self.model.evaluate_state(&features, &self.device)
    }

    /// Run V on the stored leader's state, building the leader's own
    /// `OpponentContext` from `our_state` plus the stored non-leader states.
    /// Returns `(0.0, 0.0)` if there is no leader.
    fn leader_prediction(&mut self, our_state: &State) -> (f32, f32) {
        let Some(leader) = self.leader_state else {
            return (0.0, 0.0);
        };
        // Leader's "opponents" are us + the non-leader opponents.
        let leader_ctx = build_opponent_context_for(
            leader.count_points(),
            our_state,
            &self.non_leader_states,
        );
        let features = state_features(&leader, &leader_ctx);
        let key = Self::features_key(&features);
        if let Some(&cached) = self.cache.get(&key) {
            return cached;
        }
        let value = self.model.evaluate_state(&features, &self.device);
        self.cache.insert(key, value);
        value
    }

    fn features_key(features: &[f32; NUM_FEATURES]) -> [u32; NUM_FEATURES] {
        let mut key = [0u32; NUM_FEATURES];
        for i in 0..NUM_FEATURES {
            key[i] = features[i].to_bits();
        }
        key
    }

    #[allow(dead_code)]
    fn evaluate(&mut self, state: &State) -> (f32, f32) {
        let features = state_features(state, &self.context);
        let key = Self::features_key(&features);
        if let Some(&cached) = self.cache.get(&key) {
            return cached;
        }
        let value = self.model.evaluate_state(&features, &self.device);
        self.cache.insert(key, value);
        value
    }

    /// Batch evaluate multiple states at once — single forward pass.
    /// Returns `(mean, log_var)` per state.
    fn evaluate_batch(&mut self, states: &[State]) -> Vec<(f32, f32)> {
        if states.is_empty() {
            return Vec::new();
        }

        let mut results = vec![(0.0f32, 0.0f32); states.len()];
        let mut uncached_indices = Vec::new();
        let mut uncached_features = Vec::new();

        // Check cache first
        for (i, state) in states.iter().enumerate() {
            let features = state_features(state, &self.context);
            let key = Self::features_key(&features);
            if let Some(&cached) = self.cache.get(&key) {
                results[i] = cached;
            } else {
                uncached_indices.push(i);
                uncached_features.push(features);
            }
        }

        // Batch forward pass for uncached states — output shape [n, 2], row = [mean, log_var].
        if !uncached_features.is_empty() {
            let n = uncached_features.len();
            let flat: Vec<f32> = uncached_features.iter().flat_map(|f| f.iter().copied()).collect();
            let input = Tensor::<MyBackend, 1>::from_floats(flat.as_slice(), &self.device)
                .reshape([n, NUM_FEATURES]);
            let output = self.model.forward(input);
            let values = output.into_data().to_vec::<f32>().unwrap();

            for (j, &idx) in uncached_indices.iter().enumerate() {
                let pair = (values[j * 2], values[j * 2 + 1]);
                results[idx] = pair;
                let key = Self::features_key(&uncached_features[j]);
                self.cache.insert(key, pair);
            }
        }

        results
    }
}

impl Strategy for DqnStrategy {
    fn your_move(&mut self, state: &State, dice: [u8; 6]) -> Move {
        use crate::state::MetaDecision;
        let moves = match state.apply_meta_rules(dice, self.context.score_gap_to_leader) {
            MetaDecision::Forced(mov) => return mov,
            MetaDecision::Choices(moves) => moves,
        };

        // Evaluate V on the leading opponent's actual state first, then rank
        // candidate post-move states by the full P(win) formula.
        let (opp_mean, opp_log_var) = self.leader_prediction(state);
        let result_states: Vec<State> = moves
            .iter()
            .map(|&mov| {
                let mut s = *state;
                s.apply_move(mov);
                s
            })
            .collect();
        let values = self.evaluate_batch(&result_states);

        moves
            .into_iter()
            .enumerate()
            .max_by(|(i, _), (j, _)| {
                let a = win_rank_score(values[*i].0, values[*i].1, opp_mean, opp_log_var);
                let b = win_rank_score(values[*j].0, values[*j].1, opp_mean, opp_log_var);
                a.partial_cmp(&b).unwrap()
            })
            .unwrap()
            .1
    }

    fn opponents_move(&mut self, state: &State, number: u8, locked: [bool; 4]) -> Option<Move> {
        let moves = state.generate_opponent_moves(number);

        if let Some(mov) = state.find_smart_lock(&moves, self.context.score_gap_to_leader) {
            return Some(mov);
        }

        if moves.is_empty() {
            return None;
        }

        let (opp_mean, opp_log_var) = self.leader_prediction(state);
        let mut eval_states: Vec<State> = moves
            .iter()
            .map(|&mov| {
                let mut s = *state;
                s.apply_move(mov);
                s.lock(locked);
                s
            })
            .collect();
        let mut skip_state = *state;
        skip_state.lock(locked);
        eval_states.push(skip_state);

        let values = self.evaluate_batch(&eval_states);

        let skip = values.last().unwrap();
        let skip_rank = win_rank_score(skip.0, skip.1, opp_mean, opp_log_var);

        let (best_idx, best_rank) = values[..moves.len()]
            .iter()
            .enumerate()
            .map(|(i, v)| (i, win_rank_score(v.0, v.1, opp_mean, opp_log_var)))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        if best_rank > skip_rank {
            Some(moves[best_idx])
        } else {
            None
        }
    }

    fn observe_opponents(&mut self, our_score: isize, opponents: &[State]) {
        if opponents.is_empty() {
            self.context = OpponentContext::default();
            self.leader_state = None;
            self.non_leader_states.clear();
            // Preserve the gap semantics from before (no opp → 0 gap).
            self.context.score_gap_to_leader = 0;
            return;
        }

        let leader_idx = (0..opponents.len())
            .max_by_key(|&i| opponents[i].count_points())
            .unwrap();
        let leader = opponents[leader_idx];
        let non_leader: Vec<State> = opponents
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != leader_idx)
            .map(|(_, s)| *s)
            .collect();

        self.context = build_opponent_context_for(our_score, &leader, &non_leader);
        self.leader_state = Some(leader);
        self.non_leader_states = non_leader;
    }
}
