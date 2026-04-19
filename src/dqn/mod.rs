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
pub const NUM_FEATURES: usize = 21;

/// Global seed for reproducible training. Used by `dqn::train`.
pub const TRAIN_SEED: u64 = 42;

/// Clamp bounds for `log_var`. Keeps `var ∈ [e^-5, e^10] ≈ [0.0067, 22 026]`,
/// covering all plausible score variances without allowing numerical blowup.
pub const LOG_VAR_MIN: f32 = -5.0;
pub const LOG_VAR_MAX: f32 = 10.0;

/// Context about opponents, updated via `observe_opponents`.
#[derive(Clone, Debug, Default)]
pub struct OpponentContext {
    pub num_opponents: u8,
    pub max_opponent_strikes: u8,
    pub score_gap_to_leader: isize, // positive = we're ahead
}

/// Extract features from a game state as a fixed-size float array.
pub fn state_features(state: &State, ctx: &OpponentContext) -> [f32; NUM_FEATURES] {
    let totals = state.row_totals();
    let frees = state.row_free_values();
    let locked = state.locked();

    let mut features = [0.0f32; NUM_FEATURES];
    for i in 0..4 {
        // Row progress (0=start, 1=done). Locked = 1.0 (completed).
        // Ascending (rows 0,1): (free-2)/10. Descending (rows 2,3): (12-free)/10.
        features[i] = match frees[i] {
            Some(f) if i < 2 => (f as f32 - 2.0) / 10.0,
            Some(f) => (12.0 - f as f32) / 10.0,
            None => 1.0,
        };
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
    // Strikes normalized
    features[16] = state.strikes as f32 / 4.0;
    // Blanks normalized
    features[17] = state.blanks() as f32 / 40.0;
    // Opponent context
    features[18] = ctx.num_opponents as f32 / 4.0;
    features[19] = ctx.max_opponent_strikes as f32 / 4.0;
    features[20] = (ctx.score_gap_to_leader as f32 / 100.0).clamp(-1.0, 1.0);

    features
}

/// Score a move candidate by P(win) under a Gaussian approximation, using the
/// max-opponent approach. Monotonic in `Φ((μ - c) / sqrt(2σ²))`, so argmax of
/// this function equals argmax of P(win).
///
/// We use the candidate's own variance as a proxy for both players' variance
/// (both are playing the same game from similar states), which gives a
/// ranking-stable √(2σ²) denominator up to the factor of √2. We drop the √2
/// since monotonic transforms don't change argmax — equivalent to ranking by
/// `(μ − c) / σ`.
pub fn win_rank_score(mean: f32, log_var: f32, threshold: f32) -> f32 {
    let log_var_clamped = log_var.clamp(LOG_VAR_MIN, LOG_VAR_MAX);
    // exp(-log_var / 2) = 1 / σ.
    let inv_sigma = (-0.5 * log_var_clamped).exp();
    (mean - threshold) * inv_sigma
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
            cache: std::collections::HashMap::new(),
        }
    }

    /// Evaluate a state with a custom opponent context (for state explorer).
    /// Returns `(mean, log_var)`.
    pub fn evaluate_with_context(&self, state: &State, ctx: &OpponentContext) -> (f32, f32) {
        let features = state_features(state, ctx);
        self.model.evaluate_state(&features, &self.device)
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

        // Batch-evaluate all resulting states. Rank by P(win) approximation
        // using the max-opponent threshold.
        let result_states: Vec<State> = moves
            .iter()
            .map(|&mov| {
                let mut s = *state;
                s.apply_move(mov);
                s
            })
            .collect();
        let values = self.evaluate_batch(&result_states);
        let threshold = (state.count_points() - self.context.score_gap_to_leader) as f32;

        moves
            .into_iter()
            .enumerate()
            .max_by(|(i, _), (j, _)| {
                let a = win_rank_score(values[*i].0, values[*i].1, threshold);
                let b = win_rank_score(values[*j].0, values[*j].1, threshold);
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

        // Batch-evaluate mark states + skip state; rank by P(win) approximation.
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
        let threshold = (state.count_points() - self.context.score_gap_to_leader) as f32;

        let skip = values.last().unwrap();
        let skip_rank = win_rank_score(skip.0, skip.1, threshold);

        let (best_idx, best_rank) = values[..moves.len()]
            .iter()
            .enumerate()
            .map(|(i, v)| (i, win_rank_score(v.0, v.1, threshold)))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        if best_rank > skip_rank {
            Some(moves[best_idx])
        } else {
            None
        }
    }

    fn observe_opponents(&mut self, our_score: isize, opponents: &[State]) {
        self.context.num_opponents = opponents.len() as u8;
        self.context.max_opponent_strikes = opponents.iter().map(|s| s.strikes).max().unwrap_or(0);
        let max_opp_score = opponents.iter().map(|s| s.count_points()).max().unwrap_or(0);
        self.context.score_gap_to_leader = our_score - max_opp_score;
    }
}
