// ---- Imports for inference (always available with `burn` feature) ----
use crate::state::{Move, State};
use crate::strategy::Strategy;
use burn::{
    backend::ndarray::NdArray,
    nn::{Linear, LinearConfig, Relu},
    prelude::*,
    record::{NamedMpkBytesRecorder, HalfPrecisionSettings, Recorder},
};

// ---- Imports for training (only with `dqn` feature) ----
#[cfg(feature = "dqn")]
use crate::bot::{self, DNA};
#[cfg(feature = "dqn")]
use crate::mcts::MonteCarlo;
#[cfg(feature = "dqn")]
use burn::{
    backend::Autodiff,
    data::{
        dataloader::{DataLoaderBuilder, batcher::Batcher},
        dataset::{Dataset, InMemDataset},
    },
    nn::loss::{MseLoss, Reduction::Mean},
    optim::AdamConfig,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{Learner, RegressionOutput, SupervisedTraining, TrainOutput, TrainStep, InferenceStep, metric::LossMetric},
};
#[cfg(feature = "dqn")]
use rayon::prelude::*;
#[cfg(feature = "dqn")]
use rand::{rngs::SmallRng, Rng, SeedableRng};
#[cfg(feature = "dqn")]
use std::sync::Arc;

// Backend type aliases
type MyBackend = NdArray;
#[cfg(feature = "dqn")]
type MyAutodiffBackend = Autodiff<MyBackend>;

/// Number of input features for the state representation.
pub const NUM_FEATURES: usize = 21;

/// Global seed for reproducible training. Change to get a different run.
pub const TRAIN_SEED: u64 = 42;

/// Context about opponents, updated via observe_opponents.
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

/// Build opponent context from a 2-player game state.
#[cfg(feature = "dqn")]
fn make_context(our_state: &State, opponent_state: &State) -> OpponentContext {
    OpponentContext {
        num_opponents: 1,
        max_opponent_strikes: opponent_state.strikes,
        score_gap_to_leader: our_state.count_points() - opponent_state.count_points(),
    }
}

// ---- Model (inference, always available) ----

#[derive(Module, Debug)]
pub struct QwixxModel<B: Backend> {
    layer1: Linear<B>,
    layer2: Linear<B>,
    output: Linear<B>,
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
            output: LinearConfig::new(self.hidden2, 1).with_bias(true).init(device),
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> QwixxModel<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.activation.forward(self.layer1.forward(input));
        let x = self.activation.forward(self.layer2.forward(x));
        self.output.forward(x)
    }

    /// Evaluate a single state, returning the predicted value.
    pub fn evaluate_state(&self, features: &[f32; NUM_FEATURES], device: &B::Device) -> f32 {
        let input = Tensor::<B, 1>::from_floats(features.as_slice(), device)
            .reshape([1, NUM_FEATURES]);
        let output = self.forward(input);
        output.into_data().to_vec::<f32>().unwrap()[0]
    }
}

// ---- Training step impls (dqn feature only) ----

#[cfg(feature = "dqn")]
impl<B: Backend> QwixxModel<B> {
    pub fn forward_step(&self, batch: QwixxBatch<B>) -> RegressionOutput<B> {
        let targets = batch.targets.clone().unsqueeze_dim(1);
        let output = self.forward(batch.inputs);
        let loss = MseLoss::new().forward(output.clone(), targets.clone(), Mean);
        RegressionOutput { loss, output, targets }
    }
}

#[cfg(feature = "dqn")]
impl<B: AutodiffBackend> TrainStep for QwixxModel<B> {
    type Input = QwixxBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, batch: QwixxBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_step(batch);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

#[cfg(feature = "dqn")]
impl<B: Backend> InferenceStep for QwixxModel<B> {
    type Input = QwixxBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, batch: QwixxBatch<B>) -> RegressionOutput<B> {
        self.forward_step(batch)
    }
}

// (DqnStrategy defined below after training code, with cache + batch)

// ===========================================================================
// Training code (native only, requires `dqn` feature)
// ===========================================================================

#[cfg(feature = "dqn")]
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TrainingSample {
    pub features: Vec<f32>,
    pub value: f32,
}

#[cfg(feature = "dqn")]
#[derive(Clone)]
pub struct QwixxBatcher<B: Backend> {
    _phantom: std::marker::PhantomData<B>,
}

#[cfg(feature = "dqn")]
#[derive(Clone, Debug)]
pub struct QwixxBatch<B: Backend> {
    pub inputs: Tensor<B, 2>,
    pub targets: Tensor<B, 1>,
}

#[cfg(feature = "dqn")]
impl<B: Backend> Batcher<B, TrainingSample, QwixxBatch<B>> for QwixxBatcher<B> {
    fn batch(&self, items: Vec<TrainingSample>, device: &B::Device) -> QwixxBatch<B> {
        let batch_size = items.len();
        // Seed from batch contents for deterministic augmentation
        let batch_seed = TRAIN_SEED
            .wrapping_add(items[0].value.to_bits() as u64)
            .wrapping_add(batch_size as u64);
        let mut rng = SmallRng::seed_from_u64(batch_seed);

        // Data augmentation: 3 independent swaps (8 permutations).
        // Per-row feature indices: progress [0-3], marks [4-7], locked [8-11], weighted_prob [12-15].
        let inputs: Vec<f32> = items
            .iter()
            .flat_map(|s| {
                let mut f = s.features.clone();
                // Swap red(0)↔yellow(1) within ascending pair
                if rng.gen::<bool>() {
                    for &base in &[0, 4, 8, 12] {
                        f.swap(base, base + 1);
                    }
                }
                // Swap green(2)↔blue(3) within descending pair
                if rng.gen::<bool>() {
                    for &base in &[2, 6, 10, 14] {
                        f.swap(base, base + 1);
                    }
                }
                // Swap ascending(0,1)↔descending(2,3) pairs
                if rng.gen::<bool>() {
                    for &base in &[0, 4, 8, 12] {
                        f.swap(base, base + 2);     // 0↔2, 4↔6, 8↔10, 12↔14
                        f.swap(base + 1, base + 3); // 1↔3, 5↔7, 9↔11, 13↔15
                    }
                }
                f
            })
            .collect();
        let targets: Vec<f32> = items.iter().map(|s| s.value).collect();

        let inputs = Tensor::<B, 1>::from_floats(inputs.as_slice(), device)
            .reshape([batch_size, NUM_FEATURES]);
        let targets = Tensor::<B, 1>::from_floats(targets.as_slice(), device);

        QwixxBatch { inputs, targets }
    }
}

// ---- Data generation ----

#[cfg(feature = "dqn")]
/// Generate training data by playing games and evaluating states with MC.
pub fn generate_training_data(num_games: usize, mc_sims: usize) -> Vec<TrainingSample> {
    let genes = Arc::new(bot::default_genes());
    let champion = DNA::load_weights("champion.txt", genes).expect("No champion.txt found");
    let mc = MonteCarlo::with_ga(mc_sims, champion.clone());

    println!("Generating training data: {num_games} games, {mc_sims} MC sims per state...");

    let samples: Vec<TrainingSample> = (0..num_games)
        .into_par_iter()
        .flat_map(|game_idx| {
            let mut ga = champion.clone();
            let mut rng = SmallRng::seed_from_u64(TRAIN_SEED.wrapping_add(game_idx as u64));
            let mut state = State::default();
            let mut opponent_state = State::default();
            let mut local_samples = Vec::new();

            let mut turn = 0u32;
            loop {
                if state.strikes >= 4 || opponent_state.strikes >= 4 { break; }
                if state.count_locked() >= 2 || opponent_state.count_locked() >= 2 { break; }

                let dice: [u8; 6] = core::array::from_fn(|_| rng.gen_range(1..=6));
                let on_white = dice[0] + dice[1];

                if turn % 2 == 0 {
                    // Our active turn — evaluate each candidate with MC
                    let moves = state.generate_moves(dice);
                    let mut all_moves = moves;
                    all_moves.push(Move::Strike);

                    // Evaluate all moves once, cache results
                    let evaluated: Vec<(Move, f64)> = all_moves
                        .iter()
                        .map(|&mov| {
                            let mc_value = mc.evaluate_move_public(&state, mov, &opponent_state);
                            (mov, mc_value)
                        })
                        .collect();

                    for &(mov, mc_value) in &evaluated {
                        let mut new_state = state;
                        new_state.apply_move(mov);
                        let ctx = make_context(&new_state, &opponent_state);
                        let features = state_features(&new_state, &ctx);
                        local_samples.push(TrainingSample {
                            features: features.to_vec(),
                            value: mc_value as f32,
                        });
                    }

                    // Play the MC-best move
                    let best_move = evaluated
                        .iter()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .unwrap()
                        .0;
                    state.apply_move(best_move);
                    let locked = state.locked();
                    let opp_mov = ga.opponents_move(&opponent_state, on_white, locked);
                    if let Some(mov) = opp_mov { opponent_state.apply_move(mov); }
                    opponent_state.lock(locked);
                    state.lock(opponent_state.locked());
                } else {
                    // Opponent's active turn
                    let opp_move = ga.your_move(&opponent_state, dice);
                    opponent_state.apply_move(opp_move);
                    let locked = opponent_state.locked();
                    let our_mov = ga.opponents_move(&state, on_white, locked);
                    if let Some(mov) = our_mov { state.apply_move(mov); }
                    state.lock(locked);
                    opponent_state.lock(state.locked());
                }

                turn += 1;
                if turn > 200 { break; }
            }

            if game_idx % 100 == 0 && game_idx > 0 {
                eprintln!("  {game_idx}/{num_games} games...");
            }

            local_samples
        })
        .collect();

    println!("Generated {} training samples from {num_games} games", samples.len());
    samples
}

// ---- Training ----

#[cfg(feature = "dqn")]
pub fn train(samples: Vec<TrainingSample>, artifact_dir: &str) {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;

    // Split 90/10
    let split = (samples.len() * 9) / 10;
    let train_data = InMemDataset::new(samples[..split].to_vec());
    let valid_data = InMemDataset::new(samples[split..].to_vec());

    println!("Training: {} samples, validation: {} samples", train_data.len(), valid_data.len());

    let model = QwixxModelConfig::new().init::<MyAutodiffBackend>(&device);

    let batcher_train = QwixxBatcher::<MyAutodiffBackend> { _phantom: std::marker::PhantomData };
    let batcher_valid = QwixxBatcher::<MyBackend> { _phantom: std::marker::PhantomData };

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(1024)
        .shuffle(TRAIN_SEED)
        .num_workers(2)
        .build(train_data);

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(1024)
        .shuffle(TRAIN_SEED)
        .num_workers(2)
        .build(valid_data);

    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();

    let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_valid)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(50)
        .summary();

    let result = training.launch(Learner::new(model, AdamConfig::new().init(), 1e-3));

    result
        .model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Failed to save model");

    println!("Model saved to {artifact_dir}/model");
}

// ---- Strategy using trained model ----

pub struct DqnStrategy {
    model: QwixxModel<MyBackend>,
    device: burn::backend::ndarray::NdArrayDevice,
    context: OpponentContext,
    cache: std::collections::HashMap<[u32; NUM_FEATURES], f32>,
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
        DqnStrategy { model, device, context: OpponentContext::default(), cache: std::collections::HashMap::new() }
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
        DqnStrategy { model, device, context: OpponentContext::default(), cache: std::collections::HashMap::new() }
    }

    /// Evaluate a state with a custom opponent context (for state explorer).
    pub fn evaluate_with_context(&self, state: &State, ctx: &OpponentContext) -> f32 {
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
    fn evaluate(&mut self, state: &State) -> f32 {
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
    fn evaluate_batch(&mut self, states: &[State]) -> Vec<f32> {
        if states.is_empty() {
            return Vec::new();
        }

        let mut results = vec![0.0f32; states.len()];
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

        // Batch forward pass for uncached states
        if !uncached_features.is_empty() {
            let n = uncached_features.len();
            let flat: Vec<f32> = uncached_features.iter().flat_map(|f| f.iter().copied()).collect();
            let input = Tensor::<MyBackend, 1>::from_floats(flat.as_slice(), &self.device)
                .reshape([n, NUM_FEATURES]);
            let output = self.model.forward(input).reshape([n]);
            let values = output.into_data().to_vec::<f32>().unwrap();

            for (j, &idx) in uncached_indices.iter().enumerate() {
                results[idx] = values[j];
                let key = Self::features_key(&uncached_features[j]);
                self.cache.insert(key, values[j]);
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

        // Batch: compute all resulting states, evaluate in one pass
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
            .max_by(|(i, _), (j, _)| values[*i].partial_cmp(&values[*j]).unwrap())
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

        // Batch: compute all resulting states + skip state, evaluate in one pass
        let mut eval_states: Vec<State> = moves
            .iter()
            .map(|&mov| {
                let mut s = *state;
                s.apply_move(mov);
                s.lock(locked);
                s
            })
            .collect();
        // Add skip state (no mark, just lock)
        let mut skip_state = *state;
        skip_state.lock(locked);
        eval_states.push(skip_state);

        let values = self.evaluate_batch(&eval_states);
        let skip_value = *values.last().unwrap();

        let (best_idx, best_value) = values[..moves.len()]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        if *best_value > skip_value {
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

// ---- Self-play RL training ----

#[cfg(feature = "dqn")]
/// Pick a move from candidates using the model, with epsilon-greedy exploration.
fn pick_move_with_model(
    model: &QwixxModel<MyBackend>,
    device: &burn::backend::ndarray::NdArrayDevice,
    state: &State,
    moves: &[Move],
    epsilon: f32,
    rng: &mut SmallRng,
    ctx: &OpponentContext,
) -> Move {
    if rng.gen::<f32>() < epsilon {
        return moves[rng.gen_range(0..moves.len())];
    }

    *moves
        .iter()
        .max_by(|&&a, &&b| {
            let mut sa = *state;
            sa.apply_move(a);
            let va = model.evaluate_state(&state_features(&sa, ctx), device);
            let mut sb = *state;
            sb.apply_move(b);
            let vb = model.evaluate_state(&state_features(&sb, ctx), device);
            va.partial_cmp(&vb).unwrap()
        })
        .unwrap()
}

#[cfg(feature = "dqn")]
/// Pick an opponent-turn move using the model.
fn pick_passive_move_with_model(
    model: &QwixxModel<MyBackend>,
    device: &burn::backend::ndarray::NdArrayDevice,
    state: &State,
    number: u8,
    locked: [bool; 4],
    ctx: &OpponentContext,
) -> Option<Move> {
    let moves = state.generate_opponent_moves(number);
    if moves.is_empty() {
        return None;
    }

    // Smart lock
    if let Some(mov) = state.find_smart_lock(&moves, ctx.score_gap_to_leader) {
        return Some(mov);
    }

    let best = moves
        .iter()
        .max_by(|&&a, &&b| {
            let mut sa = *state;
            sa.apply_move(a);
            sa.lock(locked);
            let va = model.evaluate_state(&state_features(&sa, ctx), device);
            let mut sb = *state;
            sb.apply_move(b);
            sb.lock(locked);
            let vb = model.evaluate_state(&state_features(&sb, ctx), device);
            va.partial_cmp(&vb).unwrap()
        })
        .copied()
        .unwrap();

    let mut mark_state = *state;
    mark_state.apply_move(best);
    mark_state.lock(locked);
    let mut skip_state = *state;
    skip_state.lock(locked);

    let mark_val = model.evaluate_state(&state_features(&mark_state, ctx), device);
    let skip_val = model.evaluate_state(&state_features(&skip_state, ctx), device);

    if mark_val > skip_val {
        Some(best)
    } else {
        None
    }
}

#[cfg(feature = "dqn")]
/// A simple wrapper to use the DQN model as an opponent (no epsilon, no recording).
struct DqnSelfPlayOpponent {
    model: QwixxModel<MyBackend>,
    device: burn::backend::ndarray::NdArrayDevice,
    rng: SmallRng,
}

#[cfg(feature = "dqn")]
impl std::fmt::Debug for DqnSelfPlayOpponent {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "DqnSelfPlayOpponent")
    }
}

#[cfg(feature = "dqn")]
impl Strategy for DqnSelfPlayOpponent {
    fn your_move(&mut self, state: &State, dice: [u8; 6]) -> Move {
        use crate::state::MetaDecision;
        let ctx = OpponentContext::default();
        match state.apply_meta_rules(dice, ctx.score_gap_to_leader) {
            MetaDecision::Forced(mov) => mov,
            MetaDecision::Choices(moves) => {
                pick_move_with_model(&self.model, &self.device, state, &moves, 0.0, &mut self.rng, &ctx)
            }
        }
    }

    fn opponents_move(&mut self, state: &State, number: u8, locked: [bool; 4]) -> Option<Move> {
        let ctx = OpponentContext::default();
        pick_passive_move_with_model(&self.model, &self.device, state, number, locked, &ctx)
    }
}

#[cfg(feature = "dqn")]
/// Build opponent context for player 0 given all player states.
fn make_context_multi(our_state: &State, all_states: &[State]) -> OpponentContext {
    let our_score = our_state.count_points();
    let mut max_strikes = 0u8;
    let mut max_score = our_score;
    let mut num_opponents = 0u8;
    for s in all_states.iter().skip(1) {
        num_opponents += 1;
        max_strikes = max_strikes.max(s.strikes);
        max_score = max_score.max(s.count_points());
    }
    OpponentContext {
        num_opponents,
        max_opponent_strikes: max_strikes,
        score_gap_to_leader: our_score - max_score,
    }
}

#[cfg(feature = "dqn")]
/// Play a training game: DQN is player 0, opponents are arbitrary strategies.
/// Records (state_features, final_score) for player 0.
fn play_training_game(
    model: &QwixxModel<MyBackend>,
    device: &burn::backend::ndarray::NdArrayDevice,
    opponents: &mut [Box<dyn Strategy>],
    epsilon: f32,
    rng: &mut SmallRng,
) -> (Vec<TrainingSample>, f32) {
    let num_players = 1 + opponents.len();
    let mut states: Vec<State> = (0..num_players).map(|_| State::default()).collect();
    let mut recorded_features: Vec<[f32; NUM_FEATURES]> = Vec::new();
    let mut turn = 0u32;

    loop {
        // Game over check
        if states.iter().any(|s| s.strikes >= 4) { break; }
        if states.iter().any(|s| s.count_locked() >= 2) { break; }

        let dice: [u8; 6] = core::array::from_fn(|_| rng.gen_range(1..=6));
        let on_white = dice[0] + dice[1];
        let active = (turn as usize) % num_players;

        if active == 0 {
            // DQN's active turn
            let ctx = make_context_multi(&states[0], &states);
            let mov = match states[0].apply_meta_rules(dice, ctx.score_gap_to_leader) {
                crate::state::MetaDecision::Forced(mov) => mov,
                crate::state::MetaDecision::Choices(moves) => {
                    pick_move_with_model(model, device, &states[0], &moves, epsilon, rng, &ctx)
                }
            };
            let mut s = states[0];
            s.apply_move(mov);
            let ctx_after = make_context_multi(&s, &states);
            recorded_features.push(state_features(&s, &ctx_after));
            states[0].apply_move(mov);
        } else {
            // Opponent's active turn
            let opp_idx = active - 1;
            let opp_mov = opponents[opp_idx].your_move(&states[active], dice);
            states[active].apply_move(opp_mov);
        }

        let mut new_locked = states[active].locked();

        // Passive turns for all non-active players
        for idx in 1..num_players {
            let passive = (active + idx) % num_players;
            if passive == 0 {
                // DQN's passive turn
                let ctx = make_context_multi(&states[0], &states);
                let our_mov = pick_passive_move_with_model(
                    model, device, &states[0], on_white, new_locked, &ctx,
                );
                if let Some(mov) = our_mov {
                    let mut s = states[0];
                    s.apply_move(mov);
                    s.lock(new_locked);
                    let ctx_after = make_context_multi(&s, &states);
                    recorded_features.push(state_features(&s, &ctx_after));
                    states[0].apply_move(mov);
                }
            } else {
                let opp_idx = passive - 1;
                let opp_mov = opponents[opp_idx].opponents_move(
                    &states[passive], on_white, new_locked,
                );
                if let Some(mov) = opp_mov {
                    states[passive].apply_move(mov);
                }
            }
            for (row, locked) in states[passive].locked().iter().enumerate() {
                new_locked[row] |= *locked;
            }
        }

        // Propagate all locks
        for s in states.iter_mut() {
            s.lock(new_locked);
        }

        turn += 1;
        if turn > 200 { break; }
    }

    let final_score = states[0].count_points() as f32;

    // Compute TD(lambda) targets backwards through the trajectory
    let lambda = 0.8f32;
    let n = recorded_features.len();
    if n == 0 {
        return (Vec::new(), final_score);
    }

    // Get model's value estimates for each recorded state
    let values: Vec<f32> = recorded_features
        .iter()
        .map(|f| model.evaluate_state(f, device))
        .collect();

    // Compute targets: G_t = (1-lambda)*V(s_{t+1}) + lambda*G_{t+1}
    // G_{n-1} = final_score
    let mut targets = vec![0.0f32; n];
    targets[n - 1] = final_score;
    for t in (0..n - 1).rev() {
        targets[t] = (1.0 - lambda) * values[t + 1] + lambda * targets[t + 1];
    }

    let samples = recorded_features
        .into_iter()
        .zip(targets)
        .map(|(features, target)| TrainingSample {
            features: features.to_vec(),
            value: target,
        })
        .collect();
    (samples, final_score)
}

#[cfg(feature = "dqn")]
/// Self-play RL training loop with replay buffer and mixed opponents.
/// Keeps samples from the last `buffer_iterations` rounds to prevent forgetting.
pub fn self_play_train(
    artifact_dir: &str,
    num_iterations: usize,
    games_per_iteration: usize,
    epochs_per_iteration: usize,
) {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    MyBackend::seed(&device, TRAIN_SEED);
    let buffer_iterations = 3;
    let mut replay_buffer: std::collections::VecDeque<Vec<TrainingSample>> = std::collections::VecDeque::new();

    let scores_log_path = format!("{artifact_dir}/training_scores.csv");
    // Write header
    std::fs::write(&scores_log_path, "iteration,avg_score\n").ok();

    let genes = Arc::new(bot::default_genes());
    let champion = DNA::load_weights("champion.txt", genes).expect("No champion.txt");

    for iteration in 0..num_iterations {
        let epsilon = (0.2 * (0.95f32).powi(iteration as i32)).max(0.07);
        println!(
            "\n=== Iteration {}/{num_iterations} (epsilon={epsilon:.3}) ===",
            iteration + 1
        );

        // Load current model for inference
        let model: QwixxModel<MyBackend> = if iteration == 0 {
            QwixxModelConfig::new()
                .init::<MyBackend>(&device)
                .load_file(format!("{artifact_dir}/model"), &CompactRecorder::new(), &device)
                .unwrap_or_else(|_| {
                    println!("  No pretrained model, starting fresh");
                    QwixxModelConfig::new().init::<MyBackend>(&device)
                })
        } else {
            QwixxModelConfig::new()
                .init::<MyBackend>(&device)
                .load_file(format!("{artifact_dir}/model"), &CompactRecorder::new(), &device)
                .expect("Failed to load model")
        };

        let games_each = games_per_iteration / 4;

        // 4 configs: 3p vs 2 GA, 4p vs 3 GA, 3p vs GA + self, 4p vs 2 GA + self
        // (1v1 configs commented out for now)
        let game_configs: Vec<u8> = (0..4)
            .flat_map(|config| std::iter::repeat(config).take(games_each))
            .collect();

        // Pre-clone models — one per game. Cloning is cheap (NdArray tensors are Arc-backed).
        let models: Vec<QwixxModel<MyBackend>> = (0..game_configs.len())
            .map(|_| model.clone())
            .collect();

        let game_results: Vec<(Vec<TrainingSample>, f32)> = game_configs
            .into_par_iter()
            .zip(models.into_par_iter())
            .enumerate()
            .map(|(game_idx, (config, thread_model))| {
                let seed = TRAIN_SEED.wrapping_add((iteration * games_per_iteration + game_idx) as u64);
                let mut rng = SmallRng::seed_from_u64(seed);
                let mut opps: Vec<Box<dyn Strategy>> = match config {
                    // // 1v1: vs GA champion
                    // 0 => vec![Box::new(champion.clone())],
                    // // 1v1: vs self
                    // 1 => vec![
                    //     Box::new(DqnSelfPlayOpponent { model: thread_model.clone(), device: device.clone(), rng: SmallRng::seed_from_u64(seed + 1_000_000) }),
                    // ],
                    // 3-player: vs 2 GA champions
                    0 => vec![Box::new(champion.clone()), Box::new(champion.clone())],
                    // 4-player: vs 3 GA champions
                    1 => vec![Box::new(champion.clone()), Box::new(champion.clone()), Box::new(champion.clone())],
                    // 3-player: vs GA + self
                    2 => vec![
                        Box::new(champion.clone()),
                        Box::new(DqnSelfPlayOpponent { model: thread_model.clone(), device: device.clone(), rng: SmallRng::seed_from_u64(seed + 1_000_000) }),
                    ],
                    // 4-player: vs 2 GA + self
                    _ => vec![
                        Box::new(champion.clone()),
                        Box::new(champion.clone()),
                        Box::new(DqnSelfPlayOpponent { model: thread_model.clone(), device: device.clone(), rng: SmallRng::seed_from_u64(seed + 1_000_000) }),
                    ],
                };
                play_training_game(&thread_model, &device, &mut opps, epsilon, &mut rng)
            })
            .collect();

        let new_samples: Vec<TrainingSample> = game_results.iter().flat_map(|(s, _)| s.iter().cloned()).collect();
        let game_scores: Vec<f32> = game_results.iter().map(|(_, score)| *score).collect();
        let avg_score = if game_scores.is_empty() {
            0.0
        } else {
            game_scores.iter().sum::<f32>() / game_scores.len() as f32
        };

        // Update replay buffer
        replay_buffer.push_back(new_samples);
        if replay_buffer.len() > buffer_iterations {
            replay_buffer.pop_front();
        }

        let all_samples: Vec<TrainingSample> = replay_buffer.iter().flatten().cloned().collect();
        println!(
            "  Generated {} new samples (avg score: {avg_score:.1}), replay buffer: {} total",
            replay_buffer.back().unwrap().len(),
            all_samples.len()
        );

        // Log avg score
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().append(true).open(&scores_log_path) {
            writeln!(f, "{},{avg_score:.2}", iteration + 1).ok();
        }

        // Train on replay buffer
        train_with_epochs(all_samples, artifact_dir, epochs_per_iteration, 4e-4);
    }

    println!("\nSelf-play training complete. Model saved to {artifact_dir}/model");
}

#[cfg(feature = "dqn")]
/// Train with a specific number of epochs, loading from existing model if present.
fn train_with_epochs(samples: Vec<TrainingSample>, artifact_dir: &str, num_epochs: usize, lr: f64) {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;

    let split = (samples.len() * 9) / 10;
    let train_data = InMemDataset::new(samples[..split].to_vec());
    let valid_data = InMemDataset::new(samples[split..].to_vec());

    // Load existing model or init fresh
    let model: QwixxModel<MyAutodiffBackend> = QwixxModelConfig::new()
        .init::<MyAutodiffBackend>(&device)
        .load_file(format!("{artifact_dir}/model"), &CompactRecorder::new(), &device)
        .unwrap_or_else(|_| QwixxModelConfig::new().init::<MyAutodiffBackend>(&device));

    let batcher_train = QwixxBatcher::<MyAutodiffBackend> { _phantom: std::marker::PhantomData };
    let batcher_valid = QwixxBatcher::<MyBackend> { _phantom: std::marker::PhantomData };

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(1024)
        .shuffle(TRAIN_SEED)
        .build(train_data);

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(1024)
        .shuffle(TRAIN_SEED)
        .build(valid_data);

    // Use a temp dir for checkpoints to avoid clobbering the main model dir
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
        .expect("Failed to save model");

    std::fs::remove_dir_all(&ckpt_dir).ok();
}
