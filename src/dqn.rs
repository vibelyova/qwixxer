use crate::bot::{self, DNA};
use crate::mcts::MonteCarlo;
use crate::state::{Move, State};
use crate::strategy::Strategy;
use burn::{
    backend::{ndarray::NdArray, Autodiff},
    data::{
        dataloader::{DataLoaderBuilder, batcher::Batcher},
        dataset::{Dataset, InMemDataset},
    },
    nn::{Linear, LinearConfig, Relu, loss::{MseLoss, Reduction::Mean}},
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{Learner, RegressionOutput, SupervisedTraining, TrainOutput, TrainStep, InferenceStep, metric::LossMetric},
};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rayon::prelude::*;
use std::sync::Arc;

// Backend type aliases
type MyBackend = NdArray;
type MyAutodiffBackend = Autodiff<MyBackend>;

/// Number of input features for the state representation.
pub const NUM_FEATURES: usize = 21;

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
        // Per-row weighted probability (may double-count overlaps)
        features[12 + i] = match frees[i] {
            Some(f) => {
                let ways = 6.0 - (7.0f32 - f as f32).abs();
                (ways / 6.0) * (totals[i] as f32 + 1.0) / 11.0
            }
            None => 0.0,
        };
    }

    // Aggregated weighted probability (correct, no double-counting):
    // find the best mark value across all markable rows, weight by P(sum).
    // Correctly handles overlapping free pointers.
    let mut agg_wp = 0.0f32;
    for white_sum in 2..=12u8 {
        let ways = 6.0 - (7.0f32 - white_sum as f32).abs();
        let prob = ways / 36.0;

        // Best value from marking this sum across all rows
        let mut best_value = 0.0f32;
        for i in 0..4 {
            let Some(free) = frees[i] else { continue };
            let can_mark = if i < 2 {
                white_sum >= free
            } else {
                white_sum <= free
            };
            if can_mark {
                // Lock requires 5+ marks
                let is_lock = (i < 2 && white_sum == 12) || (i >= 2 && white_sum == 2);
                if is_lock && totals[i] < 5 { continue; }
                let value = (totals[i] as f32 + 1.0) / 11.0;
                best_value = best_value.max(value);
            }
        }
        agg_wp += prob * best_value;
    }
    features[16] = agg_wp;

    // Strikes normalized
    features[17] = state.strikes as f32 / 4.0;
    // Blanks normalized
    features[18] = state.blanks() as f32 / 40.0;
    // Opponent context
    features[19] = ctx.num_opponents as f32 / 4.0;
    features[20] = ctx.max_opponent_strikes as f32 / 4.0;

    features
}

/// Build opponent context from a 2-player game state.
fn make_context(our_state: &State, opponent_state: &State) -> OpponentContext {
    OpponentContext {
        num_opponents: 1,
        max_opponent_strikes: opponent_state.strikes,
        score_gap_to_leader: our_state.count_points() - opponent_state.count_points(),
    }
}

// ---- Training data ----

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TrainingSample {
    pub features: Vec<f32>,
    pub value: f32,
}

#[derive(Clone)]
pub struct QwixxBatcher<B: Backend> {
    _phantom: std::marker::PhantomData<B>,
}

#[derive(Clone, Debug)]
pub struct QwixxBatch<B: Backend> {
    pub inputs: Tensor<B, 2>,
    pub targets: Tensor<B, 1>,
}

impl<B: Backend> Batcher<B, TrainingSample, QwixxBatch<B>> for QwixxBatcher<B> {
    fn batch(&self, items: Vec<TrainingSample>, device: &B::Device) -> QwixxBatch<B> {
        let batch_size = items.len();
        let inputs: Vec<f32> = items.iter().flat_map(|s| s.features.iter().copied()).collect();
        let targets: Vec<f32> = items.iter().map(|s| s.value).collect();

        let inputs = Tensor::<B, 1>::from_floats(inputs.as_slice(), device)
            .reshape([batch_size, NUM_FEATURES]);
        let targets = Tensor::<B, 1>::from_floats(targets.as_slice(), device);

        QwixxBatch { inputs, targets }
    }
}

// ---- Model ----

#[derive(Module, Debug)]
pub struct QwixxModel<B: Backend> {
    layer1: Linear<B>,
    layer2: Linear<B>,
    output: Linear<B>,
    activation: Relu,
}

#[derive(Config, Debug)]
pub struct QwixxModelConfig {
    #[config(default = 64)]
    pub hidden1: usize,
    #[config(default = 32)]
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

    pub fn forward_step(&self, batch: QwixxBatch<B>) -> RegressionOutput<B> {
        let targets = batch.targets.clone().unsqueeze_dim(1);
        let output = self.forward(batch.inputs);
        let loss = MseLoss::new().forward(output.clone(), targets.clone(), Mean);
        RegressionOutput { loss, output, targets }
    }

    /// Evaluate a single state, returning the predicted value.
    pub fn evaluate_state(&self, features: &[f32; NUM_FEATURES], device: &B::Device) -> f32 {
        let input = Tensor::<B, 1>::from_floats(features.as_slice(), device)
            .reshape([1, NUM_FEATURES]);
        let output = self.forward(input);
        output.into_data().to_vec::<f32>().unwrap()[0]
    }
}

impl<B: AutodiffBackend> TrainStep for QwixxModel<B> {
    type Input = QwixxBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, batch: QwixxBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_step(batch);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for QwixxModel<B> {
    type Input = QwixxBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, batch: QwixxBatch<B>) -> RegressionOutput<B> {
        self.forward_step(batch)
    }
}

// ---- Data generation ----

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
            let mut rng = SmallRng::from_entropy();
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
        .batch_size(256)
        .shuffle(42)
        .num_workers(2)
        .build(train_data);

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(256)
        .shuffle(42)
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
}

impl std::fmt::Debug for DqnStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "DqnStrategy")
    }
}

impl DqnStrategy {
    pub fn load(artifact_dir: &str) -> Self {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let model = QwixxModelConfig::new()
            .init::<MyBackend>(&device)
            .load_file(format!("{artifact_dir}/model"), &CompactRecorder::new(), &device)
            .expect("Failed to load model");
        DqnStrategy { model, device, context: OpponentContext::default() }
    }

    fn evaluate(&self, state: &State) -> f32 {
        let features = state_features(state, &self.context);
        self.model.evaluate_state(&features, &self.device)
    }

    fn find_locking_move(state: &State, moves: &[Move]) -> Option<Move> {
        let current_locked = state.count_locked();
        moves.iter().copied().find(|&mov| {
            let mut new_state = *state;
            new_state.apply_move(mov);
            new_state.count_locked() > current_locked
        })
    }
}

impl Strategy for DqnStrategy {
    fn your_move(&mut self, state: &State, dice: [u8; 6]) -> Move {
        let moves = state.generate_moves(dice);

        if let Some(mov) = Self::find_locking_move(state, &moves) {
            return mov;
        }

        if moves.is_empty() {
            return Move::Strike;
        }

        let mut moves = moves;
        moves.push(Move::Strike);

        moves
            .into_iter()
            .max_by(|&a, &b| {
                let mut sa = *state;
                sa.apply_move(a);
                let mut sb = *state;
                sb.apply_move(b);
                self.evaluate(&sa).partial_cmp(&self.evaluate(&sb)).unwrap()
            })
            .unwrap()
    }

    fn opponents_move(&mut self, state: &State, number: u8, locked: [bool; 4]) -> Option<Move> {
        let moves = state.generate_opponent_moves(number);

        if let Some(mov) = Self::find_locking_move(state, &moves) {
            return Some(mov);
        }

        if moves.is_empty() {
            return None;
        }

        // Evaluate mark vs skip
        let best = moves
            .into_iter()
            .max_by(|&a, &b| {
                let mut sa = *state;
                sa.apply_move(a);
                sa.lock(locked);
                let mut sb = *state;
                sb.apply_move(b);
                sb.lock(locked);
                self.evaluate(&sa).partial_cmp(&self.evaluate(&sb)).unwrap()
            })
            .unwrap();

        let mut new_state = *state;
        new_state.apply_move(best);
        new_state.lock(locked);

        let mut skip_state = *state;
        skip_state.lock(locked);

        if self.evaluate(&new_state) > self.evaluate(&skip_state) {
            Some(best)
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

/// Pick a move using the model, with epsilon-greedy exploration.
fn pick_move_with_model(
    model: &QwixxModel<MyBackend>,
    device: &burn::backend::ndarray::NdArrayDevice,
    state: &State,
    moves: &[Move],
    epsilon: f32,
    rng: &mut SmallRng,
    ctx: &OpponentContext,
) -> Move {
    if moves.is_empty() {
        return Move::Strike;
    }

    // Always lock if possible
    let current_locked = state.count_locked();
    if let Some(mov) = moves.iter().copied().find(|&mov| {
        let mut s = *state;
        s.apply_move(mov);
        s.count_locked() > current_locked
    }) {
        return mov;
    }

    // Epsilon-greedy
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

    // Always lock
    let current_locked = state.count_locked();
    if let Some(mov) = moves.iter().copied().find(|&mov| {
        let mut s = *state;
        s.apply_move(mov);
        s.count_locked() > current_locked
    }) {
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

/// A simple wrapper to use the DQN model as an opponent (no epsilon, no recording).
struct DqnSelfPlayOpponent {
    model: QwixxModel<MyBackend>,
    device: burn::backend::ndarray::NdArrayDevice,
    rng: SmallRng,
}

impl std::fmt::Debug for DqnSelfPlayOpponent {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "DqnSelfPlayOpponent")
    }
}

impl Strategy for DqnSelfPlayOpponent {
    fn your_move(&mut self, state: &State, dice: [u8; 6]) -> Move {
        let ctx = OpponentContext::default(); // opponent doesn't need perfect context
        let mut moves = state.generate_moves(dice);
        moves.push(Move::Strike);
        pick_move_with_model(&self.model, &self.device, state, &moves, 0.0, &mut self.rng, &ctx)
    }

    fn opponents_move(&mut self, state: &State, number: u8, locked: [bool; 4]) -> Option<Move> {
        let ctx = OpponentContext::default();
        pick_passive_move_with_model(&self.model, &self.device, state, number, locked, &ctx)
    }
}

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
            let mut moves = states[0].generate_moves(dice);
            moves.push(Move::Strike);
            let mov = pick_move_with_model(model, device, &states[0], &moves, epsilon, rng, &ctx);
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

/// Self-play RL training loop with replay buffer and mixed opponents.
/// Keeps samples from the last `buffer_iterations` rounds to prevent forgetting.
pub fn self_play_train(
    artifact_dir: &str,
    num_iterations: usize,
    games_per_iteration: usize,
    epochs_per_iteration: usize,
) {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    let buffer_iterations = 5;
    let mut replay_buffer: Vec<Vec<TrainingSample>> = Vec::new();

    // CSV logging
    let csv_path = format!("{artifact_dir}/training_log.csv");
    let mut csv = std::fs::File::create(&csv_path).expect("Failed to create CSV");
    use std::io::Write;
    writeln!(csv, "iteration,avg_score,num_samples,epsilon").unwrap();

    for iteration in 0..num_iterations {
        let epsilon = (0.2 * (0.95f32).powi(iteration as i32)).max(0.05);
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

        // Generate training data: 3-4 player games vs GA champions and self
        let genes = Arc::new(bot::default_genes());
        let champion = DNA::load_weights("champion.txt", genes).expect("No champion.txt");
        let games_each = games_per_iteration / 4;

        // 4 configs: 3p vs 2 GA, 4p vs 3 GA, 3p vs GA + self, 4p vs 2 GA + self
        let game_configs: Vec<u8> = (0..4)
            .flat_map(|config| std::iter::repeat(config).take(games_each))
            .collect();

        let tmp_model_path = format!("{artifact_dir}/tmp_thread_model");
        model
            .save_file(&tmp_model_path, &CompactRecorder::new())
            .expect("Failed to save temp model");

        let game_results: Vec<(Vec<TrainingSample>, f32)> = game_configs
            .par_iter()
            .map(|&config| {
                let thread_model = QwixxModelConfig::new()
                    .init::<MyBackend>(&device)
                    .load_file(&tmp_model_path, &CompactRecorder::new(), &device)
                    .expect("Failed to load thread model");
                let mut rng = SmallRng::from_entropy();
                let mut opps: Vec<Box<dyn Strategy>> = match config {
                    // 3-player: vs 2 GA champions
                    0 => vec![Box::new(champion.clone()), Box::new(champion.clone())],
                    // 4-player: vs 3 GA champions
                    1 => vec![Box::new(champion.clone()), Box::new(champion.clone()), Box::new(champion.clone())],
                    // 3-player: vs GA + self
                    2 => vec![
                        Box::new(champion.clone()),
                        Box::new(DqnSelfPlayOpponent { model: thread_model.clone(), device: device.clone(), rng: SmallRng::from_entropy() }),
                    ],
                    // 4-player: vs 2 GA + self
                    _ => vec![
                        Box::new(champion.clone()),
                        Box::new(champion.clone()),
                        Box::new(DqnSelfPlayOpponent { model: thread_model.clone(), device: device.clone(), rng: SmallRng::from_entropy() }),
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
        replay_buffer.push(new_samples);
        if replay_buffer.len() > buffer_iterations {
            replay_buffer.remove(0);
        }

        let all_samples: Vec<TrainingSample> = replay_buffer.iter().flatten().cloned().collect();
        let num_new = replay_buffer.last().unwrap().len();
        println!(
            "  Generated {num_new} new samples (avg score: {avg_score:.1}), replay buffer: {} total",
            all_samples.len()
        );
        writeln!(csv, "{},{avg_score:.2},{num_new},{epsilon:.4}", iteration + 1).unwrap();
        csv.flush().unwrap();

        // Train on the full replay buffer
        train_with_epochs(all_samples, artifact_dir, epochs_per_iteration);
    }

    println!("\nSelf-play training complete. Model saved to {artifact_dir}/model");
}

/// Train with a specific number of epochs, loading from existing model if present.
fn train_with_epochs(samples: Vec<TrainingSample>, artifact_dir: &str, num_epochs: usize) {
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
        .batch_size(256)
        .shuffle(42)
        .build(train_data);

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(256)
        .shuffle(42)
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

    let result = training.launch(Learner::new(model, AdamConfig::new().init(), 1e-4));

    result
        .model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Failed to save model");

    std::fs::remove_dir_all(&ckpt_dir).ok();
}
