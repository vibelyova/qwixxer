use crate::bot::{self, DNA};
use crate::game::{Game, Player};
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
    record::{CompactRecorder, Recorder},
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
pub const NUM_FEATURES: usize = 18;

/// Extract features from a game state as a fixed-size float array.
pub fn state_features(state: &State) -> [f32; NUM_FEATURES] {
    let totals = state.row_totals();
    let frees = state.row_free_values();
    let locked = state.locked();

    let mut features = [0.0f32; NUM_FEATURES];
    for i in 0..4 {
        // Free pointer normalized (0-1), 0 if locked
        features[i] = match frees[i] {
            Some(f) => (f as f32 - 2.0) / 10.0,
            None => 0.0,
        };
        // Row mark count normalized
        features[4 + i] = totals[i] as f32 / 11.0;
        // Row locked
        features[8 + i] = if locked[i] { 1.0 } else { 0.0 };
        // Per-row probability
        features[12 + i] = match frees[i] {
            Some(f) => (6.0 - (7.0f32 - f as f32).abs()) / 6.0,
            None => 0.0,
        };
    }
    // Strikes normalized
    features[16] = state.strikes as f32 / 4.0;
    // Blanks normalized
    features[17] = state.blanks() as f32 / 20.0;

    features
}

// ---- Training data ----

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TrainingSample {
    pub features: Vec<f32>,
    pub value: f32,
}

#[derive(Clone)]
pub struct QwixxBatcher<B: Backend> {
    device: B::Device,
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
    #[config(default = 32)]
    pub hidden1: usize,
    #[config(default = 16)]
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
    let mc = MonteCarlo::new(mc_sims, champion.clone());

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

                    for &mov in &all_moves {
                        let mut new_state = state;
                        new_state.apply_move(mov);
                        let features = state_features(&new_state);
                        let mc_value = mc.evaluate_move_public(&state, mov, &opponent_state);
                        local_samples.push(TrainingSample {
                            features: features.to_vec(),
                            value: mc_value as f32,
                        });
                    }

                    // Play the MC-best move
                    let best_move = all_moves
                        .iter()
                        .max_by(|&&a, &&b| {
                            let va = mc.evaluate_move_public(&state, a, &opponent_state);
                            let vb = mc.evaluate_move_public(&state, b, &opponent_state);
                            va.partial_cmp(&vb).unwrap()
                        })
                        .copied()
                        .unwrap();
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

    let batcher_train = QwixxBatcher::<MyAutodiffBackend> { device: device.clone() };
    let batcher_valid = QwixxBatcher::<MyBackend> { device: device.clone() };

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
        DqnStrategy { model, device }
    }

    fn evaluate(&self, state: &State) -> f32 {
        let features = state_features(state);
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
        let current_val = self.evaluate(state);
        let best = moves
            .into_iter()
            .max_by(|&a, &b| {
                let mut sa = *state;
                sa.apply_move(a);
                let mut sb = *state;
                sb.apply_move(b);
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
}
