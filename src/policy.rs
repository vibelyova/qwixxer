/// Policy network: scores (state, move) pairs directly.
/// Unlike the value net (DQN) which evaluates states, the policy net
/// sees the move itself and can learn move-specific patterns.
use crate::bot::{self, DNA};
use crate::state::{Mark, Move, State};
use crate::strategy::Strategy;
use burn::{
    backend::{ndarray::NdArray, Autodiff},
    nn::{Linear, LinearConfig, Relu},
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rayon::prelude::*;
use std::sync::Arc;

type MyBackend = NdArray;
type MyAutodiffBackend = Autodiff<MyBackend>;

pub const NUM_FEATURES: usize = 34;

// ---- Feature extraction ----

/// Opponent context for feature computation.
#[derive(Clone, Debug, Default)]
pub struct OpponentContext {
    pub num_opponents: u8,
    pub max_opponent_strikes: u8,
}

/// Extract features for a (state, move) pair.
pub fn move_features(state: &State, mov: Move, ctx: &OpponentContext) -> [f32; NUM_FEATURES] {
    let totals = state.row_totals();
    let frees = state.row_free_values();
    let locked = state.locked();

    let mut f = [0.0f32; NUM_FEATURES];

    // State features (before move): 0-11, 16-19
    for i in 0..4 {
        // Row progress
        f[i] = match frees[i] {
            Some(free) if i < 2 => (free as f32 - 2.0) / 10.0,
            Some(free) => (12.0 - free as f32) / 10.0,
            None => 1.0,
        };
        // Row mark count
        f[4 + i] = totals[i] as f32 / 11.0;
        // Row locked
        f[8 + i] = if locked[i] { 1.0 } else { 0.0 };
    }
    f[16] = state.strikes as f32 / 4.0;
    f[17] = state.blanks() as f32 / 40.0;
    f[18] = ctx.num_opponents as f32 / 4.0;
    f[19] = ctx.max_opponent_strikes as f32 / 4.0;

    // Weighted probability AFTER applying the move: 12-15
    let mut state_after = *state;
    state_after.apply_move(mov);
    let after_totals = state_after.row_totals();
    let after_frees = state_after.row_free_values();
    for i in 0..4 {
        f[12 + i] = match after_frees[i] {
            Some(free) => {
                let ways = 6.0 - (7.0f32 - free as f32).abs();
                (ways / 6.0) * (after_totals[i] as f32 + 1.0) / 11.0
            }
            None => 0.0,
        };
    }

    // Move features: 20-33
    match mov {
        Move::Strike => {
            f[20] = 1.0; // is_strike
        }
        Move::Single(mark) => {
            f[22] = 0.5; // num_marks = 1
            set_mark1_features(&mut f, state, &mark);
        }
        Move::Double(m1, m2) => {
            f[22] = 1.0; // num_marks = 2
            set_mark1_features(&mut f, state, &m1);
            // mark2 blanks computed from state AFTER mark1
            let mut s1 = *state;
            s1.apply_move(Move::Single(m1));
            set_mark2_features(&mut f, &s1, &m2);
        }
    }

    // is_pass: set externally (caller knows if it's opponent turn)
    // f[21] is left as 0.0 by default, caller sets it for pass

    // Rows locked by this move
    let locks_before = state.count_locked();
    let locks_after = state_after.count_locked();
    f[33] = (locks_after - locks_before) as f32 / 2.0;

    f
}

/// Set features for mark1 (indices 23-27).
fn set_mark1_features(f: &mut [f32; NUM_FEATURES], state: &State, mark: &Mark) {
    // One-hot row
    f[23 + mark.row] = 1.0;
    // Blanks created
    let blanks_before = state.blanks();
    let mut s = *state;
    s.apply_move(Move::Single(*mark));
    let blanks_after = s.blanks();
    f[27] = (blanks_after as f32 - blanks_before as f32).max(0.0) / 10.0;
}

/// Set features for mark2 (indices 28-32), computed from state after mark1.
fn set_mark2_features(f: &mut [f32; NUM_FEATURES], state_after_m1: &State, mark: &Mark) {
    // One-hot row
    f[28 + mark.row] = 1.0;
    // Blanks created (from state after mark1)
    let blanks_before = state_after_m1.blanks();
    let mut s = *state_after_m1;
    s.apply_move(Move::Single(*mark));
    let blanks_after = s.blanks();
    f[32] = (blanks_after as f32 - blanks_before as f32).max(0.0) / 10.0;
}

/// Create pass features (opponent turn skip).
pub fn pass_features(state: &State, ctx: &OpponentContext) -> [f32; NUM_FEATURES] {
    let mut f = [0.0f32; NUM_FEATURES];

    let totals = state.row_totals();
    let frees = state.row_free_values();
    let locked = state.locked();

    for i in 0..4 {
        f[i] = match frees[i] {
            Some(free) if i < 2 => (free as f32 - 2.0) / 10.0,
            Some(free) => (12.0 - free as f32) / 10.0,
            None => 1.0,
        };
        f[4 + i] = totals[i] as f32 / 11.0;
        f[8 + i] = if locked[i] { 1.0 } else { 0.0 };
        // Weighted prob unchanged (pass doesn't change state)
        f[12 + i] = match frees[i] {
            Some(free) => {
                let ways = 6.0 - (7.0f32 - free as f32).abs();
                (ways / 6.0) * (totals[i] as f32 + 1.0) / 11.0
            }
            None => 0.0,
        };
    }
    f[16] = state.strikes as f32 / 4.0;
    f[17] = state.blanks() as f32 / 40.0;
    f[18] = ctx.num_opponents as f32 / 4.0;
    f[19] = ctx.max_opponent_strikes as f32 / 4.0;
    f[21] = 1.0; // is_pass

    f
}

// ---- Model ----

#[derive(Module, Debug)]
pub struct PolicyModel<B: Backend> {
    layer1: Linear<B>,
    layer2: Linear<B>,
    output: Linear<B>,
    activation: Relu,
}

#[derive(Config, Debug)]
pub struct PolicyModelConfig {
    #[config(default = 64)]
    pub hidden1: usize,
    #[config(default = 32)]
    pub hidden2: usize,
}

impl PolicyModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PolicyModel<B> {
        PolicyModel {
            layer1: LinearConfig::new(NUM_FEATURES, self.hidden1).with_bias(true).init(device),
            layer2: LinearConfig::new(self.hidden1, self.hidden2).with_bias(true).init(device),
            output: LinearConfig::new(self.hidden2, 1).with_bias(true).init(device),
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> PolicyModel<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.activation.forward(self.layer1.forward(input));
        let x = self.activation.forward(self.layer2.forward(x));
        self.output.forward(x)
    }

    pub fn evaluate(&self, features: &[f32; NUM_FEATURES], device: &B::Device) -> f32 {
        let input = Tensor::<B, 1>::from_floats(features.as_slice(), device)
            .reshape([1, NUM_FEATURES]);
        let output = self.forward(input);
        output.into_data().to_vec::<f32>().unwrap()[0]
    }
}

// ---- Strategy ----

pub struct PolicyStrategy {
    model: PolicyModel<MyBackend>,
    device: burn::backend::ndarray::NdArrayDevice,
    context: OpponentContext,
}

impl std::fmt::Debug for PolicyStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "PolicyStrategy")
    }
}

impl PolicyStrategy {
    pub fn load(artifact_dir: &str) -> Self {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let model = PolicyModelConfig::new()
            .init::<MyBackend>(&device)
            .load_file(format!("{artifact_dir}/model"), &CompactRecorder::new(), &device)
            .expect("Failed to load policy model");
        PolicyStrategy { model, device, context: OpponentContext::default() }
    }

    fn score_move(&self, state: &State, mov: Move) -> f32 {
        let features = move_features(state, mov, &self.context);
        self.model.evaluate(&features, &self.device)
    }

    fn score_pass(&self, state: &State) -> f32 {
        let features = pass_features(state, &self.context);
        self.model.evaluate(&features, &self.device)
    }
}

impl Strategy for PolicyStrategy {
    fn your_move(&mut self, state: &State, dice: [u8; 6]) -> Move {
        let mut moves = state.generate_moves(dice);
        moves.push(Move::Strike);

        moves
            .into_iter()
            .max_by(|&a, &b| {
                self.score_move(state, a)
                    .partial_cmp(&self.score_move(state, b))
                    .unwrap()
            })
            .unwrap()
    }

    fn opponents_move(&mut self, state: &State, number: u8, _locked: [bool; 4]) -> Option<Move> {
        let moves = state.generate_opponent_moves(number);
        if moves.is_empty() {
            return None;
        }

        let pass_score = self.score_pass(state);
        let best = moves
            .iter()
            .max_by(|&&a, &&b| {
                self.score_move(state, a)
                    .partial_cmp(&self.score_move(state, b))
                    .unwrap()
            })
            .copied()
            .unwrap();

        if self.score_move(state, best) > pass_score {
            Some(best)
        } else {
            None
        }
    }

    fn observe_opponents(&mut self, _our_score: isize, opponents: &[State]) {
        self.context.num_opponents = opponents.len() as u8;
        self.context.max_opponent_strikes = opponents.iter().map(|s| s.strikes).max().unwrap_or(0);
    }
}

// ---- Training ----

fn make_opponent_context(all_states: &[State]) -> OpponentContext {
    let mut max_strikes = 0u8;
    let num_opponents = (all_states.len() - 1) as u8;
    for s in all_states.iter().skip(1) {
        max_strikes = max_strikes.max(s.strikes);
    }
    OpponentContext {
        num_opponents,
        max_opponent_strikes: max_strikes,
    }
}

/// Pick move using the policy model with epsilon-greedy.
#[allow(dead_code)]
fn pick_move_with_policy(
    model: &PolicyModel<MyBackend>,
    device: &burn::backend::ndarray::NdArrayDevice,
    state: &State,
    moves: &[Move],
    ctx: &OpponentContext,
    epsilon: f32,
    rng: &mut SmallRng,
) -> Move {
    if moves.is_empty() {
        return Move::Strike;
    }
    if rng.gen::<f32>() < epsilon {
        return moves[rng.gen_range(0..moves.len())];
    }
    *moves
        .iter()
        .max_by(|&&a, &&b| {
            let fa = move_features(state, a, ctx);
            let fb = move_features(state, b, ctx);
            model.evaluate(&fa, device)
                .partial_cmp(&model.evaluate(&fb, device))
                .unwrap()
        })
        .unwrap()
}

/// Pick passive move using the policy model.
#[allow(dead_code)]
fn pick_passive_with_policy(
    model: &PolicyModel<MyBackend>,
    device: &burn::backend::ndarray::NdArrayDevice,
    state: &State,
    number: u8,
    ctx: &OpponentContext,
) -> Option<Move> {
    let moves = state.generate_opponent_moves(number);
    if moves.is_empty() {
        return None;
    }
    let pass_score = {
        let f = pass_features(state, ctx);
        model.evaluate(&f, device)
    };
    let best = moves
        .iter()
        .max_by(|&&a, &&b| {
            let fa = move_features(state, a, ctx);
            let fb = move_features(state, b, ctx);
            model.evaluate(&fa, device)
                .partial_cmp(&model.evaluate(&fb, device))
                .unwrap()
        })
        .copied()
        .unwrap();
    let best_score = {
        let f = move_features(state, best, ctx);
        model.evaluate(&f, device)
    };
    if best_score > pass_score { Some(best) } else { None }
}

/// A decision point: all candidate move features + which was chosen.
#[derive(Clone)]
struct DecisionPoint {
    candidate_features: Vec<[f32; NUM_FEATURES]>, // features for each candidate
    chosen_index: usize,
}

/// Play a training game, record decision points for REINFORCE.
fn play_training_game(
    model: &PolicyModel<MyBackend>,
    device: &burn::backend::ndarray::NdArrayDevice,
    opponents: &mut [Box<dyn Strategy>],
    epsilon: f32,
    rng: &mut SmallRng,
) -> (Vec<DecisionPoint>, f32) {
    let num_players = 1 + opponents.len();
    let mut states: Vec<State> = (0..num_players).map(|_| State::default()).collect();
    let mut decisions: Vec<DecisionPoint> = Vec::new();
    let mut turn = 0u32;

    loop {
        if states.iter().any(|s| s.strikes >= 4) { break; }
        if states.iter().any(|s| s.count_locked() >= 2) { break; }

        let dice: [u8; 6] = core::array::from_fn(|_| rng.gen_range(1..=6));
        let on_white = dice[0] + dice[1];
        let active = (turn as usize) % num_players;

        if active == 0 {
            let ctx = make_opponent_context(&states);
            let mut moves = states[0].generate_moves(dice);
            moves.push(Move::Strike);

            // Compute features for ALL candidates
            let candidate_features: Vec<[f32; NUM_FEATURES]> = moves
                .iter()
                .map(|&m| move_features(&states[0], m, &ctx))
                .collect();

            // Pick move (epsilon-greedy)
            let chosen_index = if rng.gen::<f32>() < epsilon {
                rng.gen_range(0..moves.len())
            } else {
                candidate_features
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        model.evaluate(a, device)
                            .partial_cmp(&model.evaluate(b, device))
                            .unwrap()
                    })
                    .unwrap()
                    .0
            };

            decisions.push(DecisionPoint { candidate_features, chosen_index });
            states[0].apply_move(moves[chosen_index]);
        } else {
            let opp_idx = active - 1;
            let opp_mov = opponents[opp_idx].your_move(&states[active], dice);
            states[active].apply_move(opp_mov);
        }

        let mut new_locked = states[active].locked();

        for idx in 1..num_players {
            let passive = (active + idx) % num_players;
            if passive == 0 {
                let ctx = make_opponent_context(&states);
                let moves = states[0].generate_opponent_moves(on_white);
                if !moves.is_empty() {
                    // Candidates: each mark + pass
                    let mut candidate_features: Vec<[f32; NUM_FEATURES]> = moves
                        .iter()
                        .map(|&m| move_features(&states[0], m, &ctx))
                        .collect();
                    candidate_features.push(pass_features(&states[0], &ctx));

                    let chosen_index = candidate_features
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            model.evaluate(a, device)
                                .partial_cmp(&model.evaluate(b, device))
                                .unwrap()
                        })
                        .unwrap()
                        .0;

                    decisions.push(DecisionPoint { candidate_features, chosen_index });

                    // Apply if not pass (last index)
                    if chosen_index < moves.len() {
                        states[0].apply_move(moves[chosen_index]);
                    }
                }
            } else {
                let opp_idx = passive - 1;
                let opp_mov = opponents[opp_idx].opponents_move(&states[passive], on_white, new_locked);
                if let Some(m) = opp_mov { states[passive].apply_move(m); }
            }
            for (row, l) in states[passive].locked().iter().enumerate() {
                new_locked[row] |= *l;
            }
        }
        for s in states.iter_mut() { s.lock(new_locked); }

        turn += 1;
        if turn > 200 { break; }
    }

    let final_score = states[0].count_points() as f32;
    (decisions, final_score)
}

/// REINFORCE training: update policy based on advantage-weighted log-probabilities.
fn reinforce_update<O: burn::optim::Optimizer<PolicyModel<MyAutodiffBackend>, MyAutodiffBackend>>(
    model: &mut PolicyModel<MyAutodiffBackend>,
    optimizer: &mut O,
    device: &burn::backend::ndarray::NdArrayDevice,
    decisions: &[DecisionPoint],
    advantage: f32,
    lr: f64,
) {
    if decisions.is_empty() || advantage.abs() < 0.01 {
        return;
    }

    // Process decisions in mini-batches to avoid huge graphs
    for decision in decisions {
        let n = decision.candidate_features.len();
        if n <= 1 {
            continue; // no choice to make
        }

        // Build input tensor: [n_candidates, NUM_FEATURES]
        let flat: Vec<f32> = decision.candidate_features
            .iter()
            .flat_map(|f| f.iter().copied())
            .collect();
        let input = Tensor::<MyAutodiffBackend, 1>::from_floats(flat.as_slice(), device)
            .reshape([n, NUM_FEATURES]);

        // Forward pass: [n_candidates, 1]
        let scores = model.forward(input).reshape([n]);

        // Log-softmax
        let log_probs = burn::tensor::activation::log_softmax(scores, 0);

        // REINFORCE loss: -advantage * log_prob(chosen)
        let chosen_log_prob = log_probs
            .slice([decision.chosen_index..decision.chosen_index + 1]);
        let adv_tensor = Tensor::<MyAutodiffBackend, 1>::from_floats([-advantage].as_slice(), device);
        let loss = chosen_log_prob.mul(adv_tensor).sum();

        // Backward + step
        let grads = loss.backward();
        let grads = burn::optim::GradientsParams::from_grads(grads, &*model);
        *model = optimizer.step(lr, model.clone(), grads);
    }
}

/// Self-play REINFORCE training for the policy net.
pub fn self_play_train(
    artifact_dir: &str,
    num_iterations: usize,
    games_per_iteration: usize,
    _epochs_per_iteration: usize,
) {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    let mut baseline = 50.0f32; // moving average baseline
    let baseline_decay = 0.99f32;

    std::fs::create_dir_all(artifact_dir).ok();

    for iteration in 0..num_iterations {
        let epsilon = (0.3 * (0.97f32).powi(iteration as i32)).max(0.05);
        println!(
            "\n=== Iteration {}/{num_iterations} (epsilon={epsilon:.3}, baseline={baseline:.1}) ===",
            iteration + 1
        );

        // Load current model for inference
        let inference_model: PolicyModel<MyBackend> = if iteration == 0 {
            PolicyModelConfig::new()
                .init::<MyBackend>(&device)
                .load_file(format!("{artifact_dir}/model"), &CompactRecorder::new(), &device)
                .unwrap_or_else(|_| {
                    println!("  No pretrained model, starting fresh");
                    PolicyModelConfig::new().init::<MyBackend>(&device)
                })
        } else {
            PolicyModelConfig::new()
                .init::<MyBackend>(&device)
                .load_file(format!("{artifact_dir}/model"), &CompactRecorder::new(), &device)
                .expect("Failed to load model")
        };

        // Play games and collect decision points
        let genes = Arc::new(bot::default_genes());
        let champion = DNA::load_weights("champion.txt", genes).expect("No champion.txt");
        let games_each = games_per_iteration / 4;

        let game_configs: Vec<u8> = (0..4)
            .flat_map(|config| std::iter::repeat(config).take(games_each))
            .collect();

        let tmp_model_path = format!("{artifact_dir}/tmp_thread_model");
        inference_model
            .save_file(&tmp_model_path, &CompactRecorder::new())
            .expect("Failed to save temp model");

        let game_results: Vec<(Vec<DecisionPoint>, f32)> = game_configs
            .par_iter()
            .map(|&config| {
                let thread_model = PolicyModelConfig::new()
                    .init::<MyBackend>(&device)
                    .load_file(&tmp_model_path, &CompactRecorder::new(), &device)
                    .expect("Failed to load thread model");
                let mut rng = SmallRng::from_entropy();
                let mut opps: Vec<Box<dyn Strategy>> = match config {
                    0 => vec![Box::new(champion.clone()), Box::new(champion.clone())],
                    1 => vec![Box::new(champion.clone()), Box::new(champion.clone()), Box::new(champion.clone())],
                    2 => vec![Box::new(champion.clone()), Box::new(champion.clone())],
                    _ => vec![Box::new(champion.clone()), Box::new(champion.clone()), Box::new(champion.clone())],
                };
                play_training_game(&thread_model, &device, &mut opps, epsilon, &mut rng)
            })
            .collect();

        let game_scores: Vec<f32> = game_results.iter().map(|(_, score)| *score).collect();
        let avg_score = if game_scores.is_empty() { 0.0 } else {
            game_scores.iter().sum::<f32>() / game_scores.len() as f32
        };
        let total_decisions: usize = game_results.iter().map(|(d, _)| d.len()).sum();

        println!(
            "  {} games, {} decisions, avg score: {avg_score:.1}",
            game_scores.len(),
            total_decisions
        );

        // REINFORCE updates
        let mut model: PolicyModel<MyAutodiffBackend> = PolicyModelConfig::new()
            .init::<MyAutodiffBackend>(&device)
            .load_file(format!("{artifact_dir}/model"), &CompactRecorder::new(), &device)
            .unwrap_or_else(|_| PolicyModelConfig::new().init::<MyAutodiffBackend>(&device));

        let mut optimizer = AdamConfig::new().init();
        let lr = 1e-4;

        let mut update_count = 0u32;
        for (decisions, score) in &game_results {
            let advantage = score - baseline;
            if advantage.abs() > 0.5 {
                reinforce_update(&mut model, &mut optimizer, &device, decisions, advantage, lr);
                update_count += 1;
            }
        }
        println!("  Updated on {update_count}/{} games", game_scores.len());

        // Update baseline
        baseline = baseline_decay * baseline + (1.0 - baseline_decay) * avg_score;

        // Save
        {
            use burn::module::AutodiffModule;
            model.valid()
                .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
                .expect("Failed to save model");
        }
    }

    println!("\nREINFORCE training complete. Model saved to {artifact_dir}/model");
}
