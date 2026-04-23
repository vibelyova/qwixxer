//! DQN training: MC-supervised pretraining + TD(λ) self-play RL.
//!
//! Only compiled with the `dqn` feature. The inference side ([`crate::dqn`])
//! is always available so native and wasm binaries can share the same model.

use crate::bot::{self, DNA};
use crate::dqn::{
    build_opponent_context_for, rank_candidates_with_opp_context,
    state_features, DqnStrategy, MyBackend, OpponentContext, QwixxModel, QwixxModelConfig,
    LOG_VAR_MAX, LOG_VAR_MIN, NUM_FEATURES, TRAIN_SEED,
};
use crate::mcts::MonteCarlo;
use crate::state::{Mark, State};
use crate::strategy::Strategy;
use burn::{
    backend::Autodiff,
    data::{
        dataloader::{batcher::Batcher, DataLoaderBuilder},
        dataset::{Dataset, InMemDataset},
    },
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        metric::LossMetric, InferenceStep, Learner, RegressionOutput, SupervisedTraining,
        TrainOutput, TrainStep,
    },
};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rayon::prelude::*;
use std::sync::Arc;

type MyAutodiffBackend = Autodiff<MyBackend>;

/// Weight on the σ loss in the combined `L = L_μ + α · L_σ`. The μ loss (MSE
/// in raw score units) is ~100–2500 per sample; σ loss (Gaussian NLL) is
/// O(1–10). A small α prevents σ's backbone gradients from dominating μ's.
/// Tune if σ calibration isn't converging after a few iterations.
const SIGMA_LOSS_WEIGHT: f32 = 1.0;

// ---- Training step impls on the shared model ----

impl<B: Backend> QwixxModel<B> {
    /// Dual loss that decouples μ and σ training:
    ///
    /// - **μ head**: plain MSE toward the TD(λ) target `G_t`. Keeps μ sample-
    ///   efficient (bootstrap benefits) without σ interfering.
    /// - **σ head**: Gaussian NLL using the *actual final score* as the
    ///   residual reference (not `G_t`). This side-steps TD smoothing so σ
    ///   learns `Var(X_final | s)`, the quantity we actually want. μ is
    ///   detached from this term so σ gradients can't corrupt μ training.
    ///
    /// Total: `L = L_μ + α · L_σ`. See `SIGMA_LOSS_WEIGHT`.
    pub fn forward_step(&self, batch: QwixxBatch<B>) -> RegressionOutput<B> {
        let output = self.forward(batch.inputs);
        let mean = output.clone().narrow(1, 0, 1);
        let log_var = output.narrow(1, 1, 1).clamp(LOG_VAR_MIN, LOG_VAR_MAX);

        let targets = batch.targets.clone().unsqueeze_dim(1);
        let final_scores = batch.final_scores.clone().unsqueeze_dim(1);

        // μ loss: MSE toward G_t.
        let mu_residual = targets.clone() - mean.clone();
        let mu_loss = (mu_residual.clone() * mu_residual).mean();

        // σ loss: Gaussian NLL with residual measured against the real final
        // score, not the TD-smoothed target. Detach μ so σ's gradient can't
        // flow back into the mean head.
        let mean_detached = mean.clone().detach();
        let sigma_residual = final_scores - mean_detached;
        let sigma_sq = sigma_residual.clone() * sigma_residual;
        let inv_var = log_var.clone().neg().exp();
        let sigma_nll = log_var + sigma_sq * inv_var;
        let sigma_loss = sigma_nll.mean().mul_scalar(0.5);

        let loss = mu_loss + sigma_loss.mul_scalar(SIGMA_LOSS_WEIGHT);

        RegressionOutput { loss, output: mean, targets }
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

// ---- Training samples / batcher ----

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TrainingSample {
    pub features: Vec<f32>,
    /// TD(λ) target G_t used for μ regression. Smoothed, sample-efficient.
    pub value: f32,
    /// Actual final score of the trajectory this sample was recorded on. Used
    /// to train σ against an unbiased estimator of Var(X_final | s), bypassing
    /// the TD-smoothing bias that a joint NLL on `value` would introduce.
    pub final_score: f32,
}

#[derive(Clone)]
pub struct QwixxBatcher<B: Backend> {
    _phantom: std::marker::PhantomData<B>,
}

#[derive(Clone, Debug)]
pub struct QwixxBatch<B: Backend> {
    pub inputs: Tensor<B, 2>,
    /// TD(λ) targets (used to train μ).
    pub targets: Tensor<B, 1>,
    /// Actual final scores of each sample's trajectory (used to train σ).
    pub final_scores: Tensor<B, 1>,
}

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
                        f.swap(base, base + 2);
                        f.swap(base + 1, base + 3);
                    }
                }
                f
            })
            .collect();
        let targets: Vec<f32> = items.iter().map(|s| s.value).collect();
        let final_scores: Vec<f32> = items.iter().map(|s| s.final_score).collect();

        let inputs = Tensor::<B, 1>::from_floats(inputs.as_slice(), device)
            .reshape([batch_size, NUM_FEATURES]);
        let targets = Tensor::<B, 1>::from_floats(targets.as_slice(), device);
        let final_scores = Tensor::<B, 1>::from_floats(final_scores.as_slice(), device);

        QwixxBatch { inputs, targets, final_scores }
    }
}

// ---- MC-supervised data generation ----

/// Generate training data by playing games and evaluating states with MC.
pub fn generate_training_data(num_games: usize, mc_sims: usize) -> Vec<TrainingSample> {
    let genes = Arc::new(bot::default_genes());
    let champion = DNA::load_weights("champion.txt", genes).expect("No champion.txt found");
    let mc = MonteCarlo::with_ga(mc_sims, champion.clone());

    println!("Generating training data: {num_games} games, {mc_sims} MC sims per state...");

    let samples: Vec<TrainingSample> = (0..num_games)
        .into_par_iter()
        .flat_map(|game_idx| {
            use crate::state::Move;
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
                        let ctx = build_opponent_context_for(new_state.count_points(), &opponent_state, &[]);
                        let features = state_features(&new_state, &ctx);
                        local_samples.push(TrainingSample {
                            features: features.to_vec(),
                            value: mc_value as f32,
                            // MC pretraining doesn't have a sampled final score;
                            // reuse the MC mean. σ head learns ≈ 0 here and is
                            // rehydrated properly by self-play training.
                            final_score: mc_value as f32,
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
                    let opp_mov = ga.passive_phase1(&opponent_state, &[state], [on_white, 0, 0, 0, 0, 0], 0);
                    if let Some(mark) = opp_mov { opponent_state.apply_mark(mark); }
                    opponent_state.lock(locked);
                    state.lock(opponent_state.locked());
                } else {
                    // Opponent's active turn
                    let opp_phase1 = ga.active_phase1(&opponent_state, &[state], dice);
                    if let Some(mark) = opp_phase1 { opponent_state.apply_mark(mark); }
                    let opp_phase2 = ga.active_phase2(&opponent_state, &[state], dice, opp_phase1.is_some());
                    match opp_phase2 {
                        Some(mark) => { opponent_state.apply_mark(mark); }
                        None if opp_phase1.is_none() => { opponent_state.apply_strike(); }
                        None => {}
                    }
                    let locked = opponent_state.locked();
                    let our_mov = ga.passive_phase1(&state, &[opponent_state], [on_white, 0, 0, 0, 0, 0], 0);
                    if let Some(mark) = our_mov { state.apply_mark(mark); }
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

// ---- MC-supervised training ----

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

// ---- Self-play RL training ----

/// Batched forward pass: evaluates all provided feature vectors in one model
/// call. Returns `(mean, log_var)` per input (empty input → empty output).
fn batch_eval_features(
    model: &QwixxModel<MyBackend>,
    device: &burn::backend::ndarray::NdArrayDevice,
    features_list: &[[f32; NUM_FEATURES]],
) -> Vec<(f32, f32)> {
    if features_list.is_empty() {
        return Vec::new();
    }
    let n = features_list.len();
    let flat: Vec<f32> = features_list.iter().flat_map(|f| f.iter().copied()).collect();
    let input = Tensor::<MyBackend, 1>::from_floats(flat.as_slice(), device)
        .reshape([n, NUM_FEATURES]);
    let output = model.forward(input);
    let values = output.into_data().to_vec::<f32>().unwrap();
    (0..n).map(|i| (values[2 * i], values[2 * i + 1])).collect()
}

/// DQN wrapper used during self-play training. Picks moves with ε-greedy
/// exploration and records post-move features into a shared buffer that the
/// training loop drains after `Game::play` returns.
struct RecordingDqn {
    model: QwixxModel<MyBackend>,
    device: burn::backend::ndarray::NdArrayDevice,
    epsilon: f32,
    rng: SmallRng,
    /// Shared with the training loop. Stays inside one rayon closure per game
    /// so Rc<RefCell<..>> is sufficient — no synchronization needed.
    recorded: std::rc::Rc<std::cell::RefCell<Vec<[f32; NUM_FEATURES]>>>,
}

impl std::fmt::Debug for RecordingDqn {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "RecordingDqn")
    }
}

impl RecordingDqn {
    /// Build opponent context from opp_states.
    fn build_context(&self, our_state: &State, opp_states: &[State]) -> OpponentContext {
        if opp_states.is_empty() {
            return OpponentContext::default();
        }
        let leader_idx = (0..opp_states.len())
            .max_by_key(|&i| opp_states[i].count_points())
            .unwrap();
        let leader = &opp_states[leader_idx];
        let non_leaders: Vec<State> = opp_states
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != leader_idx)
            .map(|(_, s)| *s)
            .collect();
        build_opponent_context_for(our_state.count_points(), leader, &non_leaders)
    }

    /// Find leader and non-leaders from opp_states.
    fn find_leader<'a>(&self, opp_states: &'a [State]) -> (Option<&'a State>, Vec<State>) {
        if opp_states.is_empty() {
            return (None, Vec::new());
        }
        let leader_idx = (0..opp_states.len())
            .max_by_key(|&i| opp_states[i].count_points())
            .unwrap();
        let leader = &opp_states[leader_idx];
        let non_leaders: Vec<State> = opp_states
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != leader_idx)
            .map(|(_, s)| *s)
            .collect();
        (Some(leader), non_leaders)
    }

    /// Rank candidate post-move states by win probability with per-candidate
    /// opp context, matching `DqnStrategy` Bot impl.
    fn rank(&self, our_post_states: &[State], opp_states: &[State]) -> Vec<f32> {
        let (leader, non_leaders) = self.find_leader(opp_states);
        rank_candidates_with_opp_context(
            &self.model,
            &self.device,
            leader,
            &non_leaders,
            our_post_states,
        )
    }

    fn record_features(&self, state: &State, opp_states: &[State]) {
        let ctx = self.build_context(state, opp_states);
        let features = state_features(state, &ctx);
        self.recorded.borrow_mut().push(features);
    }
}

impl Strategy for RecordingDqn {
    fn active_phase1(&mut self, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
        let white_sum = dice[0] + dice[1];
        let white_marks = state.generate_white_moves(white_sum);
        let color_marks = state.generate_color_moves(dice);

        // Generate all (phase1, phase2) plans like the blanket impl
        let mut plans: Vec<(Option<Mark>, Option<Mark>, State)> = Vec::new();

        // Strike: (None, None) -> apply_strike
        {
            let mut s = *state;
            s.apply_strike();
            plans.push((None, None, s));
        }

        // Color-only singles: (None, Some(cm)) -> apply cm
        for &cm in &color_marks {
            let mut s = *state;
            s.apply_mark(cm);
            plans.push((None, Some(cm), s));
        }

        // White-only singles: (Some(wm), None) -> apply wm
        for &wm in &white_marks {
            let mut s = *state;
            s.apply_mark(wm);
            plans.push((Some(wm), None, s));
        }

        // Doubles: (Some(wm), Some(cm))
        for &wm in &white_marks {
            let mut post_white = *state;
            post_white.apply_mark(wm);
            let post_color_marks = post_white.generate_color_moves(dice);
            for &cm in &post_color_marks {
                let mut s = post_white;
                s.apply_mark(cm);
                plans.push((Some(wm), Some(cm), s));
            }
        }

        if plans.is_empty() {
            return None;
        }

        // Epsilon-greedy
        let chosen_idx = if self.rng.gen::<f32>() < self.epsilon {
            self.rng.gen_range(0..plans.len())
        } else {
            let post_states: Vec<State> = plans.iter().map(|(_, _, s)| *s).collect();
            let ranks = self.rank(&post_states, opp_states);
            ranks
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0
        };

        // Record features of the chosen plan's final state
        let (phase1, _, ref final_state) = plans[chosen_idx];
        self.record_features(final_state, opp_states);

        phase1
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
            return None;
        }

        let mut candidates: Vec<State> = marks
            .iter()
            .map(|&m| {
                let mut s = *state;
                s.apply_mark(m);
                s
            })
            .collect();
        candidates.push(no_mark_state);

        // Epsilon-greedy
        let chosen_idx = if self.rng.gen::<f32>() < self.epsilon {
            self.rng.gen_range(0..candidates.len())
        } else {
            let ranks = self.rank(&candidates, opp_states);
            ranks
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0
        };

        // Record features of chosen candidate
        self.record_features(&candidates[chosen_idx], opp_states);

        if chosen_idx < marks.len() {
            Some(marks[chosen_idx])
        } else {
            None
        }
    }

    fn passive_phase1(&mut self, state: &State, opp_states: &[State], dice: [u8; 6], _active_player: usize) -> Option<Mark> {
        let white_sum = dice[0] + dice[1];
        let marks = state.generate_white_moves(white_sum);
        if marks.is_empty() {
            return None;
        }

        let mut candidates: Vec<State> = marks
            .iter()
            .map(|&m| {
                let mut s = *state;
                s.apply_mark(m);
                s
            })
            .collect();
        candidates.push(*state); // skip

        let ranks = self.rank(&candidates, opp_states);
        let skip_rank = *ranks.last().unwrap();
        let mark_ranks = &ranks[..marks.len()];
        let (best_idx, best_rank) = mark_ranks
            .iter()
            .copied()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        if best_rank > skip_rank {
            // Record when we mark
            self.record_features(&candidates[best_idx], opp_states);
            Some(marks[best_idx])
        } else {
            None
        }
    }
}

/// Play a training game using the shared `Game::play` loop with a `RecordingDqn`
/// as player 0. Returns (TD(λ) training samples, final score for player 0).
fn play_training_game(
    model: &QwixxModel<MyBackend>,
    device: &burn::backend::ndarray::NdArrayDevice,
    opponents: Vec<Box<dyn Strategy>>,
    epsilon: f32,
    seed: u64,
) -> (Vec<TrainingSample>, f32) {
    use crate::game::{Game, Player};

    let recorded: std::rc::Rc<std::cell::RefCell<Vec<[f32; NUM_FEATURES]>>> =
        std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));

    let recording = RecordingDqn {
        model: model.clone(),
        device: device.clone(),
        epsilon,
        rng: SmallRng::seed_from_u64(seed),
        recorded: std::rc::Rc::clone(&recorded),
    };

    let mut players: Vec<Player> = Vec::with_capacity(1 + opponents.len());
    players.push(Player::new(
        Box::new(recording),
        Box::new(SmallRng::seed_from_u64(seed.wrapping_add(1))),
    ));
    for (i, opp) in opponents.into_iter().enumerate() {
        players.push(Player::new(
            opp,
            Box::new(SmallRng::seed_from_u64(seed.wrapping_add(2 + i as u64))),
        ));
    }

    let mut game = Game::new(players);
    game.play();

    let final_score = game.players[0].state.count_points() as f32;

    // Drain the recorded buffer (still shared with the RecordingDqn inside player 0).
    let recorded_features: Vec<[f32; NUM_FEATURES]> =
        std::mem::take(&mut *recorded.borrow_mut());

    let lambda = 0.8f32;
    let n = recorded_features.len();
    if n == 0 {
        return (Vec::new(), final_score);
    }

    // Batched forward pass for TD(λ) bootstrap values. We bootstrap the
    // *mean* prediction only; the variance head isn't in the bootstrap target.
    let values = batch_eval_features(model, device, &recorded_features);

    // G_t = (1-λ)·V(s_{t+1}) + λ·G_{t+1}, with G_{n-1} = final_score.
    let mut targets = vec![0.0f32; n];
    targets[n - 1] = final_score;
    for t in (0..n - 1).rev() {
        targets[t] = (1.0 - lambda) * values[t + 1].0 + lambda * targets[t + 1];
    }

    let samples = recorded_features
        .into_iter()
        .zip(targets)
        .map(|(features, target)| TrainingSample {
            features: features.to_vec(),
            value: target,
            final_score,
        })
        .collect();
    (samples, final_score)
}

// ---- Self-play benchmark + training loop ----

fn benchmark_vs_ga(artifact_dir: &str, champion: &DNA, num_games: usize) -> f64 {
    use crate::game::{Game, Player};

    // Load model once per rayon thread via map_init — shared parse, one clone per game
    let wins: u32 = (0..num_games)
        .into_par_iter()
        .map_init(
            || DqnStrategy::load(artifact_dir),
            |template, i| {
                // Cheap: model tensors are Arc-backed; fresh per game.
                let dqn = DqnStrategy {
                    model: template.model.clone(),
                    device: template.device.clone(),
                };
                let rotation = i % 2;
                let players: Vec<Player> = if rotation == 0 {
                    vec![
                        Player::new(Box::new(dqn), Box::new(SmallRng::from_entropy())),
                        Player::new(Box::new(champion.clone()), Box::new(SmallRng::from_entropy())),
                    ]
                } else {
                    vec![
                        Player::new(Box::new(champion.clone()), Box::new(SmallRng::from_entropy())),
                        Player::new(Box::new(dqn), Box::new(SmallRng::from_entropy())),
                    ]
                };
                let mut game = Game::new(players);
                game.play();
                let scores: Vec<isize> = game.players.iter().map(|p| p.state.count_points()).collect();
                let dqn_idx = rotation;
                if scores[dqn_idx] > scores[1 - dqn_idx] { 1u32 } else { 0u32 }
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
) {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    MyBackend::seed(&device, TRAIN_SEED);
    let buffer_iterations = 3;
    let mut replay_buffer: std::collections::VecDeque<Vec<TrainingSample>> =
        std::collections::VecDeque::new();

    let scores_log_path = format!("{artifact_dir}/training_scores.csv");
    // Write header
    std::fs::write(&scores_log_path, "iteration,avg_score,winrate\n").ok();

    let genes = Arc::new(bot::default_genes());
    let champion = DNA::load_weights("champion.txt", genes).expect("No champion.txt");
    let start_time = std::time::Instant::now();

    // Per-iteration stats for end-of-training summary (iter, avg_score, winrate).
    let mut iteration_stats: Vec<(usize, f32, Option<f64>)> = Vec::new();

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

        let games_each = games_per_iteration / 6;

        // 6 configs: 1v1 GA, 1v1 self, 3p vs 2 GA, 4p vs 3 GA, 3p vs GA + self, 4p vs 2 GA + self
        let game_configs: Vec<u8> = (0..6)
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
                // Deterministic DQN self-play opponents
                let dqn_self = || -> Box<dyn Strategy> {
                    Box::new(DqnStrategy {
                        model: thread_model.clone(),
                        device: device.clone(),
                    })
                };
                let opps: Vec<Box<dyn Strategy>> = match config {
                    // 1v1: vs GA champion
                    0 => vec![Box::new(champion.clone())],
                    // 1v1: vs self
                    1 => vec![dqn_self()],
                    // 3-player: vs 2 GA champions
                    2 => vec![Box::new(champion.clone()), Box::new(champion.clone())],
                    // 4-player: vs 3 GA champions
                    3 => vec![
                        Box::new(champion.clone()),
                        Box::new(champion.clone()),
                        Box::new(champion.clone()),
                    ],
                    // 3-player: vs GA + self
                    4 => vec![Box::new(champion.clone()), dqn_self()],
                    // 4-player: vs 2 GA + self
                    _ => vec![
                        Box::new(champion.clone()),
                        Box::new(champion.clone()),
                        dqn_self(),
                    ],
                };
                play_training_game(&thread_model, &device, opps, epsilon, seed)
            })
            .collect();

        let new_samples: Vec<TrainingSample> =
            game_results.iter().flat_map(|(s, _)| s.iter().cloned()).collect();
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

        // Train on replay buffer
        train_with_epochs(all_samples, artifact_dir, epochs_per_iteration, 4e-4);

        // Optional: persist a per-iteration checkpoint.
        if checkpoints {
            let src = format!("{artifact_dir}/model.mpk");
            let dst = format!("{artifact_dir}/iter-{}.mpk", iteration + 1);
            if let Err(e) = std::fs::copy(&src, &dst) {
                eprintln!("  Failed to save iter-{} checkpoint: {e}", iteration + 1);
            }
        }

        let elapsed = start_time.elapsed().as_secs();
        let mins = elapsed / 60;
        let secs = elapsed % 60;

        use std::io::Write;
        let winrate_opt = if bench_games > 0 {
            let winrate = benchmark_vs_ga(artifact_dir, &champion, bench_games);
            println!(
                "  Iteration {:>3}/{}: avg score {:.1}, winrate {:.1}%, elapsed {}m{}s",
                iteration + 1, num_iterations, avg_score, winrate * 100.0, mins, secs,
            );
            if let Ok(mut f) = std::fs::OpenOptions::new().append(true).open(&scores_log_path) {
                writeln!(f, "{},{avg_score:.2},{:.2}", iteration + 1, winrate * 100.0).ok();
            }
            Some(winrate)
        } else {
            println!(
                "  Iteration {:>3}/{}: avg score {:.1}, elapsed {}m{}s",
                iteration + 1, num_iterations, avg_score, mins, secs,
            );
            if let Ok(mut f) = std::fs::OpenOptions::new().append(true).open(&scores_log_path) {
                writeln!(f, "{},{avg_score:.2}", iteration + 1).ok();
            }
            None
        };

        iteration_stats.push((iteration + 1, avg_score, winrate_opt));
    }

    println!("\nSelf-play training complete. Model saved to {artifact_dir}/model");

    // End-of-training summary: best-by-winrate and best-by-avg-score iterations.
    if bench_games > 0 {
        let best_wr = iteration_stats
            .iter()
            .filter_map(|&(i, s, w)| w.map(|w| (i, s, w)))
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
        let best_score = iteration_stats
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        if let Some((i, s, w)) = best_wr {
            println!("  Best winrate:    iter {i:>3}: {:.1}% (avg score {s:.1})", w * 100.0);
        }
        if let Some(&(i, s, w)) = best_score {
            let wr_str = w.map(|w| format!("{:.1}%", w * 100.0)).unwrap_or_else(|| "-".into());
            println!("  Best avg score:  iter {i:>3}: {s:.1} (winrate {wr_str})");
        }
    }
}

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
