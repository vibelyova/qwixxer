//! DQN training: MC-supervised pretraining + TD(λ) self-play RL.
//!
//! Only compiled with the `dqn` feature. The inference side ([`crate::dqn`])
//! is always available so native and wasm binaries can share the same model.

use crate::bot::{self, DNA};
use crate::dqn::{
    batch_forward_features, build_opponent_context_for, state_features, DqnStrategy, MyBackend, OpponentContext,
    QwixxModel, QwixxModelConfig, LOG_VAR_MAX, LOG_VAR_MIN, NUM_FEATURES, TRAIN_SEED,
};
use crate::state::{Mark, State};
use crate::strategy::Strategy;
use burn::{
    backend::Autodiff,
    data::{
        dataloader::{batcher::Batcher, DataLoaderBuilder},
        dataset::InMemDataset,
    },
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{metric::LossMetric, InferenceStep, Learner, RegressionOutput, SupervisedTraining, TrainOutput, TrainStep},
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

        RegressionOutput {
            loss,
            output: mean,
            targets,
        }
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

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct TrainingSample {
    pub features: [f32; NUM_FEATURES],
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
                let mut f = s.features;
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

        let inputs = Tensor::<B, 1>::from_floats(inputs.as_slice(), device).reshape([batch_size, NUM_FEATURES]);
        let targets = Tensor::<B, 1>::from_floats(targets.as_slice(), device);
        let final_scores = Tensor::<B, 1>::from_floats(final_scores.as_slice(), device);

        QwixxBatch {
            inputs,
            targets,
            final_scores,
        }
    }
}

// ---- Self-play RL training ----

/// DQN wrapper used during self-play training. Picks moves with ε-greedy
/// exploration and records post-move features into a shared buffer that the
/// training loop drains after `Game::play` returns.
struct RecordingDqn {
    bot: DqnStrategy,
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
    fn record_features(&self, state: &State, opp_states: &[State]) {
        let ctx = if opp_states.is_empty() {
            OpponentContext::default()
        } else {
            let leader_idx = (0..opp_states.len())
                .max_by_key(|&i| opp_states[i].count_points())
                .unwrap();
            let non_leaders: Vec<State> = opp_states
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != leader_idx)
                .map(|(_, s)| *s)
                .collect();
            build_opponent_context_for(state.count_points(), &opp_states[leader_idx], &non_leaders)
        };
        let features = state_features(state, &ctx);
        self.recorded.borrow_mut().push(features);
    }
}

impl Strategy for RecordingDqn {
    fn active_phase1(&mut self, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
        if self.rng.gen::<f32>() < self.epsilon {
            let white_sum = dice[0] + dice[1];
            let white_marks = state.generate_white_moves(white_sum);
            if white_marks.is_empty() {
                return None;
            }
            let idx = self.rng.gen_range(0..=white_marks.len());
            if idx < white_marks.len() {
                Some(white_marks[idx])
            } else {
                None
            }
        } else {
            crate::strategy::active_phase1_impl(&self.bot, state, opp_states, dice)
        }
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
            self.record_features(&no_mark_state, opp_states);
            return None;
        }

        let mark = if self.rng.gen::<f32>() < self.epsilon {
            let idx = self.rng.gen_range(0..=marks.len());
            if idx < marks.len() {
                Some(marks[idx])
            } else {
                None
            }
        } else {
            crate::strategy::active_phase2_impl(&self.bot, state, opp_states, dice, has_marked)
        };

        let chosen_state = match mark {
            Some(m) => {
                let mut s = *state;
                s.apply_mark(m);
                s
            }
            None => no_mark_state,
        };
        self.record_features(&chosen_state, opp_states);
        mark
    }

    fn passive_phase1(
        &mut self,
        state: &State,
        opp_states: &[State],
        dice: [u8; 6],
        _active_player: usize,
    ) -> Option<Mark> {
        let white_sum = dice[0] + dice[1];
        let marks = state.generate_white_moves(white_sum);
        if marks.is_empty() {
            return None;
        }

        let mark = crate::strategy::passive_phase1_impl(&self.bot, state, opp_states, dice);

        if let Some(m) = mark {
            let mut s = *state;
            s.apply_mark(m);
            self.record_features(&s, opp_states);
        }
        mark
    }
}

/// Compute TD(λ) training samples from a recorded feature trajectory.
fn td_samples(
    model: &QwixxModel<MyBackend>,
    device: &burn::backend::ndarray::NdArrayDevice,
    features: Vec<[f32; NUM_FEATURES]>,
    final_score: f32,
) -> Vec<TrainingSample> {
    let n = features.len();
    if n == 0 {
        return Vec::new();
    }
    let lambda = 0.8f32;
    let values = batch_forward_features(model, device, &features);
    let mut targets = vec![0.0f32; n];
    targets[n - 1] = final_score;
    for t in (0..n - 1).rev() {
        targets[t] = (1.0 - lambda) * values[t + 1].0 + lambda * targets[t + 1];
    }
    features
        .into_iter()
        .zip(targets)
        .map(|(features, target)| TrainingSample {
            features,
            value: target,
            final_score,
        })
        .collect()
}

/// Play a training game with all DQN players recording experiences.
/// `num_dqn_opps` DQN opponents (greedy, epsilon=0) + remaining filled with GA.
/// Returns (TD(λ) samples from ALL DQN players, final score for player 0).
fn play_training_game(
    model: &QwixxModel<MyBackend>,
    device: &burn::backend::ndarray::NdArrayDevice,
    num_opponents: usize,
    num_dqn_opps: usize,
    champion: &DNA,
    epsilon: f32,
    seed: u64,
) -> (Vec<TrainingSample>, f32) {
    use crate::game::{Game, Player};

    let mut buffers: Vec<std::rc::Rc<std::cell::RefCell<Vec<[f32; NUM_FEATURES]>>>> = Vec::new();
    let mut recording_player_indices: Vec<usize> = Vec::new();

    // Player 0: RecordingDqn with exploration
    let buf0 = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    buffers.push(std::rc::Rc::clone(&buf0));
    recording_player_indices.push(0);
    let mut players: Vec<Player> = vec![Player::new(
        Box::new(RecordingDqn {
            bot: DqnStrategy::from_model(model.clone(), device.clone()),
            epsilon,
            rng: SmallRng::seed_from_u64(seed),
            recorded: buf0,
        }),
        Box::new(SmallRng::seed_from_u64(seed.wrapping_add(1))),
    )];

    // Opponents
    for i in 0..num_opponents {
        let opp_seed = seed.wrapping_add(2 + i as u64);
        if i < num_dqn_opps {
            let buf = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
            buffers.push(std::rc::Rc::clone(&buf));
            recording_player_indices.push(i + 1);
            players.push(Player::new(
                Box::new(RecordingDqn {
                    bot: DqnStrategy::from_model(model.clone(), device.clone()),
                    epsilon: 0.0,
                    rng: SmallRng::seed_from_u64(opp_seed.wrapping_add(100)),
                    recorded: buf,
                }),
                Box::new(SmallRng::seed_from_u64(opp_seed)),
            ));
        } else {
            players.push(Player::new(
                Box::new(champion.clone()),
                Box::new(SmallRng::seed_from_u64(opp_seed)),
            ));
        }
    }

    let mut game = Game::new(players);
    game.play();

    let p0_score = game.players[0].state.count_points() as f32;

    // Collect TD samples from all recording players
    let mut all_samples = Vec::new();
    for (buf_idx, buf) in buffers.iter().enumerate() {
        let player_idx = recording_player_indices[buf_idx];
        let player_score = game.players[player_idx].state.count_points() as f32;
        let features = std::mem::take(&mut *buf.borrow_mut());
        all_samples.extend(td_samples(model, device, features, player_score));
    }

    (all_samples, p0_score)
}

// ---- Self-play benchmark + training loop ----

fn benchmark_vs_ga(artifact_dir: &str, champion: &DNA, num_games: usize) -> f64 {
    use crate::game::{Game, Player};

    let wins: u32 = (0..num_games)
        .into_par_iter()
        .map_init(
            || DqnStrategy::load(artifact_dir),
            |template, i| {
                let dqn = DqnStrategy::from_shared(template.model.clone(), template.device.clone());
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
                if scores[dqn_idx] > scores[1 - dqn_idx] {
                    1u32
                } else {
                    0u32
                }
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
    start_iteration: usize,
) {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    MyBackend::seed(&device, TRAIN_SEED);
    let buffer_iterations = 3;
    let mut replay_buffer: std::collections::VecDeque<Vec<TrainingSample>> = std::collections::VecDeque::new();

    let scores_log_path = format!("{artifact_dir}/training_scores.csv");
    if start_iteration == 0 {
        std::fs::write(&scores_log_path, "iteration,avg_score,winrate\n").ok();
    }

    let genes = Arc::new(bot::default_genes());
    let champion = DNA::load_weights("champion.txt", genes).expect("No champion.txt");
    let start_time = std::time::Instant::now();

    // Per-iteration stats for end-of-training summary (iter, avg_score, winrate).
    let mut iteration_stats: Vec<(usize, f32, Option<f64>)> = Vec::new();

    let mut model: QwixxModel<MyBackend> = QwixxModelConfig::new()
        .init::<MyBackend>(&device)
        .load_file(format!("{artifact_dir}/model"), &CompactRecorder::new(), &device)
        .unwrap_or_else(|_| {
            println!("  No pretrained model, starting fresh");
            QwixxModelConfig::new().init::<MyBackend>(&device)
        });

    for iteration in 0..num_iterations {
        let global_iter = start_iteration + iteration;
        let epsilon = (0.2 * (0.95f32).powi(global_iter as i32)).max(0.07);
        println!(
            "\n=== Iteration {} (epsilon={epsilon:.3}) ===",
            global_iter + 1
        );

        let games_each = games_per_iteration / 3;

        // 3 configs, pure self-play: 1v1, 3p, 4p (all DQN opponents, all recording)
        let game_configs: Vec<(usize, usize)> = [
            (1, 1), // 1v1: 1 opp, 1 DQN
            (2, 2), // 3p: 2 opps, 2 DQN
            (3, 3), // 4p: 3 opps, 3 DQN
        ]
        .iter()
        .flat_map(|&cfg| std::iter::repeat(cfg).take(games_each))
        .collect();

        let models: Vec<QwixxModel<MyBackend>> = (0..game_configs.len()).map(|_| model.clone()).collect();

        let game_results: Vec<(Vec<TrainingSample>, f32)> = game_configs
            .into_par_iter()
            .zip(models.into_par_iter())
            .enumerate()
            .map(|(game_idx, ((num_opps, num_dqn), thread_model))| {
                let seed = TRAIN_SEED.wrapping_add((iteration * games_per_iteration + game_idx) as u64);
                play_training_game(&thread_model, &device, num_opps, num_dqn, &champion, epsilon, seed)
            })
            .collect();

        let game_scores: Vec<f32> = game_results.iter().map(|(_, score)| *score).collect();
        let new_samples: Vec<TrainingSample> = game_results.into_iter().flat_map(|(s, _)| s).collect();
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

        let all_samples: Vec<TrainingSample> = replay_buffer.iter().flatten().copied().collect();
        println!(
            "  Generated {} new samples (avg score: {avg_score:.1}), replay buffer: {} total",
            replay_buffer.back().unwrap().len(),
            all_samples.len()
        );

        // Train on replay buffer
        model = train_with_epochs(all_samples, artifact_dir, epochs_per_iteration, 4e-4);

        // Optional: persist a per-iteration checkpoint.
        if checkpoints {
            let src = format!("{artifact_dir}/model.mpk");
            let dst = format!("{artifact_dir}/iter-{}.mpk", global_iter + 1);
            if let Err(e) = std::fs::copy(&src, &dst) {
                eprintln!("  Failed to save iter-{} checkpoint: {e}", global_iter + 1);
            }
        }

        let elapsed = start_time.elapsed().as_secs();
        let mins = elapsed / 60;
        let secs = elapsed % 60;

        use std::io::Write;
        let winrate_opt = if bench_games > 0 {
            let winrate = benchmark_vs_ga(artifact_dir, &champion, bench_games);
            println!(
                "  Iteration {:>3}: avg score {:.1}, winrate {:.1}%, elapsed {}m{}s",
                global_iter + 1,
                avg_score,
                winrate * 100.0,
                mins,
                secs,
            );
            if let Ok(mut f) = std::fs::OpenOptions::new().append(true).open(&scores_log_path) {
                writeln!(f, "{},{avg_score:.2},{:.2}", global_iter + 1, winrate * 100.0).ok();
            }
            Some(winrate)
        } else {
            println!(
                "  Iteration {:>3}: avg score {:.1}, elapsed {}m{}s",
                global_iter + 1,
                avg_score,
                mins,
                secs,
            );
            if let Ok(mut f) = std::fs::OpenOptions::new().append(true).open(&scores_log_path) {
                writeln!(f, "{},{avg_score:.2}", global_iter + 1).ok();
            }
            None
        };

        iteration_stats.push((global_iter + 1, avg_score, winrate_opt));
    }

    println!("\nSelf-play training complete. Model saved to {artifact_dir}/model");

    // End-of-training summary: best-by-winrate and best-by-avg-score iterations.
    if bench_games > 0 {
        let best_wr = iteration_stats
            .iter()
            .filter_map(|&(i, s, w)| w.map(|w| (i, s, w)))
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
        let best_score = iteration_stats.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
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
fn train_with_epochs(samples: Vec<TrainingSample>, artifact_dir: &str, num_epochs: usize, lr: f64) -> QwixxModel<MyBackend> {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;

    let split = (samples.len() * 9) / 10;
    let train_data = InMemDataset::new(samples[..split].to_vec());
    let valid_data = InMemDataset::new(samples[split..].to_vec());

    // Load existing model or init fresh
    let model: QwixxModel<MyAutodiffBackend> = QwixxModelConfig::new()
        .init::<MyAutodiffBackend>(&device)
        .load_file(format!("{artifact_dir}/model"), &CompactRecorder::new(), &device)
        .unwrap_or_else(|_| QwixxModelConfig::new().init::<MyAutodiffBackend>(&device));

    let batcher_train = QwixxBatcher::<MyAutodiffBackend> {
        _phantom: std::marker::PhantomData,
    };
    let batcher_valid = QwixxBatcher::<MyBackend> {
        _phantom: std::marker::PhantomData,
    };

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

    QwixxModelConfig::new()
        .init::<MyBackend>(&device)
        .load_file(format!("{artifact_dir}/model"), &CompactRecorder::new(), &device)
        .expect("Failed to reload model for inference")
}
