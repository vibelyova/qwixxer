//! Training for the pairwise differential network: per-opponent TD(λ) chains,
//! board-swap sample doubling, decoupled μ/σ loss, pure self-play loop.
//!
//! Only compiled with the `dqn` feature, mirroring `dqn::train`.

use crate::bot::{self, DNA};
use crate::dqn::pair::{pair_features, PairModel, PairModelConfig, PairStrategy, BOARD_FEATURES, PAIR_FEATURES};
use crate::dqn::{MyBackend, LOG_VAR_MAX, LOG_VAR_MIN, TRAIN_SEED};
use crate::state::{Mark, State};
use crate::strategy::search::SearchBot;
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

const LAMBDA: f32 = 0.8;

/// Absolute-space TD(λ) targets over one diff chain.
///
/// `mus[t]` is the model's *future-space* μ_diff at step t (used for the
/// bootstrap at t+1), `cdiffs[t]` the current point differential. Returns
/// `G_t` in absolute-diff space:
///
/// ```text
/// G_{n−1} = final_diff
/// G_t     = (1−λ)·(μ_{t+1} + cdiff_{t+1}) + λ·G_{t+1}
/// ```
fn td_diff_targets(mus: &[f32], cdiffs: &[f32], final_diff: f32, lambda: f32) -> Vec<f32> {
    let n = cdiffs.len();
    let mut g = vec![0.0f32; n];
    g[n - 1] = final_diff;
    for t in (0..n - 1).rev() {
        g[t] = (1.0 - lambda) * (mus[t + 1] + cdiffs[t + 1]) + lambda * g[t + 1];
    }
    g
}

// ---- Training samples / loss / batcher ----

/// serde adapter for `[f32; PAIR_FEATURES]` (length 45 exceeds serde's built-in
/// array impls). Round-trips through a `Vec<f32>`.
mod pair_features_serde {
    use super::PAIR_FEATURES;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(arr: &[f32; PAIR_FEATURES], s: S) -> Result<S::Ok, S::Error> {
        arr.as_slice().serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[f32; PAIR_FEATURES], D::Error> {
        let v = Vec::<f32>::deserialize(d)?;
        <[f32; PAIR_FEATURES]>::try_from(v.as_slice()).map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct PairSample {
    // serde's built-in array impls stop at length 32; PAIR_FEATURES is 45, so
    // (de)serialize via a slice helper. (train.rs's TrainingSample escapes this
    // because NUM_FEATURES = 25.)
    #[serde(with = "pair_features_serde")]
    pub features: [f32; PAIR_FEATURES],
    /// TD(λ) target in future-diff space: `G_t − current_diff_t`.
    pub value: f32,
    /// Actual final diff in future space: `final_diff − current_diff_t`.
    /// σ's residual reference (unbiased, bypasses TD smoothing).
    pub final_diff: f32,
}

impl<B: Backend> PairModel<B> {
    /// Decoupled μ/σ loss on the differential — same recipe as the old net:
    /// μ: MSE toward the TD(λ) target; σ: Gaussian NLL against the actual
    /// final-diff residual with μ detached.
    pub fn forward_step(&self, batch: PairBatch<B>) -> RegressionOutput<B> {
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

        RegressionOutput {
            loss,
            output: mean,
            targets,
        }
    }
}

impl<B: AutodiffBackend> TrainStep for PairModel<B> {
    type Input = PairBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, batch: PairBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_step(batch);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for PairModel<B> {
    type Input = PairBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, batch: PairBatch<B>) -> RegressionOutput<B> {
        self.forward_step(batch)
    }
}

#[derive(Clone)]
pub struct PairBatcher<B: Backend> {
    _phantom: std::marker::PhantomData<B>,
}

#[derive(Clone, Debug)]
pub struct PairBatch<B: Backend> {
    pub inputs: Tensor<B, 2>,
    pub targets: Tensor<B, 1>,
    pub final_diffs: Tensor<B, 1>,
}

/// Color-permutation augmentation applied identically to BOTH board blocks.
/// Per-row groups within each block: progress [0–3], marks [4–7],
/// locked [8–11], wprob [12–15]; pair-level features 40–44 are untouched
/// (they are row-permutation invariant).
fn permute_colors(f: &mut [f32; PAIR_FEATURES], swap_ry: bool, swap_gb: bool, swap_pairs: bool) {
    for block in [0, BOARD_FEATURES] {
        if swap_ry {
            for base in [0, 4, 8, 12] {
                f.swap(block + base, block + base + 1);
            }
        }
        if swap_gb {
            for base in [2, 6, 10, 14] {
                f.swap(block + base, block + base + 1);
            }
        }
        if swap_pairs {
            for base in [0, 4, 8, 12] {
                f.swap(block + base, block + base + 2);
                f.swap(block + base + 1, block + base + 3);
            }
        }
    }
}

impl<B: Backend> Batcher<B, PairSample, PairBatch<B>> for PairBatcher<B> {
    fn batch(&self, items: Vec<PairSample>, device: &B::Device) -> PairBatch<B> {
        let batch_size = items.len();
        let batch_seed = TRAIN_SEED
            .wrapping_add(items[0].value.to_bits() as u64)
            .wrapping_add(batch_size as u64);
        let mut rng = SmallRng::seed_from_u64(batch_seed);

        let inputs: Vec<f32> = items
            .iter()
            .flat_map(|s| {
                let mut f = s.features;
                permute_colors(&mut f, rng.gen(), rng.gen(), rng.gen());
                f
            })
            .collect();
        let targets: Vec<f32> = items.iter().map(|s| s.value).collect();
        let final_diffs: Vec<f32> = items.iter().map(|s| s.final_diff).collect();

        let inputs = Tensor::<B, 1>::from_floats(inputs.as_slice(), device).reshape([batch_size, PAIR_FEATURES]);
        let targets = Tensor::<B, 1>::from_floats(targets.as_slice(), device);
        let final_diffs = Tensor::<B, 1>::from_floats(final_diffs.as_slice(), device);

        PairBatch {
            inputs,
            targets,
            final_diffs,
        }
    }
}

// ---- Self-play recording ----

/// One recorded decision: our post-decision state + all opponents' states at
/// that moment (turn-ordered relative to us, constant order per game).
type Snapshot = (State, Vec<State>);

/// Move-selection policy for training-game players: today's static net, or
/// the shipped search bot (expert iteration). The ε-coin in `RecordingPair`
/// fires BEFORE the policy, so exploring decisions never pay for (or get
/// polished by) search.
enum PairPolicy {
    Static(PairStrategy),
    Search(SearchBot<PairStrategy>),
}

impl PairPolicy {
    fn bot(&self) -> &PairStrategy {
        match self {
            PairPolicy::Static(b) => b,
            PairPolicy::Search(s) => &s.bot,
        }
    }

    fn active_phase1(&mut self, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
        match self {
            PairPolicy::Static(b) => crate::strategy::active_phase1_impl(&*b, state, opp_states, dice),
            PairPolicy::Search(s) => crate::strategy::Strategy::active_phase1(s, state, opp_states, dice),
        }
    }

    fn active_phase2(&mut self, state: &State, opp_states: &[State], dice: [u8; 6], has_marked: bool) -> Option<Mark> {
        match self {
            PairPolicy::Static(b) => crate::strategy::active_phase2_impl(&*b, state, opp_states, dice, has_marked),
            PairPolicy::Search(s) => crate::strategy::Strategy::active_phase2(s, state, opp_states, dice, has_marked),
        }
    }
}

/// Pair-bot wrapper used during self-play training. ε-greedy on active
/// decisions, greedy on passive; records snapshots for chain building.
/// Move selection runs through a [`PairPolicy`] — either the static net or
/// the search bot; the recording cadence is identical in both. Recording
/// cadence: active turns once after phase 2 (post-turn state); passive turns
/// after every real decision — including skips, which the old recorder
/// dropped (skip afterstates are evaluated at inference, so they belong in
/// the training distribution).
struct RecordingPair {
    policy: PairPolicy,
    epsilon: f32,
    rng: SmallRng,
    recorded: std::rc::Rc<std::cell::RefCell<Vec<Snapshot>>>,
}

impl std::fmt::Debug for RecordingPair {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "RecordingPair")
    }
}

impl Strategy for RecordingPair {
    fn active_phase1(&mut self, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
        if self.rng.gen::<f32>() < self.epsilon {
            let white_marks = state.generate_white_moves(dice[0] + dice[1]);
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
            self.policy.active_phase1(state, opp_states, dice)
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
            self.recorded.borrow_mut().push((no_mark_state, opp_states.to_vec()));
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
            self.policy.active_phase2(state, opp_states, dice, has_marked)
        };

        let chosen_state = match mark {
            Some(m) => {
                let mut s = *state;
                s.apply_mark(m);
                s
            }
            None => no_mark_state,
        };
        self.recorded.borrow_mut().push((chosen_state, opp_states.to_vec()));
        mark
    }

    fn passive_phase1(
        &mut self,
        state: &State,
        opp_states: &[State],
        dice: [u8; 6],
        _active_player: usize,
    ) -> Option<Mark> {
        let marks = state.generate_white_moves(dice[0] + dice[1]);
        if marks.is_empty() {
            return None;
        }

        let mark = crate::strategy::passive_phase1_impl(self.policy.bot(), state, opp_states, dice);

        let post = match mark {
            Some(m) => {
                let mut s = *state;
                s.apply_mark(m);
                s
            }
            None => *state, // record skips too
        };
        self.recorded.borrow_mut().push((post, opp_states.to_vec()));
        mark
    }
}

/// Build training samples from one player's trajectory: one TD(λ) chain per
/// opponent, every sample emitted in both board orders (swap doubling) with
/// pair-level features recomputed exactly from the swapped perspective and
/// targets negated.
fn build_pair_samples(
    net: &crate::dqn::pair::ManualPairNet,
    snapshots: &[Snapshot],
    our_final: f32,
    opp_finals: &[f32],
) -> Vec<PairSample> {
    let mut samples = Vec::new();
    if snapshots.is_empty() {
        return samples;
    }
    let num_opps = snapshots[0].1.len();
    debug_assert_eq!(num_opps, opp_finals.len());

    for k in 0..num_opps {
        let final_diff = our_final - opp_finals[k];
        let feats: Vec<[f32; PAIR_FEATURES]> = snapshots
            .iter()
            .map(|(our, opps)| pair_features(our, &opps[k], opps))
            .collect();
        let cdiffs: Vec<f32> = snapshots
            .iter()
            .map(|(our, opps)| (our.count_points() - opps[k].count_points()) as f32)
            .collect();
        let mus: Vec<f32> = net.forward(&feats).into_iter().map(|(m, _)| m).collect();
        let g = td_diff_targets(&mus, &cdiffs, final_diff, LAMBDA);

        for (t, (our, opps)) in snapshots.iter().enumerate() {
            let value = g[t] - cdiffs[t];
            let fdiff = final_diff - cdiffs[t];
            samples.push(PairSample {
                features: feats[t],
                value,
                final_diff: fdiff,
            });

            // Swapped sample: the paired opponent's perspective. Their
            // opponents are us plus the remaining opponents.
            let mut swapped_opps: Vec<State> = Vec::with_capacity(num_opps);
            swapped_opps.push(*our);
            swapped_opps.extend(opps.iter().enumerate().filter(|(j, _)| *j != k).map(|(_, s)| *s));
            samples.push(PairSample {
                features: pair_features(&opps[k], our, &swapped_opps),
                value: -value,
                final_diff: -fdiff,
            });
        }
    }
    samples
}

// ---- Self-play training loop ----

/// Play one pure-self-play training game; every player is a recording pair
/// bot (player 0 explores with ε, the rest are greedy). Returns all samples
/// plus player 0's final score.
fn play_training_game(
    model: &PairModel<MyBackend>,
    device: &burn::backend::ndarray::NdArrayDevice,
    num_opponents: usize,
    search: bool,
    epsilon: f32,
    seed: u64,
) -> (Vec<PairSample>, f32) {
    use crate::game::{Game, Player};

    let n = num_opponents + 1;
    let strategies: Vec<PairStrategy> = (0..n)
        .map(|_| PairStrategy::from_model(model.clone(), device.clone()))
        .collect();
    let boot_net = strategies[0].net.clone();

    let mut buffers = Vec::with_capacity(n);
    let mut players: Vec<Player> = Vec::with_capacity(n);
    for (i, strategy) in strategies.into_iter().enumerate() {
        let policy = if search {
            PairPolicy::Search(SearchBot::new(strategy))
        } else {
            PairPolicy::Static(strategy)
        };
        let buf = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        buffers.push(std::rc::Rc::clone(&buf));
        players.push(Player::new(
            Box::new(RecordingPair {
                policy,
                epsilon: if i == 0 { epsilon } else { 0.0 },
                rng: SmallRng::seed_from_u64(seed.wrapping_add(100 + i as u64)),
                recorded: buf,
            }),
            Box::new(SmallRng::seed_from_u64(seed.wrapping_add(i as u64))),
        ));
    }

    let mut game = Game::new(players);
    game.play();

    let finals: Vec<f32> = game.players.iter().map(|p| p.state.count_points() as f32).collect();

    let mut all_samples = Vec::new();
    for (i, buf) in buffers.iter().enumerate() {
        let snapshots = std::mem::take(&mut *buf.borrow_mut());
        // Opponent k of player i is player (i + 1 + k) % n — matches the
        // turn-ordered opp_states the game loop passes to strategies.
        let opp_finals: Vec<f32> = (1..n).map(|off| finals[(i + off) % n]).collect();
        all_samples.extend(build_pair_samples(&boot_net, &snapshots, finals[i], &opp_finals));
    }

    (all_samples, finals[0])
}

/// Same fixed-game-set paired benchmark as `train::benchmark_vs_ga`, for the
/// pair bot. Seat-swapped pairs share per-seat dice streams.
const BENCH_SEED: u64 = 0xB54C;

fn benchmark_vs_ga(artifact_dir: &str, champion: &DNA, num_games: usize) -> f64 {
    use crate::game::{Game, Player};

    let wins: u32 = (0..num_games)
        .into_par_iter()
        .map_init(
            || PairStrategy::load(artifact_dir),
            |template, i| {
                let bot = PairStrategy::from_shared(template.model.clone(), template.device.clone());
                let pair = (i / 2) as u64;
                let rotation = i % 2;
                let seat_dice = |seat: u64| Box::new(SmallRng::seed_from_u64(BENCH_SEED.wrapping_add(pair * 2 + seat)));
                let players: Vec<Player> = if rotation == 0 {
                    vec![
                        Player::new(Box::new(bot), seat_dice(0)),
                        Player::new(Box::new(champion.clone()), seat_dice(1)),
                    ]
                } else {
                    vec![
                        Player::new(Box::new(champion.clone()), seat_dice(0)),
                        Player::new(Box::new(bot), seat_dice(1)),
                    ]
                };
                let mut game = Game::new(players);
                game.play();
                let scores: Vec<isize> = game.players.iter().map(|p| p.state.count_points()).collect();
                let idx = rotation;
                if scores[idx] > scores[1 - idx] {
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
    search: bool,
) {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    MyBackend::seed(&device, TRAIN_SEED);
    std::fs::create_dir_all(artifact_dir).ok();
    let buffer_iterations = 3;
    let mut replay_buffer: std::collections::VecDeque<Vec<PairSample>> = std::collections::VecDeque::new();

    let scores_log_path = format!("{artifact_dir}/training_scores.csv");
    if start_iteration == 0 {
        std::fs::write(&scores_log_path, "iteration,avg_score,winrate\n").ok();
    }

    let genes = Arc::new(bot::default_genes());
    let champion = DNA::load_weights("champion.txt", genes).expect("No champion.txt");
    let start_time = std::time::Instant::now();

    let mut iteration_stats: Vec<(usize, f32, Option<f64>)> = Vec::new();

    let mut model: PairModel<MyBackend> = PairModelConfig::new()
        .init::<MyBackend>(&device)
        .load_file(format!("{artifact_dir}/model"), &CompactRecorder::new(), &device)
        .unwrap_or_else(|_| {
            println!("  No pretrained pair model, starting fresh");
            PairModelConfig::new().init::<MyBackend>(&device)
        });

    if search {
        println!("Expert iteration: generation uses the search bot (K=128) for all players");
    }

    for iteration in 0..num_iterations {
        let global_iter = start_iteration + iteration;
        let epsilon = (0.2 * (0.95f32).powi(global_iter as i32)).max(0.07);
        println!("\n=== Pair iteration {} (epsilon={epsilon:.3}) ===", global_iter + 1);

        let games_each = games_per_iteration / 3;
        // Pure self-play thirds: 1v1, 3p, 4p.
        let game_configs: Vec<usize> = [1usize, 2, 3]
            .iter()
            .flat_map(|&num_opps| std::iter::repeat(num_opps).take(games_each))
            .collect();

        let models: Vec<PairModel<MyBackend>> = (0..game_configs.len()).map(|_| model.clone()).collect();

        let game_results: Vec<(Vec<PairSample>, f32)> = game_configs
            .into_par_iter()
            .zip(models.into_par_iter())
            .enumerate()
            .map(|(game_idx, (num_opps, thread_model))| {
                let seed = TRAIN_SEED.wrapping_add((iteration * games_per_iteration + game_idx) as u64);
                play_training_game(&thread_model, &device, num_opps, search, epsilon, seed)
            })
            .collect();

        let game_scores: Vec<f32> = game_results.iter().map(|(_, score)| *score).collect();
        let new_samples: Vec<PairSample> = game_results.into_iter().flat_map(|(s, _)| s).collect();
        let avg_score = if game_scores.is_empty() {
            0.0
        } else {
            game_scores.iter().sum::<f32>() / game_scores.len() as f32
        };

        replay_buffer.push_back(new_samples);
        if replay_buffer.len() > buffer_iterations {
            replay_buffer.pop_front();
        }

        let all_samples: Vec<PairSample> = replay_buffer.iter().flatten().copied().collect();
        println!(
            "  Generated {} new samples (avg score: {avg_score:.1}), replay buffer: {} total",
            replay_buffer.back().unwrap().len(),
            all_samples.len()
        );

        model = train_with_epochs(all_samples, artifact_dir, epochs_per_iteration, 4e-4);

        if checkpoints {
            let src = format!("{artifact_dir}/model.mpk");
            let dst = format!("{artifact_dir}/iter-{}.mpk", global_iter + 1);
            if let Err(e) = std::fs::copy(&src, &dst) {
                eprintln!("  Failed to save iter-{} checkpoint: {e}", global_iter + 1);
            }
        }

        let elapsed = start_time.elapsed().as_secs();
        let (mins, secs) = (elapsed / 60, elapsed % 60);

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

    println!("\nPair self-play training complete. Model saved to {artifact_dir}/model");

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

fn train_with_epochs(samples: Vec<PairSample>, artifact_dir: &str, num_epochs: usize, lr: f64) -> PairModel<MyBackend> {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;

    let split = (samples.len() * 9) / 10;
    let train_data = InMemDataset::new(samples[..split].to_vec());
    let valid_data = InMemDataset::new(samples[split..].to_vec());

    let model: PairModel<MyAutodiffBackend> = PairModelConfig::new()
        .init::<MyAutodiffBackend>(&device)
        .load_file(format!("{artifact_dir}/model"), &CompactRecorder::new(), &device)
        .unwrap_or_else(|_| PairModelConfig::new().init::<MyAutodiffBackend>(&device));

    let batcher_train = PairBatcher::<MyAutodiffBackend> {
        _phantom: std::marker::PhantomData,
    };
    let batcher_valid = PairBatcher::<MyBackend> {
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
        .expect("Failed to save pair model");

    std::fs::remove_dir_all(&ckpt_dir).ok();

    PairModelConfig::new()
        .init::<MyBackend>(&device)
        .load_file(format!("{artifact_dir}/model"), &CompactRecorder::new(), &device)
        .expect("Failed to reload pair model for inference")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn td_diff_targets_hand_computed() {
        // 3 steps, λ = 0.8.
        let mus = [1.0, 2.0, 3.0];
        let cdiffs = [0.0, 5.0, 10.0];
        let g = td_diff_targets(&mus, &cdiffs, 20.0, 0.8);
        // G2 = 20
        // G1 = 0.2·(μ2 + cd2) + 0.8·G2 = 0.2·13 + 16    = 18.6
        // G0 = 0.2·(μ1 + cd1) + 0.8·G1 = 0.2·7 + 14.88  = 16.28
        assert!((g[2] - 20.0).abs() < 1e-5);
        assert!((g[1] - 18.6).abs() < 1e-5);
        assert!((g[0] - 16.28).abs() < 1e-5);
    }

    #[test]
    fn permute_colors_swaps_both_blocks_and_preserves_pair_features() {
        // Distinct values everywhere so swaps are observable.
        let mut f = [0.0f32; PAIR_FEATURES];
        for (i, v) in f.iter_mut().enumerate() {
            *v = i as f32;
        }
        let orig = f;

        permute_colors(&mut f, true, false, false); // red <-> yellow
        for block in [0, BOARD_FEATURES] {
            for base in [0usize, 4, 8, 12] {
                assert_eq!(f[block + base], orig[block + base + 1]);
                assert_eq!(f[block + base + 1], orig[block + base]);
            }
            // green/blue untouched
            for base in [2usize, 6, 10, 14] {
                assert_eq!(f[block + base], orig[block + base]);
            }
            // per-board aggregates untouched
            assert_eq!(f[block + 16..block + 20], orig[block + 16..block + 20]);
        }
        // Pair-level untouched.
        assert_eq!(f[40..45], orig[40..45]);

        // Applying the same swap again restores the original (involution).
        permute_colors(&mut f, true, false, false);
        assert_eq!(f, orig);
    }

    #[test]
    fn permute_colors_covers_gb_and_asc_desc_paths() {
        // Distinct values everywhere so swaps are observable.
        let mut f = [0.0f32; PAIR_FEATURES];
        for (i, v) in f.iter_mut().enumerate() {
            *v = i as f32;
        }
        let orig = f;

        // green <-> blue
        permute_colors(&mut f, false, true, false);
        for block in [0, BOARD_FEATURES] {
            for base in [2usize, 6, 10, 14] {
                assert_eq!(f[block + base], orig[block + base + 1]);
                assert_eq!(f[block + base + 1], orig[block + base]);
            }
            // red/yellow untouched
            for base in [0usize, 4, 8, 12] {
                assert_eq!(f[block + base], orig[block + base]);
            }
            // per-board aggregates untouched
            assert_eq!(f[block + 16..block + 20], orig[block + 16..block + 20]);
        }
        // Pair-level untouched.
        assert_eq!(f[40..45], orig[40..45]);
        // Involution.
        permute_colors(&mut f, false, true, false);
        assert_eq!(f, orig);

        // asc <-> desc pairs (two swaps per base).
        f = orig;
        permute_colors(&mut f, false, false, true);
        for block in [0, BOARD_FEATURES] {
            for base in [0usize, 4, 8, 12] {
                assert_eq!(f[block + base], orig[block + base + 2]);
                assert_eq!(f[block + base + 1], orig[block + base + 3]);
                assert_eq!(f[block + base + 2], orig[block + base]);
                assert_eq!(f[block + base + 3], orig[block + base + 1]);
            }
            // per-board aggregates untouched
            assert_eq!(f[block + 16..block + 20], orig[block + 16..block + 20]);
        }
        // Pair-level untouched.
        assert_eq!(f[40..45], orig[40..45]);
        // Involution.
        permute_colors(&mut f, false, false, true);
        assert_eq!(f, orig);
    }

    #[test]
    fn static_policy_dispatch_matches_impl_calls() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let model = PairModelConfig::new().init::<MyBackend>(&device);
        let bot = PairStrategy::from_model(model.clone(), device);
        let mut policy = PairPolicy::Static(PairStrategy::from_shared(bot.model.clone(), bot.device));

        let mut state = State::default();
        state.apply_mark(Mark { row: 1, number: 6 });
        let opps = [State::default()];
        let dice = [3, 4, 2, 3, 5, 1];

        assert_eq!(
            policy.active_phase1(&state, &opps, dice),
            crate::strategy::active_phase1_impl(&bot, &state, &opps, dice)
        );
        assert_eq!(
            policy.active_phase2(&state, &opps, dice, false),
            crate::strategy::active_phase2_impl(&bot, &state, &opps, dice, false)
        );
    }

    #[test]
    fn build_pair_samples_emits_negated_swapped_samples() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let model = PairModelConfig::new().init::<MyBackend>(&device);
        let net = crate::dqn::pair::ManualPairNet::from_model(&model);

        // 2-step 1v1 trajectory with asymmetric boards.
        let mut our1 = State::default();
        our1.apply_mark(crate::state::Mark { row: 0, number: 5 });
        let opp1 = State::default();
        let mut our2 = our1;
        our2.apply_mark(crate::state::Mark { row: 0, number: 7 });
        let mut opp2 = State::default();
        opp2.apply_mark(crate::state::Mark { row: 2, number: 10 });

        let snapshots = vec![(our1, vec![opp1]), (our2, vec![opp2])];
        let samples = build_pair_samples(&net, &snapshots, 30.0, &[20.0]);

        // 2 steps × 1 opponent × 2 orders.
        assert_eq!(samples.len(), 4);
        for pair in samples.chunks(2) {
            let (fwd, swp) = (&pair[0], &pair[1]);
            assert_eq!(swp.value, -fwd.value);
            assert_eq!(swp.final_diff, -fwd.final_diff);
            // Board blocks exchanged.
            assert_eq!(
                fwd.features[..BOARD_FEATURES],
                swp.features[BOARD_FEATURES..2 * BOARD_FEATURES]
            );
            assert_eq!(
                fwd.features[BOARD_FEATURES..2 * BOARD_FEATURES],
                swp.features[..BOARD_FEATURES]
            );
            // cdiff input negated (within clamp range here).
            assert!((fwd.features[40] + swp.features[40]).abs() < 1e-6);
        }
        // Last forward sample's targets: G_{n−1} = final_diff = 10;
        // cdiff at t=1: our 3 pts (2 marks) − opp 1 pt (1 mark) = 2.
        let last_fwd = &samples[2];
        assert!((last_fwd.value - 8.0).abs() < 1e-5);
        assert!((last_fwd.final_diff - 8.0).abs() < 1e-5);
    }

    #[test]
    fn exploring_decisions_never_invoke_search() {
        use crate::game::{Game, Player};
        use crate::strategy::search::{SearchBot, SearchStats};
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let model = PairModelConfig::new().init::<MyBackend>(&device);

        let mut players = Vec::new();
        let mut stats = Vec::new();
        for i in 0..2u64 {
            let mut sb = SearchBot::new(PairStrategy::from_model(model.clone(), device));
            let st = std::rc::Rc::new(std::cell::RefCell::new(SearchStats::default()));
            sb.stats = Some(st.clone());
            stats.push(st);
            players.push(Player::new(
                Box::new(RecordingPair {
                    policy: PairPolicy::Search(sb),
                    epsilon: 1.0, // always explore
                    rng: SmallRng::seed_from_u64(900 + i),
                    recorded: std::rc::Rc::new(std::cell::RefCell::new(Vec::new())),
                }),
                Box::new(SmallRng::seed_from_u64(910 + i)),
            ));
        }
        let mut game = Game::new(players);
        game.play();

        for st in &stats {
            assert_eq!(
                st.borrow().active_decisions,
                0,
                "search ran despite epsilon=1.0 — the ε-coin must fire first"
            );
        }
    }

    #[test]
    fn search_generation_is_deterministic() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let model = PairModelConfig::new().init::<MyBackend>(&device);

        let run = || play_training_game(&model, &device, 1, true, 0.07, 4242);
        let (s1, f1) = run();
        let (s2, f2) = run();
        assert_eq!(f1, f2);
        assert_eq!(s1.len(), s2.len());
        for (a, b) in s1.iter().zip(&s2) {
            assert_eq!(a.features, b.features);
            assert_eq!(a.value, b.value);
            assert_eq!(a.final_diff, b.final_diff);
        }
    }
}
