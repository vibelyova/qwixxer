//! Training for the aznet value net: 2-player-only plain self-play, reusing
//! the pair net's TD(λ) diff targets, decoupled μ/σ loss, and board-swap
//! doubling. No search, no distillation. Only compiled with the `dqn` feature.

use crate::dqn::aznet::{az_batch_forward, az_features, AzBatch, AzModel, AzModelConfig, AzStrategy, AZ_FEATURES, BOARD_RAW, ROW_BLOCK};
use crate::dqn::pair_train::td_diff_targets;
use crate::dqn::{MyBackend, TRAIN_SEED};
use crate::state::{Mark, State};
use crate::strategy::{active_phase1_impl, active_phase2_impl, passive_phase1_impl, Strategy};
use burn::{
    backend::Autodiff,
    data::{dataloader::batcher::Batcher, dataset::InMemDataset, dataloader::DataLoaderBuilder},
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    train::{metric::LossMetric, Learner, SupervisedTraining},
};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use crate::bot::{self, DNA};
use rayon::prelude::*;

type MyAutodiffBackend = Autodiff<MyBackend>;

const LAMBDA: f32 = 0.8;

/// One training sample: the two boards (compact `State`s — expanded to the
/// 233-float input by [`AzBatcher`] at batch time, keeping the replay buffer
/// ~20x smaller than storing the expanded features), plus the TD(λ) target
/// (`value`) and the actual final-diff residual (`final_diff`).
#[derive(Clone, Copy, Debug)]
pub struct AzSample {
    pub our: State,
    pub opp: State,
    pub value: f32,
    pub final_diff: f32,
}

/// Swap the two 28-float row blocks at offsets `a` and `b`.
fn swap_block(f: &mut [f32; AZ_FEATURES], a: usize, b: usize) {
    for k in 0..ROW_BLOCK {
        f.swap(a + k, b + k);
    }
}

/// Color-permutation augmentation: permute the four 28-dim row blocks within
/// BOTH boards identically. Strikes and cdiff are row-permutation invariant.
pub(crate) fn permute_rows(f: &mut [f32; AZ_FEATURES], swap_ry: bool, swap_gb: bool, swap_pairs: bool) {
    for board in [0usize, BOARD_RAW] {
        let blk = |r: usize| board + r * ROW_BLOCK;
        if swap_ry {
            swap_block(f, blk(0), blk(1));
        }
        if swap_gb {
            swap_block(f, blk(2), blk(3));
        }
        if swap_pairs {
            swap_block(f, blk(0), blk(2));
            swap_block(f, blk(1), blk(3));
        }
    }
}

#[derive(Clone)]
pub struct AzBatcher<B: Backend> {
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> Batcher<B, AzSample, AzBatch<B>> for AzBatcher<B> {
    fn batch(&self, items: Vec<AzSample>, device: &B::Device) -> AzBatch<B> {
        let batch_size = items.len();
        let batch_seed = TRAIN_SEED
            .wrapping_add(items[0].value.to_bits() as u64)
            .wrapping_add(batch_size as u64);
        let mut rng = SmallRng::seed_from_u64(batch_seed);

        let inputs: Vec<f32> = items
            .iter()
            .flat_map(|s| {
                let mut f = az_features(&s.our, &s.opp);
                permute_rows(&mut f, rng.gen(), rng.gen(), rng.gen());
                f
            })
            .collect();
        let targets: Vec<f32> = items.iter().map(|s| s.value).collect();
        let final_diffs: Vec<f32> = items.iter().map(|s| s.final_diff).collect();

        let inputs = Tensor::<B, 1>::from_floats(inputs.as_slice(), device).reshape([batch_size, AZ_FEATURES]);
        let targets = Tensor::<B, 1>::from_floats(targets.as_slice(), device);
        let final_diffs = Tensor::<B, 1>::from_floats(final_diffs.as_slice(), device);

        AzBatch { inputs, targets, final_diffs }
    }
}

/// One recorded decision: our post-decision state + the opponent's state.
type Snapshot = (State, Vec<State>);

/// Build training samples from one player's 2-player trajectory: a single
/// TD(λ) chain, every sample emitted in both board orders (swap doubling) with
/// negated targets. μ for the bootstrap comes from a burn forward over the
/// chain's features.
/// 2p-only: indexes `opps[0]`; any extra opponents are ignored.
fn build_az_samples(
    model: &AzModel<MyBackend>,
    device: &burn::backend::ndarray::NdArrayDevice,
    snapshots: &[Snapshot],
    our_final: f32,
    opp_final: f32,
) -> Vec<AzSample> {
    let mut samples = Vec::new();
    if snapshots.is_empty() {
        return samples;
    }
    debug_assert!(
        !snapshots[0].1.is_empty(),
        "build_az_samples is 2-player: each snapshot needs >= 1 opponent"
    );
    let final_diff = our_final - opp_final;

    let feats: Vec<[f32; AZ_FEATURES]> = snapshots
        .iter()
        .map(|(our, opps)| az_features(our, &opps[0]))
        .collect();
    let cdiffs: Vec<f32> = snapshots
        .iter()
        .map(|(our, opps)| (our.count_points() - opps[0].count_points()) as f32)
        .collect();
    let mus: Vec<f32> = az_batch_forward(model, device, &feats).into_iter().map(|(m, _)| m).collect();
    let g = td_diff_targets(&mus, &cdiffs, final_diff, LAMBDA);

    for (t, (our, opps)) in snapshots.iter().enumerate() {
        let value = g[t] - cdiffs[t];
        let fdiff = final_diff - cdiffs[t];
        samples.push(AzSample { our: *our, opp: opps[0], value, final_diff: fdiff });
        samples.push(AzSample { our: opps[0], opp: *our, value: -value, final_diff: -fdiff });
    }
    samples
}

/// Self-play recorder: ε-greedy on active decisions, greedy on passive;
/// records afterstates for chain building. The static-policy subset of the
/// pair net's `RecordingPair` (no search, no distill).
struct RecordingAz {
    bot: AzStrategy,
    epsilon: f32,
    rng: SmallRng,
    recorded: Rc<RefCell<Vec<Snapshot>>>,
}

impl std::fmt::Debug for RecordingAz {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "RecordingAz")
    }
}

impl Strategy for RecordingAz {
    fn active_phase1(&mut self, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
        // ε-uniform: random white mark or skip (recording happens post-phase-2).
        if self.rng.gen::<f32>() < self.epsilon {
            let white_marks = state.generate_white_moves(dice[0] + dice[1]);
            if white_marks.is_empty() {
                return None;
            }
            let idx = self.rng.gen_range(0..=white_marks.len());
            return (idx < white_marks.len()).then(|| white_marks[idx]);
        }
        active_phase1_impl(&self.bot, state, opp_states, dice)
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
            (idx < marks.len()).then(|| marks[idx])
        } else {
            active_phase2_impl(&self.bot, state, opp_states, dice, has_marked)
        };

        let chosen = match mark {
            Some(m) => {
                let mut s = *state;
                s.apply_mark(m);
                s
            }
            None => no_mark_state,
        };
        self.recorded.borrow_mut().push((chosen, opp_states.to_vec()));
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
        let mark = passive_phase1_impl(&self.bot, state, opp_states, dice);
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

/// Play one 1v1 plain self-play game; both players are recording aznet bots
/// (player 0 explores with ε, player 1 greedy). Returns all samples (both
/// players' trajectories, swap-doubled) plus player 0's final score.
fn play_training_game(
    model: &AzModel<MyBackend>,
    device: &burn::backend::ndarray::NdArrayDevice,
    epsilon: f32,
    seed: u64,
) -> (Vec<AzSample>, f32) {
    use crate::game::{Game, Player};

    let mut buffers = Vec::with_capacity(2);
    let mut players: Vec<Player> = Vec::with_capacity(2);
    for i in 0..2usize {
        let bot = AzStrategy::from_model(model.clone(), device.clone());
        let buf = Rc::new(RefCell::new(Vec::new()));
        buffers.push(Rc::clone(&buf));
        players.push(Player::new(
            Box::new(RecordingAz {
                bot,
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
        let opp_final = finals[(i + 1) % 2];
        all_samples.extend(build_az_samples(model, device, &snapshots, finals[i], opp_final));
    }
    (all_samples, finals[0])
}

/// Same fixed-game-set paired benchmark as the pair bot, for aznet.
const BENCH_SEED: u64 = 0xB54C;

fn benchmark_vs_ga(artifact_dir: &str, champion: &DNA, num_games: usize) -> f64 {
    use crate::game::{Game, Player};

    let wins: u32 = (0..num_games)
        .into_par_iter()
        .map_init(
            || AzStrategy::load(artifact_dir),
            |template, i| {
                let bot = AzStrategy::from_shared(template.model.clone(), template.device.clone());
                let pair = (i / 2) as u64;
                let rotation = i % 2;
                let seat_dice =
                    |seat: u64| Box::new(SmallRng::seed_from_u64(BENCH_SEED.wrapping_add(pair * 2 + seat)));
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
                (scores[idx] > scores[1 - idx]) as u32
            },
        )
        .sum();
    wins as f64 / num_games as f64
}

fn train_with_epochs(samples: Vec<AzSample>, artifact_dir: &str, num_epochs: usize, lr: f64) -> AzModel<MyBackend> {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;

    let split = (samples.len() * 9) / 10;
    let mut samples = samples;
    let valid = samples.split_off(split);
    let train_data = InMemDataset::new(samples);
    let valid_data = InMemDataset::new(valid);

    let model: AzModel<MyAutodiffBackend> = AzModelConfig::new()
        .init::<MyAutodiffBackend>(&device)
        .load_file(format!("{artifact_dir}/model"), &CompactRecorder::new(), &device)
        .unwrap_or_else(|_| AzModelConfig::new().init::<MyAutodiffBackend>(&device));

    let batcher_train = AzBatcher::<MyAutodiffBackend> { _phantom: std::marker::PhantomData };
    let batcher_valid = AzBatcher::<MyBackend> { _phantom: std::marker::PhantomData };

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
        .expect("Failed to save aznet model");
    std::fs::remove_dir_all(&ckpt_dir).ok();

    AzModelConfig::new()
        .init::<MyBackend>(&device)
        .load_file(format!("{artifact_dir}/model"), &CompactRecorder::new(), &device)
        .expect("Failed to reload aznet model for inference")
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
    std::fs::create_dir_all(artifact_dir).ok();
    let buffer_iterations = 3;
    let mut replay_buffer: std::collections::VecDeque<Vec<AzSample>> = std::collections::VecDeque::new();

    let scores_log_path = format!("{artifact_dir}/training_scores.csv");
    if start_iteration == 0 || !std::path::Path::new(&scores_log_path).exists() {
        std::fs::write(&scores_log_path, "iteration,avg_score,winrate\n").ok();
    }

    let genes = Arc::new(bot::default_genes());
    let champion = DNA::load_weights("champion.txt", genes).expect("No champion.txt");
    let start_time = std::time::Instant::now();

    let mut model: AzModel<MyBackend> = AzModelConfig::new()
        .init::<MyBackend>(&device)
        .load_file(format!("{artifact_dir}/model"), &CompactRecorder::new(), &device)
        .unwrap_or_else(|_| {
            println!("  No pretrained aznet model, starting fresh");
            AzModelConfig::new().init::<MyBackend>(&device)
        });

    for iteration in 0..num_iterations {
        let global_iter = start_iteration + iteration;
        let epsilon = (0.2 * (0.95f32).powi(global_iter as i32)).max(0.07);
        println!("\n=== Aznet iteration {} (epsilon={epsilon:.3}) ===", global_iter + 1);

        let models: Vec<AzModel<MyBackend>> = (0..games_per_iteration).map(|_| model.clone()).collect();
        let game_results: Vec<(Vec<AzSample>, f32)> = (0..games_per_iteration)
            .into_par_iter()
            .zip(models.into_par_iter())
            .map(|(game_idx, thread_model)| {
                let seed = TRAIN_SEED.wrapping_add((iteration * games_per_iteration + game_idx) as u64);
                play_training_game(&thread_model, &device, epsilon, seed)
            })
            .collect();

        let game_scores: Vec<f32> = game_results.iter().map(|(_, s)| *s).collect();
        let new_samples: Vec<AzSample> = game_results.into_iter().flat_map(|(s, _)| s).collect();
        let avg_score = if game_scores.is_empty() {
            0.0
        } else {
            game_scores.iter().sum::<f32>() / game_scores.len() as f32
        };

        replay_buffer.push_back(new_samples);
        if replay_buffer.len() > buffer_iterations {
            replay_buffer.pop_front();
        }
        let all_samples: Vec<AzSample> = replay_buffer.iter().flatten().copied().collect();
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
        use std::io::Write;
        if bench_games > 0 {
            let winrate = benchmark_vs_ga(artifact_dir, &champion, bench_games);
            println!(
                "  Iteration {:>3}: avg score {:.1}, winrate {:.1}%, elapsed {}m{}s",
                global_iter + 1,
                avg_score,
                winrate * 100.0,
                elapsed / 60,
                elapsed % 60,
            );
            if let Ok(mut f) = std::fs::OpenOptions::new().append(true).create(true).open(&scores_log_path) {
                writeln!(f, "{},{avg_score:.2},{:.2}", global_iter + 1, winrate * 100.0).ok();
            }
        } else {
            println!("  Iteration {:>3}: avg score {:.1}, elapsed {}m{}s", global_iter + 1, avg_score, elapsed / 60, elapsed % 60);
            if let Ok(mut f) = std::fs::OpenOptions::new().append(true).create(true).open(&scores_log_path) {
                writeln!(f, "{},{avg_score:.2}", global_iter + 1).ok();
            }
        }
    }

    println!("\nAznet self-play training complete. Model saved to {artifact_dir}/model");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permute_rows_swaps_both_boards_and_is_involution() {
        let mut f = [0.0f32; AZ_FEATURES];
        for (i, v) in f.iter_mut().enumerate() {
            *v = i as f32;
        }
        let orig = f;

        // red <-> yellow: blocks 0 and 1 in both boards.
        permute_rows(&mut f, true, false, false);
        for board in [0usize, BOARD_RAW] {
            for k in 0..ROW_BLOCK {
                assert_eq!(f[board + k], orig[board + ROW_BLOCK + k]);
                assert_eq!(f[board + ROW_BLOCK + k], orig[board + k]);
            }
            // green/blue untouched.
            for k in 0..ROW_BLOCK {
                assert_eq!(f[board + 2 * ROW_BLOCK + k], orig[board + 2 * ROW_BLOCK + k]);
            }
        }
        // cdiff untouched.
        assert_eq!(f[2 * BOARD_RAW], orig[2 * BOARD_RAW]);
        // involution.
        permute_rows(&mut f, true, false, false);
        assert_eq!(f, orig);

        // pair swap: {0<->2, 1<->3}, involution.
        permute_rows(&mut f, false, false, true);
        for board in [0usize, BOARD_RAW] {
            for k in 0..ROW_BLOCK {
                assert_eq!(f[board + k], orig[board + 2 * ROW_BLOCK + k]);
                assert_eq!(f[board + ROW_BLOCK + k], orig[board + 3 * ROW_BLOCK + k]);
            }
        }
        permute_rows(&mut f, false, false, true);
        assert_eq!(f, orig);

        // green <-> blue: blocks 2 and 3 in both boards, involution.
        f = orig;
        permute_rows(&mut f, false, true, false);
        for board in [0usize, BOARD_RAW] {
            for k in 0..ROW_BLOCK {
                assert_eq!(f[board + 2 * ROW_BLOCK + k], orig[board + 3 * ROW_BLOCK + k]);
                assert_eq!(f[board + 3 * ROW_BLOCK + k], orig[board + 2 * ROW_BLOCK + k]);
            }
            // red/yellow untouched.
            for k in 0..ROW_BLOCK {
                assert_eq!(f[board + k], orig[board + k]);
                assert_eq!(f[board + ROW_BLOCK + k], orig[board + ROW_BLOCK + k]);
            }
        }
        // strike one-hots untouched by an actual permutation.
        assert_eq!(f[112..116], orig[112..116]);
        assert_eq!(f[228..232], orig[228..232]);
        permute_rows(&mut f, false, true, false);
        assert_eq!(f, orig);
    }

    #[test]
    fn build_az_samples_emits_negated_swapped_pairs() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let model = AzModelConfig::new().init::<MyBackend>(&device);

        // 2-step 1v1 trajectory.
        let mut our1 = State::default();
        our1.apply_mark(Mark { row: 0, number: 5 });
        let opp1 = State::default();
        let mut our2 = our1;
        our2.apply_mark(Mark { row: 0, number: 7 });
        let mut opp2 = State::default();
        opp2.apply_mark(Mark { row: 2, number: 10 });

        let snapshots = vec![(our1, vec![opp1]), (our2, vec![opp2])];
        let samples = build_az_samples(&model, &device, &snapshots, 30.0, 20.0);

        // 2 steps × 2 orders.
        assert_eq!(samples.len(), 4);
        for pair in samples.chunks(2) {
            let (fwd, swp) = (&pair[0], &pair[1]);
            assert_eq!(swp.value, -fwd.value);
            assert_eq!(swp.final_diff, -fwd.final_diff);
            assert_eq!(fwd.our, swp.opp, "swap exchanges the two boards");
            assert_eq!(fwd.opp, swp.our, "swap exchanges the two boards");
        }
        // Last forward sample: G_{n-1} = final_diff = 10; cdiff at t=1:
        // our 3 pts (marks 5,7 -> 2 marks = 3) − opp 1 pt (1 mark) = 2.
        // value = final_diff − cdiff = 10 − 2 = 8.
        let last_fwd = &samples[2];
        assert!((last_fwd.value - 8.0).abs() < 1e-5);
        assert!((last_fwd.final_diff - 8.0).abs() < 1e-5);
    }

    #[test]
    fn play_training_game_is_deterministic() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let model = AzModelConfig::new().init::<MyBackend>(&device);

        let run = || play_training_game(&model, &device, 0.1, 4242);
        let (s1, f1) = run();
        let (s2, f2) = run();
        assert_eq!(f1, f2);
        assert_eq!(s1.len(), s2.len());
        for (a, b) in s1.iter().zip(&s2) {
            assert_eq!(a.our, b.our);
            assert_eq!(a.opp, b.opp);
            assert_eq!(a.value, b.value);
            assert_eq!(a.final_diff, b.final_diff);
        }
        assert!(!s1.is_empty(), "a full game must produce samples");
    }

    #[test]
    fn self_play_train_smoke_writes_model() {
        let dir = std::env::temp_dir().join(format!("aznet_smoke_{}", std::process::id()));
        let dir_s = dir.to_str().unwrap();
        std::fs::remove_dir_all(&dir).ok();

        // 1 iteration, 6 games, 1 epoch, no bench. Must complete and write a
        // loadable model.
        self_play_train(dir_s, 1, 6, 1, 0, false, 0);
        assert!(dir.join("model.mpk").exists(), "model artifact written");

        // The written model loads and evaluates.
        let bot = AzStrategy::load(dir_s);
        let v = crate::strategy::Bot::evaluate(&bot, &State::default(), &[State::default()]);
        assert!(v.is_finite());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn batcher_expands_states_to_inputs_deterministically() {
        use burn::data::dataloader::batcher::Batcher;
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let batcher = AzBatcher::<MyBackend> { _phantom: std::marker::PhantomData };

        let mut a = State::default();
        a.apply_mark(Mark { row: 0, number: 5 });
        let b = State::default();
        let items = vec![
            AzSample { our: a, opp: b, value: 1.0, final_diff: 2.0 },
            AzSample { our: b, opp: a, value: -1.0, final_diff: -2.0 },
        ];

        let batch1 = batcher.batch(items.clone(), &device);
        let batch2 = batcher.batch(items.clone(), &device);
        // Shape: [2, AZ_FEATURES]. (`.shape().dims` is the repo idiom — see AzModel::forward.)
        assert_eq!(batch1.inputs.shape().dims, [2, AZ_FEATURES]);
        // Deterministic (seed derived from value.to_bits + batch_size).
        let v1 = batch1.inputs.into_data().to_vec::<f32>().unwrap();
        let v2 = batch2.inputs.into_data().to_vec::<f32>().unwrap();
        assert_eq!(v1, v2);
        // targets / final_diffs carried through.
        assert_eq!(batch1.targets.into_data().to_vec::<f32>().unwrap(), vec![1.0, -1.0]);
        assert_eq!(batch1.final_diffs.into_data().to_vec::<f32>().unwrap(), vec![2.0, -2.0]);
    }
}
