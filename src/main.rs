use clap::{Parser, Subcommand, ValueEnum};
use game::Player;
use qwixxer::*;

/// Pools freed memory instead of returning it to the OS: the search bot's
/// rollout driver and burn's per-forward buffers otherwise cause constant
/// cross-thread munmap/TLB-shootdown churn (~17% of cycles in
/// smp_call_function_many_cond before this).
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::sync::Arc;

#[derive(Debug, Clone, ValueEnum)]
enum BotType {
    Ga,
    Dqn,
    Pair,
    PairSearch,
    Mcts,
    Opportunist,
    Conservative,
    Random,
}

impl std::fmt::Display for BotType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            BotType::Ga => write!(f, "GA"),
            BotType::Dqn => write!(f, "DQN"),
            BotType::Pair => write!(f, "PAIR"),
            BotType::PairSearch => write!(f, "PAIR-SEARCH"),
            BotType::Mcts => write!(f, "MCTS"),
            BotType::Opportunist => write!(f, "Opportunist"),
            BotType::Conservative => write!(f, "Conservative"),
            BotType::Random => write!(f, "Random"),
        }
    }
}

fn make_strategy(bot: &BotType) -> Box<dyn strategy::Strategy> {
    let genes = Arc::new(bot::default_genes());
    match bot {
        BotType::Ga => {
            let champion =
                bot::DNA::load_weights("champion.txt", genes).expect("No champion.txt found. Run `train ga` first.");
            Box::new(champion)
        }
        BotType::Dqn => Box::new(dqn::DqnStrategy::load("dqn_model")),
        BotType::Pair => Box::new(dqn::pair::PairStrategy::load("pair_model")),
        BotType::PairSearch => Box::new(strategy::search::SearchBot::new(dqn::pair::PairStrategy::load(
            "pair_model",
        ))),
        BotType::Mcts => {
            let champion =
                bot::DNA::load_weights("champion.txt", genes).expect("No champion.txt found. Run `train ga` first.");
            Box::new(mcts::MonteCarlo::with_ga(200, champion))
        }
        BotType::Opportunist => Box::<strategy::Opportunist>::default(),
        BotType::Conservative => Box::<strategy::Conservative>::default(),
        BotType::Random => Box::new(strategy::Random),
    }
}

struct StrategyTemplates {
    dqn: Option<dqn::DqnStrategy>,
    pair: Option<dqn::pair::PairStrategy>,
    champion: Option<bot::DNA>,
}

impl StrategyTemplates {
    fn new(bots: &[BotType]) -> Self {
        let needs_dqn = bots.iter().any(|b| matches!(b, BotType::Dqn));
        let needs_pair = bots.iter().any(|b| matches!(b, BotType::Pair | BotType::PairSearch));
        let needs_champion = bots.iter().any(|b| matches!(b, BotType::Ga | BotType::Mcts));
        let genes = Arc::new(bot::default_genes());
        StrategyTemplates {
            dqn: if needs_dqn {
                Some(dqn::DqnStrategy::load("dqn_model"))
            } else {
                None
            },
            pair: if needs_pair {
                Some(dqn::pair::PairStrategy::load("pair_model"))
            } else {
                None
            },
            champion: if needs_champion {
                Some(bot::DNA::load_weights("champion.txt", genes).expect("No champion.txt found. Run `evolve` first."))
            } else {
                None
            },
        }
    }

    fn create(&self, bot: &BotType) -> Box<dyn strategy::Strategy> {
        match bot {
            BotType::Ga => Box::new(self.champion.as_ref().unwrap().clone()),
            BotType::Dqn => {
                let t = self.dqn.as_ref().unwrap();
                Box::new(dqn::DqnStrategy::from_shared(t.model.clone(), t.device.clone()))
            }
            BotType::Pair => {
                let t = self.pair.as_ref().unwrap();
                Box::new(dqn::pair::PairStrategy::from_shared(t.model.clone(), t.device.clone()))
            }
            BotType::PairSearch => {
                let t = self.pair.as_ref().unwrap();
                Box::new(strategy::search::SearchBot::new(dqn::pair::PairStrategy::from_shared(
                    t.model.clone(),
                    t.device.clone(),
                )))
            }
            BotType::Mcts => Box::new(mcts::MonteCarlo::with_ga(200, self.champion.as_ref().unwrap().clone())),
            BotType::Opportunist => Box::<strategy::Opportunist>::default(),
            BotType::Conservative => Box::<strategy::Conservative>::default(),
            BotType::Random => Box::new(strategy::Random),
        }
    }
}

#[allow(dead_code)]
fn bot_name(bot: &BotType, index: usize, total: usize) -> String {
    if total > 1 {
        format!("{} #{}", bot, index + 1)
    } else {
        bot.to_string()
    }
}

#[derive(Parser)]
#[command(name = "qwixxer", about = "Qwixx bot framework")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Play interactively against bots
    Play {
        /// Bot types to play against
        #[arg(default_value = "mcts")]
        bots: Vec<BotType>,
        /// Show bot decisions and boards
        #[arg(short, long)]
        verbose: bool,
    },
    /// Benchmark bots against each other
    Bench {
        /// Bot types (2+)
        bots: Vec<BotType>,
        /// Number of games
        #[arg(short, long, default_value = "1000")]
        num_games: usize,
        /// Base seed for paired dice streams (same seed = identical game set)
        #[arg(short, long, default_value = "42")]
        seed: u64,
    },
    /// Single-player score benchmark
    Solo {
        /// Number of games per strategy
        #[arg(short, long, default_value = "10000")]
        num_games: usize,
    },
    /// Evolve the GA champion via genetic algorithm
    Evolve,
    /// DQN self-play reinforcement learning
    #[cfg(feature = "dqn")]
    DqnSelfplay {
        /// Number of iterations
        #[arg(short, long, default_value = "40")]
        iterations: usize,
        /// Benchmark games per iteration (0 to disable)
        #[arg(short, long, default_value = "0")]
        bench: usize,
        /// Save per-iteration checkpoints as iter-N.mpk
        #[arg(short, long)]
        checkpoints: bool,
        /// Starting iteration offset (for epsilon schedule when resuming)
        #[arg(short, long, default_value = "0")]
        start_iteration: usize,
    },
    /// Pair-network self-play reinforcement learning
    #[cfg(feature = "dqn")]
    PairTrain {
        /// Number of iterations
        #[arg(short, long, default_value = "40")]
        iterations: usize,
        /// Games per iteration
        #[arg(short, long, default_value = "20000")]
        games: usize,
        /// Training epochs per iteration over the replay buffer
        #[arg(short, long, default_value = "10")]
        epochs: usize,
        /// Benchmark games per iteration (0 to disable)
        #[arg(short, long, default_value = "0")]
        bench: usize,
        /// Save per-iteration checkpoints as iter-N.mpk
        #[arg(short, long)]
        checkpoints: bool,
        /// Starting iteration offset (for epsilon schedule when resuming)
        #[arg(long, default_value = "0")]
        start_iteration: usize,
        /// Generate games with the search bot as every player's policy
        /// (expert iteration)
        #[arg(long)]
        search: bool,
    },
}

fn run_play(bots: Vec<BotType>, verbose: bool) {
    let bot_names: Vec<String> = bots.iter().map(|b| b.to_string()).collect();
    println!("Playing against: {}\n", bot_names.join(", "));

    let mut players: Vec<Player> = bots
        .iter()
        .map(|b| Player::new(make_strategy(b), Box::new(SmallRng::from_entropy())))
        .collect();
    players.push(Player::new(
        Box::<strategy::Interactive>::default(),
        Box::new(SmallRng::from_entropy()),
    ));

    let mut game = game::Game::new(players);
    game.verbose = verbose;
    game.play();
    game.print_game_over();
}

fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Dice-stream seed for one seat of one game pair. Distinct `(pair, seat)`
/// inputs map to distinct values within a run (seats < 8); hashing `base`
/// keeps the stream sets of nearby base seeds disjoint, so different `--seed`
/// runs are independent samples. `seed_from_u64` expands the result into a
/// decorrelated stream state.
fn seat_dice_seed(base: u64, pair: usize, seat: usize) -> u64 {
    splitmix64(base).wrapping_add(pair as u64 * 8 + seat as u64)
}

fn run_bench(bots: Vec<BotType>, num_games: usize, seed: u64) {
    if bots.len() < 2 {
        eprintln!("Need at least 2 bots to benchmark");
        return;
    }

    let num_players = bots.len();
    // Games come in pairs of num_players rotations sharing per-seat dice
    // streams, so round up to complete the final pair.
    let num_games = num_games.div_ceil(num_players) * num_players;
    println!(
        "Benchmarking {} ({num_games} games, rotating seats, paired dice, seed {seed}):\n",
        bots.iter().map(|b| b.to_string()).collect::<Vec<_>>().join(" vs ")
    );

    #[cfg(feature = "parallel")]
    #[cfg(feature = "parallel")]
    use rayon::prelude::*;

    let bench_game = |templates: &StrategyTemplates, i: usize| {
        let pair = i / num_players;
        let rotation = i % num_players;
        let players: Vec<Player> = (0..num_players)
            .map(|j| {
                let bot_idx = (j + num_players - rotation) % num_players;
                Player::new(
                    templates.create(&bots[bot_idx]),
                    Box::new(SmallRng::seed_from_u64(seat_dice_seed(seed, pair, j))),
                )
            })
            .collect();

        let mut game = game::Game::new(players);
        game.play();

        let scores: Vec<isize> = game.players.iter().map(|p| p.state.count_points()).collect();
        let max = *scores.iter().max().unwrap();
        let num_winners = scores.iter().filter(|&&s| s == max).count();

        let per_bot: Vec<(usize, isize)> = (0..num_players)
            .map(|j| {
                let bot_idx = (j + num_players - rotation) % num_players;
                (bot_idx, scores[j])
            })
            .collect();

        let is_tie = num_winners > 1;
        (per_bot, is_tie)
    };

    #[cfg(feature = "parallel")]
    let results: Vec<(Vec<(usize, isize)>, bool)> = (0..num_games)
        .into_par_iter()
        .map_init(|| StrategyTemplates::new(&bots), |t, i| bench_game(t, i))
        .collect();

    #[cfg(not(feature = "parallel"))]
    let results: Vec<(Vec<(usize, isize)>, bool)> = {
        let templates = StrategyTemplates::new(&bots);
        (0..num_games).map(|i| bench_game(&templates, i)).collect()
    };

    // Aggregate
    let mut wins = vec![0u32; num_players];
    let mut total_pts = vec![0i64; num_players];
    let mut ties = 0u32;

    for (per_bot, is_tie) in &results {
        let max = per_bot.iter().map(|(_, s)| *s).max().unwrap();
        let num_winners = per_bot.iter().filter(|(_, s)| *s == max).count();
        for &(bot_idx, score) in per_bot {
            total_pts[bot_idx] += score as i64;
            if score == max && num_winners == 1 {
                wins[bot_idx] += 1;
            }
        }
        if *is_tie {
            ties += 1;
        }
    }

    // Per-player stats
    for (i, bot) in bots.iter().enumerate() {
        println!(
            "  {:<16} {:>5} wins ({:>4.1}%)  avg {:.1} pts",
            format!("{} #{}", bot, i + 1),
            wins[i],
            wins[i] as f64 / num_games as f64 * 100.0,
            total_pts[i] as f64 / num_games as f64
        );
    }
    if ties > 0 {
        println!(
            "  {:<16} {:>5}       ({:>4.1}%)",
            "Ties",
            ties,
            ties as f64 / num_games as f64 * 100.0
        );
    }

    // Aggregate stats per strategy (when multiple bots share a strategy)
    let unique_strategies: Vec<String> = bots
        .iter()
        .map(|b| b.to_string())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    if unique_strategies.len() < num_players {
        println!("\n  By strategy:");

        // For each game, determine which STRATEGY won.
        // A tie between bots of the same strategy counts as a win for that strategy.
        let mut strat_wins: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        let mut strat_pts: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
        let mut strat_count: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        let mut strat_ties = 0u32;

        for bot in &bots {
            let name = bot.to_string();
            strat_count.entry(name.clone()).or_insert(0);
            strat_wins.entry(name.clone()).or_insert(0);
            strat_pts.entry(name.clone()).or_insert(0);
        }
        for (i, bot) in bots.iter().enumerate() {
            let name = bot.to_string();
            *strat_count.entry(name).or_insert(0) += 1;
            let _ = i; // count only
        }

        for (per_bot, _) in &results {
            let max = per_bot.iter().map(|(_, s)| *s).max().unwrap();

            // Accumulate points per strategy
            for &(bot_idx, score) in per_bot {
                let name = bots[bot_idx].to_string();
                *strat_pts.entry(name).or_insert(0) += score as i64;
            }

            // Find which strategies have the max score
            let winning_strategies: std::collections::BTreeSet<String> = per_bot
                .iter()
                .filter(|(_, s)| *s == max)
                .map(|(idx, _)| bots[*idx].to_string())
                .collect();

            if winning_strategies.len() == 1 {
                let name = winning_strategies.into_iter().next().unwrap();
                *strat_wins.entry(name).or_insert(0) += 1;
            } else {
                strat_ties += 1;
            }
        }

        for strat in &unique_strategies {
            let count = *strat_count.get(strat).unwrap() as f64;
            let sw = *strat_wins.get(strat).unwrap_or(&0);
            let sp = *strat_pts.get(strat).unwrap_or(&0);
            println!(
                "  {:<16} {:>5} wins ({:>4.1}%)  avg {:.1} pts",
                strat,
                sw,
                sw as f64 / num_games as f64 * 100.0,
                sp as f64 / (num_games as f64 * count),
            );
        }
        if strat_ties > 0 {
            println!(
                "  {:<16} {:>5}       ({:>4.1}%)",
                "Ties",
                strat_ties,
                strat_ties as f64 / num_games as f64 * 100.0
            );
        }
    }

    // 99% CI on winrate for 2-strategy matchups (covers plain 1v1 and
    // aggregated NvN). Computed over pair means: the games of one pair share
    // per-seat dice streams, so their outcomes are correlated and the pair is
    // the independent sampling unit. This both keeps the CI honest and
    // captures the variance reduction from pairing.
    if unique_strategies.len() == 2 {
        // Winning strategy per game (None = tie between the two strategies).
        let game_winners: Vec<Option<String>> = results
            .iter()
            .map(|(per_bot, _)| {
                let max = per_bot.iter().map(|(_, s)| *s).max().unwrap();
                let winners: std::collections::BTreeSet<String> = per_bot
                    .iter()
                    .filter(|(_, s)| *s == max)
                    .map(|(idx, _)| bots[*idx].to_string())
                    .collect();
                if winners.len() == 1 {
                    winners.into_iter().next()
                } else {
                    None
                }
            })
            .collect();

        let count_wins = |name: &str| game_winners.iter().filter(|w| w.as_deref() == Some(name)).count();
        let (s0, s1) = (&unique_strategies[0], &unique_strategies[1]);
        let leader = if count_wins(s0) >= count_wins(s1) { s0 } else { s1 };

        let pair_means: Vec<f64> = game_winners
            .chunks(num_players)
            .map(|chunk| {
                chunk.iter().filter(|w| w.as_deref() == Some(leader.as_str())).count() as f64 / chunk.len() as f64
            })
            .collect();
        let num_pairs = pair_means.len() as f64;
        let p = pair_means.iter().sum::<f64>() / num_pairs;
        let var = pair_means.iter().map(|a| (a - p).powi(2)).sum::<f64>() / (num_pairs - 1.0);
        let se_paired = (var / num_pairs).sqrt();
        let se_naive = (p * (1.0 - p) / num_games as f64).sqrt();
        let z = 2.576;
        println!(
            "\n  99% CI (paired): {} wins {:.2}% - {:.2}%",
            leader,
            (p - z * se_paired) * 100.0,
            (p + z * se_paired) * 100.0
        );
        if se_paired > 0.0 {
            println!(
                "  Pairing efficiency: {:.2}x (naive SE {:.3}%, paired SE {:.3}%)",
                (se_naive / se_paired).powi(2),
                se_naive * 100.0,
                se_paired * 100.0
            );
        }
    }
}

fn run_solo(num_games: usize) {
    let genes = Arc::new(bot::default_genes());
    let champion = bot::DNA::load_weights("champion.txt", genes).ok();

    let all_bots: Vec<(BotType, bool)> = vec![
        (BotType::Random, true),
        (BotType::Conservative, true),
        (BotType::Opportunist, true),
        (BotType::Ga, champion.is_some()),
        (BotType::Dqn, std::path::Path::new("dqn_model/model.mpk").exists()),
    ];

    println!("Single-player scores over {num_games} games:\n");
    for (bot, available) in &all_bots {
        if !available {
            continue;
        }
        let mut total = 0i64;
        let mut min = i64::MAX;
        let mut max = i64::MIN;
        for _ in 0..num_games {
            let mut game = game::Game::new(vec![Player::new(
                make_strategy(bot),
                Box::new(SmallRng::from_entropy()),
            )]);
            game.play();
            let pts = game.players[0].state.count_points() as i64;
            total += pts;
            min = min.min(pts);
            max = max.max(pts);
        }
        let avg = total as f64 / num_games as f64;
        println!("  {:<16} avg {avg:>5.1}  min {min:>4}  max {max:>4}", bot.to_string());
    }
}

fn run_train() {
    println!("Training GA bot (population=100, 200 generations)...\n");
    let mut pop = bot::Population::new(100, bot::default_genes(), 42);
    pop.evolve(200);

    let _champion = pop.current_champion().clone();
    println!("\nBenchmarking champion vs Opportunist...\n");
    run_bench(vec![BotType::Ga, BotType::Opportunist], 100_000, 42);
}

#[cfg(feature = "dqn")]
fn run_dqn_selfplay(iterations: usize, bench_games: usize, checkpoints: bool, start_iteration: usize) {
    dqn::train::self_play_train(
        "dqn_model",
        iterations,
        20000,
        10,
        bench_games,
        checkpoints,
        start_iteration,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seat_dice_seeds_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for pair in 0..1000 {
            for seat in 0..5 {
                assert!(seen.insert(seat_dice_seed(42, pair, seat)));
            }
        }
    }

    #[test]
    fn seeded_games_are_deterministic() {
        let play = || {
            let players = vec![
                Player::new(
                    Box::<strategy::Opportunist>::default(),
                    Box::new(SmallRng::seed_from_u64(seat_dice_seed(42, 7, 0))),
                ),
                Player::new(
                    Box::<strategy::Conservative>::default(),
                    Box::new(SmallRng::seed_from_u64(seat_dice_seed(42, 7, 1))),
                ),
            ];
            let mut game = game::Game::new(players);
            game.play();
            game.players.iter().map(|p| p.state.count_points()).collect::<Vec<_>>()
        };
        assert_eq!(play(), play());
    }
}

fn main() {
    // Our matmuls are tiny (≤ a few hundred rows of a 45→128→64 MLP) and run
    // inside rayon-parallel games; matrixmultiply's own thread pool on top is
    // pure scheduler churn (measured 2x slowdown on search benches). Respect
    // an explicit user override.
    if std::env::var_os("MATMUL_NUM_THREADS").is_none() {
        std::env::set_var("MATMUL_NUM_THREADS", "1");
    }

    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Play { bots, verbose }) => run_play(bots, verbose),
        Some(Commands::Bench { bots, num_games, seed }) => run_bench(bots, num_games, seed),
        Some(Commands::Solo { num_games }) => run_solo(num_games),
        Some(Commands::Evolve) => run_train(),
        #[cfg(feature = "dqn")]
        #[cfg(feature = "dqn")]
        Some(Commands::DqnSelfplay {
            iterations,
            bench,
            checkpoints,
            start_iteration,
        }) => run_dqn_selfplay(iterations, bench, checkpoints, start_iteration),
        #[cfg(feature = "dqn")]
        Some(Commands::PairTrain {
            iterations,
            games,
            epochs,
            bench,
            checkpoints,
            start_iteration,
            search,
        }) => dqn::pair_train::self_play_train(
            "pair_model",
            iterations,
            games,
            epochs,
            bench,
            checkpoints,
            start_iteration,
            search,
        ),
        None => {
            // Default: play against MCTS
            run_play(vec![BotType::Mcts], false);
        }
    }
}
