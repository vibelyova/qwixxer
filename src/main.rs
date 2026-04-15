use clap::{Parser, Subcommand, ValueEnum};
use qwixxer::*;
use game::Player;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::sync::Arc;

#[derive(Debug, Clone, ValueEnum)]
enum BotType {
    Ga,
    #[cfg(feature = "burn")]
    Dqn,
    Mcts,
    Opportunist,
    Conservative,
    Rusher,
    Random,
    BlankSb,
    BlankRtl,
}

impl std::fmt::Display for BotType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            BotType::Ga => write!(f, "GA"),
            #[cfg(feature = "burn")]
            BotType::Dqn => write!(f, "DQN"),
            BotType::Mcts => write!(f, "MCTS"),
            BotType::Opportunist => write!(f, "Opportunist"),
            BotType::Conservative => write!(f, "Conservative"),
            BotType::Rusher => write!(f, "Rusher"),
            BotType::Random => write!(f, "Random"),
            BotType::BlankSb => write!(f, "Blank S.B."),
            BotType::BlankRtl => write!(f, "Blank R.T.L."),
        }
    }
}

fn make_strategy(bot: &BotType) -> Box<dyn strategy::Strategy> {
    let genes = Arc::new(bot::default_genes());
    match bot {
        BotType::Ga => {
            let champion = bot::DNA::load_weights("champion.txt", genes)
                .expect("No champion.txt found. Run `train ga` first.");
            Box::new(champion)
        }
        #[cfg(feature = "burn")]
        BotType::Dqn => Box::new(dqn::DqnStrategy::load("dqn_model")),
        BotType::Mcts => {
            let champion = bot::DNA::load_weights("champion.txt", genes)
                .expect("No champion.txt found. Run `train ga` first.");
            Box::new(mcts::MonteCarlo::with_ga(500, champion))
        }
        BotType::Opportunist => Box::<strategy::Opportunist>::default(),
        BotType::Conservative => Box::<strategy::Conservative>::default(),
        BotType::Rusher => Box::new(strategy::Rusher),
        BotType::Random => Box::new(strategy::Random),
        BotType::BlankSb => Box::new(blank::BlankScoreBased),
        BotType::BlankRtl => Box::new(race_to_lock::BlankRaceToLock::new()),
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
    },
    /// Single-player score benchmark
    Solo {
        /// Number of games per strategy
        #[arg(short, long, default_value = "10000")]
        num_games: usize,
    },
    /// Evolve the GA champion via genetic algorithm
    Evolve,
    /// Train DQN (MC-supervised)
    #[cfg(feature = "dqn")]
    DqnTrain,
    /// DQN self-play reinforcement learning
    #[cfg(feature = "dqn")]
    DqnSelfplay,
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

fn run_bench(bots: Vec<BotType>, num_games: usize) {
    if bots.len() < 2 {
        eprintln!("Need at least 2 bots to benchmark");
        return;
    }

    let num_players = bots.len();
    println!(
        "Benchmarking {} ({num_games} games, rotating seats):\n",
        bots.iter().map(|b| b.to_string()).collect::<Vec<_>>().join(" vs ")
    );

    #[cfg(feature = "parallel")]
    use rayon::prelude::*;

    // Run games in parallel when available
    #[cfg(feature = "parallel")]
    let iter = (0..num_games).into_par_iter();
    #[cfg(not(feature = "parallel"))]
    let iter = 0..num_games;

    let results: Vec<(Vec<(usize, isize)>, bool)> = iter
        .map(|i| {
            let rotation = i % num_players;
            let players: Vec<Player> = (0..num_players)
                .map(|j| {
                    let bot_idx = (j + num_players - rotation) % num_players;
                    Player::new(make_strategy(&bots[bot_idx]), Box::new(SmallRng::from_entropy()))
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
        })
        .collect();

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

    for (i, bot) in bots.iter().enumerate() {
        println!(
            "  {:<16} {:>5} wins ({:>4.1}%)  avg {:.1} pts",
            bot.to_string(),
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

    // For 1v1: show 99% confidence interval for the leading bot's win rate
    if num_players == 2 {
        let leader = if wins[0] >= wins[1] { 0 } else { 1 };
        let n = num_games as f64;
        let p = wins[leader] as f64 / n;
        let se = (p * (1.0 - p) / n).sqrt();
        let z = 2.576; // 99% confidence
        let moe = z * se;
        println!(
            "\n  99% CI: {} wins {:.2}% - {:.2}%",
            bots[leader],
            (p - moe) * 100.0,
            (p + moe) * 100.0
        );
    }
}

fn run_solo(num_games: usize) {
    let genes = Arc::new(bot::default_genes());
    let champion = bot::DNA::load_weights("champion.txt", genes).ok();

    let all_bots: Vec<(BotType, bool)> = vec![
        (BotType::Random, true),
        (BotType::Conservative, true),
        (BotType::Rusher, true),
        (BotType::Opportunist, true),
        (BotType::Ga, champion.is_some()),
        (BotType::BlankSb, true),
        (BotType::BlankRtl, true),
    ];

    #[cfg(feature = "burn")]
    let all_bots = {
        let mut v = all_bots;
        v.push((BotType::Dqn, std::path::Path::new("dqn_model/model.mpk").exists()));
        v
    };

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
    run_bench(vec![BotType::Ga, BotType::Opportunist], 10_000);
}

#[cfg(feature = "dqn")]
fn run_dqn_train() {
    let samples = dqn::generate_training_data(1500, 500);
    dqn::train(samples, "dqn_model");
}

#[cfg(feature = "dqn")]
fn run_dqn_selfplay() {
    std::env::set_var("OPENBLAS_NUM_THREADS", "1");
    dqn::self_play_train("dqn_model", 40, 20000, 10);
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Play { bots, verbose }) => run_play(bots, verbose),
        Some(Commands::Bench { bots, num_games }) => run_bench(bots, num_games),
        Some(Commands::Solo { num_games }) => run_solo(num_games),
        Some(Commands::Evolve) => run_train(),
        #[cfg(feature = "dqn")]
        Some(Commands::DqnTrain) => run_dqn_train(),
        #[cfg(feature = "dqn")]
        Some(Commands::DqnSelfplay) => run_dqn_selfplay(),
        None => {
            // Default: play against MCTS
            run_play(vec![BotType::Mcts], false);
        }
    }
}
