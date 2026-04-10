mod blank;
mod bot;
mod dqn;
mod game;
mod mcts;
mod race_to_lock;
mod state;
mod strategy;

use game::Player;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::sync::Arc;

fn train_and_bench() {
    println!("Training GA bot (population=100, 200 generations)...\n");
    let mut pop = bot::Population::new(100, bot::default_genes(), 42);
    pop.evolve(200);

    let champion = pop.current_champion().clone();
    println!("\nBenchmarking champion vs Opportunist (2-player, alternating seats)...\n");

    let n = 10_000;
    let mut ga_wins = 0u32;
    let mut opp_wins = 0u32;
    let mut ga_total = 0i64;
    let mut opp_total = 0i64;

    for i in 0..n {
        let (ga_seat, opp_seat) = if i % 2 == 0 { (0, 1) } else { (1, 0) };
        let mut players = vec![
            Player::new(Box::new(champion.clone()) as Box<dyn strategy::Strategy>, Box::new(SmallRng::from_entropy())),
            Player::new(Box::<strategy::Opportunist>::default(), Box::new(SmallRng::from_entropy())),
        ];
        if ga_seat == 1 {
            players.swap(0, 1);
        }

        let mut game = game::Game::new(players);
        game.play();

        let scores: Vec<isize> = game.players.iter().map(|p| p.state.count_points()).collect();
        let ga_pts = scores[ga_seat];
        let opp_pts = scores[opp_seat];
        ga_total += ga_pts as i64;
        opp_total += opp_pts as i64;
        if ga_pts > opp_pts { ga_wins += 1; }
        if opp_pts > ga_pts { opp_wins += 1; }
    }

    println!("  GA Bot:       {ga_wins} wins ({:.1}%)  avg {:.1} pts",
        ga_wins as f64 / n as f64 * 100.0, ga_total as f64 / n as f64);
    println!("  Opportunist:  {opp_wins} wins ({:.1}%)  avg {:.1} pts",
        opp_wins as f64 / n as f64 * 100.0, opp_total as f64 / n as f64);
}

fn bench() {
    let genes = Arc::new(bot::default_genes());
    let champion =
        bot::DNA::load_weights("champion.txt", genes).expect("No champion.txt found");

    let n = 10_000;

    // 1v1: GA Bot vs Blank Score-Based
    println!("=== 1v1: GA Bot vs Blank Score-Based ({n} games, alternating seats) ===\n");
    {
        let mut a_wins = 0u32;
        let mut b_wins = 0u32;
        let mut ties = 0u32;
        let mut a_total = 0i64;
        let mut b_total = 0i64;

        for i in 0..n {
            let (a_seat, b_seat) = if i % 2 == 0 { (0, 1) } else { (1, 0) };
            let mut players = vec![
                Player::new(Box::new(champion.clone()) as Box<dyn strategy::Strategy>, Box::new(SmallRng::from_entropy())),
                Player::new(Box::new(blank::BlankScoreBased) as Box<dyn strategy::Strategy>, Box::new(SmallRng::from_entropy())),
            ];
            if a_seat == 1 {
                players.swap(0, 1);
            }

            let mut game = game::Game::new(players);
            game.play();

            let scores: Vec<isize> = game.players.iter().map(|p| p.state.count_points()).collect();
            let a_pts = scores[a_seat];
            let b_pts = scores[b_seat];
            a_total += a_pts as i64;
            b_total += b_pts as i64;
            match a_pts.cmp(&b_pts) {
                std::cmp::Ordering::Greater => a_wins += 1,
                std::cmp::Ordering::Less => b_wins += 1,
                std::cmp::Ordering::Equal => ties += 1,
            }
        }

        println!(
            "  {:<14} {:>5} wins ({:>4.1}%)  avg {:.1} pts",
            "GA Bot", a_wins, a_wins as f64 / n as f64 * 100.0, a_total as f64 / n as f64
        );
        println!(
            "  {:<14} {:>5} wins ({:>4.1}%)  avg {:.1} pts",
            "Blank S.B.", b_wins, b_wins as f64 / n as f64 * 100.0, b_total as f64 / n as f64
        );
        println!(
            "  {:<14} {:>5}       ({:>4.1}%)",
            "Ties", ties, ties as f64 / n as f64 * 100.0
        );
    }

    // 1v1: GA Bot vs Blank Race-to-Lock
    println!("\n=== 1v1: GA Bot vs Blank Race-to-Lock ({n} games, alternating seats) ===\n");
    {
        let mut a_wins = 0u32;
        let mut b_wins = 0u32;
        let mut ties = 0u32;
        let mut a_total = 0i64;
        let mut b_total = 0i64;

        for i in 0..n {
            let (a_seat, b_seat) = if i % 2 == 0 { (0, 1) } else { (1, 0) };
            let mut players = vec![
                Player::new(Box::new(champion.clone()) as Box<dyn strategy::Strategy>, Box::new(SmallRng::from_entropy())),
                Player::new(Box::new(race_to_lock::BlankRaceToLock::new()) as Box<dyn strategy::Strategy>, Box::new(SmallRng::from_entropy())),
            ];
            if a_seat == 1 {
                players.swap(0, 1);
            }

            let mut game = game::Game::new(players);
            game.play();

            let scores: Vec<isize> = game.players.iter().map(|p| p.state.count_points()).collect();
            a_total += scores[a_seat] as i64;
            b_total += scores[b_seat] as i64;
            match scores[a_seat].cmp(&scores[b_seat]) {
                std::cmp::Ordering::Greater => a_wins += 1,
                std::cmp::Ordering::Less => b_wins += 1,
                std::cmp::Ordering::Equal => ties += 1,
            }
        }

        println!(
            "  {:<14} {:>5} wins ({:>4.1}%)  avg {:.1} pts",
            "GA Bot", a_wins, a_wins as f64 / n as f64 * 100.0, a_total as f64 / n as f64
        );
        println!(
            "  {:<14} {:>5} wins ({:>4.1}%)  avg {:.1} pts",
            "Blank R.T.L.", b_wins, b_wins as f64 / n as f64 * 100.0, b_total as f64 / n as f64
        );
        println!(
            "  {:<14} {:>5}       ({:>4.1}%)",
            "Ties", ties, ties as f64 / n as f64 * 100.0
        );
    }

    // 2v2: GA Bots vs Blank Race-to-Lock
    println!("\n=== 2v2: GA Bots vs Blank R.T.L. ({n} 4-player games) ===\n");
    {
        let mut ga_wins = 0u32;
        let mut blank_wins = 0u32;
        let mut ties = 0u32;
        let mut ga_total = 0i64;
        let mut blank_total = 0i64;

        for i in 0..n {
            let ga_seats: [usize; 2];
            let blank_seats: [usize; 2];
            let game = if i % 2 == 0 {
                ga_seats = [0, 2];
                blank_seats = [1, 3];
                game::Game::new(vec![
                    Player::new(Box::new(champion.clone()) as Box<dyn strategy::Strategy>, Box::new(SmallRng::from_entropy())),
                    Player::new(Box::new(race_to_lock::BlankRaceToLock::new()) as Box<dyn strategy::Strategy>, Box::new(SmallRng::from_entropy())),
                    Player::new(Box::new(champion.clone()) as Box<dyn strategy::Strategy>, Box::new(SmallRng::from_entropy())),
                    Player::new(Box::new(race_to_lock::BlankRaceToLock::new()) as Box<dyn strategy::Strategy>, Box::new(SmallRng::from_entropy())),
                ])
            } else {
                ga_seats = [1, 3];
                blank_seats = [0, 2];
                game::Game::new(vec![
                    Player::new(Box::new(race_to_lock::BlankRaceToLock::new()) as Box<dyn strategy::Strategy>, Box::new(SmallRng::from_entropy())),
                    Player::new(Box::new(champion.clone()) as Box<dyn strategy::Strategy>, Box::new(SmallRng::from_entropy())),
                    Player::new(Box::new(race_to_lock::BlankRaceToLock::new()) as Box<dyn strategy::Strategy>, Box::new(SmallRng::from_entropy())),
                    Player::new(Box::new(champion.clone()) as Box<dyn strategy::Strategy>, Box::new(SmallRng::from_entropy())),
                ])
            };
            let mut game = game;
            game.play();

            let scores: Vec<isize> = game.players.iter().map(|p| p.state.count_points()).collect();
            let ga_best = ga_seats.iter().map(|&s| scores[s]).max().unwrap();
            let blank_best = blank_seats.iter().map(|&s| scores[s]).max().unwrap();
            let ga_avg_pts = ga_seats.iter().map(|&s| scores[s] as i64).sum::<i64>();
            let blank_avg_pts = blank_seats.iter().map(|&s| scores[s] as i64).sum::<i64>();
            ga_total += ga_avg_pts;
            blank_total += blank_avg_pts;
            match ga_best.cmp(&blank_best) {
                std::cmp::Ordering::Greater => ga_wins += 1,
                std::cmp::Ordering::Less => blank_wins += 1,
                std::cmp::Ordering::Equal => ties += 1,
            }
        }

        println!(
            "  {:<14} {:>5} wins ({:>4.1}%)  avg {:.1} pts/player",
            "GA Bots", ga_wins, ga_wins as f64 / n as f64 * 100.0, ga_total as f64 / n as f64 / 2.0
        );
        println!(
            "  {:<14} {:>5} wins ({:>4.1}%)  avg {:.1} pts/player",
            "Blank R.T.L.", blank_wins, blank_wins as f64 / n as f64 * 100.0, blank_total as f64 / n as f64 / 2.0
        );
        println!(
            "  {:<14} {:>5}       ({:>4.1}%)",
            "Ties", ties, ties as f64 / n as f64 * 100.0
        );
    }
}

fn bench_mc() {
    let genes = Arc::new(bot::default_genes());
    let champion = bot::DNA::load_weights("champion.txt", genes.clone()).expect("No champion.txt found");

    let sims = 500;
    println!("Monte Carlo ({sims} sims/move) vs Opportunist (2-player, alternating seats)\n");

    let n = 1_000;
    let mut mc_wins = 0u32;
    let mut opp_wins = 0u32;
    let mut ties = 0u32;
    let mut mc_total = 0i64;
    let mut opp_total = 0i64;

    for i in 0..n {
        if i % 100 == 0 && i > 0 {
            println!("  {i}/{n}...");
        }
        let (mc_seat, opp_seat) = if i % 2 == 0 { (0, 1) } else { (1, 0) };
        let mut players = vec![
            Player::new(
                Box::new(mcts::MonteCarlo::new(sims, champion.clone())),
                Box::new(SmallRng::from_entropy()),
            ),
            Player::new(
                Box::<strategy::Opportunist>::default(),
                Box::new(SmallRng::from_entropy()),
            ),
        ];
        if mc_seat == 1 {
            players.swap(0, 1);
        }

        let mut game = game::Game::new(players);
        game.play();

        let scores: Vec<isize> = game.players.iter().map(|p| p.state.count_points()).collect();
        let mc_pts = scores[mc_seat];
        let opp_pts = scores[opp_seat];
        mc_total += mc_pts as i64;
        opp_total += opp_pts as i64;
        match mc_pts.cmp(&opp_pts) {
            std::cmp::Ordering::Greater => mc_wins += 1,
            std::cmp::Ordering::Less => opp_wins += 1,
            std::cmp::Ordering::Equal => ties += 1,
        }
    }

    println!("\nResults over {n} 2-player games (alternating seats):");
    println!(
        "  {:<14} {:>5} wins ({:>4.1}%)  avg {:.1} pts",
        "MC Bot", mc_wins, mc_wins as f64 / n as f64 * 100.0, mc_total as f64 / n as f64
    );
    println!(
        "  {:<14} {:>5} wins ({:>4.1}%)  avg {:.1} pts",
        "Opportunist", opp_wins, opp_wins as f64 / n as f64 * 100.0, opp_total as f64 / n as f64
    );
    println!(
        "  {:<14} {:>5}       ({:>4.1}%)",
        "Ties", ties, ties as f64 / n as f64 * 100.0
    );
}

fn solo() {
    let genes = Arc::new(bot::default_genes());
    let champion = bot::DNA::load_weights("champion.txt", genes).expect("No champion.txt found");

    let n = 10_000;
    let strategies: Vec<(&str, Box<dyn Fn() -> Box<dyn strategy::Strategy>>)> = vec![
        ("Random", Box::new(|| Box::new(strategy::Random))),
        ("Conservative", Box::new(|| Box::<strategy::Conservative>::default())),
        ("Rusher", Box::new(|| Box::new(strategy::Rusher))),
        ("Opportunist", Box::new(|| Box::new(strategy::Opportunist))),
        ("GA Bot", Box::new({
            let c = champion.clone();
            move || Box::new(c.clone())
        })),
        ("Blank S.B.", Box::new(|| Box::new(blank::BlankScoreBased))),
        ("Blank R.T.L.", Box::new(|| Box::new(race_to_lock::BlankRaceToLock::new()))),
    ];

    println!("Single-player scores over {n} games:\n");
    for (name, make_strategy) in &strategies {
        let mut total = 0i64;
        let mut min = i64::MAX;
        let mut max = i64::MIN;
        for _ in 0..n {
            let mut game = game::Game::new(vec![
                Player::new(make_strategy(), Box::new(SmallRng::from_entropy())),
            ]);
            game.play();
            let pts = game.players[0].state.count_points() as i64;
            total += pts;
            min = min.min(pts);
            max = max.max(pts);
        }
        let avg = total as f64 / n as f64;
        println!("  {:<14} avg {avg:>5.1}  min {min:>4}  max {max:>4}", name);
    }
}

fn main() {
    if std::env::args().any(|a| a == "--train") {
        train_and_bench();
        return;
    }

    if std::env::args().any(|a| a == "--bench") {
        bench();
        return;
    }

    if std::env::args().any(|a| a == "--mc") {
        bench_mc();
        return;
    }

    if std::env::args().any(|a| a == "--solo") {
        solo();
        return;
    }

    if std::env::args().any(|a| a == "--dqn-train") {
        let samples = dqn::generate_training_data(500, 200);
        dqn::train(samples, "dqn_model");
        return;
    }

    if std::env::args().any(|a| a == "--dqn-selfplay") {
        dqn::self_play_train("dqn_model", 80, 3000, 10);
        return;
    }

    if std::env::args().any(|a| a == "--mc-vs-ga") {
        let genes = Arc::new(bot::default_genes());
        let champion = bot::DNA::load_weights("champion.txt", genes).expect("No champion.txt");
        let n = 1_000;
        let mut mc_wins = 0u32;
        let mut ga_wins = 0u32;
        let mut ties = 0u32;
        let mut mc_total = 0i64;
        let mut ga_total = 0i64;
        println!("MC (500 sims) vs GA Champion ({n} games, alternating seats):\n");
        for i in 0..n {
            let (mc_seat, ga_seat) = if i % 2 == 0 { (0, 1) } else { (1, 0) };
            let mut players = vec![
                Player::new(Box::new(mcts::MonteCarlo::new(500, champion.clone())), Box::new(SmallRng::from_entropy())),
                Player::new(Box::new(champion.clone()) as Box<dyn strategy::Strategy>, Box::new(SmallRng::from_entropy())),
            ];
            if mc_seat == 1 { players.swap(0, 1); }
            let mut game = game::Game::new(players);
            game.play();
            let scores: Vec<isize> = game.players.iter().map(|p| p.state.count_points()).collect();
            mc_total += scores[mc_seat] as i64;
            ga_total += scores[ga_seat] as i64;
            match scores[mc_seat].cmp(&scores[ga_seat]) {
                std::cmp::Ordering::Greater => mc_wins += 1,
                std::cmp::Ordering::Less => ga_wins += 1,
                std::cmp::Ordering::Equal => ties += 1,
            }
            if i % 100 == 0 && i > 0 { eprintln!("  {i}/{n}..."); }
        }
        println!("  {:<14} {:>5} wins ({:>4.1}%)  avg {:.1} pts", "MC Bot", mc_wins, mc_wins as f64/n as f64*100.0, mc_total as f64/n as f64);
        println!("  {:<14} {:>5} wins ({:>4.1}%)  avg {:.1} pts", "GA Champion", ga_wins, ga_wins as f64/n as f64*100.0, ga_total as f64/n as f64);
        println!("  {:<14} {:>5}       ({:>4.1}%)", "Ties", ties, ties as f64/n as f64*100.0);
        return;
    }

    if std::env::args().any(|a| a == "--dqn-vs-ga-2v2") {
        let genes = Arc::new(bot::default_genes());
        let champion = bot::DNA::load_weights("champion.txt", genes).expect("No champion.txt");
        let n = 1_000;
        let mut ga_wins = 0u32;
        let mut dqn_wins = 0u32;
        let mut ties = 0u32;
        let mut ga_total = 0i64;
        let mut dqn_total = 0i64;
        println!("2v2: DQN Bots vs GA Champions ({n} games, alternating seats):\n");
        for i in 0..n {
            let (ga_seats, dqn_seats) = if i % 2 == 0 {
                ([0usize, 2], [1usize, 3])
            } else {
                ([1, 3], [0, 2])
            };
            let mut players: Vec<Player> = if i % 2 == 0 {
                vec![
                    Player::new(Box::new(champion.clone()) as Box<dyn strategy::Strategy>, Box::new(SmallRng::from_entropy())),
                    Player::new(Box::new(dqn::DqnStrategy::load("dqn_model")) as Box<dyn strategy::Strategy>, Box::new(SmallRng::from_entropy())),
                    Player::new(Box::new(champion.clone()) as Box<dyn strategy::Strategy>, Box::new(SmallRng::from_entropy())),
                    Player::new(Box::new(dqn::DqnStrategy::load("dqn_model")) as Box<dyn strategy::Strategy>, Box::new(SmallRng::from_entropy())),
                ]
            } else {
                vec![
                    Player::new(Box::new(dqn::DqnStrategy::load("dqn_model")) as Box<dyn strategy::Strategy>, Box::new(SmallRng::from_entropy())),
                    Player::new(Box::new(champion.clone()) as Box<dyn strategy::Strategy>, Box::new(SmallRng::from_entropy())),
                    Player::new(Box::new(dqn::DqnStrategy::load("dqn_model")) as Box<dyn strategy::Strategy>, Box::new(SmallRng::from_entropy())),
                    Player::new(Box::new(champion.clone()) as Box<dyn strategy::Strategy>, Box::new(SmallRng::from_entropy())),
                ]
            };
            let mut game = game::Game::new(players);
            game.play();
            let scores: Vec<isize> = game.players.iter().map(|p| p.state.count_points()).collect();
            let ga_best = ga_seats.iter().map(|&s| scores[s]).max().unwrap();
            let dqn_best = dqn_seats.iter().map(|&s| scores[s]).max().unwrap();
            ga_total += ga_seats.iter().map(|&s| scores[s] as i64).sum::<i64>();
            dqn_total += dqn_seats.iter().map(|&s| scores[s] as i64).sum::<i64>();
            match dqn_best.cmp(&ga_best) {
                std::cmp::Ordering::Greater => dqn_wins += 1,
                std::cmp::Ordering::Less => ga_wins += 1,
                std::cmp::Ordering::Equal => ties += 1,
            }
        }
        println!("  {:<14} {:>5} wins ({:>4.1}%)  avg {:.1} pts/player", "DQN Bots", dqn_wins, dqn_wins as f64/n as f64*100.0, dqn_total as f64/n as f64/2.0);
        println!("  {:<14} {:>5} wins ({:>4.1}%)  avg {:.1} pts/player", "GA Champions", ga_wins, ga_wins as f64/n as f64*100.0, ga_total as f64/n as f64/2.0);
        println!("  {:<14} {:>5}       ({:>4.1}%)", "Ties", ties, ties as f64/n as f64*100.0);
        return;
    }

    if std::env::args().any(|a| a == "--dqn-bench") {
        let genes = Arc::new(bot::default_genes());
        let champion = bot::DNA::load_weights("champion.txt", genes).expect("No champion.txt");

        let n = 1_000;

        // DQN vs Opportunist
        println!("DQN vs Opportunist ({n} games, alternating seats):\n");
        {
            let mut a_wins = 0u32;
            let mut b_wins = 0u32;
            let mut ties = 0u32;
            let mut a_total = 0i64;
            let mut b_total = 0i64;
            for i in 0..n {
                let (a_seat, b_seat) = if i % 2 == 0 { (0, 1) } else { (1, 0) };
                let mut players = vec![
                    Player::new(Box::new(dqn::DqnStrategy::load("dqn_model")) as Box<dyn strategy::Strategy>, Box::new(SmallRng::from_entropy())),
                    Player::new(Box::<strategy::Opportunist>::default(), Box::new(SmallRng::from_entropy())),
                ];
                if a_seat == 1 { players.swap(0, 1); }
                let mut game = game::Game::new(players);
                game.play();
                let scores: Vec<isize> = game.players.iter().map(|p| p.state.count_points()).collect();
                a_total += scores[a_seat] as i64;
                b_total += scores[b_seat] as i64;
                match scores[a_seat].cmp(&scores[b_seat]) {
                    std::cmp::Ordering::Greater => a_wins += 1,
                    std::cmp::Ordering::Less => b_wins += 1,
                    std::cmp::Ordering::Equal => ties += 1,
                }
            }
            println!("  {:<14} {:>5} wins ({:>4.1}%)  avg {:.1} pts", "DQN Bot", a_wins, a_wins as f64/n as f64*100.0, a_total as f64/n as f64);
            println!("  {:<14} {:>5} wins ({:>4.1}%)  avg {:.1} pts", "Opportunist", b_wins, b_wins as f64/n as f64*100.0, b_total as f64/n as f64);
            println!("  {:<14} {:>5}       ({:>4.1}%)", "Ties", ties, ties as f64/n as f64*100.0);
        }

        // DQN vs GA Champion
        println!("\nDQN vs GA Champion ({n} games, alternating seats):\n");
        {
            let mut a_wins = 0u32;
            let mut b_wins = 0u32;
            let mut ties = 0u32;
            let mut a_total = 0i64;
            let mut b_total = 0i64;
            for i in 0..n {
                let (a_seat, b_seat) = if i % 2 == 0 { (0, 1) } else { (1, 0) };
                let mut players = vec![
                    Player::new(Box::new(dqn::DqnStrategy::load("dqn_model")) as Box<dyn strategy::Strategy>, Box::new(SmallRng::from_entropy())),
                    Player::new(Box::new(champion.clone()) as Box<dyn strategy::Strategy>, Box::new(SmallRng::from_entropy())),
                ];
                if a_seat == 1 { players.swap(0, 1); }
                let mut game = game::Game::new(players);
                game.play();
                let scores: Vec<isize> = game.players.iter().map(|p| p.state.count_points()).collect();
                a_total += scores[a_seat] as i64;
                b_total += scores[b_seat] as i64;
                match scores[a_seat].cmp(&scores[b_seat]) {
                    std::cmp::Ordering::Greater => a_wins += 1,
                    std::cmp::Ordering::Less => b_wins += 1,
                    std::cmp::Ordering::Equal => ties += 1,
                }
            }
            println!("  {:<14} {:>5} wins ({:>4.1}%)  avg {:.1} pts", "DQN Bot", a_wins, a_wins as f64/n as f64*100.0, a_total as f64/n as f64);
            println!("  {:<14} {:>5} wins ({:>4.1}%)  avg {:.1} pts", "GA Champion", b_wins, b_wins as f64/n as f64*100.0, b_total as f64/n as f64);
            println!("  {:<14} {:>5}       ({:>4.1}%)", "Ties", ties, ties as f64/n as f64*100.0);
        }
        return;
    }

    let genes = Arc::new(bot::default_genes());
    let champion = bot::DNA::load_weights("champion.txt", genes)
        .expect("No champion.txt found. Run --train first.");

    let use_ga = std::env::args().any(|a| a == "--ga");
    let use_dqn = std::env::args().any(|a| a == "--dqn");
    let verbose = std::env::args().any(|a| a == "--verbose");

    let bot_name = if use_dqn { "DQN" } else if use_ga { "GA champion" } else { "MCTS (500 sims/move)" };
    println!("Playing against {bot_name} bot...\n");

    let bot_strategy: Box<dyn strategy::Strategy> = if use_dqn {
        Box::new(dqn::DqnStrategy::load("dqn_model"))
    } else if use_ga {
        Box::new(champion)
    } else {
        Box::new(mcts::MonteCarlo::new(500, champion))
    };

    let mut game = game::Game::new(vec![
        Player::new(bot_strategy, Box::new(SmallRng::from_entropy())),
        Player::new(
            Box::<strategy::Interactive>::default(),
            Box::new(SmallRng::from_entropy()),
        ),
    ]);
    game.verbose = verbose;

    game.play();
    game.print_game_over();
}
