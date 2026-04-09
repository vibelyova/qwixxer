mod bot;
mod game;
mod mcts;
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
    let mut ga_wins = 0u32;
    let mut opp_wins = 0u32;
    let mut ties = 0u32;
    let mut ga_total = 0i64;
    let mut opp_total = 0i64;

    for i in 0..n {
        // Alternate who goes first
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
        match ga_pts.cmp(&opp_pts) {
            std::cmp::Ordering::Greater => ga_wins += 1,
            std::cmp::Ordering::Less => opp_wins += 1,
            std::cmp::Ordering::Equal => ties += 1,
        }
    }

    println!("Results over {n} 2-player games (alternating seats):");
    println!(
        "  {:<14} {:>5} wins ({:>4.1}%)  avg {:.1} pts",
        "GA Bot", ga_wins, ga_wins as f64 / n as f64 * 100.0, ga_total as f64 / n as f64
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

fn bench_mc() {
    let genes = Arc::new(bot::default_genes());
    let champion = bot::DNA::load_weights("champion.txt", genes.clone()).expect("No champion.txt found");

    let sims = 500;
    println!("Monte Carlo ({sims} sims/move) vs GA Bot vs Opportunist\n");

    let n = 1_000; // fewer games since MC is slow
    let names = ["MC Bot", "GA Bot", "Opportunist"];
    let mut wins = [0u32; 3];
    let mut total_pts = [0i64; 3];

    for i in 0..n {
        if i % 100 == 0 && i > 0 {
            println!("  {i}/{n}...");
        }
        let mut game = game::Game::new(vec![
            Player::new(
                Box::new(mcts::MonteCarlo::new(sims, champion.clone())),
                Box::new(SmallRng::from_entropy()),
            ),
            Player::new(
                Box::new(champion.clone()) as Box<dyn strategy::Strategy>,
                Box::new(SmallRng::from_entropy()),
            ),
            Player::new(
                Box::<strategy::Opportunist>::default(),
                Box::new(SmallRng::from_entropy()),
            ),
        ]);
        game.play();

        let scores: Vec<isize> = game.players.iter().map(|p| p.state.count_points()).collect();
        let max = *scores.iter().max().unwrap();
        for (j, &s) in scores.iter().enumerate() {
            total_pts[j] += s as i64;
            if s == max {
                wins[j] += 1;
            }
        }
    }

    println!("\nResults over {n} 3-player games:");
    for i in 0..3 {
        println!(
            "  {:<14} {:>5} wins ({:>4.1}%)  avg {:.1} pts",
            names[i],
            wins[i],
            wins[i] as f64 / n as f64 * 100.0,
            total_pts[i] as f64 / n as f64
        );
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

    let champion = match bot::DNA::load_weights("champion.txt", Arc::new(bot::default_genes())) {
        Ok(dna) => {
            println!("Loaded champion from champion.txt");
            dna.print_weights();
            dna
        }
        Err(_) => {
            println!("No champion.txt found, training...\n");
            let mut pop = bot::Population::new(100, bot::default_genes(), 42);
            pop.evolve(200);
            pop.current_champion().clone()
        }
    };

    println!();
    let mut game = game::Game::new(vec![
        Player::new(
            Box::new(champion) as Box<dyn strategy::Strategy>,
            Box::new(SmallRng::from_entropy()),
        ),
        Player::new(
            Box::<strategy::Interactive>::default(),
            Box::new(SmallRng::from_entropy()),
        ),
    ]);

    game.play();
    game.print_game_over();
}
