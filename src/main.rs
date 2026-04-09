mod bot;
mod game;
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
    println!("\nBenchmarking champion...\n");

    let n = 10_000;
    let names = ["GA Bot", "Conservative", "Rusher"];
    let mut wins = [0u32; 3];
    let mut total_pts = [0i64; 3];

    for _ in 0..n {
        let mut game = game::Game::new(vec![
            Player::new(
                Box::new(champion.clone()) as Box<dyn strategy::Strategy>,
                Box::new(SmallRng::from_entropy()),
            ),
            Player::new(
                Box::<strategy::Conservative>::default(),
                Box::new(SmallRng::from_entropy()),
            ),
            Player::new(
                Box::<strategy::Rusher>::default(),
                Box::new(SmallRng::from_entropy()),
            ),
        ]);
        game.play();

        let scores: Vec<isize> = game.players.iter().map(|p| p.state.count_points()).collect();
        let max = *scores.iter().max().unwrap();
        for (i, &s) in scores.iter().enumerate() {
            total_pts[i] += s as i64;
            if s == max {
                wins[i] += 1;
            }
        }
    }

    println!("Results over {n} 3-player games:");
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

fn bench() {
    let genes = Arc::new(bot::default_genes());
    let champion = bot::DNA::load_weights("champion.txt", genes).expect("No champion.txt found");

    let n = 10_000;
    let names = ["Opportunist 1", "Opportunist 2", "GA Bot 1", "GA Bot 2"];
    let mut wins = [0u32; 4];
    let mut total_pts = [0i64; 4];

    for _ in 0..n {
        let mut game = game::Game::new(vec![
            Player::new(Box::<strategy::Opportunist>::default(), Box::new(SmallRng::from_entropy())),
            Player::new(Box::<strategy::Opportunist>::default(), Box::new(SmallRng::from_entropy())),
            Player::new(Box::new(champion.clone()) as Box<dyn strategy::Strategy>, Box::new(SmallRng::from_entropy())),
            Player::new(Box::new(champion.clone()) as Box<dyn strategy::Strategy>, Box::new(SmallRng::from_entropy())),
        ]);
        game.play();

        let scores: Vec<isize> = game.players.iter().map(|p| p.state.count_points()).collect();
        let max = *scores.iter().max().unwrap();
        for (i, &s) in scores.iter().enumerate() {
            total_pts[i] += s as i64;
            if s == max {
                wins[i] += 1;
            }
        }
    }

    println!("Results over {n} 4-player games:");
    for i in 0..4 {
        println!(
            "  {:<16} {:>5} wins ({:>4.1}%)  avg {:.1} pts",
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
