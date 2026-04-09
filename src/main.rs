mod bot;
mod game;
mod state;
mod strategy;

use game::Game;

fn main() {
    // let mut points = vec![];
    env_logger::init();
    // for _ in 0..1_000_000 {
    // let mut game = Game::new_rng(vec![
    //     Box::<strategy::Random>::default(),
    //     Box::<strategy::Interactive>::default(),
    // ]);
    //
    // game.play();
    //
    // println!("bot points {}", game.players[0].state.count_points());
    // println!("human points {}", game.players[1].state.count_points());
    // println!("bot state {:?}", game.players[0].state);
    // println!("human state {:?}", game.players[1].state);

    // points.push((
    //     game.players[0].state.count_points() as i64,
    //     game.players[0].state.clone(),
    // ));
    // }

    // points.sort_by_key(|(points, _)| *points);
    // let max = points[points.len() - 1].0;
    // let min = points[0].0;
    // let median = points[points.len() / 2].0;
    // let mean = points.iter().map(|(points, _)| *points as f64).sum::<f64>() / points.len() as f64;
    //
    // println!("Mean: {}", mean);
    // println!("Max: {}", max);
    // println!("Min: {}", min);
    // println!("Median: {}", median);
    //
    // let max_state = &points[points.len() - 1].1;
    // let min_state = &points[0].1;
    // let median_state = &points[points.len() / 2].1;
    //
    // println!("Max: {:?}", max_state);
    // println!("Min: {:?}", min_state);
    // println!("Median: {:?}", median_state);
}
