//! Gate-calibration diagnostic for the pair-search bot.
//!
//! Plays N 1v1 games (forced-search SearchBot vs plain pair bot), recording at
//! every eligible active decision: the top-2 static gap, which gates would
//! have fired, and whether search disagreed with the static choice. Prints a
//! summary used to tune GATE_MARGIN.
//!
//! Run: cargo run --release --example search_calibration -- 200

use qwixxer::dqn::pair::PairStrategy;
use qwixxer::game::{Game, Player};
use qwixxer::strategy::search::{SearchBot, SearchStats};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::cell::RefCell;
use std::rc::Rc;

fn main() {
    let games: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let template = PairStrategy::load("pair_model");

    let mut total = SearchStats::default();
    let start = std::time::Instant::now();
    for g in 0..games as u64 {
        let stats = Rc::new(RefCell::new(SearchStats::default()));
        let mut searcher = SearchBot::new(PairStrategy::from_shared(template.model.clone(), template.device));
        searcher.force = true;
        searcher.stats = Some(stats.clone());

        let players = vec![
            Player::new(Box::new(searcher), Box::new(SmallRng::seed_from_u64(31337 + 2 * g))),
            Player::new(
                Box::new(PairStrategy::from_shared(template.model.clone(), template.device)),
                Box::new(SmallRng::seed_from_u64(31337 + 2 * g + 1)),
            ),
        ];
        let mut game = Game::new(players);
        game.play();

        let s = stats.borrow();
        total.active_decisions += s.active_decisions;
        total.eligible += s.eligible;
        total.gate_close += s.gate_close;
        total.gate_endgame += s.gate_endgame;
        total.searched += s.searched;
        total.disagreements += s.disagreements;
        total.disagreements_gated += s.disagreements_gated;
        total.gaps.extend_from_slice(&s.gaps);
    }
    let elapsed = start.elapsed().as_secs_f64();

    let mut gaps = total.gaps.clone();
    gaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f64| -> f32 {
        if gaps.is_empty() {
            0.0
        } else {
            gaps[((gaps.len() - 1) as f64 * p) as usize]
        }
    };

    println!(
        "games:                {games}  ({:.1}s, {:.2}s/game)",
        elapsed,
        elapsed / games as f64
    );
    println!("active decisions:     {}", total.active_decisions);
    println!("eligible (>=2 cands): {}", total.eligible);
    println!("searched (forced):    {}", total.searched);
    println!(
        "gate hits: close {} ({:.1}% of eligible), endgame {} ({:.1}%)",
        total.gate_close,
        100.0 * total.gate_close as f64 / total.eligible.max(1) as f64,
        total.gate_endgame,
        100.0 * total.gate_endgame as f64 / total.eligible.max(1) as f64,
    );
    println!(
        "disagreements: {} ({:.1}% of searched); with a gate fired: {} ({:.1}% of disagreements)",
        total.disagreements,
        100.0 * total.disagreements as f64 / total.searched.max(1) as f64,
        total.disagreements_gated,
        100.0 * total.disagreements_gated as f64 / total.disagreements.max(1) as f64,
    );
    println!(
        "top-2 gap percentiles: p10 {:.3}  p25 {:.3}  p50 {:.3}  p75 {:.3}  p90 {:.3}",
        pct(0.10),
        pct(0.25),
        pct(0.50),
        pct(0.75),
        pct(0.90)
    );
}
