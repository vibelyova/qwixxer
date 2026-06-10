//! Lightweight game simulator for decision-time search. Replicates the
//! mechanics of `Game::play` (simultaneous phase 1 on pre-snapshots, lock
//! propagation, game-over ordering, phase 2) on plain `State`s, entrable
//! mid-turn, with all players' decisions driven by a `Bot`.

use super::bot_impl::{
    active_phase1_impl, active_phase2_choices, eval_decision, passive_phase1_choices, phase1_plan_choices,
};
use super::Bot;
use crate::game::DiceSource;
use crate::state::{Mark, State};
use rand::rngs::SmallRng;

/// Decision fidelity inside simulations.
#[derive(Clone, Copy, PartialEq)]
pub enum Fidelity {
    /// Identical to the blanket Strategy impl (incl. nested opponent
    /// simulation in active phase 1). Used by the Game-equivalence test.
    Full,
    /// Active phase 1 evaluates plans against current opponent states — no
    /// nested opponent simulation. Used inside search rollouts (one net
    /// evaluation per decision; sims are approximations).
    Lite,
}

pub struct SimGame {
    pub states: Vec<State>,
    /// Player to take the next turn. Player 0 is the searcher's seat.
    pub active: usize,
    /// Per-player dice streams (mirrors `Game`, where each player rolls).
    pub rngs: Vec<SmallRng>,
    pub over: bool,
}

impl SimGame {
    pub fn n(&self) -> usize {
        self.states.len()
    }

    pub fn game_over(states: &[State]) -> bool {
        states.iter().any(|s| s.strikes >= 4) || states.iter().map(|s| s.count_locked()).max().unwrap() >= 2
    }

    pub fn propagate_locks(states: &mut [State]) {
        let mut locked = [false; 4];
        for s in states.iter() {
            let l = s.locked();
            for row in 0..4 {
                locked[row] |= l[row];
            }
        }
        for s in states.iter_mut() {
            s.lock(locked);
        }
    }

    /// Turn-ordered opponent view for player `j` over `states`.
    pub fn opp_view(states: &[State], j: usize) -> Vec<State> {
        let n = states.len();
        (1..n).map(|off| states[(j + off) % n]).collect()
    }

    /// Exact outcome for player 0: 1.0 win, 0.5 top tie, 0.0 loss.
    pub fn outcome(states: &[State]) -> f32 {
        let our = states[0].count_points();
        let best_opp = states[1..].iter().map(|s| s.count_points()).max().unwrap();
        match our.cmp(&best_opp) {
            std::cmp::Ordering::Greater => 1.0,
            std::cmp::Ordering::Equal => 0.5,
            std::cmp::Ordering::Less => 0.0,
        }
    }
}

/// Advance one sim by one full turn (roll, phase 1 all players, locks,
/// game-over, phase 2, locks, game-over, advance). Reference implementation —
/// the lockstep driver must stay decision-equivalent to this.
pub fn play_sim_turn(bot: &impl Bot, sim: &mut SimGame, fidelity: Fidelity) {
    if sim.over {
        return;
    }
    let n = sim.n();
    let dice = sim.rngs[sim.active].roll();
    let snapshot = sim.states.clone();

    // Phase 1: all players decide against pre-phase1 snapshots.
    let mut marks: Vec<Option<Mark>> = Vec::with_capacity(n);
    for j in 0..n {
        let view = SimGame::opp_view(&snapshot, j);
        let m = if j == sim.active {
            match fidelity {
                Fidelity::Full => active_phase1_impl(bot, &snapshot[j], &view, dice),
                Fidelity::Lite => eval_decision(bot, phase1_plan_choices(&snapshot[j], &view, dice), &view),
            }
        } else {
            eval_decision(bot, passive_phase1_choices(&snapshot[j], &view, dice), &view)
        };
        marks.push(m);
    }
    let has_marked = marks[sim.active].is_some();
    for (j, m) in marks.iter().enumerate() {
        if let Some(m) = m {
            sim.states[j].apply_mark(*m);
        }
    }
    SimGame::propagate_locks(&mut sim.states);
    if SimGame::game_over(&sim.states) {
        sim.over = true;
        return;
    }

    // Phase 2: active player only.
    let view = SimGame::opp_view(&sim.states, sim.active);
    let decision = active_phase2_choices(&sim.states[sim.active], &view, dice, has_marked);
    match eval_decision(bot, decision, &view) {
        Some(m) => sim.states[sim.active].apply_mark(m),
        None if !has_marked => sim.states[sim.active].apply_strike(),
        None => {}
    }
    SimGame::propagate_locks(&mut sim.states);
    if SimGame::game_over(&sim.states) {
        sim.over = true;
        return;
    }

    sim.active = (sim.active + 1) % n;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dqn::pair::{PairModelConfig, PairStrategy};
    use crate::dqn::MyBackend;
    use crate::game::{Game, Player};
    use rand::SeedableRng;
    use std::sync::Arc;

    #[test]
    fn sim_turn_loop_matches_game_play() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let model = Arc::new(PairModelConfig::new().init::<MyBackend>(&device));

        for &n in &[2usize, 3, 4] {
            for seed in 0..3u64 {
                // Reference: real Game with PairStrategy players.
                let players: Vec<Player> = (0..n)
                    .map(|i| {
                        Player::new(
                            Box::new(PairStrategy::from_shared(model.clone(), device)),
                            Box::new(SmallRng::seed_from_u64(1000 * seed + i as u64)),
                        )
                    })
                    .collect();
                let mut game = Game::new(players);
                game.play();

                // Sim: same model, same per-player dice streams, full fidelity.
                let bot = PairStrategy::from_shared(model.clone(), device);
                let mut sim = SimGame {
                    states: vec![State::default(); n],
                    active: 0,
                    rngs: (0..n)
                        .map(|i| SmallRng::seed_from_u64(1000 * seed + i as u64))
                        .collect(),
                    over: false,
                };
                let mut guard = 0;
                while !sim.over {
                    play_sim_turn(&bot, &mut sim, Fidelity::Full);
                    guard += 1;
                    assert!(guard < 500, "sim did not terminate");
                }

                for (i, p) in game.players.iter().enumerate() {
                    assert_eq!(
                        sim.states[i], p.state,
                        "n={n} seed={seed} player={i}: sim diverged from Game::play"
                    );
                }
            }
        }

        // Trained weights play lock-seeking games, covering the 2-locks
        // game-over path and cross-player lock propagation (the random-init
        // model above only ever ends games by strikes).
        let trained = PairStrategy::load("pair_model");
        let model = trained.model.clone();
        let mut saw_lock_end = false;
        for &n in &[2usize, 3, 4] {
            for seed in 0..2u64 {
                let players: Vec<Player> = (0..n)
                    .map(|i| {
                        Player::new(
                            Box::new(PairStrategy::from_shared(model.clone(), device)),
                            Box::new(SmallRng::seed_from_u64(5000 * seed + i as u64)),
                        )
                    })
                    .collect();
                let mut game = Game::new(players);
                game.play();

                let bot = PairStrategy::from_shared(model.clone(), device);
                let mut sim = SimGame {
                    states: vec![State::default(); n],
                    active: 0,
                    rngs: (0..n)
                        .map(|i| SmallRng::seed_from_u64(5000 * seed + i as u64))
                        .collect(),
                    over: false,
                };
                let mut guard = 0;
                while !sim.over {
                    play_sim_turn(&bot, &mut sim, Fidelity::Full);
                    guard += 1;
                    assert!(guard < 500, "sim did not terminate");
                }
                saw_lock_end |= sim.states.iter().map(|s| s.count_locked()).max().unwrap() >= 2;
                for (i, p) in game.players.iter().enumerate() {
                    assert_eq!(sim.states[i], p.state, "trained n={n} seed={seed} player={i}");
                }
            }
        }
        assert!(saw_lock_end, "no trained game ended via locks — lock path not covered");
    }
}
