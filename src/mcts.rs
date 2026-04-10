use crate::game::{Game, Player};
use crate::state::{Move, State};
use crate::strategy::Strategy;
use rand::{rngs::SmallRng, SeedableRng};
use rayon::prelude::*;
use std::sync::Arc;

/// A factory that creates rollout strategies. Must be Send+Sync for rayon.
pub type RolloutFactory = Arc<dyn Fn() -> Box<dyn Strategy> + Send + Sync>;

#[derive(Clone)]
pub struct MonteCarlo {
    simulations: usize,
    rollout_factory: RolloutFactory,
}

impl std::fmt::Debug for MonteCarlo {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "MonteCarlo(sims={})", self.simulations)
    }
}

impl MonteCarlo {
    pub fn new(simulations: usize, rollout_factory: RolloutFactory) -> Self {
        Self {
            simulations,
            rollout_factory,
        }
    }

    /// Convenience: create MC with GA champion as rollout policy.
    pub fn with_ga(simulations: usize, champion: crate::bot::DNA) -> Self {
        Self::new(simulations, Arc::new(move || {
            Box::new(champion.clone()) as Box<dyn Strategy>
        }))
    }

    pub fn evaluate_move_public(&self, state: &State, mov: Move, opponent_state: &State) -> f64 {
        self.evaluate_move(state, mov, opponent_state)
    }

    fn evaluate_move(&self, state: &State, mov: Move, opponent_state: &State) -> f64 {
        let mut our_state = *state;
        our_state.apply_move(mov);

        let total: f64 = (0..self.simulations)
            .into_par_iter()
            .map(|_| {
                let mut rng = SmallRng::from_entropy();
                let us = Player::new_with_state(
                    (self.rollout_factory)(),
                    Box::new(SmallRng::from_rng(&mut rng).unwrap()),
                    our_state,
                );
                let them = Player::new_with_state(
                    (self.rollout_factory)(),
                    Box::new(SmallRng::from_rng(&mut rng).unwrap()),
                    *opponent_state,
                );

                let mut game = Game::new_from_turn(vec![us, them], 1);
                game.play();
                game.players[0].state.count_points() as f64
            })
            .sum();

        total / self.simulations as f64
    }
}

impl Strategy for MonteCarlo {
    fn your_move(&mut self, state: &State, dice: [u8; 6]) -> Move {
        let moves = state.generate_moves(dice);

        let current_locked = state.count_locked();
        if let Some(mov) = moves.iter().copied().find(|&mov| {
            let mut s = *state;
            s.apply_move(mov);
            s.count_locked() > current_locked
        }) {
            return mov;
        }

        if moves.is_empty() {
            return Move::Strike;
        }

        let mut moves = moves;
        moves.push(Move::Strike);

        let opponent_state = State::default();

        moves
            .into_iter()
            .map(|mov| {
                let score = self.evaluate_move(state, mov, &opponent_state);
                (score, mov)
            })
            .max_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap())
            .unwrap()
            .1
    }

    fn opponents_move(&mut self, state: &State, number: u8, _locked: [bool; 4]) -> Option<Move> {
        let moves = state.generate_opponent_moves(number);

        let current_locked = state.count_locked();
        if let Some(mov) = moves.iter().copied().find(|&mov| {
            let mut s = *state;
            s.apply_move(mov);
            s.count_locked() > current_locked
        }) {
            return Some(mov);
        }

        if moves.is_empty() {
            return None;
        }

        // Use rollout strategy for opponent-turn evaluation
        let mut eval = (self.rollout_factory)();
        eval.opponents_move(state, number, _locked)
    }
}
