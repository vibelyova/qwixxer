use crate::bot::DNA;
use crate::game::{Game, Player};
use crate::state::{Move, State};
use crate::strategy::Strategy;
use rand::{rngs::SmallRng, SeedableRng};
use rayon::prelude::*;

#[derive(Debug)]
pub struct MonteCarlo {
    simulations: usize,
    opponent: DNA,
}

impl MonteCarlo {
    pub fn new(simulations: usize, opponent: DNA) -> Self {
        Self {
            simulations,
            opponent,
        }
    }

    fn evaluate_move(&self, state: &State, mov: Move, opponent_state: &State) -> f64 {
        let mut our_state = *state;
        our_state.apply_move(mov);

        let total: f64 = (0..self.simulations)
            .into_par_iter()
            .map(|_| {
                let mut rng = SmallRng::from_entropy();
                let us = Player::new_with_state(
                    Box::new(self.opponent.clone()),
                    Box::new(SmallRng::from_rng(&mut rng).unwrap()),
                    our_state,
                );
                let them = Player::new_with_state(
                    Box::new(self.opponent.clone()),
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

        // Always lock if possible
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

        // Always lock if possible
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

        // Use GA instinct for opponent turns (fast)
        let mut states: Vec<_> = moves
            .into_iter()
            .map(|mov| {
                let mut new_state = *state;
                new_state.apply_move(mov);
                (new_state, Some(mov))
            })
            .collect();
        states.push((*state, None));

        *states
            .iter()
            .map(|(state, mov)| (self.opponent.instinct(state), mov))
            .max_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap())
            .unwrap()
            .1
    }
}
