use crate::game::{Game, Player};
use crate::state::{MetaDecision, Move, State};
use crate::strategy::Strategy;
use rand::{rngs::SmallRng, SeedableRng};
use std::sync::Arc;

/// A factory that creates rollout strategies. Must be Send+Sync for parallelism.
pub type RolloutFactory = Arc<dyn Fn() -> Box<dyn Strategy> + Send + Sync>;

#[derive(Clone)]
pub struct MonteCarlo {
    simulations: usize,
    rollout_factory: RolloutFactory,
    score_gap: isize,
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
            score_gap: 0,
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

        // Short-circuit: if the candidate already ends the game, skip rollouts.
        if our_state.strikes >= 4 || our_state.count_locked() >= 2 {
            return our_state.count_points() as f64;
        }

        let total: f64 = (0..self.simulations)
            .map(|_| {
                let mut rng = rand::thread_rng();
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
        let moves = match state.apply_meta_rules(dice, self.score_gap) {
            MetaDecision::Forced(mov) => return mov,
            MetaDecision::Choices(moves) => moves,
        };

        let opponent_state = State::default();

        moves
            .into_iter()
            .map(|mov| (self.evaluate_move(state, mov, &opponent_state), mov))
            .max_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap())
            .unwrap()
            .1
    }

    fn opponents_move(&mut self, state: &State, number: u8, locked: [bool; 4]) -> Option<Move> {
        let moves = state.generate_opponent_moves(number);

        if let Some(mov) = state.find_smart_lock(&moves, self.score_gap) {
            return Some(mov);
        }

        if moves.is_empty() {
            return None;
        }

        // Delegate passive evaluation to the rollout policy.
        let mut eval = (self.rollout_factory)();
        eval.observe_opponents(state.count_points(), &[]);
        eval.opponents_move(state, number, locked)
    }

    fn observe_opponents(&mut self, our_score: isize, opponents: &[State]) {
        let max_opp = opponents.iter().map(|s| s.count_points()).max().unwrap_or(0);
        self.score_gap = our_score - max_opp;
    }
}
