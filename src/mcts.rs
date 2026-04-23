use crate::game::{Game, Player};
use crate::state::{Mark, State};
use crate::strategy::Strategy;
use rand::{rngs::SmallRng, SeedableRng};
use std::sync::Arc;

/// A factory that creates rollout strategies. Must be Send+Sync for parallelism.
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

    /// Evaluate a post-move state by running rollout simulations from it.
    /// Creates a 2-player game from the given state and opponent state,
    /// runs N rollouts, returns the average score.
    fn evaluate_state(&self, our_state: &State, opponent_state: &State) -> f64 {
        // Short-circuit: if the state already ends the game, skip rollouts.
        if our_state.strikes >= 4 || our_state.count_locked() >= 2 {
            return our_state.count_points() as f64;
        }

        let total: f64 = (0..self.simulations)
            .map(|_| {
                let mut rng = rand::thread_rng();
                let us = Player::new_with_state(
                    (self.rollout_factory)(),
                    Box::new(SmallRng::from_rng(&mut rng).unwrap()),
                    *our_state,
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

    /// Public interface kept for backwards compatibility with training code.
    pub fn evaluate_move_public(&self, state: &State, mov: crate::state::Move, opponent_state: &State) -> f64 {
        let mut our_state = *state;
        our_state.apply_move(mov);
        self.evaluate_state(&our_state, opponent_state)
    }
}

impl Strategy for MonteCarlo {
    fn active_phase1(&mut self, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
        let white_sum = dice[0] + dice[1];
        let marks = state.generate_white_moves(white_sum);
        if marks.is_empty() {
            return None;
        }

        let opponent_state = opp_states.first().copied().unwrap_or_default();

        // Evaluate each white mark + skip
        let mut best_mark: Option<Mark> = None;
        let mut best_value = self.evaluate_state(state, &opponent_state); // skip value

        for &mark in &marks {
            let mut s = *state;
            s.apply_mark(mark);
            let value = self.evaluate_state(&s, &opponent_state);
            if value > best_value {
                best_value = value;
                best_mark = Some(mark);
            }
        }

        best_mark
    }

    fn active_phase2(&mut self, state: &State, opp_states: &[State], dice: [u8; 6], has_marked: bool) -> Option<Mark> {
        let marks = state.generate_color_moves(dice);
        if marks.is_empty() {
            return None;
        }

        let opponent_state = opp_states.first().copied().unwrap_or_default();

        // No-mark baseline
        let no_mark_value = if has_marked {
            self.evaluate_state(state, &opponent_state)
        } else {
            let mut s = *state;
            s.apply_strike();
            self.evaluate_state(&s, &opponent_state)
        };

        let mut best_mark: Option<Mark> = None;
        let mut best_value = no_mark_value;

        for &mark in &marks {
            let mut s = *state;
            s.apply_mark(mark);
            let value = self.evaluate_state(&s, &opponent_state);
            if value > best_value {
                best_value = value;
                best_mark = Some(mark);
            }
        }

        best_mark
    }

    fn passive_phase1(&mut self, state: &State, opp_states: &[State], dice: [u8; 6], _active_player: usize) -> Option<Mark> {
        let white_sum = dice[0] + dice[1];
        let marks = state.generate_white_moves(white_sum);
        if marks.is_empty() {
            return None;
        }

        let opponent_state = opp_states.first().copied().unwrap_or_default();

        // Delegate to rollout policy for passive evaluation
        let mut best_mark: Option<Mark> = None;
        let mut best_value = self.evaluate_state(state, &opponent_state); // skip value

        for &mark in &marks {
            let mut s = *state;
            s.apply_mark(mark);
            let value = self.evaluate_state(&s, &opponent_state);
            if value > best_value {
                best_value = value;
                best_mark = Some(mark);
            }
        }

        best_mark
    }
}
