use crate::state::State;
use crate::strategy::Strategy;

use rand::{prelude::*, rngs::SmallRng, Rng};

#[derive(Debug)]
pub struct Player {
    strategy: Box<dyn Strategy>,
    pub state: State,
}

impl Player {
    fn your_move(&mut self, dice: [u8; 6]) {
        let mov = self.strategy.your_move(&self.state, dice);
        self.state.apply_move(mov);
    }

    fn opponents_move(&mut self, number: u8, locked: [bool; 4]) {
        if let Some(mov) = self.strategy.opponents_move(&self.state, number, locked) {
            self.state.apply_move(mov);
        }
        self.state.lock(locked);
    }
}

#[derive(Debug)]
pub struct Game {
    pub players: Vec<Player>,
    rng: SmallRng,
}

impl Game {
    pub fn new(players: Vec<Box<dyn Strategy>>) -> Self {
        Self {
            players: players
                .into_iter()
                .map(|strategy| Player {
                    strategy,
                    state: State::default(),
                })
                .collect(),
            rng: SmallRng::from_entropy(),
        }
    }

    pub fn play(&mut self) {
        let mut active_player = 0;
        while !self.game_over() {
            let dice: [u8; 6] = core::array::from_fn(|_| self.rng.gen_range(1..=6));
            let on_white = dice[0] + dice[1];

            self.players[active_player].your_move(dice);
            let mut new_locked = self.players[active_player].state.locked();

            for index in 1..self.players.len() {
                let opponent = (active_player + index) % self.players.len();
                self.players[opponent].opponents_move(on_white, new_locked);
                new_locked = self.players[opponent].state.locked();
            }

            for player in self.players.iter_mut() {
                player.state.lock(new_locked);
            }

            // for index in 0..self.players.len() {
            //     for player in self.players.iter_mut() {
            //         player.strategy.observe(index, &self.players[index].state);
            //     }
            // }

            active_player = (active_player + 1) % self.players.len();
        }
    }

    fn game_over(&self) -> bool {
        let max_strikes = self
            .players
            .iter()
            .map(|player| player.state.strikes)
            .max()
            .unwrap();

        let total_locked = self
            .players
            .iter()
            .map(|player| player.state.count_locked())
            .max()
            .unwrap();

        max_strikes == 4 || total_locked >= 2
    }
}
