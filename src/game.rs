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

trait DiceSource: std::fmt::Debug {
    fn roll(&mut self) -> [u8; 6];
}

impl DiceSource for SmallRng {
    fn roll(&mut self) -> [u8; 6] {
        core::array::from_fn(|_| self.gen_range(1..=6))
    }
}

#[derive(Debug)]
pub struct ManualDice;

impl DiceSource for ManualDice {
    fn roll(&mut self) -> [u8; 6] {
        use std::io::{self, Write};
        print!("Enter dice WWRYGB: ");
        io::stdout().flush().unwrap();
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        input
            .trim()
            .chars()
            .map(|ch| ch.to_digit(10).unwrap() as u8)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }
}

#[derive(Debug)]
pub struct Game {
    pub players: Vec<Player>,
    dice: Box<dyn DiceSource>,
}

impl Game {
    pub fn new_rng(players: Vec<Box<dyn Strategy>>) -> Self {
        Self {
            players: players
                .into_iter()
                .map(|strategy| Player {
                    strategy,
                    state: State::default(),
                })
                .collect(),
            dice: Box::new(SmallRng::from_entropy()),
        }
    }

    pub fn new_manual(players: Vec<Box<dyn Strategy>>) -> Self {
        Self {
            players: players
                .into_iter()
                .map(|strategy| Player {
                    strategy,
                    state: State::default(),
                })
                .collect(),
            dice: Box::new(ManualDice),
        }
    }

    pub fn play(&mut self) {
        let mut active_player = 0;
        while !self.game_over() {
            let dice = self.dice.roll();
            let on_white = dice[0] + dice[1];

            self.players[active_player].your_move(dice);
            let mut new_locked = self.players[active_player].state.locked();

            for index in 1..self.players.len() {
                let opponent = (active_player + index) % self.players.len();
                self.players[opponent].opponents_move(on_white, new_locked);

                for (row, locked) in self.players[opponent].state.locked().iter().enumerate() {
                    new_locked[row] |= *locked;
                }
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
