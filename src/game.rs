use crate::state::State;
use crate::strategy::Strategy;

use rand::{prelude::*, rngs::SmallRng, Rng};

pub trait DiceSource: std::fmt::Debug {
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
pub struct Player {
    strategy: Box<dyn Strategy>,
    pub state: State,
    dice: Box<dyn DiceSource>,
}

impl Player {
    pub fn new(strategy: Box<dyn Strategy>, dice: Box<dyn DiceSource>) -> Self {
        Self {
            strategy,
            state: State::default(),
            dice,
        }
    }

    pub fn new_with_state(
        strategy: Box<dyn Strategy>,
        dice: Box<dyn DiceSource>,
        state: State,
    ) -> Self {
        Self {
            strategy,
            state,
            dice,
        }
    }

    fn roll(&mut self) -> [u8; 6] {
        self.dice.roll()
    }

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
    start_turn: Option<usize>,
}

impl Game {
    pub fn new(players: Vec<Player>) -> Self {
        Self { players, start_turn: None }
    }

    pub fn new_from_turn(players: Vec<Player>, start_turn: usize) -> Self {
        Self { players, start_turn: Some(start_turn) }
    }

    pub fn play(&mut self) {
        let mut active_player = self.start_turn.take().unwrap_or(0);
        while !self.game_over() {
            let dice = self.players[active_player].roll();
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
        self.game_over_reason().is_some()
    }

    pub fn game_over_reason(&self) -> Option<GameOverReason> {
        for (i, player) in self.players.iter().enumerate() {
            if player.state.strikes >= 4 {
                return Some(GameOverReason::Strikes(i));
            }
        }

        let total_locked = self
            .players
            .iter()
            .map(|player| player.state.count_locked())
            .max()
            .unwrap();

        if total_locked >= 2 {
            return Some(GameOverReason::TwoRowsLocked);
        }

        None
    }

    pub fn print_game_over(&self) {
        const BOLD: &str = "\x1b[1m";
        const DIM: &str = "\x1b[2m";
        const RESET: &str = "\x1b[0m";
        const YELLOW: &str = "\x1b[93m";
        const RED: &str = "\x1b[91m";

        println!("\n  {BOLD}═══════════════════════════════════════════════{RESET}");
        println!("  {BOLD}                  GAME OVER{RESET}");
        println!("  {BOLD}═══════════════════════════════════════════════{RESET}\n");

        // Reason
        match self.game_over_reason() {
            Some(GameOverReason::Strikes(i)) => {
                println!("  {RED}Player {} reached 4 strikes!{RESET}\n", i + 1);
            }
            Some(GameOverReason::TwoRowsLocked) => {
                println!("  Two rows have been locked!\n");
            }
            None => {}
        }

        // Collect scores for ranking
        let mut scores: Vec<(usize, isize)> = self
            .players
            .iter()
            .enumerate()
            .map(|(i, p)| (i, p.state.count_points()))
            .collect();
        scores.sort_by(|a, b| b.1.cmp(&a.1));

        // Print each player's board
        for (i, player) in self.players.iter().enumerate() {
            let rank = scores.iter().position(|(idx, _)| *idx == i).unwrap();
            let label = if rank == 0 {
                format!("{YELLOW}{BOLD}Player {} (Winner){RESET}", i + 1)
            } else {
                format!("{BOLD}Player {}{RESET}", i + 1)
            };
            println!("  {DIM}───{RESET} {label} {DIM}───{RESET}\n");
            println!("{}", player.state);
            println!();
        }

        // Final ranking
        println!("  {BOLD}Final Ranking{RESET}");
        println!("  {DIM}──────────────────────{RESET}");
        for (rank, (i, score)) in scores.iter().enumerate() {
            let medal = match rank {
                0 => format!("{YELLOW}{BOLD}1st{RESET}"),
                1 => format!("{DIM}2nd{RESET}"),
                2 => format!("{DIM}3rd{RESET}"),
                _ => format!("{DIM}{}th{RESET}", rank + 1),
            };
            println!("  {medal}  Player {}  {BOLD}{score}{RESET} pts", i + 1);
        }
        println!("  {DIM}──────────────────────{RESET}");
    }
}

pub enum GameOverReason {
    Strikes(usize),
    TwoRowsLocked,
}
