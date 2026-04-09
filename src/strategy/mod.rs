use crate::state::{Move, State};
use rand::Rng;

pub trait Strategy: std::fmt::Debug {
    fn your_move(&mut self, state: &State, dice: [u8; 6]) -> Move;
    fn opponents_move(&mut self, state: &State, number: u8, locked: [bool; 4]) -> Option<Move>;
    // fn observe(&mut self, _player: usize, _state: &State) {}
}

#[derive(Debug, Default, Clone)]
pub struct Interactive;

impl Strategy for Interactive {
    fn your_move(&mut self, state: &State, dice: [u8; 6]) -> Move {
        println!("Your turn!");
        println!("Current state:");
        println!("{:?}", state);
        println!("Dice: {:?}", dice);
        println!("Enter your move:");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
        input.trim().parse().unwrap()
    }

    fn opponents_move(&mut self, state: &State, number: u8, locked: [bool; 4]) -> Option<Move> {
        println!("Opponent's turn!");
        println!("Current state:");
        println!("{:?}", state);
        println!("Opponent rolled a {}", number);
        println!("Opponent locked: {:?}", locked);
        println!("Enter your move or skip:");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
        if input.trim().is_empty() {
            println!("Skipped");
            return None;
        }
        input.trim().parse().ok()
    }
}

// TODO
#[derive(Debug, Default, Clone)]
pub struct Conservative;

impl Strategy for Conservative {
    fn your_move(&mut self, state: &State, dice: [u8; 6]) -> Move {
        Move::Strike
    }

    fn opponents_move(&mut self, state: &State, number: u8, _locked: [bool; 4]) -> Option<Move> {
        None
    }
}

#[derive(Debug, Default, Clone)]
pub struct Random;

impl Strategy for Random {
    fn your_move(&mut self, state: &State, dice: [u8; 6]) -> Move {
        let moves = state.generate_moves(dice);
        if !moves.is_empty() {
            let mut rng = rand::thread_rng();
            moves[rng.gen_range(0..moves.len())]
        } else {
            Move::Strike
        }
    }

    fn opponents_move(&mut self, state: &State, number: u8, _locked: [bool; 4]) -> Option<Move> {
        let moves = state.generate_opponent_moves(number);
        if !moves.is_empty() {
            let mut rng = rand::thread_rng();
            Some(moves[rng.gen_range(0..moves.len())])
        } else {
            None
        }
    }
}
