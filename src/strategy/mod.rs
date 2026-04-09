use crate::state::{format_dice, Move, State};
use rand::Rng;

const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";

pub trait Strategy: std::fmt::Debug {
    fn your_move(&mut self, state: &State, dice: [u8; 6]) -> Move;
    fn opponents_move(&mut self, state: &State, number: u8, locked: [bool; 4]) -> Option<Move>;
    // fn observe(&mut self, _player: usize, _state: &State) {}
}

#[derive(Debug, Default, Clone)]
pub struct Interactive;

impl Interactive {
    fn show_moves(moves: &[Move]) {
        println!("  {DIM}Available moves:{RESET}");
        for (i, mov) in moves.iter().enumerate() {
            println!("    {DIM}{:>2}){RESET} {mov}", i + 1);
        }
    }

    fn read_move() -> String {
        use std::io::{self, Write};
        print!("  {BOLD}> {RESET}");
        io::stdout().flush().unwrap();
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        input
    }
}

impl Strategy for Interactive {
    fn your_move(&mut self, state: &State, dice: [u8; 6]) -> Move {
        println!("\n  {BOLD}═══ YOUR TURN ═══{RESET}\n");
        println!("{state}");
        println!("{}", format_dice(dice));
        println!();

        let mut moves = state.generate_moves(dice);
        moves.push(Move::Strike);
        Self::show_moves(&moves);

        println!("\n  {DIM}Enter move (e.g. r5, r5 g10, strike):{RESET}");
        loop {
            let input = Self::read_move();
            match input.trim().parse::<Move>() {
                Ok(mov) => return mov,
                Err(e) => println!("  {}\x1b[91m{e}{RESET}", ""),
            }
        }
    }

    fn opponents_move(&mut self, state: &State, number: u8, locked: [bool; 4]) -> Option<Move> {
        println!("\n  {BOLD}═══ OPPONENT'S TURN ═══{RESET}\n");
        println!("{state}");
        println!(
            "  White sum: {BOLD}{number}{RESET}    {DIM}Locked: [{}{}{}{}{DIM}]{RESET}",
            if locked[0] { "\x1b[91mR " } else { "· " },
            if locked[1] { "\x1b[93mY " } else { "· " },
            if locked[2] { "\x1b[92mG " } else { "· " },
            if locked[3] { "\x1b[94mB " } else { "· " },
        );
        println!();

        let moves = state.generate_opponent_moves(number);
        if moves.is_empty() {
            println!("  {DIM}No moves available.{RESET}");
            return None;
        }

        Self::show_moves(&moves);
        println!("\n  {DIM}Enter move or press Enter to skip:{RESET}");

        let input = Self::read_move();
        if input.trim().is_empty() {
            println!("  {DIM}Skipped.{RESET}");
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
