use crate::state::{format_dice, Move, State};
use rand::Rng;

const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";

pub trait Strategy: std::fmt::Debug {
    fn your_move(&mut self, state: &State, dice: [u8; 6]) -> Move;
    fn opponents_move(&mut self, state: &State, number: u8, locked: [bool; 4]) -> Option<Move>;
    fn is_interactive(&self) -> bool {
        false
    }
    /// Called before each decision with our score and the other players' states.
    /// Default: no-op. Override to use opponent information.
    fn observe_opponents(&mut self, _our_score: isize, _opponents: &[State]) {}
}

#[derive(Debug, Default, Clone)]
pub struct Interactive;

impl Interactive {
    fn show_moves(moves: &[Move]) {
        let singles: Vec<_> = moves
            .iter()
            .enumerate()
            .filter(|(_, m)| matches!(m, Move::Single(_)))
            .collect();
        let doubles: Vec<_> = moves
            .iter()
            .enumerate()
            .filter(|(_, m)| matches!(m, Move::Double(_, _)))
            .collect();

        let w = if moves.len() >= 10 { 2 } else { 1 };

        if !singles.is_empty() {
            print!("  {DIM}Single:{RESET} ");
            for (i, mov) in &singles {
                print!(" {DIM}{:>w$}){RESET} {mov}", i + 1);
            }
            println!();
        }
        if !doubles.is_empty() {
            print!("  {DIM}Both:{RESET}   ");
            let per_line = 3;
            for (j, (i, mov)) in doubles.iter().enumerate() {
                if j > 0 && j % per_line == 0 {
                    print!("\n          ");
                }
                print!(" {DIM}{:>w$}){RESET} {mov}", i + 1);
            }
            println!();
        }
        println!("  {DIM}{:>w$}) {BOLD}\x1b[91m✗ Strike{RESET}", "S");
    }

    fn read_line() -> String {
        use std::io::{self, Write};
        print!("  {BOLD}> {RESET}");
        io::stdout().flush().unwrap();
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        input
    }

    fn pick_move(moves: &[Move]) -> Move {
        println!("\n  {DIM}Enter number or s for strike:{RESET}");
        loop {
            let input = Self::read_line();
            let input = input.trim();
            if input.eq_ignore_ascii_case("s") {
                return Move::Strike;
            }
            if let Ok(n) = input.parse::<usize>() {
                if n >= 1 && n <= moves.len() {
                    if let Some(mov) = moves.get(n - 1) {
                        return *mov;
                    }
                }
            }
            println!("  \x1b[91mInvalid. Enter a number 1-{} or s.{RESET}", moves.len());
        }
    }

    fn pick_opponent_move(moves: &[Move]) -> Option<Move> {
        println!("\n  {DIM}Enter number or Enter to skip:{RESET}");
        loop {
            let input = Self::read_line();
            let input = input.trim();
            if input.is_empty() {
                println!("  {DIM}Skipped.{RESET}");
                return None;
            }
            if let Ok(n) = input.parse::<usize>() {
                if n >= 1 && n <= moves.len() {
                    if let Some(mov) = moves.get(n - 1) {
                        return Some(*mov);
                    }
                }
            }
            println!("  \x1b[91mInvalid. Enter 1-{} or Enter to skip.{RESET}", moves.len());
        }
    }
}

impl Strategy for Interactive {
    fn your_move(&mut self, state: &State, dice: [u8; 6]) -> Move {
        println!("\n  {BOLD}═══ YOUR TURN ═══{RESET}\n");
        println!("{state}");
        println!("{}", format_dice(dice));
        println!();

        let moves = state.generate_moves(dice);
        if moves.is_empty() {
            println!("  {DIM}No moves available — strike.{RESET}");
            return Move::Strike;
        }
        Self::show_moves(&moves);
        Self::pick_move(&moves)
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

        // Opponent moves are always singles, show compact
        print!("  {DIM}Mark:{RESET} ");
        for (i, mov) in moves.iter().enumerate() {
            print!(" {DIM}{}){RESET} {mov}", i + 1);
        }
        println!();

        Self::pick_opponent_move(&moves)
    }

    fn is_interactive(&self) -> bool {
        true
    }
}

/// Conservative strategy: only marks numbers that don't skip too many.
/// Picks the move creating the fewest new blanks, rejecting any that
/// would add more than `max_new_blanks`. Among moves with equal blanks,
/// picks the one with the highest resulting score.
#[derive(Debug, Clone)]
pub struct Conservative {
    max_new_blanks: u8,
    score_gap: isize,
}

impl Default for Conservative {
    fn default() -> Self {
        Self { max_new_blanks: 3, score_gap: 0 }
    }
}

impl Conservative {
    fn best_move(state: &State, moves: &[Move], max_new_blanks: u8) -> Option<Move> {
        let current_blanks = state.blanks();
        moves
            .iter()
            .filter_map(|&mov| {
                let mut new_state = *state;
                new_state.apply_move(mov);
                let new_blanks = new_state.blanks().saturating_sub(current_blanks);
                if new_blanks <= max_new_blanks {
                    Some((mov, new_blanks, new_state.count_points()))
                } else {
                    None
                }
            })
            // Minimize blanks first, then maximize resulting score
            .min_by_key(|(_, blanks, score)| (*blanks, -*score))
            .map(|(mov, _, _)| mov)
    }
}

impl Strategy for Conservative {
    fn your_move(&mut self, state: &State, dice: [u8; 6]) -> Move {
        use crate::state::MetaDecision;
        let moves = match state.apply_meta_rules(dice, self.score_gap) {
            MetaDecision::Forced(mov) => return mov,
            MetaDecision::Choices(moves) => moves,
        };
        // Evaluate only marking moves; Strike is the fallback
        let marks: Vec<Move> = moves.into_iter().filter(|m| !matches!(m, Move::Strike)).collect();
        Self::best_move(state, &marks, self.max_new_blanks).unwrap_or(Move::Strike)
    }

    fn opponents_move(&mut self, state: &State, number: u8, _locked: [bool; 4]) -> Option<Move> {
        let moves = state.generate_opponent_moves(number);
        if let Some(mov) = state.find_smart_lock(&moves, self.score_gap) {
            return Some(mov);
        }
        Self::best_move(state, &moves, 0)
    }

    fn observe_opponents(&mut self, our_score: isize, opponents: &[State]) {
        let max_opp = opponents.iter().map(|s| s.count_points()).max().unwrap_or(0);
        self.score_gap = our_score - max_opp;
    }
}

/// Opportunist strategy: optimizes for probability (keeping free pointers
/// on common 2d6 sums like 6/7/8) to maximize marks on opponent turns.
#[derive(Debug, Default, Clone)]
pub struct Opportunist {
    score_gap: isize,
}

impl Opportunist {
    /// Filter moves to those within a blank budget, then pick by best probability.
    fn best_move(state: &State, moves: &[Move], max_new_blanks: u8) -> Option<Move> {
        let current_blanks = state.blanks();
        moves
            .iter()
            .copied()
            .filter_map(|mov| {
                let mut new_state = *state;
                new_state.apply_move(mov);
                let new_blanks = new_state.blanks().saturating_sub(current_blanks);
                if new_blanks <= max_new_blanks {
                    Some((mov, new_state.probability()))
                } else {
                    None
                }
            })
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(mov, _)| mov)
    }
}

impl Strategy for Opportunist {
    fn your_move(&mut self, state: &State, dice: [u8; 6]) -> Move {
        use crate::state::MetaDecision;
        let moves = match state.apply_meta_rules(dice, self.score_gap) {
            MetaDecision::Forced(mov) => return mov,
            MetaDecision::Choices(moves) => moves,
        };
        // Evaluate only marking moves; Strike is the fallback
        let marks: Vec<Move> = moves.into_iter().filter(|m| !matches!(m, Move::Strike)).collect();
        // Allow up to 2 blanks on active turn, pick by probability
        Self::best_move(state, &marks, 2).unwrap_or(Move::Strike)
    }

    fn opponents_move(&mut self, state: &State, number: u8, _locked: [bool; 4]) -> Option<Move> {
        let moves = state.generate_opponent_moves(number);
        if let Some(mov) = state.find_smart_lock(&moves, self.score_gap) {
            return Some(mov);
        }
        // Allow up to 1 blank on opponent turn, pick by probability
        Self::best_move(state, &moves, 1)
    }

    fn observe_opponents(&mut self, our_score: isize, opponents: &[State]) {
        let max_opp = opponents.iter().map(|s| s.count_points()).max().unwrap_or(0);
        self.score_gap = our_score - max_opp;
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
