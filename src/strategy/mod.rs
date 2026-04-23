mod bot_impl;

use crate::state::{format_dice, Mark, State};
use rand::Rng;

const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";

pub trait Strategy: std::fmt::Debug {
    fn active_phase1(&mut self, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark>;
    fn active_phase2(&mut self, state: &State, opp_states: &[State], dice: [u8; 6], has_marked: bool) -> Option<Mark>;
    fn passive_phase1(&mut self, state: &State, opp_states: &[State], dice: [u8; 6], active_player: usize) -> Option<Mark>;
    fn is_interactive(&self) -> bool {
        false
    }
}

pub trait Bot: std::fmt::Debug {
    fn evaluate(&self, our_state: &State, opp_states: &[State]) -> f32;
    fn evaluate_batch(&self, candidates: &[State], opp_states: &[State]) -> Vec<f32> {
        candidates.iter().map(|s| self.evaluate(s, opp_states)).collect()
    }
}

// ---- Interactive ----

#[derive(Debug, Default, Clone)]
pub struct Interactive;

impl Interactive {
    fn read_line() -> String {
        use std::io::{self, Write};
        print!("  {BOLD}> {RESET}");
        io::stdout().flush().unwrap();
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        input
    }

    fn show_marks(marks: &[Mark]) {
        let w = if marks.len() >= 10 { 2 } else { 1 };
        for (i, mark) in marks.iter().enumerate() {
            print!(" {DIM}{:>w$}){RESET} {mark}", i + 1);
        }
        println!();
    }

    fn pick_mark(marks: &[Mark], allow_skip: bool) -> Option<Mark> {
        if allow_skip {
            println!("\n  {DIM}Enter number or Enter to skip:{RESET}");
        } else {
            println!("\n  {DIM}Enter number:{RESET}");
        }
        loop {
            let input = Self::read_line();
            let input = input.trim();
            if input.is_empty() && allow_skip {
                println!("  {DIM}Skipped.{RESET}");
                return None;
            }
            if let Ok(n) = input.parse::<usize>() {
                if n >= 1 && n <= marks.len() {
                    return Some(marks[n - 1]);
                }
            }
            if allow_skip {
                println!("  \x1b[91mInvalid. Enter 1-{} or Enter to skip.{RESET}", marks.len());
            } else {
                println!("  \x1b[91mInvalid. Enter 1-{}.{RESET}", marks.len());
            }
        }
    }
}

impl Strategy for Interactive {
    fn active_phase1(&mut self, state: &State, _opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
        println!("\n  {BOLD}=== YOUR TURN (Phase 1: White Dice) ==={RESET}\n");
        println!("{state}");
        println!("{}", format_dice(dice));
        println!();

        let white_sum = dice[0] + dice[1];
        let marks = state.generate_white_moves(white_sum);
        if marks.is_empty() {
            println!("  {DIM}No white-sum marks available.{RESET}");
            return None;
        }
        print!("  {DIM}White sum {white_sum}:{RESET}");
        Self::show_marks(&marks);
        Self::pick_mark(&marks, true)
    }

    fn active_phase2(&mut self, state: &State, _opp_states: &[State], dice: [u8; 6], has_marked: bool) -> Option<Mark> {
        println!("\n  {BOLD}=== YOUR TURN (Phase 2: Color Dice) ==={RESET}\n");
        println!("{state}");
        if !has_marked {
            println!("  {DIM}(No mark in Phase 1 -- you must mark here or take a strike){RESET}");
        }

        let marks = state.generate_color_moves(dice);
        if marks.is_empty() {
            if !has_marked {
                println!("  {DIM}No color marks available -- strike.{RESET}");
            } else {
                println!("  {DIM}No color marks available.{RESET}");
            }
            return None;
        }
        print!("  {DIM}Color marks:{RESET}");
        Self::show_marks(&marks);
        Self::pick_mark(&marks, has_marked)
    }

    fn passive_phase1(&mut self, state: &State, _opp_states: &[State], dice: [u8; 6], _active_player: usize) -> Option<Mark> {
        let white_sum = dice[0] + dice[1];
        println!("\n  {BOLD}=== OPPONENT'S TURN (White Dice) ==={RESET}\n");
        println!("{state}");
        println!("  White sum: {BOLD}{white_sum}{RESET}");
        println!();

        let marks = state.generate_white_moves(white_sum);
        if marks.is_empty() {
            println!("  {DIM}No marks available.{RESET}");
            return None;
        }
        print!("  {DIM}Mark:{RESET}");
        Self::show_marks(&marks);
        Self::pick_mark(&marks, true)
    }

    fn is_interactive(&self) -> bool {
        true
    }
}

// ---- Conservative ----

/// Conservative strategy: only marks numbers that don't skip too many.
/// Picks the move creating the fewest new blanks, rejecting any that
/// would add more than `max_new_blanks`. Among moves with equal blanks,
/// picks the one with the highest resulting score.
#[derive(Debug, Clone)]
pub struct Conservative {
    max_new_blanks: u8,
}

impl Default for Conservative {
    fn default() -> Self {
        Self { max_new_blanks: 3 }
    }
}

impl Conservative {
    fn best_mark(state: &State, marks: &[Mark], max_new_blanks: u8) -> Option<Mark> {
        let current_blanks = state.blanks();
        marks
            .iter()
            .filter_map(|&mark| {
                let mut new_state = *state;
                new_state.apply_mark(mark);
                let new_blanks = new_state.blanks().saturating_sub(current_blanks);
                if new_blanks <= max_new_blanks {
                    Some((mark, new_blanks, new_state.count_points()))
                } else {
                    None
                }
            })
            // Minimize blanks first, then maximize resulting score
            .min_by_key(|(_, blanks, score)| (*blanks, -*score))
            .map(|(mark, _, _)| mark)
    }
}

impl Strategy for Conservative {
    fn active_phase1(&mut self, state: &State, _opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
        let white_sum = dice[0] + dice[1];
        let marks = state.generate_white_moves(white_sum);
        Self::best_mark(state, &marks, self.max_new_blanks)
    }

    fn active_phase2(&mut self, state: &State, _opp_states: &[State], dice: [u8; 6], has_marked: bool) -> Option<Mark> {
        let marks = state.generate_color_moves(dice);
        let cap = if has_marked { self.max_new_blanks } else { self.max_new_blanks };
        Self::best_mark(state, &marks, cap)
    }

    fn passive_phase1(&mut self, state: &State, _opp_states: &[State], dice: [u8; 6], _active_player: usize) -> Option<Mark> {
        let white_sum = dice[0] + dice[1];
        let marks = state.generate_white_moves(white_sum);
        Self::best_mark(state, &marks, 0)
    }
}

// ---- Opportunist ----

/// Opportunist strategy: optimizes for probability (keeping free pointers
/// on common 2d6 sums like 6/7/8) to maximize marks on opponent turns.
#[derive(Debug, Default, Clone)]
pub struct Opportunist;

impl Opportunist {
    /// Filter marks to those within a blank budget, then pick by best probability.
    fn best_mark(state: &State, marks: &[Mark], max_new_blanks: u8) -> Option<Mark> {
        let current_blanks = state.blanks();
        marks
            .iter()
            .copied()
            .filter_map(|mark| {
                let mut new_state = *state;
                new_state.apply_mark(mark);
                let new_blanks = new_state.blanks().saturating_sub(current_blanks);
                if new_blanks <= max_new_blanks {
                    Some((mark, new_state.probability()))
                } else {
                    None
                }
            })
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(mark, _)| mark)
    }
}

impl Strategy for Opportunist {
    fn active_phase1(&mut self, state: &State, _opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
        let white_sum = dice[0] + dice[1];
        let marks = state.generate_white_moves(white_sum);
        Self::best_mark(state, &marks, 2)
    }

    fn active_phase2(&mut self, state: &State, _opp_states: &[State], dice: [u8; 6], _has_marked: bool) -> Option<Mark> {
        let marks = state.generate_color_moves(dice);
        Self::best_mark(state, &marks, 2)
    }

    fn passive_phase1(&mut self, state: &State, _opp_states: &[State], dice: [u8; 6], _active_player: usize) -> Option<Mark> {
        let white_sum = dice[0] + dice[1];
        let marks = state.generate_white_moves(white_sum);
        Self::best_mark(state, &marks, 1)
    }
}

// ---- Random ----

#[derive(Debug, Default, Clone)]
pub struct Random;

impl Strategy for Random {
    fn active_phase1(&mut self, state: &State, _opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
        let white_sum = dice[0] + dice[1];
        let marks = state.generate_white_moves(white_sum);
        if !marks.is_empty() {
            let mut rng = rand::thread_rng();
            if rng.gen_bool(0.5) {
                Some(marks[rng.gen_range(0..marks.len())])
            } else {
                None
            }
        } else {
            None
        }
    }

    fn active_phase2(&mut self, state: &State, _opp_states: &[State], dice: [u8; 6], has_marked: bool) -> Option<Mark> {
        let marks = state.generate_color_moves(dice);
        if !marks.is_empty() {
            let mut rng = rand::thread_rng();
            if has_marked {
                // 50% mark random, 50% skip
                if rng.gen_bool(0.5) {
                    Some(marks[rng.gen_range(0..marks.len())])
                } else {
                    None
                }
            } else {
                // Must mark to avoid strike
                Some(marks[rng.gen_range(0..marks.len())])
            }
        } else {
            None
        }
    }

    fn passive_phase1(&mut self, state: &State, _opp_states: &[State], dice: [u8; 6], _active_player: usize) -> Option<Mark> {
        let white_sum = dice[0] + dice[1];
        let marks = state.generate_white_moves(white_sum);
        if !marks.is_empty() {
            let mut rng = rand::thread_rng();
            if rng.gen_bool(0.5) {
                Some(marks[rng.gen_range(0..marks.len())])
            } else {
                None
            }
        } else {
            None
        }
    }
}
