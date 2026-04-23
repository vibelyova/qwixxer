use crate::state::{Mark, State};
use crate::strategy::Strategy;

use rand::{rngs::SmallRng, Rng};

pub trait DiceSource: std::fmt::Debug {
    fn roll(&mut self) -> [u8; 6];
}

impl DiceSource for SmallRng {
    fn roll(&mut self) -> [u8; 6] {
        core::array::from_fn(|_| self.gen_range(1..=6))
    }
}

#[derive(Debug)]
#[allow(dead_code)]
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

    pub fn is_interactive(&self) -> bool {
        self.strategy.is_interactive()
    }

    fn roll(&mut self) -> [u8; 6] {
        self.dice.roll()
    }

    fn active_phase1(&mut self, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
        self.strategy.active_phase1(&self.state, opp_states, dice)
    }

    fn active_phase2(&mut self, opp_states: &[State], dice: [u8; 6], has_marked: bool) -> Option<Mark> {
        self.strategy.active_phase2(&self.state, opp_states, dice, has_marked)
    }

    fn passive_phase1(&mut self, opp_states: &[State], dice: [u8; 6], active_player: usize) -> Option<Mark> {
        self.strategy.passive_phase1(&self.state, opp_states, dice, active_player)
    }

    fn apply_mark(&mut self, mark: Mark) {
        self.state.apply_mark(mark);
    }

    fn apply_strike(&mut self) {
        self.state.apply_strike();
    }
}

#[derive(Debug)]
pub struct Game {
    pub players: Vec<Player>,
    start_turn: Option<usize>,
    pub verbose: bool,
}

impl Game {
    pub fn new(players: Vec<Player>) -> Self {
        Self {
            players,
            start_turn: None,
            verbose: false,
        }
    }

    pub fn new_from_turn(players: Vec<Player>, start_turn: usize) -> Self {
        Self {
            players,
            start_turn: Some(start_turn),
            verbose: false,
        }
    }

    fn aggregate_locks(&self) -> [bool; 4] {
        let mut locked = [false; 4];
        for player in &self.players {
            let player_locked = player.state.locked();
            for row in 0..4 {
                locked[row] |= player_locked[row];
            }
        }
        locked
    }

    pub fn play(&mut self) {
        let n = self.players.len();
        let mut active_player = self.start_turn.take().unwrap_or(0);

        while !self.game_over() {
            let dice = self.players[active_player].roll();

            let verbose_active = self.verbose && !self.players[active_player].is_interactive();

            if verbose_active {
                println!(
                    "\n  \x1b[2m-- Player {} active turn --\x1b[0m",
                    active_player + 1
                );
                println!("  \x1b[2m{}\x1b[0m", crate::state::format_dice(dice));
            }

            // Snapshot all player states before phase 1
            let pre_phase1: Vec<State> = self.players.iter().map(|p| p.state).collect();

            // Phase 1: collect all decisions simultaneously
            let mut phase1_marks: Vec<Option<Mark>> = Vec::with_capacity(n);
            for i in 0..n {
                // Build opp_states: turn-ordered from player i's perspective
                let opp_states: Vec<State> = (1..n)
                    .map(|off| pre_phase1[(i + off) % n])
                    .collect();

                let mark = if i == active_player {
                    self.players[i].active_phase1(&opp_states, dice)
                } else {
                    // active_idx from this player's perspective:
                    // active_player is at offset (active_player + n - i) % n from i,
                    // but in the opp_states array it's 0-indexed. The active player
                    // is at position (active_player + n - i) % n - 1 in opp_states.
                    // Actually, the spec says: active_idx = (active + n - i - 1) % n
                    // This is supposed to be the index of the active player within opp_states.
                    // opp_states[0] = pre_phase1[(i+1)%n], opp_states[1] = pre_phase1[(i+2)%n], etc.
                    // active_player is at position off where (i+off)%n == active_player
                    // off = (active_player + n - i) % n
                    // In opp_states, that's at index off - 1 (since off starts at 1)
                    let active_idx = (active_player + n - i) % n - 1;
                    self.players[i].passive_phase1(&opp_states, dice, active_idx)
                };
                phase1_marks.push(mark);
            }

            // Apply all phase1 marks
            let mut has_active_marked = false;
            for (i, mark) in phase1_marks.iter().enumerate() {
                if let Some(m) = mark {
                    self.players[i].apply_mark(*m);
                    if i == active_player {
                        has_active_marked = true;
                    }
                    if self.verbose && !self.players[i].is_interactive() {
                        println!(
                            "  \x1b[2mPlayer {} Phase 1: marked {m}\x1b[0m",
                            i + 1
                        );
                    }
                } else if self.verbose && !self.players[i].is_interactive() {
                    println!(
                        "  \x1b[2mPlayer {} Phase 1: skipped\x1b[0m",
                        i + 1
                    );
                }
            }

            // Propagate phase1 locks
            let locked = self.aggregate_locks();
            for player in self.players.iter_mut() {
                player.state.lock(locked);
            }

            // Game-over check after phase1
            if self.game_over() {
                break;
            }

            // Phase 2: active player only
            let opp_states: Vec<State> = (1..n)
                .map(|off| self.players[(active_player + off) % n].state)
                .collect();
            let phase2 = self.players[active_player].active_phase2(&opp_states, dice, has_active_marked);

            match phase2 {
                Some(m) => {
                    self.players[active_player].apply_mark(m);
                    has_active_marked = true;
                    if verbose_active {
                        println!(
                            "  \x1b[2mPlayer {} Phase 2: marked {m}\x1b[0m",
                            active_player + 1
                        );
                    }
                }
                None => {
                    if !has_active_marked {
                        self.players[active_player].apply_strike();
                        if verbose_active {
                            println!(
                                "  \x1b[2mPlayer {} Phase 2: strike\x1b[0m",
                                active_player + 1
                            );
                        }
                    } else if verbose_active {
                        println!(
                            "  \x1b[2mPlayer {} Phase 2: skipped\x1b[0m",
                            active_player + 1
                        );
                    }
                }
            }

            // Propagate phase2 locks
            let locked = self.aggregate_locks();
            for player in self.players.iter_mut() {
                player.state.lock(locked);
            }

            if verbose_active {
                println!("{}", self.players[active_player].state);
            }

            // Suppress unused variable warning
            let _ = has_active_marked;

            active_player = (active_player + 1) % n;
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

        println!("\n  {BOLD}==============================================={RESET}");
        println!("  {BOLD}                  GAME OVER{RESET}");
        println!("  {BOLD}==============================================={RESET}\n");

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
            println!("  {DIM}---{RESET} {label} {DIM}---{RESET}\n");
            println!("{}", player.state);
            println!();
        }

        // Final ranking
        println!("  {BOLD}Final Ranking{RESET}");
        println!("  {DIM}----------------------{RESET}");
        for (rank, (i, score)) in scores.iter().enumerate() {
            let medal = match rank {
                0 => format!("{YELLOW}{BOLD}1st{RESET}"),
                1 => format!("{DIM}2nd{RESET}"),
                2 => format!("{DIM}3rd{RESET}"),
                _ => format!("{DIM}{}th{RESET}", rank + 1),
            };
            println!("  {medal}  Player {}  {BOLD}{score}{RESET} pts", i + 1);
        }
        println!("  {DIM}----------------------{RESET}");
    }
}

pub enum GameOverReason {
    Strikes(usize),
    TwoRowsLocked,
}
