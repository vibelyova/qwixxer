use itertools::Itertools;
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub struct State {
    pub strikes: u8,
    rows: [Row; 4],
}

#[derive(Debug, Clone, Copy, Default)]
struct Row {
    ascending: bool,
    total: u8,
    // None if locked, otherwise next free number
    free: Option<u8>,
}

impl Default for State {
    fn default() -> Self {
        let ascending = Row {
            ascending: true,
            total: 0,
            free: Some(2),
        };
        let descending = Row {
            ascending: false,
            total: 0,
            free: Some(12),
        };
        Self {
            strikes: 0,
            rows: [ascending, ascending, descending, descending],
        }
    }
}

impl State {
    pub fn apply_move(&mut self, mov: impl Into<Move>) {
        match mov.into() {
            Move::Strike => {
                assert!(self.strikes < 4);
                self.strikes += 1;
            }
            Move::Single(mark) => {
                self.rows[mark.row].mark(mark.number);
            }
            Move::Double(mark1, mark2) => {
                self.rows[mark1.row].mark(mark1.number);
                self.rows[mark2.row].mark(mark2.number);
            }
        }
    }

    pub fn row_totals(&self) -> [u8; 4] {
        core::array::from_fn(|i| self.rows[i].total)
    }

    pub fn row_free_values(&self) -> [Option<u8>; 4] {
        core::array::from_fn(|i| self.rows[i].free)
    }

    pub fn count_locked(&self) -> u8 {
        self.rows.iter().map(|row| row.free.is_none() as u8).sum()
    }

    #[allow(dead_code)]
    pub fn lockable_rows(&self) -> u8 {
        self.rows
            .iter()
            .filter(|row| row.free.is_some() && row.total >= 5)
            .count() as u8
    }

    pub fn locked(&self) -> [bool; 4] {
        self.rows
            .iter()
            .map(|row| row.free.is_none())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    pub fn lock(&mut self, locked: [bool; 4]) {
        self.rows
            .iter_mut()
            .zip(locked)
            .filter(|(_, lock)| *lock)
            .for_each(|(row, _)| row.free = None);
    }

    /// Reconstruct a State from a marks array and strike count.
    /// marks[row][i] corresponds to the numbers in display order:
    ///   rows 0,1 (ascending): index 0=number 2, index 10=number 12
    ///   rows 2,3 (descending): index 0=number 12, index 10=number 2
    pub fn from_marks(marks: &[[bool; 11]; 4], strikes: u8) -> State {
        let mut rows = [Row::default(); 4];
        for (i, row_marks) in marks.iter().enumerate() {
            let ascending = i < 2;
            let total: u8 = row_marks.iter().filter(|&&m| m).count() as u8;

            // Find the free pointer: one past the rightmost marked number.
            // For ascending rows: numbers are 2..=12 (indices 0..=10)
            // For descending rows: numbers are 12..=2 (indices 0..=10, so index 0=12, index 10=2)
            let last_marked_index = row_marks.iter().rposition(|&m| m);

            let lock_number_index = 10; // last position is always the lock number
            let is_locked = last_marked_index == Some(lock_number_index) && total >= 5;

            let free = if is_locked {
                None
            } else if let Some(last_idx) = last_marked_index {
                // Convert index to number
                let last_number = if ascending {
                    (last_idx as u8) + 2
                } else {
                    12 - (last_idx as u8)
                };
                // Free is one past the last marked number
                if ascending {
                    Some(last_number + 1)
                } else {
                    Some(last_number - 1)
                }
            } else {
                // No marks: default free pointer
                if ascending { Some(2) } else { Some(12) }
            };

            // If locked, add 1 to total for the lock bonus (the lock symbol)
            let total = if is_locked { total + 1 } else { total };

            rows[i] = Row {
                ascending,
                total,
                free,
            };
        }
        State { strikes, rows }
    }

    pub fn count_points(&self) -> isize {
        self.rows
            .iter()
            .map(|row| (row.total * (row.total + 1) / 2) as isize)
            .sum::<isize>()
            - self.strikes as isize * 5
    }

    // Generate moves for an opponent's turn (white dice sum only, single mark).
    pub fn generate_opponent_moves(&self, number: u8) -> Vec<Move> {
        self.rows
            .iter()
            .enumerate()
            .filter(|(_, row)| row.can_mark(number))
            .map(|(i, _)| Move::from((i, number)))
            .collect()
    }

    // Generate moves for the active player's turn. Does not include `Strike`.
    pub fn generate_moves(&self, dice: [u8; 6]) -> Vec<Move> {
        let (white, color) = dice.split_at(2);
        let on_white: u8 = white.iter().sum();
        let mut white_moves = Vec::new();
        let mut color_moves = Vec::new();
        let mut double_moves = Vec::new();

        // Generates possible moves using color die for `row` with index `i`.
        let on_color = |i: usize, row: Row| {
            white
                .iter()
                .filter(move |&w| row.can_mark(w + color[i]))
                .map(move |w| (i, w + color[i]))
        };

        for (i, &row) in self.rows.iter().enumerate() {
            if row.can_mark(on_white) {
                let white_mov = (i, on_white);
                white_moves.push(white_mov);

                let mut state_after = self.clone();
                state_after.apply_move(white_mov);

                on_color(i, state_after.rows[i])
                    .for_each(|color_mov| double_moves.push((white_mov, color_mov)));
            }
            on_color(i, row).for_each(|color_mov| color_moves.push(color_mov));
        }

        let double_moves = white_moves
            .clone()
            .into_iter()
            .cartesian_product(color_moves.clone())
            // here we only care about double moves on different rows,
            // since we already collected ones on the same row
            .filter(|(white, color)| white.0 != color.0)
            .chain(double_moves)
            .unique()
            .map(Move::from);

        white_moves
            .into_iter()
            .chain(color_moves)
            .unique()
            .map(Move::from)
            .chain(double_moves)
            .collect()
    }

    /// Prune dominated single moves. Doubles and Strike pass through unchanged.
    ///
    /// Dominance rules for singles:
    /// 1. Same row: keep only the mark closest to free pointer (fewer blanks, more options).
    /// 2. Cross-row: if two singles create equal blanks and equal resulting progress
    ///    (direction-adjusted), prefer the row with higher total marks (more marginal score).
    pub fn prune_dominated(&self, moves: &[Move]) -> Vec<Move> {
        // Strike always passes through; prune the rest by post-state dominance.
        let (strikes, markers): (Vec<_>, Vec<_>) = moves
            .iter()
            .copied()
            .partition(|m| matches!(m, Move::Strike));

        let post_states: Vec<State> = markers
            .iter()
            .map(|&m| {
                let mut s = *self;
                s.apply_move(m);
                s
            })
            .collect();

        // Rule 1: strict post-state dominance (applies to singles and doubles uniformly)
        let mut surviving: Vec<Move> = markers
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                !(0..markers.len())
                    .any(|j| j != *i && post_state_dominates(&post_states[j], &post_states[*i]))
            })
            .map(|(_, m)| *m)
            .collect();

        // Rule 2: cross-row single heuristic — prefer higher pre-move total when blanks+progress match
        surviving = self.apply_single_marginal_dominance(&surviving);

        let mut result = surviving;
        result.extend(strikes);
        result
    }

    /// Rule 2 — among singles, prune if another single with same blanks and progress
    /// is on a row with higher current total (higher marginal value from triangular scoring).
    fn apply_single_marginal_dominance(&self, moves: &[Move]) -> Vec<Move> {
        let info: Vec<(Move, Option<(usize, u8, u8)>)> = moves
            .iter()
            .map(|&m| {
                let tag = match m {
                    Move::Single(mark) => {
                        let free = self.rows[mark.row].free.unwrap();
                        let blanks = if mark.row < 2 { mark.number - free } else { free - mark.number };
                        let progress = if mark.row < 2 { mark.number - 1 } else { 13 - mark.number };
                        Some((mark.row, blanks, progress))
                    }
                    _ => None,
                };
                (m, tag)
            })
            .collect();

        info.iter()
            .filter(|(_, tag)| match tag {
                None => true, // non-singles pass through
                Some((row, blanks, progress)) => !info.iter().any(|(_, other)| match other {
                    Some((other_row, b, p)) => {
                        other_row != row
                            && b == blanks
                            && p == progress
                            && self.rows[*other_row].total > self.rows[*row].total
                    }
                    None => false,
                }),
            })
            .map(|(m, _)| *m)
            .collect()
    }

    //////////////////////////////////////
    // Metrics ///////////////////////////
    //////////////////////////////////////

    pub fn blanks(&self) -> u8 {
        self.rows
            .iter()
            .map(|row| {
                let Some(free) = row.free else {
                    // if the row is locked, we don't care
                    return 0;
                };
                if row.ascending {
                    (free - 2) - row.total
                } else {
                    (12 - free) - row.total
                }
            })
            .sum()
    }

    // Probability that opponent's dice will have a number I can mark with no blanks.
    pub fn probability(&self) -> f32 {
        self.rows
            .iter()
            .flat_map(|row| row.free)
            .unique()
            .map(|x| 6 - 7usize.abs_diff(x as usize))
            .sum::<usize>() as f32
            / 36.0
    }

    // Probability that at least one of N opponents will roll a number I can mark with no blanks.
    #[allow(dead_code)]
    pub fn probability_n(&self, n: u8) -> f32 {
        let p = self.probability();
        1.0 - (1.0 - p).powi(n as i32)
    }
}

impl Row {
    fn can_mark(&self, number: u8) -> bool {
        if !(2..=12).contains(&number) {
            return false;
        }
        let Some(free) = self.free else {
            return false;
        };
        if self.ascending && number == 12 && self.total < 5 {
            return false;
        }
        if !self.ascending && number == 2 && self.total < 5 {
            return false;
        }
        self.ascending && free <= number || !self.ascending && free >= number
    }

    fn mark(&mut self, number: u8) {
        assert!(self.can_mark(number));
        self.total += 1;
        self.free = if self.ascending && number == 12 || !self.ascending && number == 2 {
            self.total += 1;
            None
        } else if self.ascending {
            Some(number + 1)
        } else {
            Some(number - 1)
        };
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Mark {
    pub row: usize,
    pub number: u8,
}

impl std::str::FromStr for Mark {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (row, number) = s.split_at(1);
        anyhow::ensure!(row.len() == 1, "Invalid row format");
        let row = match row.chars().next().unwrap() {
            'r' | 'R' => 0,
            'y' | 'Y' => 1,
            'g' | 'G' => 2,
            'b' | 'B' => 3,
            _ => anyhow::bail!("Invalid row format"),
        };
        let number = number.parse()?;
        Ok(Mark { row, number })
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Move {
    Strike,
    Single(Mark),
    Double(Mark, Mark),
}

/// Result of applying meta-rules to a move list.
pub enum MetaDecision {
    /// A forced move (smart lock or smart strike) — use immediately.
    Forced(Move),
    /// Filtered move list for the strategy to evaluate.
    Choices(Vec<Move>),
}

/// True if state `better` strictly dominates `worse`: no worse on any row and strictly better on one.
/// Per row: higher total is better; among unlocked rows, free closer to start (asc) / end (desc) is better.
/// Locked vs unlocked is incomparable (locked gains mark bonus but loses all future options).
fn post_state_dominates(better: &State, worse: &State) -> bool {
    let mut any_strict = false;
    for row in 0..4 {
        let t1 = better.rows[row].total;
        let t2 = worse.rows[row].total;
        if t1 < t2 { return false; }

        let f1 = better.rows[row].free;
        let f2 = worse.rows[row].free;
        let ascending = row < 2;

        let free_cmp = match (f1, f2) {
            (None, None) => std::cmp::Ordering::Equal,
            (None, Some(_)) | (Some(_), None) => return false,
            (Some(a), Some(b)) => {
                if ascending { b.cmp(&a) } else { a.cmp(&b) }
            }
        };
        if free_cmp == std::cmp::Ordering::Less { return false; }
        if free_cmp == std::cmp::Ordering::Greater || t1 > t2 {
            any_strict = true;
        }
    }
    any_strict
}

impl State {
    /// Apply meta-rules: smart lock, smart strike, don't-strike-into-loss, prune dominated.
    /// `score_gap` is our_score - max_opponent_score.
    pub fn apply_meta_rules(&self, dice: [u8; 6], score_gap: isize) -> MetaDecision {
        let mut moves = self.generate_moves(dice);
        moves.push(Move::Strike);

        let opp_score = self.count_points() - score_gap;

        // Smart strike: end the game if ahead with 3 strikes
        if self.strikes == 3 {
            if self.count_points() - 5 > opp_score {
                return MetaDecision::Forced(Move::Strike);
            }
            // Don't strike into a loss (unless forced)
            if moves.len() > 1 {
                moves.retain(|m| !matches!(m, Move::Strike));
            }
        }

        // Smart lock: lock if possible, but not into a loss
        if let Some(mov) = self.find_smart_lock(&moves, score_gap) {
            return MetaDecision::Forced(mov);
        }

        // Filter out moves that lock into a game-ending loss (the model mustn't pick them).
        // Ties are allowed — neither player wins in a tie, so it's not a loss.
        let current_locked = self.count_locked();
        moves.retain(|&mov| {
            let mut s = *self;
            s.apply_move(mov);
            !(s.count_locked() > current_locked
                && s.count_locked() >= 2
                && s.count_points() < opp_score)
        });

        // Prune dominated moves
        let moves = self.prune_dominated(&moves);
        if moves.is_empty() {
            return MetaDecision::Forced(Move::Strike);
        }

        MetaDecision::Choices(moves)
    }

    /// Check if any move in the list is a smart lock (locks a row beneficially).
    /// Returns the locking move if found.
    pub fn find_smart_lock(&self, moves: &[Move], score_gap: isize) -> Option<Move> {
        let opp_score = self.count_points() - score_gap;
        let current_locked = self.count_locked();
        moves
            .iter()
            .copied()
            .filter(|&mov| {
                let mut s = *self;
                s.apply_move(mov);
                if s.count_locked() <= current_locked { return false; }
                if s.count_locked() >= 2 {
                    return s.count_points() > opp_score;
                }
                true
            })
            .max_by_key(|&mov| {
                let mut s = *self;
                s.apply_move(mov);
                s.count_points()
            })
    }
}

impl From<(usize, u8)> for Move {
    fn from((row, number): (usize, u8)) -> Self {
        Move::Single(Mark { row, number })
    }
}

impl From<((usize, u8), (usize, u8))> for Move {
    fn from((first, second): ((usize, u8), (usize, u8))) -> Self {
        Move::Double(
            Mark {
                row: first.0,
                number: first.1,
            },
            Mark {
                row: second.0,
                number: second.1,
            },
        )
    }
}

impl std::str::FromStr for Move {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<_> = s.split_whitespace().collect();
        anyhow::ensure!(parts.len() == 1 || parts.len() == 2, "Invalid move format");

        match parts[0] {
            "strike" => Ok(Move::Strike),
            _ => {
                let mark: Mark = parts[0].parse()?;
                if parts.len() == 1 {
                    Ok(Move::Single(mark))
                } else {
                    let mark2: Mark = parts[1].parse()?;
                    Ok(Move::Double(mark, mark2))
                }
            }
        }
    }
}

// ---- Display ----

const ROW_COLORS: [&str; 4] = ["\x1b[91m", "\x1b[93m", "\x1b[92m", "\x1b[94m"];
const ROW_NAMES: [&str; 4] = ["RED", "YLW", "GRN", "BLU"];
const DIM: &str = "\x1b[2m";
const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "{DIM}──────────────────────────────────────────────{RESET}")?;

        for (i, row) in self.rows.iter().enumerate() {
            let numbers: Vec<u8> = if row.ascending {
                (2..=12).collect()
            } else {
                (2..=12).rev().collect()
            };

            write!(f, "  {BOLD}{}{:<4}{RESET}", ROW_COLORS[i], ROW_NAMES[i])?;

            for &n in &numbers {
                let is_available = match row.free {
                    None => false,
                    Some(free) => {
                        if row.ascending {
                            n >= free
                        } else {
                            n <= free
                        }
                    }
                };

                let is_lock_number = (row.ascending && n == 12) || (!row.ascending && n == 2);

                if !is_available {
                    write!(f, " {DIM}·{RESET} ")?;
                } else if is_lock_number && row.total < 5 {
                    write!(f, "{DIM}{:>2}{RESET} ", n)?;
                } else {
                    write!(f, "{}{BOLD}{:>2}{RESET} ", ROW_COLORS[i], n)?;
                }
            }

            if row.free.is_none() {
                writeln!(f, "  {}{}{RESET} {DIM}LOCKED{RESET}", ROW_COLORS[i], row.total)?;
            } else {
                writeln!(f, "  {}{}{RESET}", ROW_COLORS[i], row.total)?;
            }
        }

        writeln!(f, "{DIM}──────────────────────────────────────────────{RESET}")?;

        write!(f, "  Strikes ")?;
        for i in 0..4u8 {
            if i < self.strikes {
                write!(f, "\x1b[91m✗{RESET} ")?;
            } else {
                write!(f, "{DIM}·{RESET} ")?;
            }
        }

        let points = self.count_points();
        writeln!(f, "                Score: {BOLD}{points}{RESET}")?;

        write!(f, "{DIM}──────────────────────────────────────────────{RESET}")?;

        Ok(())
    }
}

impl fmt::Display for Mark {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let color = ROW_COLORS[self.row];
        let name = ROW_NAMES[self.row];
        write!(f, "{BOLD}{color}{name} {:>2}{RESET}", self.number)
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Move::Strike => write!(f, "{BOLD}\x1b[91m✗ Strike{RESET}"),
            Move::Single(mark) => write!(f, "{mark}"),
            Move::Double(m1, m2) => write!(f, "{m1} + {m2}"),
        }
    }
}

const DICE_COLORS: [&str; 6] = [
    "\x1b[97m", "\x1b[97m", // white dice
    "\x1b[91m", "\x1b[93m", // red, yellow
    "\x1b[92m", "\x1b[94m", // green, blue
];
#[allow(dead_code)]
const DICE_NAMES: [&str; 6] = ["W", "W", "R", "Y", "G", "B"];

pub fn format_dice(dice: [u8; 6]) -> String {
    let mut s = String::from("  Dice  ");
    for (i, &d) in dice.iter().enumerate() {
        s += &format!("{BOLD}{}{d}{RESET} ", DICE_COLORS[i]);
    }
    let white_sum = dice[0] + dice[1];
    s += &format!(" {DIM}(white sum: {BOLD}{white_sum}{RESET}{DIM}){RESET}");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Row basics ----

    #[test]
    fn ascending_row_marks_in_order() {
        let mut row = Row { ascending: true, total: 0, free: Some(2) };
        assert!(row.can_mark(2));
        assert!(row.can_mark(7));
        assert!(row.can_mark(12) == false); // need 5 marks first
        row.mark(3);
        assert_eq!(row.free, Some(4));
        assert_eq!(row.total, 1);
        assert!(!row.can_mark(3)); // already past
        assert!(row.can_mark(4));
    }

    #[test]
    fn descending_row_marks_in_order() {
        let mut row = Row { ascending: false, total: 0, free: Some(12) };
        assert!(row.can_mark(12));
        assert!(row.can_mark(5));
        assert!(!row.can_mark(2)); // need 5 marks first
        row.mark(10);
        assert_eq!(row.free, Some(9));
        assert_eq!(row.total, 1);
        assert!(!row.can_mark(11)); // already past
        assert!(row.can_mark(9));
    }

    #[test]
    fn cannot_mark_out_of_range() {
        let row = Row { ascending: true, total: 0, free: Some(2) };
        assert!(!row.can_mark(0));
        assert!(!row.can_mark(1));
        assert!(!row.can_mark(13));
    }

    #[test]
    fn cannot_mark_locked_row() {
        let row = Row { ascending: true, total: 6, free: None };
        assert!(!row.can_mark(7));
    }

    #[test]
    fn locking_ascending_row() {
        let mut row = Row { ascending: true, total: 5, free: Some(11) };
        assert!(row.can_mark(12));
        row.mark(12);
        assert_eq!(row.free, None); // locked
        assert_eq!(row.total, 7); // 5 + 1 for mark + 1 bonus for lock
    }

    #[test]
    fn locking_descending_row() {
        let mut row = Row { ascending: false, total: 5, free: Some(3) };
        assert!(row.can_mark(2));
        row.mark(2);
        assert_eq!(row.free, None);
        assert_eq!(row.total, 7);
    }

    #[test]
    fn cannot_lock_with_fewer_than_5_marks() {
        let row = Row { ascending: true, total: 4, free: Some(11) };
        assert!(!row.can_mark(12));

        let row = Row { ascending: false, total: 4, free: Some(3) };
        assert!(!row.can_mark(2));
    }

    // ---- State scoring ----

    #[test]
    fn empty_state_scores_zero() {
        let state = State::default();
        assert_eq!(state.count_points(), 0);
    }

    #[test]
    fn scoring_triangular_numbers() {
        let mut state = State::default();
        // Mark 3 numbers in red (row 0): 2, 3, 4
        state.apply_move((0usize, 2u8));
        state.apply_move((0usize, 3u8));
        state.apply_move((0usize, 4u8));
        // 3 marks = 3*4/2 = 6 points
        assert_eq!(state.count_points(), 6);
    }

    #[test]
    fn strikes_subtract_five_each() {
        let mut state = State::default();
        state.apply_move(Move::Strike);
        assert_eq!(state.count_points(), -5);
        state.apply_move(Move::Strike);
        assert_eq!(state.count_points(), -10);
    }

    // ---- State locking ----

    #[test]
    fn lock_propagates_to_rows() {
        let mut state = State::default();
        assert_eq!(state.count_locked(), 0);
        state.lock([true, false, false, true]);
        assert_eq!(state.count_locked(), 2);
        assert_eq!(state.locked(), [true, false, false, true]);
    }

    #[test]
    fn locking_via_mark_locks_row() {
        let mut state = State::default();
        // Mark 5 numbers in red then lock with 12
        for n in 2..=6 {
            state.apply_move((0usize, n as u8));
        }
        assert_eq!(state.locked(), [false, false, false, false]);
        state.apply_move((0usize, 12u8));
        assert_eq!(state.locked(), [true, false, false, false]);
    }

    // ---- Move application ----

    #[test]
    fn double_move_marks_two_rows() {
        let mut state = State::default();
        state.apply_move(((0usize, 5u8), (2usize, 10u8)));
        // Red row: 1 mark = 1 point. Green row: 1 mark = 1 point.
        assert_eq!(state.count_points(), 2);
    }

    #[test]
    #[should_panic]
    fn fifth_strike_panics() {
        let mut state = State::default();
        for _ in 0..5 {
            state.apply_move(Move::Strike);
        }
    }

    // ---- generate_opponent_moves ----

    #[test]
    fn opponent_moves_are_all_singles() {
        let state = State::default();
        let moves = state.generate_opponent_moves(7);
        for mov in &moves {
            assert!(matches!(mov, Move::Single(_)));
        }
    }

    #[test]
    fn opponent_moves_fresh_state() {
        let state = State::default();
        // 7 is markable in all 4 rows on a fresh state
        let moves = state.generate_opponent_moves(7);
        assert_eq!(moves.len(), 4);
    }

    #[test]
    fn opponent_moves_respects_free_pointer() {
        let mut state = State::default();
        // Mark red 5, so red free = 6
        state.apply_move((0usize, 5u8));
        // 4 is no longer markable in red, but still in yellow
        let moves = state.generate_opponent_moves(4);
        let rows: Vec<usize> = moves.iter().map(|m| match m {
            Move::Single(mark) => mark.row,
            _ => panic!("expected single"),
        }).collect();
        assert!(!rows.contains(&0)); // red excluded
        assert!(rows.contains(&1));  // yellow still open
    }

    #[test]
    fn opponent_moves_skips_locked_row() {
        let mut state = State::default();
        state.lock([true, false, false, false]);
        let moves = state.generate_opponent_moves(7);
        let rows: Vec<usize> = moves.iter().map(|m| match m {
            Move::Single(mark) => mark.row,
            _ => panic!("expected single"),
        }).collect();
        assert!(!rows.contains(&0));
        assert_eq!(moves.len(), 3);
    }

    // ---- generate_moves (active player) ----

    #[test]
    fn generate_moves_includes_singles_and_doubles() {
        let state = State::default();
        // dice: W1=3, W2=4, R=1, Y=2, G=3, B=4
        let moves = state.generate_moves([3, 4, 1, 2, 3, 4]);
        let has_single = moves.iter().any(|m| matches!(m, Move::Single(_)));
        let has_double = moves.iter().any(|m| matches!(m, Move::Double(_, _)));
        assert!(has_single);
        assert!(has_double);
    }

    #[test]
    fn generate_moves_no_strikes() {
        let state = State::default();
        let moves = state.generate_moves([3, 4, 1, 2, 3, 4]);
        assert!(moves.iter().all(|m| !matches!(m, Move::Strike)));
    }

    #[test]
    fn generate_moves_empty_when_nothing_markable() {
        let mut state = State::default();
        state.lock([true, true, true, true]);
        let moves = state.generate_moves([3, 4, 1, 2, 3, 4]);
        assert!(moves.is_empty());
    }

    // ---- Parsing ----

    #[test]
    fn parse_mark() {
        let mark: Mark = "r5".parse().unwrap();
        assert_eq!(mark.row, 0);
        assert_eq!(mark.number, 5);

        let mark: Mark = "G10".parse().unwrap();
        assert_eq!(mark.row, 2);
        assert_eq!(mark.number, 10);
    }

    #[test]
    fn parse_move_strike() {
        let mov: Move = "strike".parse().unwrap();
        assert!(matches!(mov, Move::Strike));
    }

    #[test]
    fn parse_move_single() {
        let mov: Move = "b7".parse().unwrap();
        match mov {
            Move::Single(m) => { assert_eq!(m.row, 3); assert_eq!(m.number, 7); }
            _ => panic!("expected single"),
        }
    }

    #[test]
    fn parse_move_double() {
        let mov: Move = "r4 g10".parse().unwrap();
        match mov {
            Move::Double(m1, m2) => {
                assert_eq!(m1.row, 0);
                assert_eq!(m1.number, 4);
                assert_eq!(m2.row, 2);
                assert_eq!(m2.number, 10);
            }
            _ => panic!("expected double"),
        }
    }

    // ---- Metrics ----

    #[test]
    fn blanks_fresh_state() {
        let state = State::default();
        assert_eq!(state.blanks(), 0);
    }

    #[test]
    fn blanks_after_skip() {
        let mut state = State::default();
        // Mark red 5, skipping 2,3,4 = 3 blanks
        state.apply_move((0usize, 5u8));
        assert_eq!(state.blanks(), 3);
    }

    #[test]
    fn probability_fresh_state() {
        let state = State::default();
        // Fresh: free values are 2 and 12
        // 2 has 1 way (1+1), 12 has 1 way (6+6) → 2/36
        let p = state.probability();
        assert!((p - 2.0 / 36.0).abs() < 1e-6);
    }
}
