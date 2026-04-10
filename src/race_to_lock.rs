/// Reimplementation of the race-to-lock strategy from Joshua Blank's thesis.
/// Uses absorbing Markov chains per row to restrict moves to only those that
/// decrease expected rolls to lock. Uses the "2 sums from 3 dice" criterion
/// (the best-performing variant from the thesis).
use crate::state::{Move, State};
use crate::strategy::Strategy;

/// Row state: (free pointer, mark count). Count < 5 is transient, >= 5 is lockable.
/// For ascending rows (Red/Yellow): free ranges 2-12, mark towards 12.
/// Descending rows are handled by mirroring.
// RowState removed — was unused placeholder from initial design

/// Precomputed allowed-transitions table for one row direction.
/// allowed[free_idx][count] is a bitmask of which numbers (2-12) are allowed to mark.
#[derive(Debug)]
struct AllowedTable {
    allowed: [[u16; 5]; 11], // [free-2][count] -> bitmask over numbers 2-12
}

impl AllowedTable {
    fn is_allowed(&self, free: u8, count: u8, number: u8) -> bool {
        if count >= 5 {
            return number == 12; // only lock move
        }
        let free_idx = (free - 2) as usize;
        let bit = 1u16 << (number - 2);
        self.allowed[free_idx][count as usize] & bit != 0
    }

    /// Compute the allowed-transitions table using value iteration and
    /// iterative restriction (per thesis methodology).
    fn compute() -> Self {
        // Start: allow marks that don't lead to dead ends
        let mut allowed = [[0u16; 5]; 11];
        for free_idx in 0..11usize {
            let free = free_idx as u8 + 2;
            for count in 0..5usize {
                for n in free..=11u8 {
                    let _new_free_idx = (n + 1 - 2) as usize; // n+1-2
                    let new_count = count + 1;
                    // Only allow if the resulting state can eventually lock
                    // Dead end: free=12 (new_free_idx=10) with count < 5
                    // Need at least (5 - new_count) more marks with only (12 - (n+1)) numbers left
                    let remaining_numbers = 12u8.saturating_sub(n + 1); // numbers n+1..=11
                    let marks_needed = 5usize.saturating_sub(new_count);
                    if remaining_numbers as usize >= marks_needed || new_count >= 5 {
                        allowed[free_idx][count] |= 1u16 << (n - 2);
                    }
                }
                if count + 1 >= 5 {
                    allowed[free_idx][count] |= 1u16 << 10; // mark 12 = lock
                }
            }
        }

        // Iteratively restrict: remove transitions that increase expected rolls
        loop {
            let expected = Self::compute_expected(&allowed);
            let mut changed = false;

            for free_idx in 0..11usize {
                let free = free_idx as u8 + 2;
                for count in 0..5usize {
                    let current_e = expected[free_idx][count];
                    if !current_e.is_finite() {
                        continue;
                    }
                    for n in free..=11u8 {
                        let bit = 1u16 << (n - 2);
                        if allowed[free_idx][count] & bit == 0 {
                            continue;
                        }
                        let new_count = count + 1;
                        let new_free_idx = (n + 1 - 2) as usize;
                        let after_e = if new_count >= 5 {
                            0.0
                        } else if new_free_idx < 11 {
                            expected[new_free_idx][new_count]
                        } else {
                            f64::INFINITY
                        };

                        if after_e >= current_e {
                            allowed[free_idx][count] &= !bit;
                            changed = true;
                        }
                    }
                }
            }

            if !changed {
                break;
            }
        }

        AllowedTable { allowed }
    }

    /// Compute expected rolls to lockable (count >= 5) for each state,
    /// given the current allowed-transitions table.
    fn compute_expected(allowed: &[[u16; 5]; 11]) -> [[f64; 5]; 11] {
        let mut expected = [[50.0f64; 5]; 11];

        // Value iteration: E[s] = (1 + sum_s' p(s->s') * E[s']) / p_any_transition
        for _ in 0..10000 {
            let prev = expected;
            for free_idx in 0..11usize {
                let free = free_idx as u8 + 2;
                for count in 0..5usize {
                    // Count transitions over all 216 dice outcomes
                    let mut weighted_e = 0.0f64;
                    let mut n_mark = 0u32;

                    for w1 in 1..=6u8 {
                        for w2 in 1..=6u8 {
                            for c in 1..=6u8 {
                                let sums = [w1 + w2, w1 + c, w2 + c];
                                let best = Self::best_mark(
                                    free,
                                    count as u8,
                                    &sums,
                                    &allowed[free_idx][count],
                                );

                                if let Some(n) = best {
                                    let new_count = count + 1;
                                    if n == 12 && new_count >= 5 {
                                        // Lock
                                        n_mark += 1;
                                    } else if new_count >= 5 {
                                        // Lockable (count reaches 5)
                                        n_mark += 1;
                                    } else {
                                        let new_free_idx = (n + 1 - 2) as usize;
                                        if new_free_idx < 11 {
                                            n_mark += 1;
                                            weighted_e += prev[new_free_idx][new_count];
                                        }
                                        // dead end (free=12, count<5): treat as no mark
                                    }
                                }
                            }
                        }
                    }

                    if n_mark > 0 {
                        let _p = n_mark as f64 / 216.0;
                        // E = 1/p + (weighted_e / n_mark)
                        // Derived from: E = 1 + (n_mark/216)*avg_E_next + ((216-n_mark)/216)*E
                        // => E * (n_mark/216) = 1 + (weighted_e/216)
                        // => E = 216/n_mark + weighted_e/n_mark
                        expected[free_idx][count] =
                            (216.0 + weighted_e) / n_mark as f64;
                    } else {
                        expected[free_idx][count] = f64::INFINITY;
                    }
                }
            }

            let max_diff: f64 = expected
                .iter()
                .flatten()
                .zip(prev.iter().flatten())
                .filter(|(a, _)| a.is_finite())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            if max_diff < 1e-6 {
                break;
            }
        }

        expected
    }

    /// Given available dice sums and allowed marks for a state, find the best mark
    /// (smallest gap from free, i.e., closest to free).
    fn best_mark(free: u8, count: u8, sums: &[u8; 3], allowed_mask: &u16) -> Option<u8> {
        let mut best: Option<u8> = None;
        for &s in sums {
            if s < free || s > 12 {
                continue;
            }
            if s == 12 {
                // Lock: only if count + 1 >= 5
                if count + 1 >= 5 {
                    // Locking is always best
                    return Some(12);
                }
                continue;
            }
            let bit = 1u16 << (s - 2);
            if allowed_mask & bit != 0 {
                if best.map_or(true, |b| s < b) {
                    best = Some(s);
                }
            }
        }
        best
    }
}

/// The race-to-lock strategy.
#[derive(Debug)]
pub struct BlankRaceToLock {
    table: AllowedTable,
}

impl Clone for BlankRaceToLock {
    fn clone(&self) -> Self {
        Self::new()
    }
}

impl Default for BlankRaceToLock {
    fn default() -> Self {
        Self::new()
    }
}

impl BlankRaceToLock {
    pub fn new() -> Self {
        Self {
            table: AllowedTable::compute(),
        }
    }

    /// Check if marking `number` in `row` is allowed given the current state.
    fn is_mark_allowed(&self, state: &State, row: usize, number: u8) -> bool {
        let totals = state.row_totals();
        let frees = state.row_free_values();
        let Some(free) = frees[row] else {
            return false; // locked
        };
        let count = totals[row];

        // For descending rows (2, 3), mirror the number and free
        let (norm_free, norm_number) = if row < 2 {
            (free, number)
        } else {
            // Mirror: 2↔12, 3↔11, etc.
            (14 - free, 14 - number)
        };

        if count >= 5 {
            // Lockable — only allow the lock number
            return norm_number == 12;
        }

        self.table.is_allowed(norm_free, count, norm_number)
    }

    /// Score a move: prefer moves that advance rows with most marks (smallest gap).
    fn score_move(&self, state: &State, mov: Move) -> (bool, i32) {
        match mov {
            Move::Strike => (false, i32::MIN),
            Move::Single(mark) => {
                let allowed = self.is_mark_allowed(state, mark.row, mark.number);
                let gap = Self::compute_gap(state, mark.row, mark.number) as i32;
                (allowed, -gap)
            }
            Move::Double(m1, m2) => {
                let a1 = self.is_mark_allowed(state, m1.row, m1.number);
                // For second mark in same row, check after first mark applied
                let a2 = if m1.row == m2.row {
                    let mut new_state = *state;
                    new_state.apply_move(Move::Single(m1));
                    self.is_mark_allowed(&new_state, m2.row, m2.number)
                } else {
                    self.is_mark_allowed(state, m2.row, m2.number)
                };
                let gap1 = Self::compute_gap(state, m1.row, m1.number) as i32;
                let gap2 = Self::compute_gap(state, m2.row, m2.number) as i32;
                (a1 && a2, -(gap1 + gap2))
            }
        }
    }

    fn compute_gap(state: &State, row: usize, number: u8) -> u8 {
        let frees = state.row_free_values();
        let Some(free) = frees[row] else {
            return 0;
        };
        if row < 2 {
            number.saturating_sub(free)
        } else {
            free.saturating_sub(number)
        }
    }
}

impl Strategy for BlankRaceToLock {
    fn your_move(&mut self, state: &State, dice: [u8; 6]) -> Move {
        let moves = state.generate_moves(dice);

        if moves.is_empty() {
            return Move::Strike;
        }

        // Never take a 4th strike if any move exists
        if state.strikes >= 3 {
            // Find any non-strike move (allowed or not)
            if let Some(&mov) = moves.iter().find(|m| !matches!(m, Move::Strike)) {
                // Even if not race-to-lock-allowed, better than game over
                let allowed_moves: Vec<_> = moves
                    .iter()
                    .filter(|m| {
                        let (allowed, _) = self.score_move(state, **m);
                        allowed
                    })
                    .collect();
                if let Some(&&best) = allowed_moves
                    .iter()
                    .max_by_key(|m| self.score_move(state, ***m).1)
                {
                    return best;
                }
                return mov; // fallback: any non-strike move
            }
        }

        // Prefer allowed moves by smallest gap; fall back to any move (smallest gap)
        let scored: Vec<_> = moves
            .iter()
            .map(|&m| {
                let (allowed, gap_score) = self.score_move(state, m);
                (m, allowed, gap_score)
            })
            .collect();

        // First pick from allowed moves
        if let Some(&(mov, _, _)) = scored
            .iter()
            .filter(|(_, allowed, _)| *allowed)
            .max_by_key(|(_, _, gap)| *gap)
        {
            return mov;
        }

        // No allowed moves — fall back to smallest-gap mark rather than striking
        scored
            .iter()
            .filter(|(m, _, _)| !matches!(m, Move::Strike))
            .max_by_key(|(_, _, gap)| *gap)
            .map(|(m, _, _)| *m)
            .unwrap_or(Move::Strike)
    }

    fn opponents_move(&mut self, state: &State, number: u8, _locked: [bool; 4]) -> Option<Move> {
        let moves = state.generate_opponent_moves(number);
        if moves.is_empty() {
            return None;
        }

        // Only mark if it's an allowed transition, pick smallest gap
        let best = moves
            .iter()
            .filter(|m| match m {
                Move::Single(mark) => self.is_mark_allowed(state, mark.row, mark.number),
                _ => false,
            })
            .min_by_key(|m| match m {
                Move::Single(mark) => Self::compute_gap(state, mark.row, mark.number),
                _ => u8::MAX,
            });

        best.copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allowed_table_not_empty() {
        let table = AllowedTable::compute();
        // Initial state (free=2, count=0) should have some allowed marks
        let mask = table.allowed[0][0];
        println!("State (free=2, count=0) allowed mask: {mask:011b}");
        assert!(mask != 0, "Initial state should have allowed marks");
    }

    #[test]
    fn best_mark_works() {
        // State: free=2, count=0, all marks 2-11 allowed
        let mask = 0b01111111111u16; // bits 0-9 = numbers 2-11
        // Roll (3, 4, 2): sums = [7, 5, 6]
        let best = AllowedTable::best_mark(2, 0, &[7, 5, 6], &mask);
        assert_eq!(best, Some(5), "Should pick 5 (smallest >= free=2)");

        // Roll (1, 1, 1): sums = [2, 2, 2]
        let best = AllowedTable::best_mark(2, 0, &[2, 2, 2], &mask);
        assert_eq!(best, Some(2), "Should pick 2");

        // free=7, roll gives [3, 4, 5] — nothing >= 7
        let best = AllowedTable::best_mark(7, 2, &[3, 4, 5], &mask);
        assert_eq!(best, None, "Nothing >= free=7");
    }

    #[test]
    fn expected_rolls_single_state() {
        // Simple case: free=11, count=4. Only need to mark 11 (->count=5, lockable).
        let mut allowed = [[0u16; 5]; 11];
        // Allow mark 11 from state (free=11, count=4)
        allowed[9][4] = 1u16 << 9; // bit 9 = number 11
        let expected = AllowedTable::compute_expected(&allowed);
        let e = expected[9][4];
        println!("Expected rolls from (free=11, count=4): {e:.1}");
        // P(rolling 11 from 3 dice) should be computable
        assert!(e > 1.0 && e < 50.0, "Should be finite, got {e}");
    }

    #[test]
    fn debug_before_after_restrictions() {
        // Show expected rolls before and after restrictions
        let mut allowed = [[0u16; 5]; 11];
        for free_idx in 0..11usize {
            let free = free_idx as u8 + 2;
            for count in 0..5usize {
                for n in free..=11u8 {
                    let new_count = count + 1;
                    let remaining = 12u8.saturating_sub(n + 1);
                    let needed = 5usize.saturating_sub(new_count);
                    if remaining as usize >= needed || new_count >= 5 {
                        allowed[free_idx][count] |= 1u16 << (n - 2);
                    }
                }
                if count + 1 >= 5 {
                    allowed[free_idx][count] |= 1u16 << 10;
                }
            }
        }
        let before = AllowedTable::compute_expected(&allowed);
        let table = AllowedTable::compute();
        let after = AllowedTable::compute_expected(&table.allowed);

        println!("\nState (free, count) | Before E | After E | Before mask | After mask");
        for count in 0..5 {
            for free in 2..=8u8 {
                let fi = (free - 2) as usize;
                println!(
                    "  ({free:>2}, {count})             {:>6.1}    {:>6.1}   {:011b}    {:011b}",
                    before[fi][count], after[fi][count],
                    allowed[fi][count], table.allowed[fi][count],
                );
            }
        }
    }

    #[test]
    fn debug_expected_chain() {
        let mut allowed = [[0u16; 5]; 11];
        for free_idx in 0..11usize {
            let free = free_idx as u8 + 2;
            for count in 0..5usize {
                for n in free..=11u8 {
                    let new_count = count + 1;
                    let remaining = 12u8.saturating_sub(n + 1);
                    let needed = 5usize.saturating_sub(new_count);
                    if remaining as usize >= needed || new_count >= 5 {
                        allowed[free_idx][count] |= 1u16 << (n - 2);
                    }
                }
                if count + 1 >= 5 {
                    allowed[free_idx][count] |= 1u16 << 10;
                }
            }
        }
        let expected = AllowedTable::compute_expected(&allowed);
        // Print a selection of states
        for count in 0..5 {
            for free in [2u8, 5, 7, 9, 11] {
                let fi = (free - 2) as usize;
                let mask = allowed[fi][count];
                println!(
                    "  (free={free:>2}, count={count}) mask={mask:011b} E={:.1}",
                    expected[fi][count]
                );
            }
        }
    }

    #[test]
    fn expected_rolls_reasonable() {
        let table = AllowedTable::compute();
        let expected = AllowedTable::compute_expected(&table.allowed);
        let e0 = expected[0][0]; // (free=2, count=0)
        println!("Expected rolls from initial state (after restrictions): {e0:.1}");
        assert!(e0 > 5.0 && e0 < 100.0, "Expected rolls should be reasonable, got {e0}");
    }

    #[test]
    fn count_strikes_in_solo() {
        use crate::game::{Game, Player};
        use rand::{rngs::SmallRng, SeedableRng};
        let mut total_strikes = 0u32;
        let n = 1000;
        for _ in 0..n {
            let mut game = Game::new(vec![Player::new(
                Box::new(BlankRaceToLock::new()),
                Box::new(SmallRng::from_entropy()),
            )]);
            game.play();
            total_strikes += game.players[0].state.strikes as u32;
        }
        let avg = total_strikes as f64 / n as f64;
        println!("RTL avg strikes per game: {avg:.2}");
    }
}
