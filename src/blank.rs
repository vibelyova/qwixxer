/// Reimplementation of strategies from Joshua Blank's thesis:
/// "Qwixx Strategies Using Simulation and MCMC Methods" (Cal Poly, 2024)
use crate::state::{Mark, Move, State};
use crate::strategy::Strategy;

/// Blank's score-based strategy.
/// Tuned parameters from the thesis:
/// - W_G = W_C = W_P = 1 (equal weights, scores scaled 0-10)
/// - skip_score = 5
/// - skip_adjustment = 0 (no penalty-based adjustment)
/// - W_L = 1/300 (likelihood weight relative to others)
/// - Never take a 4th strike if any move exists
#[derive(Debug, Default, Clone)]
pub struct BlankScoreBased;

impl BlankScoreBased {
    /// Gap score for a sub-move: 0 (9 spaces skipped) to 9 (no gap), scaled to 0-10.
    /// `gap` is the number of spaces between the last marked space and this move.
    fn gap_score(gap: u8) -> f64 {
        let g = (9u8.saturating_sub(gap)) as f64;
        g * 10.0 / 9.0 // scale from 0-9 to 0-10
    }

    /// Position score: 10 for leftmost (position 0), 0 for rightmost (position 10).
    /// `pos` is the 0-indexed position from the left of the row.
    fn position_score(pos: u8) -> f64 {
        (10u8.saturating_sub(pos)) as f64
    }

    /// Count score: number of marks already in the row (0-10).
    fn count_score(total: u8) -> f64 {
        total as f64
    }

    /// Likelihood score: rarer rolls score higher. Position 0 or 10 (sums 2 or 12)
    /// score 5, position 5 (sum 7) scores 0.
    fn likelihood_score(pos: u8) -> f64 {
        (pos as f64 - 5.0).abs()
    }

    /// Compute the position (0-10) of a number within a row.
    /// Ascending (Red/Yellow): number 2 = pos 0, number 12 = pos 10
    /// Descending (Green/Blue): number 12 = pos 0, number 2 = pos 10
    fn number_to_pos(row: usize, number: u8) -> u8 {
        if row < 2 {
            // ascending
            number - 2
        } else {
            // descending
            12 - number
        }
    }

    /// Compute the gap for marking `number` in `row`.
    /// Gap = spaces skipped between the last marked position and this mark.
    fn compute_gap(state: &State, row: usize, number: u8) -> u8 {
        let free_values = state.row_free_values();
        let Some(free) = free_values[row] else {
            return 9; // locked, shouldn't happen
        };
        if row < 2 {
            // ascending: gap = number - free
            number.saturating_sub(free)
        } else {
            // descending: gap = free - number
            free.saturating_sub(number)
        }
    }

    /// Score a single mark (sub-move).
    fn score_mark(state: &State, mark: &Mark) -> (f64, f64, f64, f64) {
        let gap = Self::compute_gap(state, mark.row, mark.number);
        let pos = Self::number_to_pos(mark.row, mark.number);
        let total = state.row_totals()[mark.row];

        let g = Self::gap_score(gap);
        let p = Self::position_score(pos);
        let c = Self::count_score(total);
        let l = Self::likelihood_score(pos) * (g * 9.0 / 10.0 + c); // use unscaled gap for likelihood product

        (g, p, c, l)
    }

    /// Score a skip sub-move.
    fn score_skip() -> (f64, f64, f64, f64) {
        let skip = 5.0;
        let g = skip * 10.0 / 9.0; // scaled like gap scores: skip * 9/10 then * 10/9
        (g, skip, skip, 0.0)
    }

    /// Score a complete move (1 or 2 sub-moves).
    fn score_move(state: &State, mov: Move) -> f64 {
        const LIKELIHOOD_WEIGHT: f64 = 1.0 / 300.0;

        match mov {
            Move::Strike => {
                // Strike = both sub-moves are skips
                let (g, p, c, l) = Self::score_skip();
                let total = (g + g) + (p + p) + (c + c) + (l + l) * LIKELIHOOD_WEIGHT;
                total
            }
            Move::Single(mark) => {
                let (g1, p1, c1, l1) = Self::score_mark(state, &mark);
                let (g2, p2, c2, l2) = Self::score_skip(); // second sub-move is skip
                (g1 + g2) + (p1 + p2) + (c1 + c2) + (l1 + l2) * LIKELIHOOD_WEIGHT
            }
            Move::Double(mark1, mark2) => {
                let (g1, p1, c1, l1) = Self::score_mark(state, &mark1);
                // For second mark, if same row, count is +1 and gap is from first mark
                let (g2, p2, c2, l2) = if mark1.row == mark2.row {
                    let gap = if mark1.row < 2 {
                        mark2.number.saturating_sub(mark1.number + 1)
                    } else {
                        mark1.number.saturating_sub(mark2.number + 1)
                    };
                    let pos = Self::number_to_pos(mark2.row, mark2.number);
                    let total = state.row_totals()[mark2.row] + 1; // +1 because first mark was in same row
                    let g = Self::gap_score(gap);
                    let p = Self::position_score(pos);
                    let c = Self::count_score(total);
                    let l = Self::likelihood_score(pos) * (g * 9.0 / 10.0 + c);
                    (g, p, c, l)
                } else {
                    Self::score_mark(state, &mark2)
                };
                (g1 + g2) + (p1 + p2) + (c1 + c2) + (l1 + l2) * LIKELIHOOD_WEIGHT
            }
        }
    }
}

impl Strategy for BlankScoreBased {
    fn your_move(&mut self, state: &State, dice: [u8; 6]) -> Move {
        let mut moves = state.generate_moves(dice);
        moves.push(Move::Strike);

        // Never take a 4th strike if any non-strike move exists
        if state.strikes >= 3 && moves.len() > 1 {
            moves.retain(|m| !matches!(m, Move::Strike));
        }

        moves
            .into_iter()
            .max_by(|a, b| {
                Self::score_move(state, *a)
                    .partial_cmp(&Self::score_move(state, *b))
                    .unwrap()
            })
            .unwrap()
    }

    fn opponents_move(&mut self, state: &State, number: u8, _locked: [bool; 4]) -> Option<Move> {
        // On opponent turns, only the white sum is available (single mark or skip).
        // Use the same scoring but with the second sub-move always being skip.
        let moves = state.generate_opponent_moves(number);
        if moves.is_empty() {
            return None;
        }

        // Compare best mark vs skip (None)
        let best_mark = moves
            .iter()
            .max_by(|a, b| {
                Self::score_move(state, **a)
                    .partial_cmp(&Self::score_move(state, **b))
                    .unwrap()
            })
            .copied()
            .unwrap();

        let mark_score = Self::score_move(state, best_mark);
        let skip_score = {
            let (g, p, c, l) = Self::score_skip();
            g + p + c // single skip sub-move
        };

        if mark_score >= skip_score {
            Some(best_mark)
        } else {
            None
        }
    }
}
