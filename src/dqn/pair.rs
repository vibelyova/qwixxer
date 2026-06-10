//! Pairwise differential network ("pair" bot): joint two-board evaluation.
//!
//! The model sees both players' boards and predicts the distribution of the
//! *future* score differential `final_diff − current_diff` as `(μ, log σ²)`.
//! Move selection ranks candidates by `(current_diff + μ) / σ`, monotone in
//! the Gaussian P(win) against the leading opponent.
//!
//! Design: docs/superpowers/specs/2026-06-10-pair-network-design.md

use super::{
    aggregate_weighted_probability, lockable_rows, row_progress, total_progress, MyBackend, LOG_VAR_MAX, LOG_VAR_MIN,
};
use crate::state::State;
use crate::strategy::Bot;
use burn::{
    nn::{Linear, LinearConfig, Relu},
    prelude::*,
    record::CompactRecorder,
};
use std::sync::Arc;

/// Per-board feature block size.
pub const BOARD_FEATURES: usize = 20;
/// Full pair input: two board blocks + 5 pair-level features.
pub const PAIR_FEATURES: usize = 45;

/// Per-board feature block. Same layout as the old net's per-board features,
/// plus per-board aggregates, and with the lock-rule fix: a free pointer
/// sitting on the row's terminal number (12/2) cannot be marked while the row
/// has <5 marks, so it contributes 0 to the weighted probability.
pub fn board_features(state: &State) -> [f32; BOARD_FEATURES] {
    let totals = state.row_totals();
    let frees = state.row_free_values();
    let locked = state.locked();

    let mut f = [0.0f32; BOARD_FEATURES];
    for i in 0..4 {
        f[i] = row_progress(frees[i], i < 2);
        f[4 + i] = totals[i] as f32 / 11.0;
        f[8 + i] = if locked[i] { 1.0 } else { 0.0 };
        f[12 + i] = match frees[i] {
            Some(fr) if fr == State::row_terminal(i) && totals[i] < 5 => 0.0,
            Some(fr) => {
                let ways = 6.0 - (7.0f32 - fr as f32).abs();
                (ways / 6.0) * (totals[i] as f32 + 1.0) / 11.0
            }
            None => 0.0,
        };
    }
    f[16] = state.strikes as f32 / 3.0;
    f[17] = state.blanks() as f32 / 40.0;
    f[18] = aggregate_weighted_probability(state);
    f[19] = lockable_rows(state) as f32 / 4.0;
    f
}

/// Full pair input. `paired` must be one of `all_opps` (the opponent we are
/// being compared against); `all_opps` are ALL opponents of the `our` player —
/// the pair-level features 42–44 are computed over all of them (uniform
/// semantics in 1v1 and multiplayer; redundant with the opp block in 1v1).
///
/// Layout: `[our 20 | paired 20 | cdiff/100, num_opps/4, max opp progress,
/// max opp strikes/3, opp lockable-rows sum/8]`.
pub fn pair_features(our: &State, paired: &State, all_opps: &[State]) -> [f32; PAIR_FEATURES] {
    // Invariant: `paired` must be one of `all_opps`.
    debug_assert!(all_opps.contains(paired));
    let mut f = [0.0f32; PAIR_FEATURES];
    f[..BOARD_FEATURES].copy_from_slice(&board_features(our));
    f[BOARD_FEATURES..2 * BOARD_FEATURES].copy_from_slice(&board_features(paired));

    let cdiff = (our.count_points() - paired.count_points()) as f32;
    f[40] = (cdiff / 100.0).clamp(-1.0, 1.0);
    f[41] = all_opps.len() as f32 / 4.0;
    f[42] = all_opps.iter().map(total_progress).fold(0.0, f32::max);
    f[43] = all_opps.iter().map(|s| s.strikes).max().unwrap_or(0) as f32 / 3.0;
    f[44] = all_opps.iter().map(lockable_rows).sum::<u8>() as f32 / 8.0;
    f
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::Mark;

    #[test]
    fn wprob_zero_when_free_pointer_on_unlockable_terminal() {
        // Red (ascending): mark 2, 3, 11 -> free = 12 (terminal) with only 3 marks.
        let mut s = State::default();
        for n in [2u8, 3, 11] {
            s.apply_mark(Mark { row: 0, number: n });
        }
        assert_eq!(s.row_free_values()[0], Some(12));
        assert_eq!(s.row_totals()[0], 3);
        let f = board_features(&s);
        assert_eq!(f[12], 0.0, "terminal with <5 marks must contribute 0 ways");

        // A reachable state with 5 marks whose free pointer rests on the
        // terminal (2,3,4,5,11 -> free = 12, total = 5): terminal now lockable,
        // so it contributes nonzero ways. (We rebuild because once the free
        // pointer reaches 12 the earlier numbers can no longer be marked.)
        let mut s = State::default();
        for n in [2u8, 3, 4, 5, 11] {
            s.apply_mark(Mark { row: 0, number: n });
        }
        assert_eq!(s.row_free_values()[0], Some(12));
        assert_eq!(s.row_totals()[0], 5);
        let f = board_features(&s);
        assert!(f[12] > 0.0);

        // Row 2 (descending): descending rows mark high->low, so mark
        // 12, 11, 3 in that order -> free = 2 (terminal) with only 3 marks.
        let mut s = State::default();
        for n in [12u8, 11, 3] {
            s.apply_mark(Mark { row: 2, number: n });
        }
        assert_eq!(s.row_free_values()[2], Some(2));
        assert_eq!(s.row_totals()[2], 3);
        let f = board_features(&s);
        assert_eq!(f[14], 0.0, "descending terminal with <5 marks must contribute 0 ways");
    }

    #[test]
    fn pair_features_fresh_1v1_golden() {
        let a = State::default();
        let b = State::default();
        let f = pair_features(&a, &b, &[b]);

        // Both board blocks identical for fresh boards.
        assert_eq!(f[..BOARD_FEATURES], f[BOARD_FEATURES..2 * BOARD_FEATURES]);
        // Fresh rows: progress 0, marks 0, not locked.
        for i in 0..4 {
            assert_eq!(f[i], 0.0);
            assert_eq!(f[4 + i], 0.0);
            assert_eq!(f[8 + i], 0.0);
        }
        // Fresh free pointers sit on 2 (asc) / 12 (desc): 1 way each ->
        // (1/6) * (0+1)/11 = 1/66.
        for i in 0..4 {
            assert!((f[12 + i] - 1.0 / 66.0).abs() < 1e-6);
        }
        assert_eq!(f[16], 0.0); // strikes
        assert_eq!(f[17], 0.0); // blanks
        assert!(f[18] > 0.0); // aggregate wprob
        assert_eq!(f[19], 0.0); // lockable rows
                                // Pair-level: equal scores, 1 opponent, fresh opponent stats.
        assert_eq!(f[40], 0.0);
        assert_eq!(f[41], 0.25);
        assert_eq!(f[42], 0.0);
        assert_eq!(f[43], 0.0);
        assert_eq!(f[44], 0.0);
    }

    #[test]
    fn pair_features_swap_relationship() {
        // Asymmetric position: A has marks + a strike, B is fresh.
        let mut a = State::default();
        for n in [2u8, 3, 5] {
            a.apply_mark(Mark { row: 0, number: n });
        }
        a.apply_strike();
        let b = State::default();

        let ab = pair_features(&a, &b, &[b]);
        let ba = pair_features(&b, &a, &[a]);

        // Board blocks exchange.
        assert_eq!(ab[..BOARD_FEATURES], ba[BOARD_FEATURES..2 * BOARD_FEATURES]);
        assert_eq!(ab[BOARD_FEATURES..2 * BOARD_FEATURES], ba[..BOARD_FEATURES]);
        // current_diff negates.
        assert!((ab[40] + ba[40]).abs() < 1e-6);
        assert!(ab[40] != 0.0);
        // Opponent summaries describe the respective opponent.
        assert_eq!(ab[42], 0.0); // A's opponent (B) is fresh
        assert!((ba[42] - total_progress(&a)).abs() < 1e-6);
        assert!((ba[43] - 1.0 / 3.0).abs() < 1e-6); // A has 1 strike
    }
}
