//! "aznet" value net: AlphaZero-style one-hot crossing-order representation
//! run through a shared per-row encoder. Afterstate value net like the pair
//! net (diff-space μ/σ heads, `(cdiff+μ)/σ` ranking), differing only in the
//! board encoding. 2-player only. Design:
//! docs/superpowers/specs/2026-06-15-aznet-representation-design.md

use crate::state::State;

/// Per-row one-hot block width.
pub const ROW_BLOCK: usize = 28;
/// Per-board raw width: 4 row blocks + one-hot strikes.
pub const BOARD_RAW: usize = 4 * ROW_BLOCK + 4; // 116
/// Full input: two boards + cdiff.
pub const AZ_FEATURES: usize = 2 * BOARD_RAW + 1; // 233

/// One-hot crossing-order block for row `i` of `state` (afterstate; no roll).
pub fn az_row_block(state: &State, i: usize) -> [f32; ROW_BLOCK] {
    let total = state.row_totals()[i];
    let free = state.row_free_values()[i];
    let ascending = i < 2;
    let terminal = State::row_terminal(i);
    let mut b = [0.0f32; ROW_BLOCK];

    // count one-hot [0..13)
    b[(total as usize).min(12)] = 1.0;

    // free-pointer slot one-hot [13..24); all-zero when locked (free == None)
    if let Some(fr) = free {
        let slot = if ascending { fr as usize - 2 } else { 12 - fr as usize };
        b[13 + slot] = 1.0;
    }

    // is_locked [24]
    b[24] = if free.is_none() { 1.0 } else { 0.0 };

    // is_lockable [25]: free pointer on the terminal AND >= 5 marks
    b[25] = if free == Some(terminal) && total >= 5 { 1.0 } else { 0.0 };

    // wprob scalar [26] (mirror of pair::board_features f[12+i])
    b[26] = match free {
        Some(fr) if fr == terminal && total < 5 => 0.0,
        Some(fr) => {
            let ways = 6.0 - (7.0f32 - fr as f32).abs();
            (ways / 6.0) * (total as f32 + 1.0) / 11.0
        }
        None => 0.0,
    };

    // blanks scalar [27]: cells skipped left of the pointer; 0 if locked
    b[27] = match free {
        Some(fr) => {
            let slot = if ascending { fr as f32 - 2.0 } else { 12.0 - fr as f32 };
            (slot - total as f32).max(0.0) / 10.0
        }
        None => 0.0,
    };

    b
}

/// Per-board block: 4 row blocks followed by one-hot strikes (index clamped to
/// 3 — a terminal 4th-strike afterstate maps to the strikes=3 slot; harmless,
/// terminal states train on the realized outcome).
pub fn az_board(state: &State) -> [f32; BOARD_RAW] {
    let mut b = [0.0f32; BOARD_RAW];
    for i in 0..4 {
        b[i * ROW_BLOCK..(i + 1) * ROW_BLOCK].copy_from_slice(&az_row_block(state, i));
    }
    b[4 * ROW_BLOCK + (state.strikes as usize).min(3)] = 1.0;
    b
}

/// Full 2-player input: own board, opponent board, clamped cdiff.
pub fn az_features(our: &State, opp: &State) -> [f32; AZ_FEATURES] {
    let mut f = [0.0f32; AZ_FEATURES];
    f[0..BOARD_RAW].copy_from_slice(&az_board(our));
    f[BOARD_RAW..2 * BOARD_RAW].copy_from_slice(&az_board(opp));
    let cdiff = (our.count_points() - opp.count_points()) as f32;
    f[2 * BOARD_RAW] = (cdiff / 100.0).clamp(-1.0, 1.0);
    f
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::Mark;

    // Block-local indices for readability.
    const FREE0: usize = 13; // first free-slot index
    const LOCKED: usize = 24;
    const LOCKABLE: usize = 25;
    const WPROB: usize = 26;
    const BLANKS: usize = 27;

    #[test]
    fn fresh_board_block() {
        let s = State::default();
        for i in 0..4 {
            let b = az_row_block(&s, i);
            assert_eq!(b[0], 1.0, "count one-hot at 0");
            assert_eq!(b[FREE0], 1.0, "free slot 0 set");
            assert_eq!(b[LOCKED], 0.0);
            assert_eq!(b[LOCKABLE], 0.0);
            assert!((b[WPROB] - 1.0 / 66.0).abs() < 1e-6, "fresh wprob = 1/66");
            assert_eq!(b[BLANKS], 0.0);
            // exactly two bits set (count + free slot)
            assert_eq!(b.iter().filter(|&&x| x == 1.0).count(), 2);
        }
    }

    #[test]
    fn frontier_one_hot_ascending() {
        let mut s = State::default();
        for n in [2u8, 3, 5] {
            s.apply_mark(Mark { row: 0, number: n });
        }
        // free = 6 -> slot 4; count = 3.
        let b = az_row_block(&s, 0);
        assert_eq!(b[3], 1.0, "count index 3");
        assert_eq!(b[FREE0 + 4], 1.0, "free slot 4");
        assert!((b[BLANKS] - 0.1).abs() < 1e-6, "blanks (4-3)/10 = 0.1");
    }

    #[test]
    fn frontier_one_hot_descending() {
        let mut s = State::default();
        for n in [12u8, 11, 9] {
            s.apply_mark(Mark { row: 2, number: n });
        }
        // descending free = 8 -> slot 12-8 = 4; count = 3.
        let b = az_row_block(&s, 2);
        assert_eq!(b[3], 1.0, "count index 3");
        assert_eq!(b[FREE0 + 4], 1.0, "free slot 4 (12-8)");
    }

    #[test]
    fn locked_row_block() {
        // R 2..=6 (5 marks) then mark 12 -> safe lock; total = 7.
        let mut s = State::default();
        for n in 2..=6 {
            s.apply_mark(Mark { row: 0, number: n });
        }
        s.apply_mark(Mark { row: 0, number: 12 });
        assert_eq!(s.row_free_values()[0], None, "row locked");
        let b = az_row_block(&s, 0);
        assert_eq!(b[7], 1.0, "count index = post-lock total 7");
        assert_eq!(b[LOCKED], 1.0);
        // free-slot one-hot all zero
        assert!(b[FREE0..FREE0 + 11].iter().all(|&x| x == 0.0), "no free slot when locked");
        assert_eq!(b[LOCKABLE], 0.0);
        assert_eq!(b[BLANKS], 0.0);
    }

    #[test]
    fn count_one_hot_reaches_twelve() {
        // Mark 2..=11 (10 marks, free=12) then 12 -> lock; total = 12.
        let mut s = State::default();
        for n in 2..=11 {
            s.apply_mark(Mark { row: 0, number: n });
        }
        s.apply_mark(Mark { row: 0, number: 12 });
        let b = az_row_block(&s, 0);
        assert_eq!(b[12], 1.0, "count index 12 (lock bonus) must exist");
    }

    #[test]
    fn is_lockable_set_on_terminal_with_five() {
        // R 2..=6 (5 marks) without the lock: free should sit on 12 only after
        // reaching it; build a row whose free is on the terminal with >=5 marks.
        let mut s = State::default();
        for n in [2u8, 3, 4, 5, 11] {
            s.apply_mark(Mark { row: 0, number: n });
        }
        // free = 12 (terminal), total = 5 -> lockable.
        assert_eq!(s.row_free_values()[0], Some(12));
        let b = az_row_block(&s, 0);
        assert_eq!(b[LOCKABLE], 1.0, "free on terminal + 5 marks => lockable");
    }

    #[test]
    fn is_lockable_present_on_both_boards() {
        let mut lockable = State::default();
        for n in [2u8, 3, 4, 5, 11] {
            lockable.apply_mark(Mark { row: 0, number: n });
        }
        let fresh = State::default();
        // our = lockable, opp = fresh: own block 0 lockable bit set.
        let f = az_features(&lockable, &fresh);
        assert_eq!(f[25], 1.0, "own row 0 lockable");
        // swap: opp = lockable -> the opponent's block 0 lockable bit set too.
        let f2 = az_features(&fresh, &lockable);
        assert_eq!(f2[BOARD_RAW + 25], 1.0, "opponent row 0 lockable (not zeroed)");
    }

    #[test]
    fn swap_relationship() {
        let mut a = State::default();
        for n in [2u8, 3, 5] {
            a.apply_mark(Mark { row: 0, number: n });
        }
        a.apply_strike();
        let b = State::default();

        let ab = az_features(&a, &b);
        let ba = az_features(&b, &a);
        // Board halves exchange.
        assert_eq!(ab[0..BOARD_RAW], ba[BOARD_RAW..2 * BOARD_RAW]);
        assert_eq!(ab[BOARD_RAW..2 * BOARD_RAW], ba[0..BOARD_RAW]);
        // cdiff negates and is nonzero.
        assert!((ab[2 * BOARD_RAW] + ba[2 * BOARD_RAW]).abs() < 1e-6);
        assert!(ab[2 * BOARD_RAW] != 0.0);
    }
}
