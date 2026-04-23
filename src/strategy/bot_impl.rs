use crate::state::{Mark, State};
use super::Bot;

fn passive_phase1_impl(bot: &impl Bot, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
    let white_sum = dice[0] + dice[1];
    let marks = state.generate_white_moves(white_sum);
    if marks.is_empty() {
        return None;
    }
    let mut candidates: Vec<State> = marks
        .iter()
        .map(|&m| {
            let mut s = *state;
            s.apply_mark(m);
            s
        })
        .collect();
    candidates.push(*state); // skip
    let values = bot.evaluate_batch(&candidates, opp_states);
    let skip_value = *values.last().unwrap();
    let mark_values = &values[..marks.len()];
    let best_idx = mark_values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0;
    if mark_values[best_idx] > skip_value {
        Some(marks[best_idx])
    } else {
        None
    }
}

fn active_phase2_impl(
    bot: &impl Bot,
    state: &State,
    opp_states: &[State],
    dice: [u8; 6],
    has_marked: bool,
) -> Option<Mark> {
    let marks = state.generate_color_moves(dice);
    let no_mark_state = if has_marked {
        *state
    } else {
        let mut s = *state;
        s.apply_strike();
        s
    };
    if marks.is_empty() {
        return None;
    }
    let mut candidates: Vec<State> = marks
        .iter()
        .map(|&m| {
            let mut s = *state;
            s.apply_mark(m);
            s
        })
        .collect();
    candidates.push(no_mark_state);
    let values = bot.evaluate_batch(&candidates, opp_states);
    let no_mark_value = *values.last().unwrap();
    let mark_values = &values[..marks.len()];
    let best_idx = mark_values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0;
    if mark_values[best_idx] > no_mark_value {
        Some(marks[best_idx])
    } else {
        None
    }
}

/// Filter plans based on Phase 1 locking risks. Only relevant when
/// white_sum == 2 or 12 (terminal numbers that can trigger locks).
///
/// Returns `Some(mark)` if a winning lock should be forced immediately.
/// Otherwise mutates `plans` in place and returns `None`.
fn filter_risky_plans(
    plans: &mut Vec<(Option<Mark>, Option<Mark>, State)>,
    state: &State,
    opp_states: &[State],
    white_sum: u8,
) -> Option<Mark> {
    if white_sum != 2 && white_sum != 12 {
        return None;
    }

    // RISKY: rows where at least one opponent could lock by marking white_sum.
    // can_mark on a terminal number already requires total >= 5, so the mark
    // will always trigger a lock — no need for would_lock_row.
    let risky: Vec<usize> = (0..4)
        .filter(|&row| {
            State::row_terminal(row) == white_sum
                && opp_states.iter().any(|opp| opp.can_mark(row, white_sum))
        })
        .collect();

    // MAX_RISKY: maximum rows opponents can collectively lock in phase1.
    // Each opponent marks at most once, so this is min(opponents who can
    // lock *something*, distinct risky rows).
    let opponents_who_can_lock = opp_states
        .iter()
        .filter(|opp| risky.iter().any(|&row| opp.can_mark(row, white_sum)))
        .count();
    let max_risky = opponents_who_can_lock.min(risky.len()) as u8;

    // LOCKABLE: rows WE can lock in phase1 by marking white_sum.
    let lockable: Vec<usize> = (0..4)
        .filter(|&row| {
            State::row_terminal(row) == white_sum && state.can_mark(row, white_sum)
        })
        .collect();

    // LOCKED: rows already globally locked (before this turn).
    let locked = {
        let our = state.locked();
        (0..4)
            .filter(|&r| our[r] || opp_states.iter().any(|o| o.locked()[r]))
            .count() as u8
    };

    // Best possible opponent score after phase1: each opponent picks their
    // highest-scoring white_sum mark (or skips if no mark improves their score).
    let opp_best_score = opp_states
        .iter()
        .map(|opp| {
            let base = opp.count_points();
            opp.generate_white_moves(white_sum)
                .iter()
                .map(|&m| {
                    let mut s = *opp;
                    s.apply_mark(m);
                    s.count_points()
                })
                .max()
                .unwrap_or(base)
                .max(base)
        })
        .max()
        .unwrap_or(0);

    if locked + max_risky >= 2 {
        // Game can end in phase1 from opponent locks alone (+ already locked).
        // Remove ALL plans that rely on phase2.
        plans.retain(|(_, phase2, _)| phase2.is_none());
    } else if max_risky == 1 {
        // Implies LOCKED == 0 (otherwise locked + max_risky >= 2).
        if lockable.is_empty() {
            // We can't lock anything. Game cannot end in phase1.
            // But the risky row(s) might get locked → deny phase2 on those rows.
            plans.retain(|(_, phase2, _)| match phase2 {
                Some(cm) => !risky.contains(&cm.row),
                None => true,
            });
        } else {
            // We CAN lock. Game ends if we lock a DIFFERENT row than the one
            // the opponent locks. Since MAX_RISKY == 1 and LOCKED == 0, game
            // ending requires our lock + opponent's lock on distinct rows = 2.
            let game_can_end = if lockable.len() == 1 && risky.len() == 1 && lockable[0] == risky[0]
            {
                // Our only lockable IS their only risky row. Locking the same
                // row → 1 total lock. Game safe either way.
                false
            } else {
                true
            };

            if game_can_end {
                // If we lock and the game ends, check if we win/tie.
                // Compare against opponent's best possible post-phase1 score.
                let winning_lock = lockable.iter().find(|&&row| {
                    let mut s = *state;
                    s.apply_mark(Mark { row, number: white_sum });
                    s.count_points() >= opp_best_score
                });
                if let Some(&row) = winning_lock {
                    return Some(Mark { row, number: white_sum });
                }
                // No winning lock — remove locking plans on non-risky rows
                // (those guarantee 2 distinct locks → game ends as a loss).
                // Keep locking plans on risky rows (might overlap with
                // opponent's lock → only 1 lock → game safe).
                plans.retain(|(phase1, _, _)| match phase1 {
                    Some(m) => {
                        if state.would_lock_row(*m) {
                            risky.contains(&m.row)
                        } else {
                            true
                        }
                    }
                    None => true,
                });
            }
            // Whether or not game can end from us, the risky rows still
            // threaten phase2 (opponent might lock one of them).
            plans.retain(|(_, phase2, _)| match phase2 {
                Some(cm) => !risky.contains(&cm.row),
                None => true,
            });
        }
    } else if locked >= 1 && max_risky == 0 {
        // No opponent can lock anything, but rows are already locked.
        // We can lock → locked + 1 >= 2 → game ends.
        if !lockable.is_empty() {
            let winning_lock = lockable.iter().find(|&&row| {
                let mut s = *state;
                s.apply_mark(Mark { row, number: white_sum });
                s.count_points() >= opp_best_score
            });
            if let Some(&row) = winning_lock {
                return Some(Mark { row, number: white_sum });
            }
            // No winning lock — remove all our locking plans.
            plans.retain(|(phase1, _, _)| match phase1 {
                Some(m) => !state.would_lock_row(*m),
                None => true,
            });
        }
    }
    // else: LOCKED == 0 && MAX_RISKY == 0. Game cannot end. Proceed normally.

    // Fallback: ensure we always have at least the strike plan.
    if plans.is_empty() {
        let mut s = *state;
        s.apply_strike();
        plans.push((None, None, s));
    }

    None
}

/// Generate all possible (phase1, phase2) plans, apply RISKY filtering,
/// evaluate, and return the phase1 part of the best plan.
fn active_phase1_impl(
    bot: &impl Bot,
    state: &State,
    opp_states: &[State],
    dice: [u8; 6],
) -> Option<Mark> {
    let white_sum = dice[0] + dice[1];
    let white_marks = state.generate_white_moves(white_sum);
    let color_marks = state.generate_color_moves(dice);

    // A plan is (Option<Mark>, Option<Mark>, post_state)
    // where first is phase1 (white), second is phase2 (color)
    let mut plans: Vec<(Option<Mark>, Option<Mark>, State)> = Vec::new();

    // 1. Strike: (None, None) -> apply_strike
    {
        let mut s = *state;
        s.apply_strike();
        plans.push((None, None, s));
    }

    // 2. Color-only singles: (None, Some(cm)) -> apply cm
    for &cm in &color_marks {
        let mut s = *state;
        s.apply_mark(cm);
        plans.push((None, Some(cm), s));
    }

    // 3. White-only singles: (Some(wm), None) -> apply wm
    for &wm in &white_marks {
        let mut s = *state;
        s.apply_mark(wm);
        plans.push((Some(wm), None, s));
    }

    // 4. Doubles: (Some(wm), Some(cm)) -> for each wm, generate color moves from POST-white state
    for &wm in &white_marks {
        let mut post_white = *state;
        post_white.apply_mark(wm);
        let post_color_marks = post_white.generate_color_moves(dice);
        for &cm in &post_color_marks {
            let mut s = post_white;
            s.apply_mark(cm);
            plans.push((Some(wm), Some(cm), s));
        }
    }

    if plans.is_empty() {
        return None;
    }

    // RISKY/LOCKABLE filtering — only when white_sum hits a terminal number.
    if let Some(forced) = filter_risky_plans(&mut plans, state, opp_states, white_sum) {
        return Some(forced);
    }

    // Evaluate all plans
    let post_states: Vec<State> = plans.iter().map(|(_, _, s)| *s).collect();
    let values = bot.evaluate_batch(&post_states, opp_states);

    let best_idx = values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0;

    plans[best_idx].0
}

impl<T: Bot + std::fmt::Debug> super::Strategy for T {
    fn active_phase1(
        &mut self,
        state: &State,
        opp_states: &[State],
        dice: [u8; 6],
    ) -> Option<Mark> {
        active_phase1_impl(self, state, opp_states, dice)
    }

    fn active_phase2(
        &mut self,
        state: &State,
        opp_states: &[State],
        dice: [u8; 6],
        has_marked: bool,
    ) -> Option<Mark> {
        active_phase2_impl(self, state, opp_states, dice, has_marked)
    }

    fn passive_phase1(
        &mut self,
        state: &State,
        opp_states: &[State],
        dice: [u8; 6],
        _active_player: usize,
    ) -> Option<Mark> {
        passive_phase1_impl(self, state, opp_states, dice)
    }
}
