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

    // RISKY filtering (only when white_sum == 2 or white_sum == 12)
    if white_sum == 2 || white_sum == 12 {
        // Compute which rows opponents might lock
        let mut at_risk_rows: Vec<usize> = Vec::new();
        for opp in opp_states {
            for row in 0..4 {
                let terminal = State::row_terminal(row);
                if terminal == white_sum && opp.can_mark(row, terminal) && opp.would_lock_row(Mark { row, number: terminal }) {
                    if !at_risk_rows.contains(&row) {
                        at_risk_rows.push(row);
                    }
                }
            }
        }

        if !at_risk_rows.is_empty() {
            // Count how many rows are currently globally locked
            let mut global_locked = 0u8;
            for row in 0..4 {
                let locked_by_any = state.locked()[row]
                    || opp_states.iter().any(|o| o.locked()[row]);
                if locked_by_any {
                    global_locked += 1;
                }
            }

            // max_new_locks = number of at-risk rows
            let max_new_locks = at_risk_rows.len() as u8;

            if global_locked + max_new_locks >= 2 {
                // Game can end in phase1: remove plans with phase2 dependency
                let filtered: Vec<_> = plans
                    .iter()
                    .filter(|(_, phase2, _)| phase2.is_none())
                    .cloned()
                    .collect();
                if !filtered.is_empty() {
                    plans = filtered;
                }
                // If all plans removed, keep strike as fallback (already have it)
            } else {
                // Remove plans that use at-risk rows in phase2
                let filtered: Vec<_> = plans
                    .iter()
                    .filter(|(_, phase2, _)| {
                        match phase2 {
                            Some(cm) => !at_risk_rows.contains(&cm.row),
                            None => true,
                        }
                    })
                    .cloned()
                    .collect();
                if !filtered.is_empty() {
                    plans = filtered;
                }
            }
        }
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
