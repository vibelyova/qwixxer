use crate::state::{post_state_dominates, Mark, State};
use super::Bot;

/// If any mark locks a row without ending the game, force it (pick highest
/// scoring lock). Locking a first row is pure upside — no reason to skip it.
fn find_safe_lock(state: &State, marks: &[Mark]) -> Option<Mark> {
    marks
        .iter()
        .copied()
        .filter(|&m| state.would_lock_row(m))
        .filter(|&m| {
            let mut s = *state;
            s.apply_mark(m);
            !would_end_game(&s)
        })
        .max_by_key(|&m| {
            let mut s = *state;
            s.apply_mark(m);
            s.count_points()
        })
}

/// Would a game end if `our_state` were the final state? Checks for 4+ strikes
/// or 2+ locked rows. Locks are globally consistent (propagated by the game
/// loop), so our state already reflects all global locks.
fn would_end_game(our_state: &State) -> bool {
    our_state.strikes >= 4 || our_state.count_locked() >= 2
}

/// Apply game-ending meta-rules to a list of (mark, post_state) candidates.
///
/// - If any candidate ends the game and guarantees a win (our score >
///   `opp_best_score`), returns it immediately.
/// - Removes candidates that end the game as a loss (our score < `opp_best_score`).
///
/// `opp_best_score` should be the worst-case (highest) opponent score —
/// for phase1 that means their best possible post-phase1 score,
/// for phase2 it's their current score (phase1 already applied).
fn apply_endgame_rules(
    marks: &[Mark],
    candidates: &[State],
    opp_best_score: isize,
) -> Option<Mark> {
    for (i, post) in candidates.iter().enumerate() {
        if would_end_game(post) && post.count_points() > opp_best_score {
            return Some(marks[i]);
        }
    }
    None
}

fn remove_losing_endgame(
    candidates: &[State],
    opp_best_score: isize,
) -> Vec<bool> {
    candidates
        .iter()
        .map(|post| !(would_end_game(post) && post.count_points() < opp_best_score))
        .collect()
}

fn passive_phase1_impl(bot: &impl Bot, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
    let white_sum = dice[0] + dice[1];
    let marks = state.generate_white_moves(white_sum);
    if marks.is_empty() {
        return None;
    }

    if let Some(m) = find_safe_lock(state, &marks) {
        return Some(m);
    }

    let mark_states: Vec<State> = marks
        .iter()
        .map(|&m| {
            let mut s = *state;
            s.apply_mark(m);
            s
        })
        .collect();

    // Opponent best score: they might also mark in phase1 (simultaneous).
    let opp_best = opp_best_phase1_score(opp_states, white_sum);

    // Force a winning game-end.
    if let Some(m) = apply_endgame_rules(&marks, &mark_states, opp_best) {
        return Some(m);
    }

    let keep = remove_losing_endgame(&mark_states, opp_best);
    let marks: Vec<Mark> = marks.iter().copied().zip(keep.iter()).filter(|(_, &k)| k).map(|(m, _)| m).collect();
    let mark_states: Vec<State> = mark_states.into_iter().zip(keep.iter()).filter(|(_, &k)| k).map(|(s, _)| s).collect();

    if marks.is_empty() {
        return None;
    }

    // Evaluate: marks + skip
    let mut candidates = mark_states;
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
    if marks.is_empty() {
        return None;
    }

    if let Some(m) = find_safe_lock(state, &marks) {
        return Some(m);
    }

    let mark_states: Vec<State> = marks
        .iter()
        .map(|&m| {
            let mut s = *state;
            s.apply_mark(m);
            s
        })
        .collect();

    let opp_best = opp_states
        .iter()
        .map(|s| s.count_points())
        .max()
        .unwrap_or(0);

    if let Some(m) = apply_endgame_rules(&marks, &mark_states, opp_best) {
        return Some(m);
    }

    let keep = remove_losing_endgame(&mark_states, opp_best);
    let marks: Vec<Mark> = marks.iter().copied().zip(keep.iter()).filter(|(_, &k)| k).map(|(m, _)| m).collect();
    let mark_states: Vec<State> = mark_states.into_iter().zip(keep.iter()).filter(|(_, &k)| k).map(|(s, _)| s).collect();

    if marks.is_empty() {
        return None;
    }

    let no_mark_state = if has_marked {
        *state
    } else {
        let mut s = *state;
        s.apply_strike();
        s
    };

    // Prune dominated: build full candidate list (marks + no-mark), prune,
    // then split back. No-mark is always at the end.
    let mut candidates = mark_states;
    candidates.push(no_mark_state);
    let keep: Vec<bool> = (0..candidates.len())
        .map(|i| {
            !candidates.iter().enumerate().any(|(j, s)| {
                j != i && post_state_dominates(s, &candidates[i])
            })
        })
        .collect();
    // If no-mark got pruned, it's fine — a dominating mark exists.
    // If all marks got pruned but no-mark survives, we skip.
    let mut surviving_marks: Vec<Mark> = Vec::new();
    let mut surviving_states: Vec<State> = Vec::new();
    let mut no_mark_survived = false;
    for (idx, &kept) in keep.iter().enumerate() {
        if !kept { continue; }
        if idx < marks.len() {
            surviving_marks.push(marks[idx]);
            surviving_states.push(candidates[idx]);
        } else {
            no_mark_survived = true;
        }
    }
    if surviving_marks.is_empty() {
        return None;
    }
    let mut candidates = surviving_states;
    let marks = surviving_marks;
    if no_mark_survived {
        candidates.push(no_mark_state);
    }
    let values = bot.evaluate_batch(&candidates, opp_states);
    if no_mark_survived {
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
    } else {
        // No-mark was dominated — a mark is strictly better. Pick the best.
        let best_idx = values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        Some(marks[best_idx])
    }
}

/// Best possible opponent score after phase1: each opponent picks their
/// highest-scoring white_sum mark (or skips if no mark improves their score).
fn opp_best_phase1_score(opp_states: &[State], white_sum: u8) -> isize {
    opp_states
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
        .unwrap_or(0)
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
    // Locks are consistent across all players after propagation.
    let locked = state.count_locked();

    let opp_best_score = opp_best_phase1_score(opp_states, white_sum);

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

    // --- Meta-rules: game-ending checks ---
    let opp_best = opp_best_phase1_score(opp_states, white_sum);

    let mut best_winning_plan: Option<(usize, isize)> = None;
    for (i, (_, _, post)) in plans.iter().enumerate() {
        if would_end_game(post) && post.count_points() > opp_best {
            let score = post.count_points();
            if best_winning_plan.map_or(true, |(_, s)| score > s) {
                best_winning_plan = Some((i, score));
            }
        }
    }
    if let Some((i, _)) = best_winning_plan {
        return plans[i].0;
    }

    plans.retain(|(_, _, post)| {
        !(would_end_game(post) && post.count_points() < opp_best)
    });
    if plans.is_empty() {
        let mut s = *state;
        s.apply_strike();
        plans.push((None, None, s));
    }

    // RISKY/LOCKABLE filtering — only when white_sum hits a terminal number.
    if let Some(forced) = filter_risky_plans(&mut plans, state, opp_states, white_sum) {
        return Some(forced);
    }

    // Force a non-game-ending lock if any surviving plan's phase1 mark locks.
    // Check on post-phase1 state only (not full plan end-state).
    for (phase1, _, _) in &plans {
        if let Some(m) = phase1 {
            if state.would_lock_row(*m) && {
                let mut s = *state;
                s.apply_mark(*m);
                !would_end_game(&s)
            } {
                return Some(*m);
            }
        }
    }

    // Prune plans whose end-state is strictly dominated by another plan's.
    {
        let keep: Vec<bool> = (0..plans.len())
            .map(|i| {
                !plans.iter().enumerate().any(|(j, (_, _, s))| {
                    j != i && post_state_dominates(s, &plans[i].2)
                })
            })
            .collect();
        let mut idx = 0;
        plans.retain(|_| { let k = keep[idx]; idx += 1; k });
    }

    // Evaluate remaining plans
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
