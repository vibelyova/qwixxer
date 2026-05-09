use super::Bot;
use crate::state::{Mark, State};
use std::cmp::Ordering;

fn prune_dominated<T>(items: &mut Vec<T>, state_of: impl Fn(&T) -> &State) {
    let n = items.len();
    let mut dominated = vec![false; n];
    for i in 0..n {
        if dominated[i] {
            continue;
        }
        for j in (i + 1)..n {
            if dominated[j] {
                continue;
            }
            match state_of(&items[i]).partial_cmp(state_of(&items[j])) {
                Some(Ordering::Greater) => dominated[j] = true,
                Some(Ordering::Less) => {
                    dominated[i] = true;
                    break;
                }
                _ => {}
            }
        }
    }
    let mut idx = 0;
    items.retain(|_| {
        let keep = !dominated[idx];
        idx += 1;
        keep
    });
}

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

/// If any mark locks a row without ending the game, force it.
fn find_safe_lock(state: &State, marks: &[Mark]) -> Option<Mark> {
    marks
        .iter()
        .copied()
        .filter(|&m| state.would_lock_row(m))
        .filter(|&m| {
            let mut s = *state;
            s.apply_mark(m);
            !s.would_end_game()
        })
        .max_by_key(|&m| {
            let mut s = *state;
            s.apply_mark(m);
            s.count_points()
        })
}

/// Core decision logic shared by passive_phase1 and active_phase2.
///
/// Given candidate marks and a baseline state (skip for passive, skip-or-strike
/// for phase2), applies meta-rules then evaluates with the bot:
///   1. Force winning baseline (e.g. winning 4th strike) → None
///   2. Force safe lock → Some(mark)
///   3. Force winning game-ending mark → Some(mark)
///   4. Filter out all losing game-ends (marks AND baseline)
///   5. Post-state dominance pruning
///   6. Evaluate survivors, return best
fn pick_best_mark(
    bot: &impl Bot,
    state: &State,
    opp_states: &[State],
    marks: &[Mark],
    baseline: State,
    opp_best: isize,
) -> Option<Mark> {
    if marks.is_empty() {
        return None;
    }

    if let Some(m) = find_safe_lock(state, marks) {
        return Some(m);
    }
    // TODO: smart strike

    let mark_states: Vec<State> = marks
        .iter()
        .map(|&m| {
            let mut s = *state;
            s.apply_mark(m);
            s
        })
        .collect();

    // Force highest-scoring winning game-end (mark or baseline).
    // Marks first — a winning lock beats a winning strike (avoids -5 penalty).
    if let Some((mark, _)) = mark_states
        .iter()
        .enumerate()
        .map(|(i, &s)| (Some(marks[i]), s))
        .chain(std::iter::once((None, baseline)))
        .filter(|(_, post)| post.would_end_game() && post.count_points() > opp_best)
        .max_by_key(|(_, post)| post.count_points())
    {
        return mark;
    }

    // Build candidates: marks + baseline. Filter out losing game-ends.
    // This handles "must mark" automatically: if baseline is a losing strike,
    // it gets filtered and only marks survive.
    let mut cands: Vec<(Option<Mark>, State)> = mark_states
        .iter()
        .enumerate()
        .map(|(i, &s)| (Some(marks[i]), s))
        .chain(std::iter::once((None, baseline)))
        .filter(|(_, s)| !(s.would_end_game() && s.count_points() < opp_best))
        .collect();

    if cands.is_empty() {
        return None;
    }
    if cands.len() == 1 {
        return cands[0].0;
    }

    prune_dominated(&mut cands, |(_, s)| s);

    if cands.is_empty() {
        return None;
    }
    if cands.len() == 1 {
        return cands[0].0;
    }

    // Evaluate all survivors
    let states: Vec<State> = cands.iter().map(|(_, s)| *s).collect();
    let values = bot.evaluate_batch(&states, opp_states);
    let best_idx = values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0;
    cands[best_idx].0
}

// ---------------------------------------------------------------------------

pub(crate) fn passive_phase1_impl(bot: &impl Bot, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
    let white_sum = dice[0] + dice[1];
    let marks = state.generate_white_moves(white_sum);
    let opp_best = opp_best_phase1_score(opp_states, white_sum);
    pick_best_mark(bot, state, opp_states, &marks, *state, opp_best)
}

pub(crate) fn active_phase2_impl(
    bot: &impl Bot,
    state: &State,
    opp_states: &[State],
    dice: [u8; 6],
    has_marked: bool,
) -> Option<Mark> {
    let opp_best = opp_states.iter().map(|s| s.count_points()).max().unwrap_or(0);
    let marks = state.generate_color_moves(dice);
    let baseline = if has_marked {
        *state
    } else {
        let mut s = *state;
        s.apply_strike();
        s
    };
    pick_best_mark(bot, state, opp_states, &marks, baseline, opp_best)
}

// ---------------------------------------------------------------------------

/// Simulate each opponent's likely phase1 decision by running the full
/// passive_phase1 pipeline (safe lock, endgame rules, evaluate) from
/// their perspective.
fn simulate_opp_phase1(bot: &impl Bot, state: &State, opp_states: &[State], dice: [u8; 6]) -> Vec<State> {
    opp_states
        .iter()
        .enumerate()
        .map(|(i, opp)| {
            let their_opps: Vec<State> = std::iter::once(*state)
                .chain(opp_states.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, s)| *s))
                .collect();
            match passive_phase1_impl(bot, opp, &their_opps, dice) {
                Some(m) => {
                    let mut s = *opp;
                    s.apply_mark(m);
                    s
                }
                None => *opp,
            }
        })
        .collect()
}

pub(crate) fn active_phase1_impl(bot: &impl Bot, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
    let white_sum = dice[0] + dice[1];

    // Simulate opponents' likely phase1 marks to get predicted post-phase1
    // opponent states. This replaces the RISKY filter: instead of conservatively
    // removing plans that might be invalidated, we predict what opponents will
    // do and plan around it.
    let sim_opp = simulate_opp_phase1(bot, state, opp_states, dice);
    let opp_best = sim_opp.iter().map(|s| s.count_points()).max().unwrap_or(0);

    let white_marks = state.generate_white_moves(white_sum);
    let color_marks = state.generate_color_moves(dice);

    let mut plans: Vec<(Option<Mark>, Option<Mark>, State)> = Vec::new();

    // Strike
    {
        let mut s = *state;
        s.apply_strike();
        plans.push((None, None, s));
    }
    // Color-only singles
    for &cm in &color_marks {
        let mut s = *state;
        s.apply_mark(cm);
        plans.push((None, Some(cm), s));
    }
    // White-only singles
    for &wm in &white_marks {
        let mut s = *state;
        s.apply_mark(wm);
        plans.push((Some(wm), None, s));
    }
    // Doubles
    for &wm in &white_marks {
        let mut post_white = *state;
        post_white.apply_mark(wm);
        for &cm in &post_white.generate_color_moves(dice) {
            let mut s = post_white;
            s.apply_mark(cm);
            plans.push((Some(wm), Some(cm), s));
        }
    }

    if plans.is_empty() {
        return None;
    }

    // Force best winning game-end
    let winning = plans
        .iter()
        .enumerate()
        .filter(|(_, (_, _, post))| post.would_end_game() && post.count_points() > opp_best)
        .max_by_key(|(_, (_, _, post))| post.count_points());
    if let Some((i, _)) = winning {
        return plans[i].0;
    }

    // Remove losing game-ends
    plans.retain(|(_, _, post)| !(post.would_end_game() && post.count_points() < opp_best));
    if plans.is_empty() {
        let mut s = *state;
        s.apply_strike();
        plans.push((None, None, s));
    }

    // Force safe lock
    for (phase1, _, _) in &plans {
        if let Some(m) = phase1 {
            if state.would_lock_row(*m) && {
                let mut s = *state;
                s.apply_mark(*m);
                !s.would_end_game()
            } {
                return Some(*m);
            }
        }
    }

    prune_dominated(&mut plans, |(_, _, s)| s);

    // Evaluate plans against simulated post-opponent states
    let post_states: Vec<State> = plans.iter().map(|(_, _, s)| *s).collect();
    let values = bot.evaluate_batch(&post_states, &sim_opp);
    let best_idx = values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0;
    plans[best_idx].0
}

// ---------------------------------------------------------------------------

impl<T: Bot + std::fmt::Debug> super::Strategy for T {
    fn active_phase1(&mut self, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
        active_phase1_impl(self, state, opp_states, dice)
    }

    fn active_phase2(&mut self, state: &State, opp_states: &[State], dice: [u8; 6], has_marked: bool) -> Option<Mark> {
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
