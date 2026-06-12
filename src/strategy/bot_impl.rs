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

pub(crate) fn opp_best_phase1_score(opp_states: &[State], white_sum: u8) -> isize {
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
pub(crate) fn find_safe_lock(state: &State, marks: &[Mark]) -> Option<Mark> {
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

/// Outcome of the pure-logic half of a decision pipeline: either the meta
/// rules fully determine the move, or a filtered+pruned candidate list
/// remains for value-based selection.
/// Pub for analysis examples (see examples/divergence.rs).
pub enum Decision {
    Forced(Option<Mark>),
    /// (move, post-move state) pairs; `None` = skip/baseline.
    Choices(Vec<(Option<Mark>, State)>),
}

/// Index of the maximum value, matching `Iterator::max_by` semantics
/// (last maximum wins) so refactored paths pick identical moves.
pub fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0
}

pub fn eval_decision(bot: &impl Bot, decision: Decision, eval_opps: &[State]) -> Option<Mark> {
    match decision {
        Decision::Forced(m) => m,
        Decision::Choices(cands) => {
            let states: Vec<State> = cands.iter().map(|(_, s)| *s).collect();
            let values = bot.evaluate_batch(&states, eval_opps);
            cands[argmax(&values)].0
        }
    }
}

/// Core decision logic shared by passive_phase1 and active_phase2.
///
/// Given candidate marks and a baseline state (skip for passive, skip-or-strike
/// for phase2), applies meta-rules and returns either a forced move or a
/// filtered+pruned candidate list for value-based selection.
pub(crate) fn mark_choices(state: &State, marks: &[Mark], baseline: State, opp_best: isize) -> Decision {
    mark_choices_with(state, marks, baseline, opp_best, true)
}

/// `mark_choices` with the safe-lock force gated by `force_lock`. With
/// `force_lock = false` the lock becomes an ordinary candidate (rule-free
/// pipeline used by search-value distillation); `true` reproduces production.
pub(crate) fn mark_choices_with(
    state: &State,
    marks: &[Mark],
    baseline: State,
    opp_best: isize,
    force_lock: bool,
) -> Decision {
    if marks.is_empty() {
        return Decision::Forced(None);
    }

    if force_lock {
        if let Some(m) = find_safe_lock(state, marks) {
            return Decision::Forced(Some(m));
        }
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
        return Decision::Forced(mark);
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
        return Decision::Forced(None);
    }
    if cands.len() == 1 {
        return Decision::Forced(cands[0].0);
    }

    prune_dominated(&mut cands, |(_, s)| s);

    if cands.is_empty() {
        return Decision::Forced(None);
    }
    if cands.len() == 1 {
        return Decision::Forced(cands[0].0);
    }

    Decision::Choices(cands)
}

// ---------------------------------------------------------------------------

pub(crate) fn passive_phase1_choices(state: &State, opp_states: &[State], dice: [u8; 6]) -> Decision {
    let white_sum = dice[0] + dice[1];
    let marks = state.generate_white_moves(white_sum);
    let opp_best = opp_best_phase1_score(opp_states, white_sum);
    mark_choices(state, &marks, *state, opp_best)
}

pub(crate) fn passive_phase1_impl(bot: &impl Bot, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
    eval_decision(bot, passive_phase1_choices(state, opp_states, dice), opp_states)
}

pub fn active_phase2_choices(state: &State, opp_states: &[State], dice: [u8; 6], has_marked: bool) -> Decision {
    let opp_best = opp_states.iter().map(|s| s.count_points()).max().unwrap_or(0);
    let marks = state.generate_color_moves(dice);
    let baseline = if has_marked {
        *state
    } else {
        let mut s = *state;
        s.apply_strike();
        s
    };
    mark_choices(state, &marks, baseline, opp_best)
}

pub(crate) fn active_phase2_impl(
    bot: &impl Bot,
    state: &State,
    opp_states: &[State],
    dice: [u8; 6],
    has_marked: bool,
) -> Option<Mark> {
    eval_decision(
        bot,
        active_phase2_choices(state, opp_states, dice, has_marked),
        opp_states,
    )
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

/// Pure-logic phase-1 plan pipeline. `comparison_opps` supplies `opp_best`
/// for the winning/losing endgame filters — predicted post-phase1 states in
/// the full pipeline, current states in the simulator's lite mode. Returned
/// `Choices` carry (phase1 mark, plan end-state); the phase-2 part of each
/// plan is internal (the chooser only commits phase 1).
pub(crate) fn phase1_plan_choices(state: &State, comparison_opps: &[State], dice: [u8; 6]) -> Decision {
    phase1_plan_choices_with(state, comparison_opps, dice, true)
}

/// `phase1_plan_choices` with the safe-lock force gated by `force_lock`. With
/// `force_lock = false` the lock stays an ordinary candidate (rule-free
/// pipeline used by search-value distillation); `true` reproduces production.
pub(crate) fn phase1_plan_choices_with(
    state: &State,
    comparison_opps: &[State],
    dice: [u8; 6],
    force_lock: bool,
) -> Decision {
    let white_sum = dice[0] + dice[1];
    let opp_best = comparison_opps.iter().map(|s| s.count_points()).max().unwrap_or(0);

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
        return Decision::Forced(None);
    }

    // Force best winning game-end
    let winning = plans
        .iter()
        .enumerate()
        .filter(|(_, (_, _, post))| post.would_end_game() && post.count_points() > opp_best)
        .max_by_key(|(_, (_, _, post))| post.count_points());
    if let Some((i, _)) = winning {
        return Decision::Forced(plans[i].0);
    }

    // Remove losing game-ends
    plans.retain(|(_, _, post)| !(post.would_end_game() && post.count_points() < opp_best));
    if plans.is_empty() {
        let mut s = *state;
        s.apply_strike();
        plans.push((None, None, s));
    }

    // Force safe lock
    if force_lock {
        for (phase1, _, _) in &plans {
            if let Some(m) = phase1 {
                if state.would_lock_row(*m) && {
                    let mut s = *state;
                    s.apply_mark(*m);
                    !s.would_end_game()
                } {
                    return Decision::Forced(Some(*m));
                }
            }
        }
    }

    prune_dominated(&mut plans, |(_, _, s)| s);

    Decision::Choices(plans.into_iter().map(|(p1, _, s)| (p1, s)).collect())
}

/// Full phase-1 pipeline: simulate opponents' phase-1 responses, then run the
/// plan pipeline against them. Returns the decision plus the simulated
/// post-phase1 opponent states (also the evaluation context for Choices).
pub fn active_phase1_choices(
    bot: &impl Bot,
    state: &State,
    opp_states: &[State],
    dice: [u8; 6],
) -> (Decision, Vec<State>) {
    // Simulate opponents' likely phase1 marks to get predicted post-phase1
    // opponent states. This replaces the RISKY filter: instead of conservatively
    // removing plans that might be invalidated, we predict what opponents will
    // do and plan around it.
    let sim_opp = simulate_opp_phase1(bot, state, opp_states, dice);
    let decision = phase1_plan_choices(state, &sim_opp, dice);
    (decision, sim_opp)
}

pub(crate) fn active_phase1_impl(bot: &impl Bot, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
    let (decision, sim_opp) = active_phase1_choices(bot, state, opp_states, dice);
    eval_decision(bot, decision, &sim_opp)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::Mark;

    /// A state with a safe lock available: red 2..6 marked, white sum 12
    /// completes it (first lock, does not end the game).
    fn lockable_state() -> State {
        let mut s = State::default();
        for n in 2..=6 {
            s.apply_mark(Mark { row: 0, number: n });
        }
        s
    }

    #[test]
    fn mark_choices_with_force_matches_production_and_without_skips_lock() {
        let s = lockable_state();
        let marks = vec![Mark { row: 0, number: 12 }, Mark { row: 1, number: 5 }];
        // Forced path: identical to mark_choices.
        match mark_choices_with(&s, &marks, s, 0, true) {
            Decision::Forced(Some(m)) => assert_eq!(m, Mark { row: 0, number: 12 }),
            d => panic!("expected forced lock, got {:?}", matches!(d, Decision::Choices(_))),
        }
        // Rule-free path: the lock is a candidate, not forced.
        match mark_choices_with(&s, &marks, s, 0, false) {
            Decision::Choices(c) => {
                assert!(c.iter().any(|(m, _)| *m == Some(Mark { row: 0, number: 12 })));
                assert!(c.len() >= 2);
            }
            Decision::Forced(_) => panic!("rule-free pipeline must not force here"),
        }
        // find_safe_lock is exposed and agrees.
        assert_eq!(find_safe_lock(&s, &marks), Some(Mark { row: 0, number: 12 }));
    }

    #[test]
    fn phase1_plan_choices_with_force_matches_production() {
        let s = lockable_state();
        let opps = [State::default()];
        let dice = [6, 6, 1, 1, 1, 1]; // white sum 12 completes the red lock
        match phase1_plan_choices_with(&s, &opps, dice, true) {
            Decision::Forced(Some(m)) => assert_eq!(m, Mark { row: 0, number: 12 }),
            _ => panic!("expected forced phase-1 lock"),
        }
        match phase1_plan_choices_with(&s, &opps, dice, false) {
            Decision::Choices(c) => {
                assert!(c.iter().any(|(m, _)| *m == Some(Mark { row: 0, number: 12 })))
            }
            Decision::Forced(_) => panic!("rule-free phase-1 pipeline must not force here"),
        }
    }
}
