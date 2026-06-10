# Decision-Time Search ("pair-search" bot) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `pair-search` bot that, at gated active-turn decisions, ranks the top-2 candidates by truncated lockstep-batched rollouts bootstrapped with the pair net's win probability, instead of static value.

**Architecture:** Generic `SearchBot<B: WinProb>` (any `Bot` with a win-probability capability) implementing `Strategy`. Candidates come from refactored shared `*_choices()` pipelines in `bot_impl.rs` (meta-rules single-source). A dedicated lockstep simulator (`strategy/sim.rs`) advances all `2×64` rollouts phase-by-phase, batching every phase's net evaluations through a new `Bot::evaluate_batch_multi`. Dice seeds are hashed from the decision context: stateless-deterministic + CRN across candidates. Spec: `docs/superpowers/specs/2026-06-10-pair-search-design.md`.

**Tech Stack:** Rust, burn 0.20 NdArray. New files in `src/strategy/` (the machinery is Bot-generic — conscious amendment of the spec's `src/dqn/search.rs` placement; the pair-specific `WinProb` impl lives in `src/dqn/pair.rs`).

---

## File structure

```
Modify: src/strategy/mod.rs      — Bot::evaluate_batch_multi default; register sim + search modules
Modify: src/strategy/bot_impl.rs — Decision enum, *_choices() pipelines, argmax/eval_decision helpers
Create: src/strategy/sim.rs      — SimGame, Fidelity, play_sim_turn, BatchedRollouts (lockstep driver)
Create: src/strategy/search.rs   — phi, context_seed, WinProb trait, SearchBot + gating + SearchStats
Modify: src/dqn/pair.rs          — evaluate_batch_multi override, WinProb impl for PairStrategy
Modify: src/main.rs              — BotType::PairSearch
Create: examples/search_calibration.rs — gate-tuning diagnostic
Modify: README.md, docs/ARCHITECTURE.md, docs/EXPERIMENTS.md — final task
```

Conventions: `cargo fmt` before every commit; default features throughout. The repo currently has 52 passing tests — they must stay green after every task.

---

### Task 1: `Bot::evaluate_batch_multi` + PairStrategy batched override

**Files:**
- Modify: `src/strategy/mod.rs` (Bot trait)
- Modify: `src/dqn/pair.rs`

- [ ] **Step 1: Add the default method to the `Bot` trait in `src/strategy/mod.rs`**

```rust
pub trait Bot: std::fmt::Debug {
    fn evaluate(&self, our_state: &State, opp_states: &[State]) -> f32;
    fn evaluate_batch(&self, candidates: &[State], opp_states: &[State]) -> Vec<f32> {
        candidates.iter().map(|s| self.evaluate(s, opp_states)).collect()
    }
    /// Evaluate several independent (candidates, opp_states) groups. The
    /// default loops `evaluate_batch`; bots with batched inference override
    /// this to run all groups in a single forward pass.
    fn evaluate_batch_multi(&self, groups: &[(&[State], &[State])]) -> Vec<Vec<f32>> {
        groups.iter().map(|(c, o)| self.evaluate_batch(c, o)).collect()
    }
}
```

- [ ] **Step 2: Rework `impl Bot for PairStrategy` in `src/dqn/pair.rs`**

Replace the body of `evaluate_batch` and add the override so all ranking flows
through one helper (the existing per-candidate logic moves into the multi form):

```rust
impl Bot for PairStrategy {
    fn evaluate(&self, our_state: &State, opp_states: &[State]) -> f32 {
        self.evaluate_batch(&[*our_state], opp_states)[0]
    }

    fn evaluate_batch(&self, candidates: &[State], opp_states: &[State]) -> Vec<f32> {
        self.evaluate_batch_multi(&[(candidates, opp_states)]).pop().unwrap()
    }

    fn evaluate_batch_multi(&self, groups: &[(&[State], &[State])]) -> Vec<Vec<f32>> {
        let default_opps = [State::default()];
        // Per group: (leader points, slice of opp states actually used).
        let mut leaders: Vec<isize> = Vec::with_capacity(groups.len());
        let mut feats: Vec<[f32; PAIR_FEATURES]> = Vec::new();
        for (candidates, opp_states) in groups {
            // Solo fallback: rank against an empty default board. Solo play is
            // officially unsupported for the pair bot (the old DQN covers it).
            let opps: &[State] = if opp_states.is_empty() { &default_opps } else { opp_states };
            let leader = opps.iter().max_by_key(|s| s.count_points()).unwrap();
            for c in *candidates {
                feats.push(pair_features(c, leader, opps));
            }
            leaders.push(leader.count_points());
        }
        let values = pair_batch_forward(&self.model, &self.device, &feats);

        let mut out = Vec::with_capacity(groups.len());
        let mut idx = 0;
        for ((candidates, _), leader_points) in groups.iter().zip(leaders) {
            let group = candidates
                .iter()
                .map(|cand| {
                    let (mu, log_var) = values[idx];
                    idx += 1;
                    let cdiff = (cand.count_points() - leader_points) as f32;
                    let sigma = (0.5 * log_var.clamp(LOG_VAR_MIN, LOG_VAR_MAX)).exp();
                    (cdiff + mu) / sigma
                })
                .collect();
            out.push(group);
        }
        out
    }
}
```

Delete the old `evaluate_batch` body (the leader/feats/zip logic) — it is now the
single-group case of the multi form. The doc comment about the solo fallback moves
with it.

- [ ] **Step 3: Add a segmentation test to the `tests` module in `src/dqn/pair.rs`**

```rust
    #[test]
    fn evaluate_batch_multi_matches_per_group_calls() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let bot = PairStrategy::from_model(PairModelConfig::new().init::<MyBackend>(&device), device);

        let mut a = State::default();
        a.apply_mark(Mark { row: 0, number: 4 });
        let mut b = State::default();
        b.apply_mark(Mark { row: 2, number: 9 });
        let mut opp = State::default();
        opp.apply_mark(Mark { row: 1, number: 6 });

        let g1_c = [State::default(), a];
        let g1_o = [opp];
        let g2_c = [b];
        let g2_o = [State::default(), a]; // different leader situation

        let multi = bot.evaluate_batch_multi(&[(&g1_c[..], &g1_o[..]), (&g2_c[..], &g2_o[..])]);
        let solo1 = bot.evaluate_batch(&g1_c, &g1_o);
        let solo2 = bot.evaluate_batch(&g2_c, &g2_o);
        assert_eq!(multi.len(), 2);
        assert_eq!(multi[0], solo1);
        assert_eq!(multi[1], solo2);
    }
```

- [ ] **Step 4: Run tests**

Run: `cargo test --lib dqn::pair`
Expected: 6 pair tests + 4 pair_train tests pass. Then `cargo test 2>&1 | grep "test result"` — all green.

- [ ] **Step 5: Commit**

```bash
cargo fmt && git add src/strategy/mod.rs src/dqn/pair.rs && git commit -m "feat(search): Bot::evaluate_batch_multi with batched PairStrategy override"
```

---

### Task 2: bot_impl choices refactor (meta-rules single-source) + byte-identical regression gate

**Files:**
- Modify: `src/strategy/bot_impl.rs`
- Modify: `src/strategy/mod.rs` (re-exports)

- [ ] **Step 1: Capture BEFORE outputs of the regression benches**

```bash
cargo build --release
./target/release/qwixxer bench pair ga -n 10000 > /tmp/before_pair_ga.txt
./target/release/qwixxer bench dqn ga -n 10000 > /tmp/before_dqn_ga.txt
./target/release/qwixxer bench ga opportunist -n 10000 > /tmp/before_ga_opp.txt
```

- [ ] **Step 2: Add `Decision`, `argmax`, `eval_decision` to `src/strategy/bot_impl.rs`**

```rust
/// Outcome of the pure-logic half of a decision pipeline: either the meta
/// rules fully determine the move, or a filtered+pruned candidate list
/// remains for value-based selection.
pub(crate) enum Decision {
    Forced(Option<Mark>),
    /// (move, post-move state) pairs; `None` = skip/baseline.
    Choices(Vec<(Option<Mark>, State)>),
}

/// Index of the maximum value, matching `Iterator::max_by` semantics
/// (last maximum wins) so refactored paths pick identical moves.
pub(crate) fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0
}

pub(crate) fn eval_decision(bot: &impl Bot, decision: Decision, eval_opps: &[State]) -> Option<Mark> {
    match decision {
        Decision::Forced(m) => m,
        Decision::Choices(cands) => {
            let states: Vec<State> = cands.iter().map(|(_, s)| *s).collect();
            let values = bot.evaluate_batch(&states, eval_opps);
            cands[argmax(&values)].0
        }
    }
}
```

- [ ] **Step 3: Split `pick_best_mark` into pure logic + evaluation**

Replace `pick_best_mark` with `mark_choices` + a thin wrapper. The logic and its
ORDER must be copied exactly from the current `pick_best_mark` (safe lock first,
then winning game-end, then losing-end filter, the two len-checks, then dominance
pruning) — only the final evaluate block moves out:

```rust
/// Pure-logic half of the passive/phase-2 decision (see `pick_best_mark`'s
/// old doc comment, steps 1–5). Evaluation is the caller's job.
pub(crate) fn mark_choices(state: &State, marks: &[Mark], baseline: State, opp_best: isize) -> Decision {
    if marks.is_empty() {
        return Decision::Forced(None);
    }

    if let Some(m) = find_safe_lock(state, marks) {
        return Decision::Forced(Some(m));
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
```

Rewire the two consumers (public signatures unchanged):

```rust
pub(crate) fn passive_phase1_choices(state: &State, opp_states: &[State], dice: [u8; 6]) -> Decision {
    let white_sum = dice[0] + dice[1];
    let marks = state.generate_white_moves(white_sum);
    let opp_best = opp_best_phase1_score(opp_states, white_sum);
    mark_choices(state, &marks, *state, opp_best)
}

pub(crate) fn passive_phase1_impl(bot: &impl Bot, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
    eval_decision(bot, passive_phase1_choices(state, opp_states, dice), opp_states)
}

pub(crate) fn active_phase2_choices(state: &State, opp_states: &[State], dice: [u8; 6], has_marked: bool) -> Decision {
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
    eval_decision(bot, active_phase2_choices(state, opp_states, dice, has_marked), opp_states)
}
```

Delete the old `pick_best_mark`.

- [ ] **Step 4: Split `active_phase1_impl` the same way**

The plan-building + endgame filters + safe-lock + pruning move into a pure
function parameterized by the comparison opponents (the full pipeline passes the
simulated post-phase1 states; the simulator's lite mode will pass current states):

```rust
/// Pure-logic phase-1 plan pipeline. `comparison_opps` supplies `opp_best`
/// for the winning/losing endgame filters — predicted post-phase1 states in
/// the full pipeline, current states in the simulator's lite mode. Returned
/// `Choices` carry (phase1 mark, plan end-state); the phase-2 part of each
/// plan is internal (the chooser only commits phase 1).
pub(crate) fn phase1_plan_choices(state: &State, comparison_opps: &[State], dice: [u8; 6]) -> Decision {
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

    prune_dominated(&mut plans, |(_, _, s)| s);

    Decision::Choices(plans.into_iter().map(|(p1, _, s)| (p1, s)).collect())
}

/// Full phase-1 pipeline: simulate opponents' phase-1 responses, then run the
/// plan pipeline against them. Returns the decision plus the simulated
/// post-phase1 opponent states (also the evaluation context for Choices).
pub(crate) fn active_phase1_choices(
    bot: &impl Bot,
    state: &State,
    opp_states: &[State],
    dice: [u8; 6],
) -> (Decision, Vec<State>) {
    let sim_opp = simulate_opp_phase1(bot, state, opp_states, dice);
    let decision = phase1_plan_choices(state, &sim_opp, dice);
    (decision, sim_opp)
}

pub(crate) fn active_phase1_impl(bot: &impl Bot, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
    let (decision, sim_opp) = active_phase1_choices(bot, state, opp_states, dice);
    eval_decision(bot, decision, &sim_opp)
}
```

Note the one intentional behavior-preserving detail: the old code, when `plans`
became a single strike plan after the losing-end filter, still went through
pruning+evaluation of that single plan — the result is identical to evaluating a
1-element `Choices`, which is what now happens. The old single-plan case never
returned early, and `eval_decision` of 1 choice returns it; behavior identical.

- [ ] **Step 5: Update re-exports in `src/strategy/mod.rs`**

```rust
pub(crate) use bot_impl::{
    active_phase1_choices, active_phase1_impl, active_phase2_choices, active_phase2_impl, argmax, eval_decision,
    mark_choices, passive_phase1_choices, passive_phase1_impl, phase1_plan_choices, Decision,
};
```

- [ ] **Step 6: Verify byte-identical behavior**

```bash
cargo test 2>&1 | grep "test result"        # all green
cargo build --release
./target/release/qwixxer bench pair ga -n 10000 > /tmp/after_pair_ga.txt
./target/release/qwixxer bench dqn ga -n 10000 > /tmp/after_dqn_ga.txt
./target/release/qwixxer bench ga opportunist -n 10000 > /tmp/after_ga_opp.txt
diff /tmp/before_pair_ga.txt /tmp/after_pair_ga.txt
diff /tmp/before_dqn_ga.txt /tmp/after_dqn_ga.txt
diff /tmp/before_ga_opp.txt /tmp/after_ga_opp.txt
```

Expected: all three diffs EMPTY. If any differ, the refactor changed decision
behavior — find and fix the divergence before proceeding (order of meta-rule
checks and `max_by` tie-breaking are the usual suspects).

- [ ] **Step 7: Commit**

```bash
cargo fmt && git add src/strategy/ && git commit -m "refactor(strategy): split decision pipelines into pure choices + evaluation"
```

---

### Task 3: phi, context seeds, `WinProb` trait + PairStrategy impl

**Files:**
- Create: `src/strategy/search.rs` (scaffold — SearchBot comes in Task 6)
- Modify: `src/strategy/mod.rs` (register module)
- Modify: `src/dqn/pair.rs` (WinProb impl)

- [ ] **Step 1: Create `src/strategy/search.rs`**

```rust
//! Decision-time search: gated truncated rollouts bootstrapped by a win
//! probability. Generic over any [`WinProb`] bot.
//!
//! Design: docs/superpowers/specs/2026-06-10-pair-search-design.md

use super::Bot;
use crate::state::State;

/// Candidates searched per decision (top by static value).
pub const K_CANDIDATES: usize = 2;
/// Sampled futures per candidate (CRN: dice shared across candidates).
pub const K_SAMPLES: usize = 64;
/// Top-2 static value gap (bot's evaluate units) below which search triggers.
pub const GATE_MARGIN: f32 = 0.15;

/// Capability for search leaf scoring: a calibrated win probability, batched
/// over independent (our_state, opp_states) groups in one inference call.
pub trait WinProb: Bot {
    /// P(group's player finishes ahead of its leading opponent), one per group.
    fn win_prob_multi(&self, groups: &[(&State, &[State])]) -> Vec<f32>;
}

/// Standard normal CDF via the Abramowitz–Stegun erf approximation
/// (7.1.26, |error| < 1.5e-7).
pub fn phi(z: f32) -> f32 {
    0.5 * (1.0 + erf(z as f64 / std::f64::consts::SQRT_2)) as f32
}

fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn absorb_state(mut h: u64, s: &State) -> u64 {
    let totals = s.row_totals();
    let frees = s.row_free_values();
    for i in 0..4 {
        // (total, free) fully determines a row; locked == free.is_none().
        h = splitmix64(h ^ (((totals[i] as u64) << 8) | frees[i].map_or(0xFF, |f| f as u64)));
    }
    splitmix64(h ^ s.strikes as u64)
}

/// Deterministic seed from the decision context. Candidate-independent by
/// construction (the candidate is not hashed), so all candidates of one
/// decision share dice streams (common random numbers), and identical
/// contexts replay identically (reproducible benches).
pub fn context_seed(our: &State, opps: &[State], dice: [u8; 6]) -> u64 {
    let mut h = absorb_state(0x9E3779B97F4A7C15, our);
    for o in opps {
        h = absorb_state(h, o);
    }
    let packed = dice.iter().fold(0u64, |acc, &d| (acc << 8) | d as u64);
    splitmix64(h ^ packed)
}

/// Per-player dice-stream seed for one rollout sample.
pub fn sample_player_seed(decision_seed: u64, sample: usize, player: usize) -> u64 {
    splitmix64(decision_seed ^ ((sample as u64) << 16) ^ player as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::Mark;

    #[test]
    fn phi_matches_known_values() {
        assert!((phi(0.0) - 0.5).abs() < 1e-6);
        assert!((phi(1.0) - 0.8413).abs() < 1e-3);
        assert!((phi(1.96) - 0.9750).abs() < 1e-3);
        assert!((phi(-1.0) - (1.0 - phi(1.0))).abs() < 1e-6);
        assert!(phi(10.0) > 0.9999);
        assert!(phi(-10.0) < 0.0001);
    }

    #[test]
    fn context_seed_is_deterministic_and_sensitive() {
        let a = State::default();
        let mut b = State::default();
        b.apply_mark(Mark { row: 0, number: 5 });
        let dice = [1, 2, 3, 4, 5, 6];

        assert_eq!(context_seed(&a, &[b], dice), context_seed(&a, &[b], dice));
        assert_ne!(context_seed(&a, &[b], dice), context_seed(&b, &[a], dice));
        assert_ne!(context_seed(&a, &[b], dice), context_seed(&a, &[b], [2, 1, 3, 4, 5, 6]));
        // Sample/player streams are distinct.
        let s = context_seed(&a, &[b], dice);
        assert_ne!(sample_player_seed(s, 0, 0), sample_player_seed(s, 0, 1));
        assert_ne!(sample_player_seed(s, 0, 0), sample_player_seed(s, 1, 0));
    }
}
```

- [ ] **Step 2: Register in `src/strategy/mod.rs`**

```rust
pub mod search;
```

(after the `mod bot_impl;` line; keep the existing re-exports unchanged).

- [ ] **Step 3: Implement `WinProb` for `PairStrategy` in `src/dqn/pair.rs`**

```rust
impl crate::strategy::search::WinProb for PairStrategy {
    fn win_prob_multi(&self, groups: &[(&State, &[State])]) -> Vec<f32> {
        let default_opps = [State::default()];
        let mut feats = Vec::with_capacity(groups.len());
        let mut cdiffs = Vec::with_capacity(groups.len());
        for (our, opps) in groups {
            let opps: &[State] = if opps.is_empty() { &default_opps } else { opps };
            let leader = opps.iter().max_by_key(|s| s.count_points()).unwrap();
            feats.push(pair_features(our, leader, opps));
            cdiffs.push((our.count_points() - leader.count_points()) as f32);
        }
        pair_batch_forward(&self.model, &self.device, &feats)
            .into_iter()
            .zip(cdiffs)
            .map(|((mu, log_var), cdiff)| {
                let sigma = (0.5 * log_var.clamp(LOG_VAR_MIN, LOG_VAR_MAX)).exp();
                crate::strategy::search::phi((cdiff + mu) / sigma)
            })
            .collect()
    }
}
```

Add a test in the pair tests module:

```rust
    #[test]
    fn win_prob_multi_is_probability_and_monotone_in_score() {
        use crate::strategy::search::WinProb;
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let bot = PairStrategy::from_model(PairModelConfig::new().init::<MyBackend>(&device), device);

        // Same boards, but in g2 we are 30 points ahead via a locked-ish row.
        let behind = State::default();
        let mut ahead = State::default();
        for n in 2..=8 {
            ahead.apply_mark(Mark { row: 0, number: n });
        }
        let opp = State::default();

        let p = bot.win_prob_multi(&[(&behind, &[opp]), (&ahead, &[opp])]);
        assert_eq!(p.len(), 2);
        assert!(p.iter().all(|x| (0.0..=1.0).contains(x)));
        // 28-point lead with same opponent must not be rated worse.
        assert!(p[1] >= p[0]);
    }
```

- [ ] **Step 4: Run tests, commit**

Run: `cargo test --lib 2>&1 | grep "test result"` — all green (2 new search tests, 1 new pair test).

```bash
cargo fmt && git add src/strategy/ src/dqn/pair.rs && git commit -m "feat(search): WinProb trait, normal CDF, deterministic context seeds"
```

---

### Task 4: `SimGame` + per-sim turn loop + Game-equivalence test

**Files:**
- Create: `src/strategy/sim.rs`
- Modify: `src/strategy/mod.rs` (register)

- [ ] **Step 1: Create `src/strategy/sim.rs`**

```rust
//! Lightweight game simulator for decision-time search. Replicates the
//! mechanics of `Game::play` (simultaneous phase 1 on pre-snapshots, lock
//! propagation, game-over ordering, phase 2) on plain `State`s, entrable
//! mid-turn, with all players' decisions driven by a `Bot`.

use super::bot_impl::{
    active_phase1_impl, active_phase2_choices, eval_decision, passive_phase1_choices, phase1_plan_choices, Decision,
};
use super::Bot;
use crate::game::DiceSource;
use crate::state::{Mark, State};
use rand::rngs::SmallRng;

/// Decision fidelity inside simulations.
#[derive(Clone, Copy, PartialEq)]
pub enum Fidelity {
    /// Identical to the blanket Strategy impl (incl. nested opponent
    /// simulation in active phase 1). Used by the Game-equivalence test.
    Full,
    /// Active phase 1 evaluates plans against current opponent states — no
    /// nested opponent simulation. Used inside search rollouts (one net
    /// evaluation per decision; sims are approximations).
    Lite,
}

pub struct SimGame {
    pub states: Vec<State>,
    /// Player to take the next turn. Player 0 is the searcher's seat.
    pub active: usize,
    /// Per-player dice streams (mirrors `Game`, where each player rolls).
    pub rngs: Vec<SmallRng>,
    pub over: bool,
}

impl SimGame {
    pub fn n(&self) -> usize {
        self.states.len()
    }

    pub fn game_over(states: &[State]) -> bool {
        states.iter().any(|s| s.strikes >= 4)
            || states.iter().map(|s| s.count_locked()).max().unwrap() >= 2
    }

    pub fn propagate_locks(states: &mut [State]) {
        let mut locked = [false; 4];
        for s in states.iter() {
            let l = s.locked();
            for row in 0..4 {
                locked[row] |= l[row];
            }
        }
        for s in states.iter_mut() {
            s.lock(locked);
        }
    }

    /// Turn-ordered opponent view for player `j` over `states`.
    pub fn opp_view(states: &[State], j: usize) -> Vec<State> {
        let n = states.len();
        (1..n).map(|off| states[(j + off) % n]).collect()
    }

    /// Exact outcome for player 0: 1.0 win, 0.5 top tie, 0.0 loss.
    pub fn outcome(states: &[State]) -> f32 {
        let our = states[0].count_points();
        let best_opp = states[1..].iter().map(|s| s.count_points()).max().unwrap();
        match our.cmp(&best_opp) {
            std::cmp::Ordering::Greater => 1.0,
            std::cmp::Ordering::Equal => 0.5,
            std::cmp::Ordering::Less => 0.0,
        }
    }
}

/// Advance one sim by one full turn (roll, phase 1 all players, locks,
/// game-over, phase 2, locks, game-over, advance). Reference implementation —
/// the lockstep driver must stay decision-equivalent to this.
pub fn play_sim_turn(bot: &impl Bot, sim: &mut SimGame, fidelity: Fidelity) {
    if sim.over {
        return;
    }
    let n = sim.n();
    let dice = sim.rngs[sim.active].roll();
    let snapshot = sim.states.clone();

    // Phase 1: all players decide against pre-phase1 snapshots.
    let mut marks: Vec<Option<Mark>> = Vec::with_capacity(n);
    for j in 0..n {
        let view = SimGame::opp_view(&snapshot, j);
        let m = if j == sim.active {
            match fidelity {
                Fidelity::Full => active_phase1_impl(bot, &snapshot[j], &view, dice),
                Fidelity::Lite => eval_decision(bot, phase1_plan_choices(&snapshot[j], &view, dice), &view),
            }
        } else {
            eval_decision(bot, passive_phase1_choices(&snapshot[j], &view, dice), &view)
        };
        marks.push(m);
    }
    let has_marked = marks[sim.active].is_some();
    for (j, m) in marks.iter().enumerate() {
        if let Some(m) = m {
            sim.states[j].apply_mark(*m);
        }
    }
    SimGame::propagate_locks(&mut sim.states);
    if SimGame::game_over(&sim.states) {
        sim.over = true;
        return;
    }

    // Phase 2: active player only.
    let view = SimGame::opp_view(&sim.states, sim.active);
    let decision = active_phase2_choices(&sim.states[sim.active], &view, dice, has_marked);
    match eval_decision(bot, decision, &view) {
        Some(m) => sim.states[sim.active].apply_mark(m),
        None if !has_marked => sim.states[sim.active].apply_strike(),
        None => {}
    }
    SimGame::propagate_locks(&mut sim.states);
    if SimGame::game_over(&sim.states) {
        sim.over = true;
        return;
    }

    sim.active = (sim.active + 1) % n;
}
```

Note: passive decisions go through `passive_phase1_choices` + `eval_decision`,
which is exactly `passive_phase1_impl` (post-Task-2) — written this way so Task 5's
lockstep driver can batch the same `Decision` values.

- [ ] **Step 2: Register in `src/strategy/mod.rs`**

```rust
pub mod sim;
```

- [ ] **Step 3: Add the Game-equivalence test to `src/strategy/sim.rs`**

This is the key correctness gate: identical strategies + identical dice streams
through `Game::play` and the sim loop must yield identical final states. It needs
a `PairStrategy`, so it lives behind the same module (dqn is always compiled).

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::dqn::pair::{PairModelConfig, PairStrategy};
    use crate::dqn::MyBackend;
    use crate::game::{Game, Player};
    use rand::SeedableRng;
    use std::sync::Arc;

    #[test]
    fn sim_turn_loop_matches_game_play() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let model = Arc::new(PairModelConfig::new().init::<MyBackend>(&device));

        for &n in &[2usize, 3, 4] {
            for seed in 0..3u64 {
                // Reference: real Game with PairStrategy players.
                let players: Vec<Player> = (0..n)
                    .map(|i| {
                        Player::new(
                            Box::new(PairStrategy::from_shared(model.clone(), device)),
                            Box::new(SmallRng::seed_from_u64(1000 * seed + i as u64)),
                        )
                    })
                    .collect();
                let mut game = Game::new(players);
                game.play();

                // Sim: same model, same per-player dice streams, full fidelity.
                let bot = PairStrategy::from_shared(model.clone(), device);
                let mut sim = SimGame {
                    states: vec![State::default(); n],
                    active: 0,
                    rngs: (0..n).map(|i| SmallRng::seed_from_u64(1000 * seed + i as u64)).collect(),
                    over: false,
                };
                let mut guard = 0;
                while !sim.over {
                    play_sim_turn(&bot, &mut sim, Fidelity::Full);
                    guard += 1;
                    assert!(guard < 500, "sim did not terminate");
                }

                for (i, p) in game.players.iter().enumerate() {
                    assert_eq!(
                        sim.states[i], p.state,
                        "n={n} seed={seed} player={i}: sim diverged from Game::play"
                    );
                }
            }
        }
    }
}
```

(`NdArrayDevice` is `Copy`; if the compiler disagrees, `.clone()` it.)

- [ ] **Step 4: Run, fix any divergence, commit**

Run: `cargo test --lib strategy::sim`
Expected: PASS. If it fails, the sim's mechanics differ from `Game::play` — diff
the orderings (snapshot timing, lock propagation points, game-over checks,
`has_marked` propagation) until identical. Do not weaken the test.

```bash
cargo fmt && git add src/strategy/ && git commit -m "feat(search): SimGame turn loop equivalent to Game::play"
```

---

### Task 5: Lockstep batched driver

**Files:**
- Modify: `src/strategy/sim.rs`

- [ ] **Step 1: Append the lockstep driver**

```rust
/// Advances many sims in lockstep, batching each phase's net evaluations of
/// ALL live sims into one `evaluate_batch_multi` call. Decisions inside are
/// Lite-fidelity (see [`Fidelity::Lite`]).
pub struct BatchedRollouts<'a, B: Bot> {
    pub bot: &'a B,
    pub sims: Vec<SimGame>,
}

/// One unevaluated decision gathered from a sim.
struct PendingChoice {
    sim: usize,
    player: usize,
    cands: Vec<(Option<Mark>, State)>,
    states: Vec<State>,
    view: Vec<State>,
}

impl<'a, B: Bot> BatchedRollouts<'a, B> {
    pub fn new(bot: &'a B, sims: Vec<SimGame>) -> Self {
        Self { bot, sims }
    }

    pub fn all_over(&self) -> bool {
        self.sims.iter().all(|s| s.over)
    }

    /// Resolve a batch of pending choices with one multi-group evaluation.
    /// Returns (sim, player, chosen mark) triples.
    fn resolve(&self, pending: Vec<PendingChoice>) -> Vec<(usize, usize, Option<Mark>)> {
        let groups: Vec<(&[State], &[State])> = pending
            .iter()
            .map(|p| (p.states.as_slice(), p.view.as_slice()))
            .collect();
        let values = self.bot.evaluate_batch_multi(&groups);
        pending
            .into_iter()
            .zip(values)
            .map(|(p, vals)| (p.sim, p.player, p.cands[super::bot_impl::argmax(&vals)].0))
            .collect()
    }

    /// Advance every live sim by one full turn.
    pub fn step_turn(&mut self) {
        let live: Vec<usize> = (0..self.sims.len()).filter(|&i| !self.sims[i].over).collect();
        if live.is_empty() {
            return;
        }

        // ---- Phase 1: roll + gather all players' decisions ----
        let mut dice_of: Vec<[u8; 6]> = Vec::with_capacity(live.len());
        let mut snapshots: Vec<Vec<State>> = Vec::with_capacity(live.len());
        for &si in &live {
            let sim = &mut self.sims[si];
            let active = sim.active;
            dice_of.push(sim.rngs[active].roll());
            snapshots.push(sim.states.clone());
        }

        let mut marks: Vec<Vec<Option<Mark>>> = live.iter().map(|&si| vec![None; self.sims[si].n()]).collect();
        let mut pending: Vec<PendingChoice> = Vec::new();
        for (li, &si) in live.iter().enumerate() {
            let sim = &self.sims[si];
            let n = sim.n();
            let dice = dice_of[li];
            for j in 0..n {
                let view = SimGame::opp_view(&snapshots[li], j);
                let decision = if j == sim.active {
                    phase1_plan_choices(&snapshots[li][j], &view, dice)
                } else {
                    passive_phase1_choices(&snapshots[li][j], &view, dice)
                };
                match decision {
                    Decision::Forced(m) => marks[li][j] = m,
                    Decision::Choices(cands) => {
                        let states: Vec<State> = cands.iter().map(|(_, s)| *s).collect();
                        pending.push(PendingChoice { sim: li, player: j, cands, states, view });
                    }
                }
            }
        }
        for (li, j, m) in self.resolve(pending) {
            marks[li][j] = m;
        }

        // ---- Apply phase 1, locks, game-over ----
        let mut has_marked: Vec<bool> = Vec::with_capacity(live.len());
        for (li, &si) in live.iter().enumerate() {
            let sim = &mut self.sims[si];
            has_marked.push(marks[li][sim.active].is_some());
            for (j, m) in marks[li].iter().enumerate() {
                if let Some(m) = m {
                    sim.states[j].apply_mark(*m);
                }
            }
            SimGame::propagate_locks(&mut sim.states);
            if SimGame::game_over(&sim.states) {
                sim.over = true;
            }
        }

        // ---- Phase 2: active players of still-live sims ----
        let mut pending: Vec<PendingChoice> = Vec::new();
        let mut phase2_marks: Vec<Option<Option<Mark>>> = vec![None; live.len()]; // outer None = pending
        for (li, &si) in live.iter().enumerate() {
            let sim = &self.sims[si];
            if sim.over {
                continue;
            }
            let view = SimGame::opp_view(&sim.states, sim.active);
            match active_phase2_choices(&sim.states[sim.active], &view, dice_of[li], has_marked[li]) {
                Decision::Forced(m) => phase2_marks[li] = Some(m),
                Decision::Choices(cands) => {
                    let states: Vec<State> = cands.iter().map(|(_, s)| *s).collect();
                    pending.push(PendingChoice { sim: li, player: sim.active, cands, states, view });
                }
            }
        }
        for (li, _, m) in self.resolve(pending) {
            phase2_marks[li] = Some(m);
        }

        for (li, &si) in live.iter().enumerate() {
            let sim = &mut self.sims[si];
            if sim.over {
                continue;
            }
            match phase2_marks[li].expect("phase 2 decision missing") {
                Some(m) => sim.states[sim.active].apply_mark(m),
                None if !has_marked[li] => sim.states[sim.active].apply_strike(),
                None => {}
            }
            SimGame::propagate_locks(&mut sim.states);
            if SimGame::game_over(&sim.states) {
                sim.over = true;
                continue;
            }
            sim.active = (sim.active + 1) % sim.n();
        }
    }
}
```

- [ ] **Step 2: Add the lockstep-equals-sequential test**

```rust
    #[test]
    fn lockstep_driver_matches_sequential_lite_sims() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let model = Arc::new(PairModelConfig::new().init::<MyBackend>(&device));
        let bot = PairStrategy::from_shared(model, device);

        let make_sims = || -> Vec<SimGame> {
            (0..8u64)
                .map(|k| SimGame {
                    states: vec![State::default(); 2],
                    active: 0,
                    rngs: (0..2).map(|i| SmallRng::seed_from_u64(7000 + 10 * k + i)).collect(),
                    over: false,
                })
                .collect()
        };

        // Sequential reference (Lite fidelity).
        let mut seq = make_sims();
        for _ in 0..6 {
            for sim in seq.iter_mut() {
                play_sim_turn(&bot, sim, Fidelity::Lite);
            }
        }

        // Lockstep.
        let mut batched = BatchedRollouts::new(&bot, make_sims());
        for _ in 0..6 {
            batched.step_turn();
        }

        for (a, b) in seq.iter().zip(&batched.sims) {
            assert_eq!(a.over, b.over);
            assert_eq!(a.active, b.active);
            assert_eq!(a.states, b.states);
        }
    }
```

- [ ] **Step 3: Run, commit**

Run: `cargo test --lib strategy::sim`
Expected: 2 tests pass (equivalence + lockstep). Any mismatch means the batched
path's argmax or phase ordering deviates from the reference — fix the driver, not
the test. Then `cargo test 2>&1 | grep "test result"` — all green.

```bash
cargo fmt && git add src/strategy/sim.rs && git commit -m "feat(search): lockstep batched rollout driver"
```

---

### Task 6: `SearchBot` — gating, entries, rollout scoring

**Files:**
- Modify: `src/strategy/search.rs`

- [ ] **Step 1: Append SearchBot and helpers to `src/strategy/search.rs`**

```rust
use super::bot_impl::{active_phase1_choices, active_phase2_choices, eval_decision, passive_phase1_impl, Decision};
use super::sim::{BatchedRollouts, SimGame};
use super::Strategy;
use crate::state::Mark;
use rand::rngs::SmallRng;
use rand::SeedableRng;

/// Why search fired (for diagnostics).
#[derive(Default, Debug, Clone)]
pub struct SearchStats {
    pub active_decisions: u32,
    /// Decisions that reached the gate (>= 2 candidates, not meta-forced).
    pub eligible: u32,
    pub gate_close: u32,
    pub gate_endgame: u32,
    pub searched: u32,
    /// Search picked a different move than static evaluation.
    pub disagreements: u32,
    /// ...of which a gate (close OR endgame) had fired.
    pub disagreements_gated: u32,
    /// Top-2 static value gaps at eligible decisions.
    pub gaps: Vec<f32>,
}

pub struct SearchBot<B: WinProb> {
    pub bot: B,
    /// Search every eligible decision regardless of gates (diagnostics).
    pub force: bool,
    pub stats: Option<std::rc::Rc<std::cell::RefCell<SearchStats>>>,
}

impl<B: WinProb> SearchBot<B> {
    pub fn new(bot: B) -> Self {
        Self { bot, force: false, stats: None }
    }
}

impl<B: WinProb> std::fmt::Debug for SearchBot<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "SearchBot({:?})", self.bot)
    }
}

/// A shortlisted candidate: the move, its static value, its post-move state.
struct Candidate {
    mark: Option<Mark>,
    value: f32,
    post: State,
}

/// Gate predicate. `our` is the pre-decision state (lock comparisons are
/// relative to it).
fn gates(cands: &[Candidate], our: &State, opps: &[State]) -> (bool, bool) {
    let close = cands[0].value - cands[1].value < GATE_MARGIN;
    let endgame = our.count_locked() >= 1
        || opps.iter().any(|s| s.count_locked() >= 1)
        || our.strikes >= 3
        || opps.iter().any(|s| s.strikes >= 3)
        || cands
            .iter()
            .take(K_CANDIDATES)
            .any(|c| c.post.count_locked() > our.count_locked());
    (close, endgame)
}

impl<B: WinProb> SearchBot<B> {
    /// Shared post-static-eval flow: search the top-K entries, pick.
    /// `entries[i]` = (full player states after candidate i's deterministic
    /// completion of the current turn, game-already-over flag). Returns the
    /// index of the winning candidate. Takes `&self` (stats use RefCell) so
    /// `self.bot` can be borrowed by the rollout driver without fuss.
    fn search_pick(&self, cands: &[Candidate], entries: Vec<(Vec<State>, bool)>, seed: u64) -> usize {
        debug_assert_eq!(entries.len(), cands.len().min(K_CANDIDATES));
        let n = entries[0].0.len();

        // Build rollouts: K_SAMPLES per non-ended entry, CRN dice across entries.
        let mut sims: Vec<SimGame> = Vec::new();
        let mut sim_owner: Vec<usize> = Vec::new(); // entry index per sim
        for (ei, (states, ended)) in entries.iter().enumerate() {
            if *ended {
                continue;
            }
            for k in 0..K_SAMPLES {
                sims.push(SimGame {
                    states: states.clone(),
                    active: 1 % n,
                    rngs: (0..n)
                        .map(|p| SmallRng::seed_from_u64(sample_player_seed(seed, k, p)))
                        .collect(),
                    over: false,
                });
                sim_owner.push(ei);
            }
        }

        let mut driver = BatchedRollouts::new(&self.bot, sims);
        for _ in 0..n {
            // HORIZON_TURNS = one full round
            if driver.all_over() {
                break;
            }
            driver.step_turn();
        }

        // Score: exact outcome for finished sims, win prob for survivors
        // (one batched call over all survivors of all entries).
        let mut scores: Vec<f32> = vec![0.0; entries.len()];
        let mut counts: Vec<u32> = vec![0; entries.len()];
        let survivor_idx: Vec<usize> = (0..driver.sims.len()).filter(|&i| !driver.sims[i].over).collect();
        let survivor_views: Vec<Vec<State>> = survivor_idx
            .iter()
            .map(|&i| SimGame::opp_view(&driver.sims[i].states, 0))
            .collect();
        let groups: Vec<(&State, &[State])> = survivor_idx
            .iter()
            .zip(&survivor_views)
            .map(|(&i, view)| (&driver.sims[i].states[0], view.as_slice()))
            .collect();
        let probs = self.bot.win_prob_multi(&groups);
        let mut prob_iter = probs.into_iter();
        for (i, sim) in driver.sims.iter().enumerate() {
            let p = if sim.over {
                SimGame::outcome(&sim.states)
            } else {
                prob_iter.next().unwrap()
            };
            scores[sim_owner[i]] += p;
            counts[sim_owner[i]] += 1;
        }
        for (ei, (states, ended)) in entries.iter().enumerate() {
            if *ended {
                scores[ei] = SimGame::outcome(states);
                counts[ei] = 1;
            }
        }

        // Highest mean wins; ties keep the static (lower-index) candidate.
        let mut best = 0;
        let mut best_score = scores[0] / counts[0] as f32;
        for ei in 1..entries.len() {
            let s = scores[ei] / counts[ei] as f32;
            if s > best_score {
                best = ei;
                best_score = s;
            }
        }
        best
    }

    /// Gate bookkeeping. Returns true if search should run. Never searches
    /// with no opponents (solo is unsupported; the simulator's outcome and
    /// leader logic require at least one opponent).
    fn gate_and_record(&self, cands: &[Candidate], our: &State, opps: &[State]) -> bool {
        if opps.is_empty() {
            return false;
        }
        if let Some(stats) = &self.stats {
            stats.borrow_mut().eligible += 1;
            stats.borrow_mut().gaps.push(cands[0].value - cands[1].value);
        }
        let (close, endgame) = gates(cands, our, opps);
        if let Some(stats) = &self.stats {
            let mut s = stats.borrow_mut();
            if close {
                s.gate_close += 1;
            }
            if endgame {
                s.gate_endgame += 1;
            }
        }
        self.force || close || endgame
    }

    fn record_search_result(&self, gated: bool, disagreed: bool) {
        if let Some(stats) = &self.stats {
            let mut s = stats.borrow_mut();
            s.searched += 1;
            if disagreed {
                s.disagreements += 1;
                if gated {
                    s.disagreements_gated += 1;
                }
            }
        }
    }
}
```

- [ ] **Step 2: Append the `Strategy` impl**

```rust
impl<B: WinProb> Strategy for SearchBot<B> {
    fn active_phase1(&mut self, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
        if let Some(stats) = &self.stats {
            stats.borrow_mut().active_decisions += 1;
        }
        let (decision, sim_opp) = active_phase1_choices(&self.bot, state, opp_states, dice);
        let plans = match decision {
            Decision::Forced(m) => return m,
            Decision::Choices(c) => c,
        };
        let states: Vec<State> = plans.iter().map(|(_, s)| *s).collect();
        let values = self.bot.evaluate_batch(&states, &sim_opp);

        // Collapse plans to distinct phase-1 marks: each keeps its best plan's
        // static value (for shortlisting) and that plan's end-state (for the
        // lock gate). The rollout re-decides phase 2 itself.
        let mut cands: Vec<Candidate> = Vec::new();
        for (i, (p1, _)) in plans.iter().enumerate() {
            match cands.iter_mut().find(|c| c.mark == *p1) {
                Some(c) if values[i] > c.value => {
                    c.value = values[i];
                    c.post = states[i];
                }
                Some(_) => {}
                None => cands.push(Candidate { mark: *p1, value: values[i], post: states[i] }),
            }
        }
        cands.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap());

        if cands.len() < 2 || !self.gate_and_record(&cands, state, opp_states) {
            return cands[0].mark;
        }
        let (close, endgame) = gates(&cands, state, opp_states);

        // Entry per shortlisted phase-1 mark: opponents' (shared, simultaneous)
        // phase-1 marks are sim_opp; complete OUR turn deterministically.
        let shortlist = &cands[..cands.len().min(K_CANDIDATES)];
        let entries: Vec<(Vec<State>, bool)> = shortlist
            .iter()
            .map(|c| {
                let mut our = *state;
                if let Some(m) = c.mark {
                    our.apply_mark(m);
                }
                let mut all: Vec<State> = std::iter::once(our).chain(sim_opp.iter().copied()).collect();
                SimGame::propagate_locks(&mut all);
                if SimGame::game_over(&all) {
                    return (all, true);
                }
                // Phase 2, re-decided exactly like real play would follow up.
                let view = SimGame::opp_view(&all, 0);
                let d = active_phase2_choices(&all[0], &view, dice, c.mark.is_some());
                match eval_decision(&self.bot, d, &view) {
                    Some(m) => all[0].apply_mark(m),
                    None if c.mark.is_none() => all[0].apply_strike(),
                    None => {}
                }
                SimGame::propagate_locks(&mut all);
                let ended = SimGame::game_over(&all);
                (all, ended)
            })
            .collect();

        let seed = context_seed(state, opp_states, dice);
        let pick = self.search_pick(shortlist, entries, seed);
        self.record_search_result(close || endgame, pick != 0);
        shortlist[pick].mark
    }

    fn active_phase2(&mut self, state: &State, opp_states: &[State], dice: [u8; 6], has_marked: bool) -> Option<Mark> {
        if let Some(stats) = &self.stats {
            stats.borrow_mut().active_decisions += 1;
        }
        let choices = match active_phase2_choices(state, opp_states, dice, has_marked) {
            Decision::Forced(m) => return m,
            Decision::Choices(c) => c,
        };
        let states: Vec<State> = choices.iter().map(|(_, s)| *s).collect();
        let values = self.bot.evaluate_batch(&states, opp_states);
        let mut cands: Vec<Candidate> = choices
            .iter()
            .zip(&values)
            .map(|((m, s), &v)| Candidate { mark: *m, value: v, post: *s })
            .collect();
        cands.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap());

        if cands.len() < 2 || !self.gate_and_record(&cands, state, opp_states) {
            return cands[0].mark;
        }
        let (close, endgame) = gates(&cands, state, opp_states);

        // Entry per candidate: post state + lock propagation; turn is then over.
        let shortlist = &cands[..cands.len().min(K_CANDIDATES)];
        let entries: Vec<(Vec<State>, bool)> = shortlist
            .iter()
            .map(|c| {
                let mut all: Vec<State> = std::iter::once(c.post).chain(opp_states.iter().copied()).collect();
                SimGame::propagate_locks(&mut all);
                let ended = SimGame::game_over(&all);
                (all, ended)
            })
            .collect();

        let seed = context_seed(state, opp_states, dice);
        let pick = self.search_pick(shortlist, entries, seed);
        self.record_search_result(close || endgame, pick != 0);
        shortlist[pick].mark
    }

    fn passive_phase1(
        &mut self,
        state: &State,
        opp_states: &[State],
        dice: [u8; 6],
        _active_player: usize,
    ) -> Option<Mark> {
        passive_phase1_impl(&self.bot, state, opp_states, dice)
    }
}
```

Note the phase-2 candidate set: `active_phase2_choices` already includes the
skip/strike baseline as a `(None, baseline)` entry, so search compares "mark"
vs "don't mark" naturally; for a `None` candidate `c.post` IS the baseline
(strike already applied when `!has_marked`), so the entry construction is uniform.

- [ ] **Step 3: Add tests to the search tests module**

```rust
    use crate::dqn::pair::{PairModelConfig, PairStrategy};
    use crate::dqn::MyBackend;
    use crate::strategy::Strategy;

    fn test_bot() -> PairStrategy {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        PairStrategy::from_model(PairModelConfig::new().init::<MyBackend>(&device), device)
    }

    #[test]
    fn search_decisions_are_deterministic() {
        // SearchBot holds no RNG state: repeating the same decision context on
        // the same instance must reproduce the same move (context-hashed dice).
        let mut bot = SearchBot::new(test_bot());
        bot.force = true; // exercise the search path regardless of gates
        let state = State::default();
        let opps = [State::default()];
        let dice = [3, 4, 2, 3, 5, 1];
        let a = bot.active_phase1(&state, &opps, dice);
        let b = bot.active_phase1(&state, &opps, dice);
        assert_eq!(a, b);
        let c = bot.active_phase2(&state, &opps, dice, false);
        let d = bot.active_phase2(&state, &opps, dice, false);
        assert_eq!(c, d);
    }

    #[test]
    fn gate_predicate_truth_table() {
        let mk = |value, post: State| Candidate { mark: None, value, post };
        let fresh = State::default();
        let mut locked_one = State::default();
        for n in 2..=6 {
            locked_one.apply_mark(Mark { row: 0, number: n });
        }
        locked_one.apply_mark(Mark { row: 0, number: 12 }); // locks red

        // Close gap, no endgame: close gate only.
        let cands = [mk(1.00, fresh), mk(0.95, fresh)];
        assert_eq!(gates(&cands, &fresh, &[fresh]), (true, false));
        // Wide gap, no endgame: nothing fires.
        let cands = [mk(1.00, fresh), mk(0.50, fresh)];
        assert_eq!(gates(&cands, &fresh, &[fresh]), (false, false));
        // Wide gap, opponent has a locked row: endgame fires.
        assert_eq!(gates(&cands, &fresh, &[locked_one]), (false, true));
        // Wide gap, our candidate would lock: endgame fires.
        let cands = [mk(1.00, locked_one), mk(0.50, fresh)];
        assert_eq!(gates(&cands, &fresh, &[fresh]), (false, true));
        // Wide gap, we sit at 3 strikes: endgame fires.
        let mut striked = State::default();
        for _ in 0..3 {
            striked.apply_strike();
        }
        let cands = [mk(1.00, fresh), mk(0.50, fresh)];
        assert_eq!(gates(&cands, &striked, &[fresh]), (false, true));
    }

    #[test]
    fn forced_decisions_bypass_search() {
        // A state with a safe lock available must return the lock without
        // search: 5 marks in red, white sum 12 completes it (first lock,
        // doesn't end the game).
        let mut bot = SearchBot::new(test_bot());
        bot.force = true;
        let mut state = State::default();
        for n in 2..=6 {
            state.apply_mark(Mark { row: 0, number: n });
        }
        let opps = [State::default()];
        let dice = [6, 6, 1, 1, 1, 1]; // white sum 12
        let m = bot.active_phase1(&state, &opps, dice);
        assert_eq!(m, Some(Mark { row: 0, number: 12 }));
        // And stats must not count a search for it.
        // (force=true but Forced short-circuits before gating)
    }

    #[test]
    fn fallback_when_gates_closed() {
        // With gates impossible to fire (huge margin → close-gate fires
        // easily; so instead verify force=false + GATE off behavior via a
        // a fresh-board decision where endgame gate is off): if the close
        // gate fires the bot searches; either way the move must be legal and
        // deterministic. Sanity smoke:
        let mut bot = SearchBot::new(test_bot());
        let state = State::default();
        let opps = [State::default()];
        let m1 = bot.active_phase2(&state, &opps, [3, 4, 2, 3, 5, 1], true);
        let m2 = bot.active_phase2(&state, &opps, [3, 4, 2, 3, 5, 1], true);
        assert_eq!(m1, m2);
    }

    #[test]
    fn stats_are_recorded() {
        let stats = std::rc::Rc::new(std::cell::RefCell::new(SearchStats::default()));
        let mut bot = SearchBot::new(test_bot());
        bot.force = true;
        bot.stats = Some(stats.clone());
        let state = State::default();
        let opps = [State::default()];
        bot.active_phase1(&state, &opps, [3, 4, 2, 3, 5, 1]);
        let s = stats.borrow();
        assert_eq!(s.active_decisions, 1);
        assert!(s.eligible <= 1);
        assert_eq!(s.gaps.len() as u32, s.eligible);
    }
```

- [ ] **Step 4: Run, commit**

Run: `cargo test --lib strategy::search` — all pass; then full `cargo test`.

```bash
cargo fmt && git add src/strategy/search.rs && git commit -m "feat(search): SearchBot with gated CRN rollout selection"
```

---

### Task 7: CLI integration + determinism smoke

**Files:**
- Modify: `src/main.rs`

- [ ] **Step 1: Wire the bot type**

`BotType` enum: add `PairSearch,` after `Pair,`. Display: `BotType::PairSearch => write!(f, "PAIR-SEARCH"),` (clap's ValueEnum derives the CLI name `pair-search` automatically).

`make_strategy` arm:

```rust
        BotType::PairSearch => Box::new(strategy::search::SearchBot::new(dqn::pair::PairStrategy::load("pair_model"))),
```

`StrategyTemplates::new`: change `needs_pair` to

```rust
        let needs_pair = bots.iter().any(|b| matches!(b, BotType::Pair | BotType::PairSearch));
```

`StrategyTemplates::create` arm:

```rust
            BotType::PairSearch => {
                let t = self.pair.as_ref().unwrap();
                Box::new(strategy::search::SearchBot::new(dqn::pair::PairStrategy::from_shared(
                    t.model.clone(),
                    t.device.clone(),
                )))
            }
```

Do NOT add PairSearch to `run_solo`.

- [ ] **Step 2: Verify + determinism smoke**

```bash
cargo test 2>&1 | grep "test result"
cargo build --release
./target/release/qwixxer bench pair-search pair -n 200 > /tmp/ps1.txt
./target/release/qwixxer bench pair-search pair -n 200 > /tmp/ps2.txt
diff /tmp/ps1.txt /tmp/ps2.txt && echo DETERMINISTIC
./target/release/qwixxer bench pair-search ga -n 200
```

Expected: all tests green; diff empty (stateless-deterministic search); both
benches complete. Note the per-game time — if a 200-game bench takes more than
~2 minutes, flag it (cost model assumed ~1 ms/searched decision).

- [ ] **Step 3: Commit**

```bash
cargo fmt && git add src/main.rs && git commit -m "feat(search): pair-search bot type"
```

---

### Task 8: Calibration diagnostic

**Files:**
- Create: `examples/search_calibration.rs`

- [ ] **Step 1: Create the example**

```rust
//! Gate-calibration diagnostic for the pair-search bot.
//!
//! Plays N 1v1 games (forced-search SearchBot vs plain pair bot), recording at
//! every eligible active decision: the top-2 static gap, which gates would
//! have fired, and whether search disagreed with the static choice. Prints a
//! summary used to tune GATE_MARGIN.
//!
//! Run: cargo run --release --example search_calibration -- 200

use qwixxer::dqn::pair::PairStrategy;
use qwixxer::game::{Game, Player};
use qwixxer::strategy::search::{SearchBot, SearchStats};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::cell::RefCell;
use std::rc::Rc;

fn main() {
    let games: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let template = PairStrategy::load("pair_model");

    let mut total = SearchStats::default();
    let start = std::time::Instant::now();
    for g in 0..games as u64 {
        let stats = Rc::new(RefCell::new(SearchStats::default()));
        let mut searcher = SearchBot::new(PairStrategy::from_shared(template.model.clone(), template.device));
        searcher.force = true;
        searcher.stats = Some(stats.clone());

        let players = vec![
            Player::new(Box::new(searcher), Box::new(SmallRng::seed_from_u64(31337 + 2 * g))),
            Player::new(
                Box::new(PairStrategy::from_shared(template.model.clone(), template.device)),
                Box::new(SmallRng::seed_from_u64(31337 + 2 * g + 1)),
            ),
        ];
        let mut game = Game::new(players);
        game.play();

        let s = stats.borrow();
        total.active_decisions += s.active_decisions;
        total.eligible += s.eligible;
        total.gate_close += s.gate_close;
        total.gate_endgame += s.gate_endgame;
        total.searched += s.searched;
        total.disagreements += s.disagreements;
        total.disagreements_gated += s.disagreements_gated;
        total.gaps.extend_from_slice(&s.gaps);
    }
    let elapsed = start.elapsed().as_secs_f64();

    let mut gaps = total.gaps.clone();
    gaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f64| -> f32 {
        if gaps.is_empty() {
            0.0
        } else {
            gaps[((gaps.len() - 1) as f64 * p) as usize]
        }
    };

    println!("games:                {games}  ({:.1}s, {:.2}s/game)", elapsed, elapsed / games as f64);
    println!("active decisions:     {}", total.active_decisions);
    println!("eligible (>=2 cands): {}", total.eligible);
    println!("searched (forced):    {}", total.searched);
    println!(
        "gate hits: close {} ({:.1}% of eligible), endgame {} ({:.1}%)",
        total.gate_close,
        100.0 * total.gate_close as f64 / total.eligible.max(1) as f64,
        total.gate_endgame,
        100.0 * total.gate_endgame as f64 / total.eligible.max(1) as f64,
    );
    println!(
        "disagreements: {} ({:.1}% of searched); with a gate fired: {} ({:.1}% of disagreements)",
        total.disagreements,
        100.0 * total.disagreements as f64 / total.searched.max(1) as f64,
        total.disagreements_gated,
        100.0 * total.disagreements_gated as f64 / total.disagreements.max(1) as f64,
    );
    println!(
        "top-2 gap percentiles: p10 {:.3}  p25 {:.3}  p50 {:.3}  p75 {:.3}  p90 {:.3}",
        pct(0.10),
        pct(0.25),
        pct(0.50),
        pct(0.75),
        pct(0.90)
    );
}
```

Required visibility check: the example needs `qwixxer::dqn::pair::PairStrategy`,
`qwixxer::strategy::search::{SearchBot, SearchStats}`, `qwixxer::game::{Game, Player}` —
all already `pub`. `SearchBot.force`/`stats` fields are `pub` (Task 6). If `lib.rs`
needs no change, don't change it.

- [ ] **Step 2: Run + commit**

```bash
cargo run --release --example search_calibration -- 50
```

Expected: completes; prints plausible numbers (eligible > 0, gaps sorted,
seconds/game noted). Then:

```bash
cargo fmt && git add examples/search_calibration.rs && git commit -m "feat(search): gate calibration diagnostic"
```

---

### Task 9: Calibrate, acceptance benchmarks, docs

**Files:**
- Modify: `src/strategy/search.rs` (GATE_MARGIN only, if calibration says so)
- Modify: `README.md`, `docs/ARCHITECTURE.md`, `docs/EXPERIMENTS.md`

- [ ] **Step 1: Calibration run**

```bash
cargo run --release --example search_calibration -- 300
```

Read the output. Decision rule: choose `GATE_MARGIN` ≈ the gap percentile that
captures ≥80% of disagreements via the close gate (look at "with a gate fired" —
if it's already ≥80% at the default 0.15, keep 0.15). If the endgame gate alone
captures most disagreements, consider lowering the margin to cut cost. Record the
numbers; if `GATE_MARGIN` changes, update the const, re-run the Task 7 determinism
smoke, and commit `chore(search): calibrate GATE_MARGIN to <value>`.

- [ ] **Step 2: Acceptance benchmarks**

```bash
./target/release/qwixxer bench pair-search pair -n 50000
./target/release/qwixxer bench pair-search pair -n 50000 --seed 7
./target/release/qwixxer bench pair-search ga -n 50000
./target/release/qwixxer bench pair-search dqn -n 50000
```

Acceptance: pair-search vs pair > 50% with the paired 99% CI excluding 50%, on
both seeds; pair-search vs GA > 59.2%. Record ALL results (positive or negative).

- [ ] **Step 3: Documentation**

`README.md` Strategies table, after the Pair row:

```markdown
| **Pair-Search** | Pair net + gated truncated rollouts at active decisions | see EXPERIMENTS.md |
```

`docs/ARCHITECTURE.md`, after the Pair Network section:

```markdown
### Pair-Search (`strategy/search.rs`, `strategy/sim.rs`)

Decision-time search on top of the pair network. At gated active decisions
(top-2 static gap below GATE_MARGIN, or endgame proximity), the top-2
candidates are re-ranked by truncated rollouts: complete the current turn
deterministically, simulate one full round with all players played greedily by
the value net (lockstep-batched through Bot::evaluate_batch_multi), then score
each sampled future as the exact outcome if the game ended or the net's win
probability (WinProb trait) at the horizon. 64 samples per candidate with
common-random-number dice derived by hashing the decision context, so the bot
is stateless-deterministic. Meta-rules stay single-source: the shared
bot_impl pipelines expose pure-logic `*_choices()` consumed by both the
blanket Bot->Strategy impl and the search bot. The simulator is pinned to
`Game::play` by an equivalence test. Calibration diagnostic:
`cargo run --release --example search_calibration -- 300`.
```

`docs/EXPERIMENTS.md`: append a "Phase 13: Decision-time search" section with the
design one-liner, calibration numbers, all four benchmark results, and the
verdict (including, if negative, the spec's "record it and stop the search line"
conclusion).

- [ ] **Step 4: Final commit**

```bash
cargo fmt && cargo test 2>&1 | grep "test result"   # all green
git add README.md docs/ && git commit -m "docs: pair-search results and architecture notes"
```

---

## Execution notes

- Task order is strict: 2 depends on 1 only trivially, but 4–6 depend hard on 2–3.
- **Task 2's byte-identical gate is non-negotiable** — it proves the refactor
  preserved every bot's behavior. Diff failures there are bugs to fix, never to
  accept.
- The spec's `win_prob_batch(candidates, opp_states)` signature was amended to
  `win_prob_multi(groups)` during planning: at the rollout leaf every sim has its
  own opponent set, so the batched form must be per-group. Same intent, corrected
  shape.
- Search code placement amended from the spec's `src/dqn/search.rs` to
  `src/strategy/{search,sim}.rs` — the machinery is Bot-generic and lives with
  the Strategy/Bot infrastructure; only the WinProb impl is pair-specific.
- Cost watch: if Task 7's 200-game smoke is dramatically slower than the ~1 ms/
  searched-decision model predicts, profile before Task 9 (the usual suspect is
  per-row tensor overhead — check that `evaluate_batch_multi` really receives
  multi-group batches from the driver).
- If acceptance fails at every reasonable gate setting, that outcome goes into
  EXPERIMENTS.md as a decisive negative result per the spec — do not silently
  tune until something looks positive (selection noise).
