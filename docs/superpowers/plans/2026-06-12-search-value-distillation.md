# Search-Value Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `pair-train --distill` emits rollout-derived value targets at gate-firing and lock-firing decisions (both candidates / lock trio), plus targeted ε-decline of forced locks during generation; smoke locally, hand the full run to the user.

**Architecture:** Detection happens inside `RecordingPair` (cheap context capture); rollouts + sample emission run post-game inside the existing parallel game closure. New `pub(crate)` surface: parameterized rule pipelines in `bot_impl.rs`, extracted entries builders + a full-game rollout scorer in `search.rs`. Distill samples are ordinary `PairSample`s through the unchanged loss.

**Spec:** `docs/superpowers/specs/2026-06-12-search-value-distillation-design.md`

**Key facts** (verified):
- `pair_train.rs`: `Snapshot = (State, Vec<State>)`; `RecordingPair { policy, epsilon, rng, recorded }` records post-decision states (active: post-turn after phase 2; passive: every decision incl. skips); `play_training_game(model, device, num_opponents, search, epsilon, seed) -> (Vec<PairSample>, f32)` (player 0 explores, others greedy; per-player buffers; `build_pair_samples` does per-opponent TD chains + swap-doubled negated samples); `self_play_train(artifact_dir, num_iterations, games_per_iteration, epochs_per_iteration, bench_games, checkpoints, start_iteration, search)`; thirds 1v1/3p/4p. `PairSample { features, value (μ target), final_diff (σ reference) }`; loss decoupled (μ MSE / σ NLL, μ detached).
- `bot_impl.rs`: `mark_choices` forces `find_safe_lock` FIRST (line ~113); `phase1_plan_choices` scans for the first safe-locking phase-1 mark AFTER the losing-end retain, BEFORE pruning (~300); `eval_decision`, `argmax`, `active_phase1_choices`, `active_phase2_choices` pub; `passive_phase1_choices`, `*_impl` pub(crate); `find_safe_lock`, `opp_best_phase1_score`, `prune_dominated` private.
- `search.rs`: entries closures inline in `SearchBot::active_phase1` (~314-338: apply mark → chain sim_opp → propagate → over-check → re-decide phase 2 → propagate → ended) and `active_phase2` (~377-385: post + propagate); `gates(cands, our, opps)` private; `sample_player_seed`, `context_seed` pub; `K_CANDIDATES=2`, `GATE_MARGIN` pub. `Candidate { mark, value, post }` private.
- `sim.rs`: `SimGame { pub states, active, rngs, over }`, `BatchedRollouts`, `propagate_locks`, `game_over`, `opp_view` — all pub. States ordered `[us, opps in turn order]` in search rollouts; `active: 1 % n` = the player after us.
- `game.rs` passes `active_player` to `passive_phase1`; verify its frame (absolute seat vs relative) by reading `Game::play` before using it (Task 3 step notes this).
- `main.rs` `PairTrain` subcommand (~187-210) with `search: bool` flag, dispatch at ~597-605.
- Reference implementations to adapt (do NOT import from the example): `examples/divergence.rs` `rollout_scores_full` (full-game scorer with 200-turn cap) and `passive_entries` (1v1 active-turn completion).

---

### Task 1: bot_impl — parameterized rule pipelines

**Files:**
- Modify: `src/strategy/bot_impl.rs`

- [ ] **Step 1: Write the failing test** (append to a `#[cfg(test)] mod tests` in bot_impl.rs; create the module if absent — check `grep -n "mod tests" src/strategy/bot_impl.rs`):

```rust
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
```

- [ ] **Step 2:** `cargo test mark_choices_with 2>&1 | tail -3` → compile error (functions don't exist).

- [ ] **Step 3: Refactor.** Rename the bodies, keep the old names as thin wrappers (NO behavior change):
  - `pub(crate) fn mark_choices_with(state, marks, baseline, opp_best, force_lock: bool) -> Decision` — the existing `mark_choices` body with the lock force wrapped: `if force_lock { if let Some(m) = find_safe_lock(state, marks) { return Decision::Forced(Some(m)); } }`.
  - `pub(crate) fn mark_choices(...)` → `mark_choices_with(..., true)`.
  - `pub(crate) fn phase1_plan_choices_with(state, comparison_opps, dice, force_lock: bool) -> Decision` — existing body; the lock-scan block (after retain, before prune) wrapped in `if force_lock { ... }`.
  - `pub(crate) fn phase1_plan_choices(...)` → `phase1_plan_choices_with(..., true)`.
  - `fn find_safe_lock` → `pub(crate) fn find_safe_lock`.
  - `fn opp_best_phase1_score` → `pub(crate) fn opp_best_phase1_score` (needed by pp1 detection in Task 3).
  - Add `#[derive(Debug)]`-compatible note: `Decision` has no Debug; the test above avoids needing it (uses `matches!`). Do not add derives.

- [ ] **Step 4:** `cargo test --release 2>&1 | grep "test result"` → all green (existing 68 + 2 new).

- [ ] **Step 5: Commit:** `git add src/strategy/bot_impl.rs && git commit -m "refactor(bot_impl): parameterize the safe-lock force; expose helpers pub(crate)"`

---

### Task 2: search.rs — extracted entries builders + full-game rollout scorer

**Files:**
- Modify: `src/strategy/search.rs`

- [ ] **Step 1: Extract the entries builders** as `pub(crate)` free functions, and make `SearchBot` use them (pure refactor; the existing search tests are the regression net):

```rust
/// Deterministic completion of our turn after a phase-1 mark choice:
/// opponents' (simultaneous) phase-1 marks are `sim_opp`; our phase 2 is
/// re-decided. Returns (all-player states, game-ended). Extracted from
/// `SearchBot::active_phase1` for reuse by training-time distillation.
pub(crate) fn phase1_entry(
    bot: &impl Bot,
    state: &State,
    sim_opp: &[State],
    dice: [u8; 6],
    mark: Option<Mark>,
) -> (Vec<State>, bool) {
    let mut our = *state;
    if let Some(m) = mark {
        our.apply_mark(m);
    }
    let mut all: Vec<State> = std::iter::once(our).chain(sim_opp.iter().copied()).collect();
    SimGame::propagate_locks(&mut all);
    if SimGame::game_over(&all) {
        return (all, true);
    }
    let view = SimGame::opp_view(&all, 0);
    let d = active_phase2_choices(&all[0], &view, dice, mark.is_some());
    match eval_decision(bot, d, &view) {
        Some(m) => all[0].apply_mark(m),
        None if mark.is_none() => all[0].apply_strike(),
        None => {}
    }
    SimGame::propagate_locks(&mut all);
    let ended = SimGame::game_over(&all);
    (all, ended)
}

/// Post-phase-2 entry: candidate post-state + current opponents.
/// Extracted from `SearchBot::active_phase2`.
pub(crate) fn phase2_entry(post: State, opps: &[State]) -> (Vec<State>, bool) {
    let mut all: Vec<State> = std::iter::once(post).chain(opps.iter().copied()).collect();
    SimGame::propagate_locks(&mut all);
    let ended = SimGame::game_over(&all);
    (all, ended)
}
```

In `SearchBot::active_phase1`, replace the entries closure body with `phase1_entry(&self.bot, state, &sim_opp, dice, c.mark)`; in `active_phase2` with `phase2_entry(c.post, opp_states)`. Also change `fn gates` → `pub(crate) fn gates` (Task 3 needs it for detection).

- [ ] **Step 2: Add the full-game scorer** (adapted from the example's `rollout_scores_full`, returning final states so callers compute per-opponent diffs):

```rust
/// Full-game CRN rollouts: every sample rolled to completion; returns the
/// final all-player states per entry per sample (ended entries: one element,
/// the entry itself). `first_active` = first player to act in the rollout.
/// Used by training-time distillation; the truncated production scorer in
/// `search_pick` is unaffected.
pub(crate) fn rollout_final_states(
    bot: &impl Bot,
    entries: &[(Vec<State>, bool)],
    seed: u64,
    k: usize,
    first_active: usize,
) -> Vec<Vec<Vec<State>>> {
    let n = entries[0].0.len();
    let mut sims: Vec<SimGame> = Vec::new();
    let mut sim_owner: Vec<usize> = Vec::new();
    for (ei, (states, ended)) in entries.iter().enumerate() {
        if *ended {
            continue;
        }
        for s in 0..k {
            sims.push(SimGame {
                states: states.clone(),
                active: first_active,
                rngs: (0..n)
                    .map(|p| SmallRng::seed_from_u64(sample_player_seed(seed, s, p)))
                    .collect(),
                over: false,
            });
            sim_owner.push(ei);
        }
    }
    let mut driver = BatchedRollouts::new(bot, sims);
    let mut turns = 0;
    while !driver.all_over() {
        driver.step_turn();
        turns += 1;
        assert!(turns <= 200, "rollout exceeded 200 turns — game-end invariant violated");
    }
    let mut out: Vec<Vec<Vec<State>>> = entries
        .iter()
        .map(|(states, ended)| if *ended { vec![states.clone()] } else { Vec::with_capacity(k) })
        .collect();
    for (i, sim) in driver.sims.iter().enumerate() {
        out[sim_owner[i]].push(sim.states.clone());
    }
    out
}
```

(Needs `use super::sim::BatchedRollouts;` adjustments — `SimGame`/`BatchedRollouts` are already imported in search.rs.)

- [ ] **Step 3: Test** (append to search.rs tests):

```rust
#[test]
fn rollout_final_states_full_games_deterministic() {
    let bot = test_bot();
    let a = State::default();
    let b = State::default();
    let entries = vec![
        phase2_entry(a, &[b]),
        {
            let mut marked = a;
            marked.apply_mark(Mark { row: 0, number: 5 });
            phase2_entry(marked, &[b])
        },
    ];
    let r1 = rollout_final_states(&bot, &entries, 7, 8, 1 % 2);
    let r2 = rollout_final_states(&bot, &entries, 7, 8, 1 % 2);
    assert_eq!(r1.len(), 2);
    assert_eq!(r1[0].len(), 8);
    // Every rollout reached a real game end and is reproducible.
    for (e1, e2) in r1.iter().zip(&r2) {
        for (s1, s2) in e1.iter().zip(e2) {
            assert!(SimGame::game_over(s1));
            assert_eq!(
                s1.iter().map(|s| s.count_points()).collect::<Vec<_>>(),
                s2.iter().map(|s| s.count_points()).collect::<Vec<_>>()
            );
        }
    }
}
```

- [ ] **Step 4:** `cargo test --release 2>&1 | grep "test result"` → all green (search tests prove the extraction didn't change behavior; the new test passes).

- [ ] **Step 5: Commit:** `git add src/strategy/search.rs && git commit -m "refactor(search): extract entry builders pub(crate); add full-game rollout scorer"`

---

### Task 3: pair_train — detection, ε-decline, emission, wiring

**Files:**
- Modify: `src/dqn/pair_train.rs`
- Modify: `src/main.rs`

- [ ] **Step 1: Config plumbing.** Add to `pair_train.rs`:

```rust
/// Distillation configuration (all off ⇒ behavior identical to before).
#[derive(Clone, Copy)]
pub struct DistillCfg {
    pub enabled: bool,
    /// Full-game rollouts per candidate.
    pub k: usize,
    /// Samples emitted per (candidate, opponent) pair.
    pub m: usize,
    /// Probability of declining a forced safe lock during generation
    /// (player 0 only, like uniform ε).
    pub epsilon_lock: f32,
}
```

`self_play_train` gains a `distill: DistillCfg` parameter; `play_training_game` likewise. `main.rs` `PairTrain` gains:

```rust
        /// Emit rollout-derived value targets at gated and lock-forced
        /// decisions (search-value distillation)
        #[arg(long)]
        distill: bool,
        /// Full-game rollouts per distill candidate
        #[arg(long, default_value = "32")]
        distill_k: usize,
        /// Distill samples per (candidate, opponent)
        #[arg(long, default_value = "4")]
        distill_m: usize,
        /// Probability of declining a forced safe lock during generation
        /// (0 = current behavior; the run recipe uses 0.05)
        #[arg(long, default_value = "0.0")]
        epsilon_lock: f32,
```

and the dispatch passes `DistillCfg { enabled: distill, k: distill_k, m: distill_m, epsilon_lock }`. (CLI default for `epsilon_lock` is 0.0 — flag absent ⇒ today's behavior; the documented run recipe sets 0.05. This intentionally differs from the spec's "default 0.05", which describes the campaign recipe, not the CLI.)

- [ ] **Step 2: Distill context capture in `RecordingPair`.** Add:

```rust
/// A decision worth distilling, captured during play; rollouts happen
/// post-game.
enum DistillCtx {
    /// Gated (close/endgame) active decision: top-2 candidates by static value.
    /// `sim_opp` is the evaluation context (ap1: simulated post-phase1 opps;
    /// ap2: current opps). `cands` = (mark, post/end state, value) sorted desc.
    Gated {
        phase: u8,
        state: State,
        opps: Vec<State>,
        sim_opp: Vec<State>,
        dice: [u8; 6],
        cands: Vec<(Option<Mark>, State)>,
    },
    /// Safe-lock firing: lock + best non-lock + runner-up lock indices into
    /// `cands` (the rule-free candidate list).
    Lock {
        ctx: LockCtx,
        state: State,
        opps: Vec<State>,
        sim_opp: Vec<State>, // ap1 only; empty otherwise
        dice: [u8; 6],
        active_player: usize, // pp1 only; 0 otherwise
        cands: Vec<(Option<Mark>, State)>,
        lock: Mark,
    },
}

#[derive(Clone, Copy, PartialEq)]
enum LockCtx {
    Ap1,
    Ap2,
    Pp1,
}
```

`RecordingPair` gains `distill: Option<std::rc::Rc<std::cell::RefCell<Vec<DistillCtx>>>>` and `epsilon_lock: f32`.

Detection logic, per method (player 0's `RecordingPair` only gets `distill: Some(...)` and `epsilon_lock > 0` — others `None`/0.0, mirroring uniform-ε):

- **active_phase1** (non-ε branch): call `active_phase1_choices`; if `Decision::Choices(plans)`, evaluate (`self.policy.bot().evaluate_batch`), and:
  - Gated capture: build the collapsed top-2 (reuse the collapse loop from `SearchBot::active_phase1` — copy the ~12-line block; in-crate, no guard needed beyond a debug_assert) and if ≥2 distinct candidates and `crate::strategy::search::gates(...)` fires (build the `Candidate`-equivalents inline — `gates` takes `&[Candidate]`; since `Candidate` is private to search.rs, change `gates` signature during Task 2's visibility edit to `pub(crate) fn gates(values: &[(f32, State)], our, opps) -> (bool, bool)` taking (value, post) pairs — adjust SearchBot's two call sites accordingly), push `DistillCtx::Gated { phase: 1, .. }` with the top-2 `(mark, plan-end-state)`.
  - If `Decision::Forced(Some(m))` where `state.would_lock_row(m)` and the post-state doesn't end the game: a lock firing. Rebuild the rule-free set via `phase1_plan_choices_with(state, &sim_opp, dice, false)`; if `Choices` with ≥2 distinct phase-1 marks (collapse first), push `DistillCtx::Lock { ctx: Ap1, .. }`. **ε-decline:** with probability `epsilon_lock`, play `eval_decision` over the rule-free choices instead of the lock (and still record the snapshot as usual — the normal recording cadence is downstream of the returned move and unchanged).
- **active_phase2** (non-ε branch): same shape with `active_phase2_choices` / `mark_choices_with(..., false)` for the rule-free set / `find_safe_lock(state, &generate_color_moves(dice))` for firing detection; gated capture from the `Choices` list directly (no collapse), sim_opp = opps.
- **passive_phase1**: production move via `passive_phase1_impl` as today; firing detection via `find_safe_lock(state, &generate_white_moves(white_sum))`; rule-free via `mark_choices_with(state, &marks, *state, opp_best_phase1_score(opps, white_sum), false)`; ε-decline same pattern; **no Gated capture on passive decisions** (search never gated them; out of scope).
- Candidate indices for Lock ctxs are computed at emission time (Task 3 Step 3), not stored.

- [ ] **Step 3: Post-game emission.** New function in pair_train.rs:

```rust
/// Emit distill samples for one game's captured contexts. Per candidate:
/// complete the turn deterministically (entries), run K full-game CRN
/// rollouts, and emit `m` samples per opponent pairing — value = mean
/// future-diff, final_diff = an individual rollout's future-diff — plus the
/// swap-doubled negated sample, matching build_pair_samples' conventions.
fn build_distill_samples(
    boot: &PairStrategy,
    ctxs: &[DistillCtx],
    cfg: DistillCfg,
    seed: u64,
) -> Vec<PairSample>
```

Implementation outline (complete the details from these exact semantics):
1. For each ctx, assemble the compared candidate list:
   - `Gated`: the stored top-2.
   - `Lock`: from `cands`, the lock (position by mark equality), the best non-safe-lock candidate, and the best other safe lock if any (same selection semantics as the Phase 16 builder: candidates sorted desc by value first — sort here using `evaluate_batch` values computed at emission time).
2. Entries per candidate by context: ap1 → `crate::strategy::search::phase1_entry(bot, state, sim_opp, dice, mark)`; ap2 → `phase2_entry(post, opps)`; pp1 → an n-player completion: apply our mark; simulate every OTHER passive player's phase-1 via `passive_phase1_impl`, the active player's phase 1 + phase 2 via `active_phase1_choices`/`eval_decision` + `active_phase2_choices`/`eval_decision` (READ `Game::play` first to fix the `active_player` index frame and the relative ordering of `opps`; document the mapping in a comment); propagate locks between steps; `first_active` = the player after the active one in our-relative frame.
3. `rollout_final_states(bot, &entries, context_seed(state, opps, dice), cfg.k, first_active)`.
4. Per candidate entry, per opponent k (entry states `[our, opps...]`): `cdiff = entry_our.points − entry_opps[k].points`; rollout future-diffs `d_r = (final_our − final_k) − cdiff`; `value = mean(d_r)`; emit `cfg.m` samples (rollout indices `0..m`, deterministic) with `features = pair_features(&entry_our, &entry_opps[k], &entry_opps_view)`, `value`, `final_diff = d_r`; plus the swapped negated sample per `build_pair_samples`' convention (swapped features via `pair_features(&entry_opps[k], &entry_our, &swapped_view)`, value and final_diff negated). Ended entries (single deterministic outcome): emit ONE sample (+swap), value = final_diff = the deterministic future-diff.
5. The boot net for any value evaluations is `boot.net` (the same convention `play_training_game` uses).

Wire into `play_training_game`: create the distill buffer for player 0 when `cfg.enabled`; after `game.play()`, `all_samples.extend(build_distill_samples(...))` with a per-game seed derived from the game seed. Print per-iteration distill sample counts in `self_play_train`'s iteration summary.

- [ ] **Step 4: Tests** (pair_train.rs tests module):

```rust
#[test]
fn distill_emission_counts_and_semantics() {
    // Tiny deterministic game with distill on: counts = Σ over ctxs of
    // (#compared candidates × #opponents × m × 2 swap), values are finite,
    // each candidate's m samples share `value` and differ in `final_diff`,
    // and the swapped twin is the negation.
}

#[test]
fn epsilon_lock_declines_only_at_firings() {
    // With epsilon_lock = 1.0 and a constructed lockable state, the
    // RecordingPair returns a non-lock move at a firing; with 0.0 it
    // returns the lock. Non-firing decisions are unaffected (same move
    // with either setting).
}

#[test]
fn distill_off_is_bit_identical() {
    // play_training_game with cfg.enabled=false, epsilon_lock=0.0 produces
    // identical samples to the pre-change behavior (same seed). Guard: run
    // it twice with the same seed and compare counts + a few sample values
    // (regression-by-determinism; the true pre-change comparison is the
    // existing test suite staying green).
}
```

Write real assertions — the comments above are the contracts; the implementing engineer builds the fixtures (lockable states as in Task 1's tests; small games with fixed seeds).

- [ ] **Step 5:** Full suite green; `cargo build --release` clean. Commit: `git add src/dqn/pair_train.rs src/main.rs && git commit -m "feat(pair): search-value distillation targets + targeted lock-decline exploration"`

---

### Task 4: smoke + handoff

**Files:**
- Modify: `docs/EXPERIMENTS.md` (phase stub + run recipe)

- [ ] **Step 1: Local smoke run** (foreground, minutes):

```bash
cargo run --release -- pair-train -i 2 -g 300 -e 2 -b 20000 --distill --epsilon-lock 0.05
```

Checks: distill sample counts printed and plausible (~300 games × ~8.5 ctx × ~2.2 cands × opp-pairings × m×2 — compute the expected order of magnitude from the printed ctx counts); generation wall time consistent with the ~8-10 min/iteration extrapolation at 5k games; loss not NaN; bench number sane (~57-59%). NOTE: this trains in `pair_model/` — afterwards, restore the committed model: `git checkout pair_model/model.mpk` and remove stray checkpoints (`rm -f pair_model/iter-*.mpk`), leaving the tree clean.

- [ ] **Step 2: K diagnostic.** From the smoke logs, compare the distill-sample μ-loss component trend across the 2 iterations (add a temporary or permanent per-iteration printout of TD-vs-distill sample MSE if not trivially visible — a permanent split metric is acceptable and useful; note in report). If distill MSE is flat at iteration 2 while TD MSE declines, flag "raise --distill-k" in the recipe.

- [ ] **Step 3: Write the EXPERIMENTS.md stub** (next phase number): one paragraph of setup (mechanism summary, pools, flags), the smoke numbers, the run recipe:

```bash
rm -f pair_model/iter-*.mpk
cargo run --release -- pair-train --distill --epsilon-lock 0.05 -g 5000 -e 5 -b 500000 -c --start-iteration 20
```

(start-iteration 20 keeps ε at floor + checkpoint numbering distinct, as Phase 14), what to watch (the curve; the +0.5% bar; the K escalation rule), and the post-run acceptance steps (lock-adjudicate + lock-ab + search-on bench on the selected checkpoint — exact commands). Mark results "pending user run".

- [ ] **Step 4: Commit:** `git add docs/EXPERIMENTS.md && git commit -m "docs: distillation campaign stub + run recipe (run pending)"`

---

## Self-review notes

- Spec coverage: target semantics/K/m → Task 3 Step 3 + CLI defaults; pools + candidates → DistillCtx capture (Gated top-2, Lock trio); static generation → no `--search` interaction (flags independent; recipe omits --search); ε-decline all three contexts player-0 → Step 2; src-change list matches spec §src (bot_impl Task 1, search Task 2, pair_train/main Task 3); evaluation/handoff → Task 4; examples' mirrors untouched (no edits to examples/ anywhere in this plan).
- The `gates` signature change (private `Candidate` → `(f32, State)` pairs) is the one production-code touch beyond pure extraction in Task 2 — SearchBot's two call sites adjust trivially; search tests are the regression net.
- Deviation from spec noted inline: CLI `epsilon_lock` defaults 0.0 (recipe sets 0.05) so the flag-absent path is bit-identical to current behavior, which `distill_off_is_bit_identical` pins.
- Type consistency: `DistillCfg` fields used in main dispatch and play_training_game match; `phase1_entry`/`phase2_entry`/`rollout_final_states`/`gates` names consistent between Task 2 definitions and Task 3 uses.
