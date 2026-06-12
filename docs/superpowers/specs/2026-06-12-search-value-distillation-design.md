# Search-Value Distillation with Lock-State Targets — Design

**Goal:** Train the pair net on rollout-derived value targets emitted at the
decision states where its miscalibration is measured to live — gate-firing
(close/endgame) decisions and rule-forced safe-lock firings — closing the two
known calibration gaps: the diffuse pool behind pair-search's +2.4pp edge
(Phase 15) and the lock blind spot the unconditional rule trained into the net
(Phases 16–17).

**This is the remaining rung of Phase 14's ladder**, with the σ-target and
mixing-weight questions resolved and the lock pool folded in. Honest prior
(Phase 14's own estimate): ~30–40% odds of +0.3–0.5% static-V.

## Why this differs from expert iteration (Phase 14)

Expert iteration transmitted ~one bit per searched decision (which candidate
got played), diluted into trajectory TD noise — Phase 14 measured ~93% of TD
targets indistinguishable from the net's current fit, and the campaign
plateaued at +0.4% (below the +0.5% bar). Distillation changes what crosses
the channel:

1. **Target SNR:** a distill μ-target is a K-rollout mean (≈√K less noise
   than a single game outcome), placed exactly at decision states.
2. **Direct ranking supervision:** both shortlisted candidates get targets —
   the ranking, the only thing that changes play, is supervised directly, with
   CRN making the contrast precise.
3. **Off-trajectory coverage:** the *unchosen* candidate's post-state is
   labeled. For lock firings this is qualitatively new: the rule fires before
   the search gate, so no amount of expert iteration could ever label a
   declined lock — the pool is structurally unreachable by Phase 14's
   mechanism and disjoint from (additive to) the diffuse pool.

Known limit: rollouts are played by the same net (circularity), so targets
estimate value-under-current-policy. Distillation fixes *inconsistency* (the
net disagreeing with its own measured consequences — exactly what Phases 15/16
quantified); the flywheel (better V → better rollout policy → better targets
next iteration) is the lever against the bias itself.

## Decisions made during brainstorming

- **Targets in diff space, decoupled per the existing loss** (resolves Phase
  14's σ question): per candidate post-state, emit `m` `PairSample`s with
  `value` = mean future-diff over K full-game rollouts (μ via MSE) and
  `final_diff` = an individual rollout's future-diff (σ via NLL, spread
  preserved). Zero loss-code changes; distill samples take the same
  swap/color augmentation as TD samples.
- **K = 32 (default, `--distill-k`).** Target noise is unbiased and averaged
  out by sample count; for fixed compute, more states beat more precision per
  state, and K cannot reduce the circularity bias — only iteration can.
  Diagnostic at smoke time: if the distill-sample loss component sits pinned
  at target-noise level instead of declining, raise K at the run handoff (a
  flag, not a redesign). Higher K becomes necessary only if targets are ever
  confidence-filtered (not in v1).
- **Generation policy: static** (not `--search`). Phase 14 already measured
  search-on generation alone; static keeps attribution clean and spends the
  rollout compute on targets instead of play.
- **Targeted ε-decline at lock firings** (user choice): with probability
  `--epsilon-lock` (default 0.05), a firing is declined and the move is
  value-selected over the rule-free candidate set (A/B-variant semantics), in
  ALL three contexts — including passive phase 1, which currently has zero
  exploration. Puts declined-lock continuations on-trajectory (~3–5% of
  games); injected targets alone are off-trajectory.
- **Adoption bar (user choice): +0.5% static-V over re-baseline** on the
  per-iteration 500k bench (unchanged from Phase 14 — no post-hoc lowering),
  **plus lock acceptance as secondary evidence** (informs interpretation, not
  adoption): on the selected checkpoint, `lock-adjudicate` lock_wrong rate
  materially down (target <5% from 8.8%) and `lock-ab` suppression edge
  shrunk toward zero. Uniquely, these secondaries distinguish "mechanism
  failed" from "ceiling reached" even if the headline misses.
- **Run logistics (user choice): handoff.** Implementation ends at the smoke
  test (1–2 short local iterations); the user executes the full ~20-iteration
  run elsewhere with a documented recipe and reports the curve.

## Design

### Target states and candidates (~8.5/game combined)

| pool | detection (during generation) | candidates receiving targets |
|---|---|---|
| gate-firing decisions | close/endgame gates recomputed on static candidate values (no search policy needed) | top-2 by static value |
| safe-lock firings | rule fires (ap1/ap2/pp1, same semantics as Phase 16) | lock, best non-lock, runner-up lock (if any) |

Deterministic turn completion per candidate (the search-entries semantics),
then K full-game CRN rollouts per candidate (shared dice streams across
candidates of one decision); each rollout's final states yield the future-diff
sample. `m` (default 4, `--distill-m`) samples emitted per candidate.

### src changes (production training feature — invasive changes appropriate)

- `src/main.rs`: `--distill`, `--distill-k`, `--distill-m`, `--epsilon-lock`
  flags on the `pair-train` subcommand, threaded into `self_play_train`.
- `src/dqn/pair_train.rs`: target-state detection, candidate building, target
  emission inside the generation loop; ε-decline in `RecordingPair` (all
  three contexts; the existing uniform-ε machinery is untouched and still
  bypasses distill emission, as it bypasses search today).
- `src/strategy/search.rs`: extract the entries-construction closures into
  `pub(crate)` functions; add a `pub(crate)` full-game rollout scorer
  returning per-sample final diffs (the truncated production scorer is
  untouched).
- `src/strategy/bot_impl.rs`: `find_safe_lock` and the rule-free candidate
  enumeration exposed `pub(crate)` for firing detection/decline.
- The examples' guarded mirrors stay as-is (their drift guards keep
  protecting the Phase 15–17 analysis pipelines; consolidation is out of
  scope).

### Volume & mixing

~8.5 target states/game × ~2.2 candidates × m=4 ≈ 75 distill samples/game ≈
375k/iteration at 5k games — vs ~1.5M TD samples/iteration. Mixing is
controlled by `m` (and implicitly K); both are run-recipe knobs. Per the
Phase 12/14 lessons, the full TD buffer is retained — distill samples are
additive, never a replacement.

### Per-iteration cost estimate

Target rollouts: 5k games × 8.5 × 2.2 × 32 full games (~10–20 plies) — on the
order of Phase 14's all-player K=128 truncated-search generation; expect
~8–10 min/iteration generation + the usual training/bench, ~20 iterations
≈ 4–5 h on the user's machine.

### Evaluation & acceptance

- Per-iteration: 500k static bench vs GA (paired SE ≈ 0.05%) — the curve.
- Checkpoint selection: best static-V, as Phase 14.
- On the selected checkpoint: `lock-adjudicate` (lock_wrong rate), `lock-ab`
  (suppression edge), search-on bench vs GA. The Phase 16/17 harnesses are
  the acceptance tests; no new eval code.
- Findings: next EXPERIMENTS.md phase; must report the curve, the bar
  verdict, both lock secondaries, and the mechanism interpretation
  (calibration fixed vs ceiling).

### Smoke (end of implementation, before handoff)

- Unit: distill-sample emission counts match the detected target states ×
  candidates × m; ε-decline fires at the configured rate and only at firings;
  full-game scorer's diffs are consistent with final states; existing tests
  green.
- 1–2 iterations at small `-g` locally: non-degenerate loss curve; distill
  loss component declines (else the K diagnostic fires); bench sanity.
- Handoff recipe documented in the findings stub (flags, iteration count,
  what to watch, the K escalation rule).

## Out of scope

- Policy head / full AlphaZero (rejected in Phase 14's design).
- Confidence-filtered or gap-weighted targets (would force higher K).
- Consolidating the examples' mirrors onto the new `pub(crate)` functions.
- Changing the safe-lock rule itself (Phase 17's verdict stands; revisit only
  after a retrained net via the lock-ab harness).
