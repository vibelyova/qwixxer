# Safe-Lock Adjudication — Design

**Goal:** Determine whether the "safe lock" meta-rule (`find_safe_lock` in
`bot_impl.rs`: force a lock whenever one is available that doesn't end the
game) is ever wrong — when, how often, and at what cost. The rule returns
`Decision::Forced`, so it bypasses both the value net and search; Phase 15
deliberately excluded forced decisions, so nothing measured so far covers it.

**Method:** Phase-15-style shadow study. Production play is unchanged (rule
on, trajectories real); every rule firing is logged; a second pass adjudicates
the forced lock against alternatives with full-game CRN-paired rollouts.

## Suspected failure modes (drive the conditioning analysis)

1. **Game-shortening while behind.** The first lock moves the game one lock
   from ending; when trailing, deferring the lock buys catch-up time.
2. **Wrong lock chosen.** With 2+ safe locks the rule takes
   `max_by_key(count_points)`, ignoring opponent-denial.
3. **Phase-1 opportunity cost.** Forcing the white-dice lock short-circuits
   the plan comparison.

Code-reading note (to verify formally in analysis, not a campaign): `mark_choices`
tests the safe lock *before* the winning-game-end force while
`phase1_plan_choices` tests it *after*; via marks the two forces are mutually
exclusive (a game-ending mark requires an existing lock, making every further
lock unsafe), so the asymmetry is believed immaterial.

## Decisions made during brainstorming

- **Adjudicate first; A/B only if the data warrants it** (user choice). The
  conditional A/B is pre-scoped below but not built in this campaign.
- **Full-game rollouts** (user choice): score = exact outcome, no truncation,
  no win-prob bootstrap. Rationale: the bootstrap leaf is the pair net, which
  was trained entirely on rule-on trajectories — structurally biased toward
  the rule at exactly these states. Full-game scoring removes leaf-value
  circularity (the net still drives rollout *policy*; unavoidable, same as
  production search).
- **Adjudication set** (user choice): forced lock vs best non-lock alternative
  (by static value over the rule-free candidate set), plus the runner-up safe
  lock when 2+ exist. Targets failure modes 1–2 at ~2–3 entries/event.
- **Implementation: third mode-pair in `examples/divergence.rs`** (user
  choice) — shares StateJson/seeding/sim machinery; no new example file; src/
  untouched.
- **No-contamination constraint** (user): `find_safe_lock` lives in code paths
  shared by all `Bot` strategies including GA. Nothing in this campaign
  modifies the rule; the conditional A/B implements its variant as an
  example-side `Strategy`, leaving src/ and hence GA untouched.

## Architecture

```
divergence lock-run ──► lock-events.jsonl ──► divergence lock-adjudicate ──► adjudicated jsonl
(10k games, static     (1 line per rule        (K=2048 FULL-GAME CRN          │
 pair vs GA, rule ON,   firing + game           rollouts: lock vs each        ▼
 firings detected)      summaries)              alternative)              analysis/lock_analysis.py
                                                                              │
                                                                              ▼
                                                                   EXPERIMENTS.md findings
                                                                   → (maybe) A/B campaign
```

### 1. Production-tree changes

**None.** Everything needed is already public (Task 1 of the Phase-15
campaign): `bot_impl::{Decision, argmax, eval_decision, active_phase1_choices,
active_phase2_choices}`, `State` methods (`generate_white_moves`,
`generate_color_moves`, `would_lock_row`, `would_end_game`, `count_points`,
`from_parts`, …), `strategy::sim`, `search::{context_seed,
sample_player_seed}`. Passive phase 1 uses the same public surface
(`generate_white_moves` + opponent context).

### 2. Collection — `lock-run` mode

```
cargo run --release --example divergence -- lock-run -n 10000 --seed 0 --out lock-events.jsonl
```

- Game setup identical to `run` (GA vs PAIR, rotating seats, paired per-seat
  dice, rayon): a `LockShadowPair` strategy plays the **unmodified production
  static pair** (delegates every decision to the same
  `active_phase1_choices`/`active_phase2_choices`/`eval_decision`/passive path
  the blanket impl uses — moves are production-identical, rule on).
- **Firing detection** at each decision: production decision is
  `Forced(Some(m))` where `our.would_lock_row(m)` and the post-mark state
  doesn't end the game ⇒ the safe-lock force fired (winning-end forces always
  end the game; single-candidate forces are excluded by the alternative check
  below).
- **Rule-free candidate set** (the new mirror surface, ~100 lines): copies of
  `mark_choices` and `phase1_plan_choices` minus the `find_safe_lock` force —
  same winning-end force, losing-end filter, domination pruning. Both mirrors
  return the candidate `(Option<Mark>, State)` list. Passive phase 1 reuses
  the `mark_choices` mirror (with `opp_best_phase1_score`-equivalent context —
  mirrored too, ~15 lines).
- An event is logged only when the rule fired **and** the rule-free set has
  ≥2 entries (otherwise there was no real decision).
- **Equivalence guard at collection time:** re-applying the lock force on top
  of the rule-free mirror must reproduce production's exact forced mark `m`
  for every logged event; hard panic otherwise. (Same role as Phase 15's
  drift guards: the mirror is fenced, not trusted.)
- Event fields: context (`game`, `turn`, `ctx ∈ {ap1, ap2, pp1}`,
  `has_marked` where applicable, `dice`, `our`/`opps` as StateJson,
  `our_points`/`opp_points`), `lock_mark`, all rule-free candidates with
  static values (sorted desc), indices of: the lock, the best non-lock, the
  runner-up safe lock (or null), `n_safe_locks`, `seed = context_seed(...)`.
  Game summary lines as in `run`.
- Expected volume: safe-lock firings ≈ 0.5–0.9/game for our seat → ~6–9k
  events from 10k games; collection ≈ the Phase-15 run cost (~6 min) since
  no shadow search runs.

### 3. Adjudication — `lock-adjudicate` mode

```
cargo run --release --example divergence -- lock-adjudicate --input lock-events.jsonl --out lock-events.adjudicated.jsonl -k 2048
```

- For each event: rebuild context from StateJson (assert recomputed
  `context_seed` == logged seed), rebuild the rule-free candidates (assert
  they match the logged list — marks, order, values within 1e-4).
- Entries: deterministic turn completion per compared candidate, reusing the
  Phase-15 entry builders — `phase1_entries` (ap1), `phase2_entries` (ap2) —
  plus a new passive builder (pp1). **pp1 entry construction** (no production
  mirror exists — search never handles passive decisions): apply OUR candidate
  mark to our state, then deterministically complete the ACTIVE player's turn
  from their perspective using the same public pipeline
  (`active_phase1_choices` + `eval_decision`, then `active_phase2_choices` +
  `eval_decision`), propagate locks after each step, and start the rollout
  with `active` = the player after the active one. This completion is an
  approximation (the real active player decided phase 1 simultaneously with
  us, not after seeing our mark), but it is *identical across the compared
  candidates*' shared opponent model, so the CRN-paired gaps remain valid;
  state it in the findings. Compared candidates: lock, best non-lock,
  runner-up lock (if present).
- **Full-game scoring:** a `rollout_scores_full` variant of `rollout_scores`
  with no horizon cap — `while !driver.all_over() { driver.step_turn() }` —
  and exact `SimGame::outcome` per sample; no `win_prob_multi` call. Same CRN
  seeding (`sample_player_seed(seed, s, p)`).
- Safety: a generous iteration cap (e.g. 200 turns) with a hard panic if any
  sim is still unfinished — a finished-game invariant violation must be loud,
  not a silent mis-score.
- Output per event: per-pair `(gap_mean, gap_se, z)` for lock-vs-best-non-lock
  and lock-vs-runner-up-lock (when present), with paired stats as in Phase 15
  (`paired_stats`), gaps oriented **alternative − lock** (positive = rule
  wrong), and verdicts `lock_wrong` (z > 2), `lock_right` (z < −2), else
  `coinflip`. Note: unlike Phase 15 there is no K=128 "original pick" — the
  rule is unconditional — so the CRN-reuse selection-bias caveat does not
  apply; the estimator is clean.
- Cost: ~8k events × ~3 entries × 2048 full games (≈10–30 plies each) ≈
  15–25 min on 8 cores.

### 4. Analysis — `analysis/lock_analysis.py`

- Loader for the new event type (same JSONL conventions; reuses `load.py`
  helper functions where applicable — `points`, board parsing).
- Report: verdict split overall and per context (ap1/ap2/pp1); conditioning on
  the failure-mode features — `cdiff` (behind/ahead), game stage (turn, total
  marks), locks already on board, `n_safe_locks`, lock row/points, gap to the
  best non-lock's static value; cost quantification — Σ gap over `lock_wrong`
  events / 10k games = win-prob points/game the rule costs (upper bound);
  rendered top-|z| `lock_wrong` positions (reuse `examples.py` board
  renderer via import or copy).
- Findings: next EXPERIMENTS.md phase. Must state the estimator differences
  vs Phase 15: full-game scoring (no truncation/bootstrap), no CRN-selection
  caveat; rollout-policy circularity (net trained rule-on drives rollout
  moves) remains and must be stated as the residual bias, direction:
  rollouts *play* locks the rule's way, likely flattering the rule.

### 5. Conditional A/B (pre-scoped, NOT built in this campaign)

If adjudication shows material cost: implement the variant rule (e.g. "lock
only when `cdiff ≥ 0`" or whatever the data suggests) as an **example-side
`Strategy`** that mirrors the production decision pipeline with the modified
force — src/ and GA untouched. Bench: variant-pair vs baseline-pair
head-to-head (paired dice, most sensitive) and variant-pair vs production GA
(500k, SE ≈ 0.05%). Equivalence guard: with the variant disabled, the mirror
strategy must reproduce baseline-pair moves on a replay set.

## Error handling

- Both modes refuse `--out` overwrite without `--force-overwrite`.
- Collection equivalence guard and adjudication rebuild/seed guards hard-panic
  with line/game context.
- Full-game iteration cap panic (no silent truncation).

## Testing

- **Smoke:** `lock-run -n 100` → firing rate ≈ 0.5–0.9/game, all contexts
  represented, zero equivalence-guard panics (the guard runs on every event —
  this is the mirror's correctness proof); determinism (same seed → identical
  file); `lock-adjudicate` on that file completes with zero guard panics.
- **Cross-check:** total locks by our seat per game (from final states)
  vs logged firings + game-ending locks — counts must reconcile approximately
  (sanity, reported not asserted).
- **Scale rehearsal:** 500-game run + full-K adjudication before the 10k run;
  verify wall-time extrapolation < ~30 min.

## Out of scope (v1)

- Any rule modification (production or example-side) — that's the conditional
  A/B campaign.
- Multiplayer, non-GA opponents.
- Near-miss states (lockable row exists but the rule didn't fire).
- Re-examining the `mark_choices` vs `phase1_plan_choices` force-ordering
  asymmetry beyond the one formal check noted above.
