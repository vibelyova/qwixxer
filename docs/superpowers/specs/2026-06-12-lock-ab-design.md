# Safe-Lock Suppression A/B — Design

**Goal:** Measure, in real play, whether making the safe-lock force *conditional
on not being behind* beats the unconditional rule. This is the conditional A/B
pre-scoped in `2026-06-12-safe-lock-adjudication-design.md` §5, triggered by
Phase 16's findings (rule wrong on 8.8% of firings, concentrated at `cdiff<0`;
suppression upper bound ~0.007 wp/game under perfect substitution).

**Why a bench and not more analysis:** suppression doesn't ban the lock — it
demotes it to a normal candidate the value net may still pick. Phase 16's
precision (16%) means 84% of interventions depend on the net re-picking good
locks unaided; whether it does is unmeasurable offline (rollout-policy
circularity) and decides the sign of the net effect.

## Decisions

- **Arms (user choice): sweep `cdiff<0`, `cdiff<=0`, `cdiff<-5`** — locating
  the boundary, not just testing one point. `cdiff` = our points − leading
  opponent's points, evaluated at decision time (same definition as Phase 16).
- **Zero src contamination** (standing constraint): the variant is an
  example-side `Strategy`; production code and GA untouched.
- **Primary metric:** variant vs baseline-pair head-to-head, paired dice +
  seat rotation, 1M games per arm (paired SE ≈ 0.05pp vs the ≤0.7pp expected
  ceiling). Secondary: each arm vs untouched GA (1M), compared against
  baseline-vs-GA from the production bench.
- **Decision rule:** adopt-the-variant requires a head-to-head win rate
  significantly above 50% (z>2) AND no regression vs GA. A flat result =
  keep the unconditional rule (Phase 16 stands as the explanation of its
  known, tolerable cost).

## Design

### VariantPair (in examples/divergence.rs)

`VariantPair { bot: PairStrategy, suppress_below: Option<isize> }` — suppress
the lock force when `Some(t)` and `cdiff < t`. Arms: `t=0` (`cdiff<0`), `t=1`
(`cdiff<=0`), `t=-5` (`cdiff<-5`); `None` = never (baseline-equivalent, used
by the equivalence check).

Per decision (reusing the lock-campaign mirrors, which are already fenced by
Phase 16's per-firing equivalence guards):
- **ap1:** `(decision, sim_opp) = active_phase1_choices(...)`;
  `(scan_lock, rule_free) = phase1_plans_mirror(state, &sim_opp, dice)`. If
  `scan_lock.is_some()` and suppressing → `eval_decision(bot, rule_free,
  &sim_opp)`. Else → `eval_decision(bot, decision, &sim_opp)` — production
  play by construction.
- **ap2:** `find_safe_lock(state, generate_color_moves)`; if firing and
  suppressing → `eval_decision` over `mark_choices_nolock(...)`; else
  `eval_decision(active_phase2_choices(...), opps)`.
- **pp1:** `find_safe_lock(state, generate_white_moves)`; if firing and
  suppressing → `eval_decision` over `mark_choices_nolock(...)` (baseline
  `*state`, `opp_best_phase1_score` mirror); else delegate to the production
  `passive_phase1`.

Only the suppression branch depends on mirror code; all non-suppressed play
calls the production pipeline verbatim.

### lock-ab mode

```
divergence lock-ab -n 1000000 --seed 0 --suppress-below 0 [--opponent pair|ga]
divergence lock-ab --equivalence-check 2000 --seed 0
```

- Bench harness: rotation pairs sharing per-seat dice (`seat_dice_seed`, as in
  the other modes). Opponent `pair` (default) = baseline `PairStrategy`;
  `ga` = champion. Reports wins/ties/avg points, suppression-fire count per
  game, and the paired win-rate diff with SE (per rotation-pair outcome
  differences), plus z vs 50%.
- **Equivalence check:** for n games, play `[VariantPair(None), baseline]`
  and `[baseline, baseline]` with identical seat dice; assert per-game final
  scores identical. Deterministic strategies + same dice ⇒ any mismatch is a
  variant bug; hard panic with game index.

### Benches & findings

Rehearsal at 10k/arm first (sanity: suppression fire-rate ≈ Phase 16's
behind-state firing share ≈ 0.25/game for t=0). Then 1M/arm head-to-head,
1M/arm vs GA, 1M baseline-vs-GA via the production `bench ga pair` (same
machine, same session). Findings as the next EXPERIMENTS.md phase: per-arm
head-to-head result with CI, vs-GA table, suppression fire rates, verdict per
the decision rule, and the explicit caveat that a flat result does NOT mean
Phase 16 was wrong — it means the net fails to re-pick good locks often enough
to cancel the gains (the 84%-precision gap realized).

## Error handling / testing

- Equivalence check is the gate: must pass at 2k games before any bench runs.
- Suppression fire counter sanity vs Phase 16 rates at rehearsal.
- Existing modes untouched; `cargo test` green.

## Out of scope

- Changing production rules regardless of outcome (that's a follow-up decision
  for the user with the A/B numbers in hand).
- Multiplayer, other opponents, other suppression conditions (P(win)-based
  variants etc.).
