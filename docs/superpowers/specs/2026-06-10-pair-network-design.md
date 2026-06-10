# Pairwise Differential Network ("pair" bot) — Design

**Goal:** Replace the single-board value network + hand-built opponent context with a
network that sees *both* boards and predicts the **score differential** distribution
directly. This gives the net row-level opponent visibility (lock races, game-length
prediction) and handles score correlation implicitly — `Var(A−B)` is the calibrated
quantity, with no independence assumption.

**Primary objective:** 1v1 winrate vs the GA champion. 3–4p stays supported but is not
tuned for in v1.

**Migration strategy:** Coexist. The new bot is a separate bot type (`pair`) with its
own model dir; the existing DQN stays untouched as a benchmark opponent and the
solo/web bot. Replacement happens only after the pair bot proves itself.

**Baseline to beat:** old DQN = 59.10% vs GA on the canonical paired game set
(`bench dqn ga -n 100000`, seed 42; 99% CI 58.75–59.46%).

---

## Architecture decision

Concatenated joint MLP (approach A). One trunk over
`[our board | opponent board | pair-level features]`. Chosen over a twin-tower
(siamese) encoder because the trunk can form cross-board features directly from raw
inputs (e.g. "my red progress minus their red progress" — the lock-race signal this
project exists for), with no embedding bottleneck. Pairwise consistency is trained in
via board-swap augmentation rather than enforced structurally. Twin-tower remains a
future ablation inside the same pipeline.

## Input features (45)

Per-board block (20), computed by a new `board_features(state: &State) -> [f32; 20]`:

| Idx | Feature | Notes |
|-----|---------|-------|
| 0–3 | row progress | locked = 1.0, as today |
| 4–7 | row marks / 11 | |
| 8–11 | row locked flags | |
| 12–15 | per-row weighted probability | **with lock-rule fix**: terminal number (12/2) contributes 0 ways while total < 5 (the old net's features 12–15 ignore this) |
| 16 | strikes / 3 | |
| 17 | blanks / 40 | |
| 18 | aggregate weighted probability | existing `aggregate_weighted_probability` |
| 19 | lockable rows / 4 | |

Pair-level (5):

| Idx | Feature | Notes |
|-----|---------|-------|
| 40 | `current_diff` = (our pts − opp pts)/100, clamped [−1,1] | known information given explicitly; the net never reconstructs triangular scores |
| 41 | num_opponents / 4 | |
| 42 | max `total_progress` over **all** opponents | includes the paired opponent — uniform semantics in every format (avoids the absent-vs-zero ambiguity); redundant with the opp block in 1v1, which is harmless |
| 43 | max strikes / 3 over all opponents | same |
| 44 | lockable-rows sum / 8 over all opponents | same; /8 matches the old net's normalization |

Layout: `[our 20 | opp 20 | pair 5]`. `OpponentContext` is not used by this bot; all
cross-player reasoning happens inside the net.

## Model

MLP `45 → 128 → 64 → 2` with ReLU (~14.3k params). Outputs:

- `μ_diff` — expected **future** differential: `E[final_diff − current_diff]`
- `log σ²_diff` — log-variance of the differential, clamped to the existing
  `[LOG_VAR_MIN, LOG_VAR_MAX]`

No auxiliary heads in v1 (win-probability head and absolute-score heads are future
ablations — see below). Loss is the proven decoupled recipe, on the differential:

- `L_μ = MSE(μ_diff, G_t − current_diff_t)` — TD(λ=0.8) target shifted to future space
- `L_σ = Gaussian NLL` with residual measured against the **actual** final diff:
  `(final_diff − current_diff_t) − μ_diff.detach()`
- `L = L_μ + 1.0 · L_σ`

Future-space targets mean the net learns only the *remaining* differential;
`current_diff` is added back at inference. Note `Var(A−B) = Var(A)+Var(B)−2Cov(A,B)`:
with Qwixx's strongly positive score correlation the diff target is likely less noisy
than the absolute-score target used today.

## Recording & targets

`RecordingPair` wrapper (mirrors `RecordingDqn`: ε-greedy on top of the shared
`bot_impl` selection functions, per-game `Rc<RefCell<…>>` buffer). Records
`(our post-decision State, Vec<State> of all opponents at that moment)`:

- active turn: one record after phase 2 (post-turn state), as today
- passive turn: one record after the decision — **including skips** (the old recorder
  only logged passive marks; skip afterstates are evaluated at inference, so they
  belong in the training distribution, and chains get denser)

Post-game, per recording player × per opponent: one coherent TD chain
("per-opponent trajectories"):

```
final_diff   = our_final − that_opp_final
G_{n−1}      = final_diff
G_t          = (1−λ)·(μ̂_diff(s_{t+1}) + current_diff_{t+1}) + λ·G_{t+1}     λ = 0.8
sample_t     = { pair_features_t,
                 value      = G_t − current_diff_t,          // μ target, future space
                 final_diff = final_diff − current_diff_t }  // σ residual ref, future space
```

The bootstrap `μ̂_diff(s_{t+1})` comes from one batched forward pass over the chain,
as `td_samples` does today. In 1v1 this is exactly one chain per player; in 3p/4p,
2–3 chains per player per game (~+60% sample yield).

## Augmentation

1. **Board swap — at sample construction** (not in the batcher): when building each
   chain sample, also emit the swapped sample — board blocks exchanged, pair-level
   features 40–44 recomputed exactly from the swapped perspective (the boards are in
   hand at this point), both targets negated. Doubles the sample count and trains in
   the pairwise consistency `eval(A,B) = −eval(B,A)` with exact feature semantics.
   (The negated TD target was bootstrapped from unswapped next states — that gap *is*
   the consistency-training signal, not an error.)
2. **Color permutation — in the batcher**: the existing 3-swap/8-permutation scheme
   applied to *both* board blocks identically (per-row groups at offsets 0 and 20;
   aggregate features are permutation-invariant). Deterministic per-batch seeding as
   today (`TRAIN_SEED`-derived).

## Inference

`PairStrategy` (in `src/dqn/pair.rs`) implements `Bot`:

- `evaluate_batch(candidates, opp_states)`:
  - leader = opponent with max current points; one pair-feature row per candidate
    (vs leader), single batched forward pass
  - rank score `z = (current_diff + μ_diff) / σ_diff` — monotone in Gaussian P(win);
    `current_diff` varies per candidate, subsuming the old per-candidate context
    rebuild
- `opp_states` empty (solo): rank vs a default empty board. Documented as
  unsupported; the old DQN remains the solo bot.

The blanket `Bot → Strategy` impl provides move selection, meta-rules, and opponent
phase-1 simulation unchanged. Model loading mirrors `DqnStrategy` (`load`,
`from_model`, `from_shared`, `load_from_bytes`).

## Training loop & CLI

`pair-train` subcommand mirroring `dqn-selfplay` (iterations, `-b` bench games,
checkpoints, start-iteration): same 1v1/3p/4p self-play thirds (prior evidence:
multiplayer games improved even the 1v1 benchmark), same ε schedule
(0.2·0.95^iter, floor 0.07), replay buffer of 3 iterations, Adam lr 4e-4, 10
epochs/iter, batch 1024, per-iteration checkpoint `pair_model/iter-N.mpk`, and
per-iteration benchmark vs GA on the fixed paired game set (`BENCH_SEED`), logged to
`pair_model/training_scores.csv`. All RNG seeded from `TRAIN_SEED`.

## Acceptance

1. `bench pair ga -n 100000` (seed 42) ≥ **59.10%** — parity with old DQN; stretch 60%+.
2. `bench pair dqn -n 100000` (seed 42) > 50% — direct head-to-head on identical
   games; the cleanest signal that joint evaluation beats compressed context.
3. Existing test suite untouched and green.

Honest expectation: the suspected 60–62% structural ceiling vs GA may hold. Success =
clear head-to-head win over the old net + at-least-parity vs GA, plus a stronger
evaluation platform for decision-time search later.

## Testing

Unit:
- per-row weighted-prob lock-rule fix (row at terminal with <5 marks contributes 0)
- pair-feature layout golden test (hand-built states → expected vector)
- TD diff chain on a hand-computed 3-step toy trajectory (future-space values)
- board-swap augmentation roundtrip (swapped features + negated targets consistent)
- color permutation applied identically to both blocks

Integration:
- tiny end-to-end training run (few games, 1 iteration) completes; model save/load
- `bench pair ga` smoke test with an untrained model (plays legal games via blanket impl)

## Future ablations (explicitly out of v1 scope)

- **Win-probability head** (BCE on outcome, tie=0.5) as alternative ranker — the only
  head matching the true objective; one experiment, added later so v1 stays the
  proven 2-head recipe
- TTA antisymmetrization (average both board orderings at inference)
- Twin-tower encoder variant (same pipeline, swapped model)
- 1v1-only training mix
- Decision-time expectimax search on top of the pair net (Tier 3b of the masterplan)
