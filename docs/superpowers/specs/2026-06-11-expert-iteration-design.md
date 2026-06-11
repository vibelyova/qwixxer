# Expert Iteration ("pair-train --search") — Design

**Goal:** Close the loop that decision-time search opened: generate self-play training
games with the search bot (the "expert") as every player's policy, so the value net
distills search-improved play into its static evaluation. Each iteration is policy
improvement (search) followed by policy evaluation (TD) — the TD-Gammon 3.0 recipe,
value-only AlphaZero skeleton.

**Why justified now:** Search demonstrably beats static play (+2.3–2.5% head-to-head
margin at K=128, replicated; EXPERIMENTS.md Phase 13). Distilling a policy that isn't
better than the student would be pointless; this one is.

**Primary success metric (user decision): static-V strength.** The per-iteration
static `benchmark_vs_ga` curve (fixed game set) is the selection signal — it directly
tests the distillation thesis "did the net absorb what search knows?" Search-on
benchmarks run once at the end, on the selected checkpoint.

## Approach (v1: trajectory-level only)

Generation runs with search ON for **all players**; recording, TD(λ) targets, replay
buffer, augmentation, losses, and the per-iteration benchmark are byte-for-byte the
existing pipeline. The net learns V of the search policy purely through better
trajectories and better-correlated outcomes. Train on **all samples** — a corrected
decision changes the returns of every sample on its trajectory (TD chains propagate
the credit), and V must stay accurate on the bulk state distribution it is queried on
(see the curriculum-randomization and pure-win-signal failures in EXPERIMENTS.md).

Pre-planned escalations, in order, only if the v1 curve is flat (each gated on the
previous one stalling): (1) disagreement-weighted trajectory sampling; (2)
search-value distillation (regress V toward rollout means at searched decisions).
Out of scope for v1. A policy head (full AlphaZero) is explicitly rejected — this
codebase's policy-gradient history (Phase 5) and no current need.

## Generation budget (user decisions)

- **5,000 games/iteration** (`-g 5000`). Generation cost with all players searching
  across the 1v1/3p/4p thirds ≈ 5 min/iteration. Fresh samples ≈ 1.5M/iteration;
  the 3-iteration replay buffer spans ~15k expert games at any moment. We have never
  been sample-starved; wall-clock is the binding resource.
- **Expert = the shipped configuration, K_SAMPLES=128** — no reduced generation
  budget. The expert's strength bounds the distillation target (V converges toward
  the value of the *generating* policy), and at 5k games the cost argument for a
  weaker expert evaporates. No `SearchBot` API change needed.
- **Epochs: `-e 5`.** Total gradient passes per iteration ≈ 4.5M-sample buffer × 5
  ≈ 22M — inside the empirically healthy band (old DQN recipe ≈ 54M over a 5.4M
  buffer: fine; pair at 185M over 18.5M: overfit plateau; at 55M: healthy). The
  `--epochs` flag exists; tune live off the curve if needed.
- ~8–9 min/iteration total; default run is 20 iterations ≈ 3 hours.

## Design

### CLI

`pair-train` gains a `--search` flag. Everything else reuses existing flags. The
documented run recipe:

```bash
rm -f pair_model/iter-*.mpk     # clear stale checkpoints from the previous run
cargo run --release -- pair-train --search -g 5000 -e 5 -b 500000 -c --start-iteration 20
```

The per-iteration benchmark runs at **500k games** (paired SE ≈ 0.05%): the expected
per-iteration improvement is small, and the static bench is cheap (~2 min at static
speed), so we buy the precision to see it. **The full training run is executed by
the user on a separate machine** — the implementation work ends at the smoke test;
the run recipe above is the handoff.

- Trains **in place** on `pair_model/` — warm start is automatic (the loop loads
  `pair_model/model` if present); the git-committed `model.mpk` is the recovery
  point; the hardcoded `BotType` loaders keep working.
- `--start-iteration 20` puts ε at its 0.07 floor immediately (a converged model
  should not re-suffer ε=0.2) and keeps checkpoint numbering distinct from the old
  run's cleared checkpoints.

### The expert policy in RecordingPair

`RecordingPair`'s move selection becomes a small policy enum:

```rust
enum PairPolicy {
    Static(PairStrategy),            // today's behavior (--search absent)
    Search(SearchBot<PairStrategy>), // expert iteration
}
```

- **Active phases:** the ε-coin flips FIRST. Explore → play the random mark and skip
  search entirely (exploration shouldn't be search-polished, and it saves the cost).
  Greedy → delegate to `SearchBot::active_phase1/active_phase2`, which internally
  does gating, Forced bypass, and static fallback — identical to the shipped bot.
- **Passive phase:** unchanged (`passive_phase1_impl` on the inner `PairStrategy`;
  search never applies to passive decisions).
- **Recording cadence:** byte-for-byte identical to today (snapshots at the same
  decision points); only the chosen moves differ.
- All players use the search policy; opponents at ε=0. Trajectories are fully
  expert-played and `opp_finals` reflect expert play.
- The inner `PairStrategy` is accessible via `SearchBot.bot` for the passive path
  and the TD bootstrap.

### TD bootstrap on the manual net

`build_pair_samples` switches its bootstrap forward pass from
`pair_batch_forward(model, device, …)` to `ManualPairNet::forward`, taking
`&ManualPairNet` instead of the model+device pair; `play_training_game` passes a
clone of one player's `PairStrategy.net` Arc — same ~1.5× inference win for
target computation. Tolerance-level numeric
drift vs burn is acceptable here (targets are estimates by definition).

### Benchmarks, success, failure

- **Per-iteration:** existing static `benchmark_vs_ga` (500k, fixed `BENCH_SEED`
  set, paired SE ≈ 0.05%) — the primary curve. Comparable to the previous run's
  history (old peak: 59.02% at iteration 19, measured at 100k — re-baseline the
  current model at 500k in iteration "0" terms by reading the warm-start model's
  first benchmark).
- **Success:** the static curve exceeds the old peak by ≥ +0.5% at some checkpoint.
  Selection by best static winrate; confirm the chosen checkpoint on the seed-42
  CLI set (baseline 59.2%), then run the search-on acceptance benches on it
  (`bench pair-search ga -n 50000`, baseline 60.1%; head-to-head margin vs the
  static net via `bench pair-search pair`).
- **Failure:** flat curve after ~10 iterations → record the negative result in
  EXPERIMENTS.md (Phase 14) with escalation (1) as the documented next step. Do not
  tune until something looks positive.

### Determinism

Search is context-hash deterministic and all RNG is `TRAIN_SEED`-derived, so
expert-iteration games are exactly as reproducible as static training games.

### Testing

Unit: ε-short-circuit (an exploring decision never invokes search — assert via a
stats hook or a counting wrapper); `PairPolicy` dispatch (Static path identical to
today on a seeded game); one-game generation determinism (same seed → identical
snapshot sequence). Integration: `pair-train --search -i 1 -g 60 -b 200` end-to-end
smoke; full existing suite untouched.

## Out of scope

- Escalations (1) and (2) above; policy head
- Generation-budget knobs (`SearchBot.samples` field, gate tightening) — add only
  if cost becomes binding again
- Search on passive decisions during generation
