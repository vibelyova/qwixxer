# Arm A — Shared-Encoder One-Hot Value Net (`aznet`) Design

**Date:** 2026-06-15
**Status:** design approved; implementation plan pending
**Goal in one line:** Test whether AlphaZero's *representation* lever — a one-hot
crossing-order encoding run through a shared per-row encoder — beats the current
hand-scalar pair net, isolated from every other variable, by dropping a new
value net into the existing pair-train pipeline and benching it **plain** (no
distillation) against a matched plain baseline.

This is **Arm A** of a staged campaign. Arm B (a directly-learned win-probability
head) and a distillation leg are deliberately out of scope here and only proceed
if Arm A pays (see §9).

---

## 1. Motivation & context

The pair-search campaign (EXPERIMENTS.md Phases 12–19) concluded that the
~60% vs-GA ceiling is *structural* — within-decision σ across candidates is
~4–5%, deeper search hurts, and the lock/endgame blind spot proved "not fixable
by data alone **under the current representation**." That caveat is the opening:
the current net (`src/dqn/pair.rs`) reads boards through hand-engineered scalars
(row progress, marks/11, a weighted-probability scalar). The `alpha-qwixx.txt`
design argues for a richer **one-hot crossing-order** encoding — explicit count,
explicit frontier, explicit lockability — which is exactly a *representation
change* and therefore the most plausible route past the wall.

The full AlphaZero stack (MCTS + policy head + win-prob value) is **not** built
here. Prior evidence says its headline levers (a policy prior over a tiny action
space; deeper tree search) are low-value or actively harmful for Qwixx, and the
expensive MCTS data-generation loop is a poor fit. We extract only the lever the
evidence points to — the representation — and test it in the proven, fast
pair-train harness.

### Established baselines (vs GA, 1v1, seed 42, paired CRN)

| bot | static | search |
|---|---|---|
| old pair net, **plain** (pre-distillation, Phase 12) | ~59.2% | ~60.1% |
| old pair net, **distilled** (iter-57, committed `pair_model/model.mpk`) | 59.85% | 60.7% |

Arm A compares against the **plain** row (apples-to-apples; distillation is a
later, separate leg). We do **not** retrain or modify the old net's training
code; the control is the **recorded** plain pair numbers above. This is weaker
isolation (the recorded run used the 1v1/3p/4p-thirds recipe, not 2-player-only)
— accepted deliberately to avoid touching `pair_train.rs`'s behavior. The
comparison still shares `Game`/`State`/seeds/loss-recipe/TD/bench.

---

## 2. Scope — what stays identical (the spine)

The entire training/eval pipeline is reused unchanged. Only **feature
extraction, the model, inference, and the augmentation index-map** change. These
components are kept bit-for-bit equivalent in behavior:

- **Diff-space afterstate contract.** Each candidate *move* is applied; the
  resulting state is evaluated. Input = (own board, the single opponent board,
  `cdiff`). Output = `(μ_diff, log σ²_diff)` predicting the distribution of
  `final_diff − current_diff`.
- **TD(λ=0.8) diff targets** (`td_diff_targets`, identical formula).
- **Decoupled μ/σ loss**: μ = MSE toward the TD target; σ = Gaussian NLL against
  the *final-diff* residual with μ detached (identical recipe to
  `PairModel::forward_step`).
- **Board-swap doubling**: every sample emitted in both board orders with
  negated targets.
- **Ranking**: `(cdiff + μ) / σ`. **WinProb**: `Φ((cdiff + μ) / σ)`.
- **`SearchBot`** (generic over `WinProb`): wraps the new net unchanged.
- **Bench harness**: same fixed-game-set paired CRN benchmark.

Out of scope for Arm A: distillation/expert-iteration (`DistillCtx`,
`build_distill_samples`), the hand-rolled inference kernel (`ManualPairNet`
analogue), multiplayer (>2 players), a policy head, a win head, any web/WASM
wiring.

---

## 3. Representation

### 3.1 Canonical crossing-order

Rows are indexed `[R=0 (asc), Y=1 (asc), G=2 (desc), B=3 (desc)]`. Each row is
encoded by **crossing order**, slot 0 = first crossable cell, slot 10 = the
lockable (terminal) cell:

- ascending (R, Y): slot `i` ↔ number `i + 2`.
- descending (G, B): slot `i` ↔ number `12 − i`.

The **free-pointer slot** of an unlocked row is derived from `State`'s
`row_free_values()[i] = Some(free)`:

- ascending: `slot = free − 2`.
- descending: `slot = 12 − free`.

`free` is always the slot immediately after the rightmost mark (marking advances
the pointer past the marked cell, including over skips), so the free-pointer slot
fully determines future legality. A locked row has `free = None`.

### 3.2 Per-row block — 28 dims

| group | dims | semantics |
|---|---|---|
| one-hot count | 13 | `total ∈ 0..=12`. **13, not 12**: a locked row gets the lock bonus (`total += 2`), so `total` reaches 12; triangular scoring makes 11→12 a real **+12-point** jump, so the 12th index must exist. |
| one-hot free-pointer slot | 11 | slot `0..=10`; **all-zero ⇒ locked** (no free pointer). |
| is_locked | 1 | explicit; locked rows are global game-enders (2 locks = game over). |
| is_lockable | 1 | `total ≥ 5 ∧ free on terminal slot`. **Kept on BOTH boards** — for the opponent it is a genuine denial/endgame threat signal (the `total≥5 ⇔ lockable` rule handed to the net), not a "what I can do this turn" leak. |
| weighted-prob scalar | 1 | today's `board_features` `f[12+i]`: `0` if the free pointer rests on an unlockable terminal, else `(ways/6) · (total+1)/11`. Retained as a proven EV-of-future-markability signal. |
| blanks scalar | 1 | `(free_slot − count)` (cells skipped left of the pointer; always ≥0; **0 if locked**), normalized `/10`. A genuine `free_slot × count` interaction the encoder would otherwise have to learn, and directly value-relevant (skipped cells = permanently lost potential). Reintroduces, at per-row resolution, the board-level `blanks` signal the old net used (dropped by the globals-collapse in §3.4). |

**Total: 28 dims/row.** The AZ doc's 35-dim block had an 11-dim
`is_markable_this_turn` group; that is dropped because afterstate evaluation has
no associated roll. A per-row **score** scalar was considered and **rejected** as
redundant: `triangular(count)` is a function of `count` alone, which the one-hot
already lets the encoder represent exactly.

### 3.3 Per-board block

- 4 row blocks × 28 = 112.
- one-hot strikes: `strikes ∈ 0..=3` → 4 dims (4 ends the game; never an
  afterstate input).
- **Per board raw = 116.**

### 3.4 Global features

- `cdiff = (our.count_points() − opp.count_points()) as f32`, then
  `(cdiff / 100).clamp(-1, 1)` — 1 dim.

Dropped vs the old net (justified by 2-player-only, §6): `num_opponents`
(constant 1), and the opponent-summary globals (max-opp-progress,
max-opp-strikes, opp-lockable-sum) — all redundant with the single full opponent
board now present at full resolution.

### 3.5 Flat feature vector — `AZ_FEATURES = 233`

Fixed layout (the model slices this; the batcher permutes row blocks within it):

```
[  0.. 28)  own row 0 (R)        \
[ 28.. 56)  own row 1 (Y)         |  own 4 row blocks (112)
[ 56.. 84)  own row 2 (G)         |
[ 84..112)  own row 3 (B)        /
[112..116)  own strikes one-hot (4)
[116..144)  opp row 0 (R)        \
[144..172)  opp row 1 (Y)         |  opp 4 row blocks (112)
[172..200)  opp row 2 (G)         |
[200..228)  opp row 3 (B)        /
[228..232)  opp strikes one-hot (4)
[232]       cdiff/100 (clamped)
```

Constants: `ROW_BLOCK = 28`, `BOARD_RAW = 116` (112 rows + 4 strikes),
`AZ_FEATURES = 233`.

**Swap doubling** = swap `[0..116)` ↔ `[116..232)` and negate `[232]`.
**Color augmentation** = permute the four 28-dim row blocks within own `[0..112)`
and within opp `[116..228)` *identically*; strikes and cdiff untouched.

---

## 4. Model architecture

```
AZ_FEATURES (233)
  ├─ slice 8 row blocks (own 4 @ [0..112), opp 4 @ [116..228)) → [batch*8, 28]
  │     shared f_row encoder:  Linear(28→32) → ReLU → Linear(32→16) → ReLU
  │     → [batch, 8*16 = 128]
  ├─ own strikes [112..116] (4) ─┐
  ├─ opp strikes [228..232] (4) ─┤ concat
  └─ cdiff [232] (1) ───────────┘
        → trunk input (128 + 4 + 4 + 1 = 137)
        Linear(137→128) → ReLU → Linear(128→64) → ReLU
        ├─ output_mean   Linear(64→1)
        └─ output_log_var Linear(64→1)
  forward → cat([mean, log_var], dim=1) → [batch, 2]
```

- **Shared encoder** `f_row`: one `Linear(28→32)` + one `Linear(32→16)` with
  tied weights, applied to all 8 row blocks by batching them as `[batch*8, 28]`.
- **Parameter count** ≈ 28·32+32 + 32·16+16 (encoder, 1 456) + 137·128+128 +
  128·64+64 + 64+1 + 64+1 (trunk + 2 heads, 26 050) ≈ **27.5k** (~2× the old
  net's 14k; chosen "modest headroom" — see §9 capacity-control).
- **No normalization** between trunk layers — matches the proven pair net's
  recipe (inputs are bounded one-hots/scalars; a steady LR is reused). LayerNorm
  is held in reserve (§11), added only if training shows instability; BatchNorm
  is avoided regardless (tiny, varying inference batches).
- **Inference: burn only.** No hand-rolled kernel in Arm A.

### Module name & types

`src/dqn/aznet.rs`:
- `pub const ROW_BLOCK`, `BOARD_RAW`, `AZ_FEATURES`.
- `pub fn az_row_block(state, row_idx) -> [f32; ROW_BLOCK]`
- `pub fn az_board(state) -> [f32; BOARD_RAW]`
- `pub fn az_features(our: &State, opp: &State) -> [f32; AZ_FEATURES]`
- `#[derive(Module)] pub struct AzModel`, `#[derive(Config)] pub struct AzModelConfig`
  (fields: `encoder_hidden=32`, `encoder_out=16`, `trunk1=128`, `trunk2=64`).
- `AzModel::forward(Tensor<B,2>) -> Tensor<B,2>` (cols: μ, log σ²).
- `pub fn az_batch_forward(model, device, &[[f32; AZ_FEATURES]]) -> Vec<(f32,f32)>`.
- `pub struct AzStrategy { model: Arc<AzModel<MyBackend>>, device }`
  - `load(dir)`, `from_model`, `from_shared`.
  - `impl Bot for AzStrategy` — `evaluate_batch_multi` ranks by `(cdiff+μ)/σ`
    against the single opponent (1v1: that opponent is the leader trivially).
  - `impl WinProb for AzStrategy` — `win_prob_multi` returns `Φ((cdiff+μ)/σ)`.

`AzModel` mirrors `PairModel`'s `forward_step`/`TrainStep`/`InferenceStep`
(decoupled μ/σ loss, identical math) — re-implemented on `AzModel` (small,
self-contained).

---

## 5. Training samples, batcher, loss

`src/dqn/aznet_train.rs`:

- `AzSample { features: [f32; AZ_FEATURES], value: f32, final_diff: f32 }` with a
  serde slice adapter (length 233 > serde's array cap), modeled on
  `pair_features_serde`.
- `AzBatcher` — color permutation over 28-dim row blocks (own + opp identically),
  seeded exactly as `PairBatcher` (`TRAIN_SEED ⊕ value.to_bits ⊕ batch_size`).
- `permute_rows(&mut [f32; AZ_FEATURES], swap_ry, swap_gb, swap_pairs)` — swaps
  whole 28-dim blocks at offsets `{0,28,56,84}` (own) and `{116,144,172,200}`
  (opp). R↔Y swaps blocks 0↔1; G↔B swaps 2↔3; pair-swap swaps {0↔2, 1↔3}.
- Reuse `td_diff_targets` — promote it to `pub(crate)` in `pair_train.rs` and
  import it (DRY; identical formula, already unit-tested).
- `build_az_samples(net_forward, snapshots, our_final, opp_final)` — the
  2-player specialization of `build_pair_samples`: one TD chain, μ from a forward
  pass over the chain's features, swap-doubled negated emission.

---

## 6. Self-play loop (2-player only, plain)

`aznet_train::self_play_train(artifact_dir, num_iterations, games_per_iteration,
epochs_per_iteration, bench_games, checkpoints, start_iteration)`:

- **Every game is 1v1** (drop the 1v1/3p/4p thirds). All players are recording
  `AzStrategy` bots; player 0 explores with the same ε schedule
  (`0.2·0.95^iter`, floor 0.07); others greedy.
- **No search, no distillation** (plain). The recorder is the static-policy
  subset of `RecordingPair` — a slimmer `RecordingAz` that records afterstates
  for active phase-2 (post-turn) and passive phase-1 (including skips), matching
  the existing recording cadence.
- Replay buffer = last 3 iterations; retrain `epochs_per_iteration` epochs over
  the full buffer (Adam lr 4e-4, batch 1024, 90/10 train/valid split) — same as
  the pair trainer.
- Per-iteration: save `model`, optional `iter-N.mpk` checkpoint, append
  `training_scores.csv`, optional `benchmark_vs_ga` at `bench_games`.
- Seeds: identical scheme (`TRAIN_SEED + iteration*games + idx`), so runs are
  reproducible.

Model dir: **`aznet_model/`**.

### Control (no old-net retraining)

We do **not** retrain the old net or touch `pair_train.rs`'s behavior. The
control is the **recorded** plain pair numbers (~59.2% static / ~60.1% search).
Weaker isolation is accepted: the recorded run used the 1v1/3p/4p-thirds recipe,
not 2-player-only, so the comparison carries a recipe difference on top of the
representation+capacity difference. This is the deliberate trade for leaving the
pair trainer untouched.

---

## 7. CLI wiring (`src/main.rs`)

- `BotType::Aznet`, `BotType::AznetSearch` added to the enum.
- `make_strategy` / `run_bench` template loader: load `AzStrategy::load("aznet_model")`;
  `AznetSearch` wraps it in `SearchBot::new(...)` (mirrors `Pair`/`PairSearch`).
- New `Commands::AznetTrain` mirroring `PairTrain` (iterations, games, epochs,
  bench, checkpoints, start-iteration) **without** `--search`/`--distill`
  (plain only) → calls `aznet_train::self_play_train`.
- No changes to the `PairTrain` command.

---

## 8. Testing strategy (TDD)

Feature extraction (goldens, the highest-value tests — they pin the lever):

1. **fresh board** — all count one-hots at index 0; free-pointer slot at index 0
   for every row (asc free=2→slot0; desc free=12→slot0); is_locked=0;
   is_lockable=0; wprob = `1/66`; strikes one-hot at index 0; cdiff=0.
2. **frontier one-hot, ascending** — mark R {2,3,5}; free=6 → slot 4 set, others
   0; count one-hot index 3.
3. **frontier one-hot, descending** — mark G {12,11,9}; free=8 → slot `12−8=4`
   set; count index 3.
4. **locked row** — build a safe lock (e.g. R 2..6 then mark 12): is_locked=1,
   free-pointer one-hot **all-zero**, count one-hot index = post-lock total
   (incl. bonus, reaches up to 12), is_lockable=0.
5. **count one-hot reaches 12** — a fully-marked+locked row sets index 12 (guards
   the 13-wide choice).
6. **is_lockable on both boards** — `total≥5 ∧ free on terminal` sets the bit for
   own *and* opponent boards.
7. **swap relationship** — `az_features(a,b)` vs `az_features(b,a)`: board halves
   exchange (`[0..116)`↔`[116..232)`), cdiff negates.
7b. **blanks scalar** — fresh row = 0; R{2,3,5} (count 3, free_slot 4) → `1/10`;
    locked row → 0.

Model / batcher:

8. `permute_rows` — each generator swaps the right 28-dim blocks in both halves;
   strikes/cdiff invariant; **involution** (apply twice = identity).
9. `az_batch_forward` — shape `[n,2]`, finite, deterministic across calls.
10. `AzModel::forward_step` — loss finite; μ output column matches `forward`'s
    col 0 (sanity that the decoupled loss is wired to the right head).

Pipeline:

11. reuse `td_diff_targets` hand-computed test (already exists; just confirm the
    promoted `pub(crate)` import compiles).
12. `build_az_samples` — 2-step 1v1 trajectory: 2 steps × 1 opp × 2 orders = 4
    samples; swap value/final_diff negated; last forward sample value =
    `final_diff − cdiff_t` (hand-computed).
13. `WinProb` monotonicity — a points lead with the same opponent is not rated
    lower.
14. determinism — `play_training_game` (the aznet variant) produces identical
    samples run-to-run on a fixed seed.

The existing `pair.rs`/`pair_train.rs` suites must stay green (only the
behavior-preserving `pub(crate)` promotion of `td_diff_targets` touches that
file).

---

## 9. Pre-registration & acceptance

**Primary metric:** new-repr **plain** win rate vs GA @ **1,000,000** games,
seed 42 (paired CRN), both **static** and **search** (`SearchBot`).

**Control:** the **recorded** plain pair numbers — ~59.2% static / ~60.1%
search (§6; weaker isolation, accepted). No old-net retraining.

**Checkpoint selection:** per-iteration **200k** static bench curve (SE ≈ 0.1pp
paired — enough resolution to track a +0.3pp signal; 50k's ~0.22pp SE is too
coarse). Final **1M** benches (static + search) on the selected checkpoint only.
Burn-only inference is slow but tolerable at these sizes.

| outcome | rule |
|---|---|
| **PAY** | new-repr static **or** search beats its plain control by **≥ +0.3pp** at 1M, CI-separated (99% CI, SE over per-pair means). ⇒ build the hand-rolled inference kernel, run a distillation leg, proceed to Arm B (win head). |
| **KILL** | improvement **< +0.1pp** ⇒ representation is not the lever; stop before Arm B; record and reconsider. |
| between | judgment call; lean toward a capacity-control run before deciding. |

**Capacity-control (run only if PAY):** the chosen net is ~27k vs the old ~14k,
so a win confounds representation with capacity. Disentangle by re-running either
the **old repr scaled to ~27k** or the **new repr shrunk to ~14k** under the same
plain recipe; attribute the gain accordingly. (Cheap; one extra training run.)

**Guards:** the per-iteration static curve must show a sane learning trajectory
(no divergence); 90/10 valid loss watched for overfit — if the valid gap blows
up, reduce epochs (old net needed 3 to break a plateau) before resizing.

---

## 10. File map

| file | change |
|---|---|
| `src/dqn/aznet.rs` | **new** — features, `AzModel`/`AzModelConfig`, `az_batch_forward`, `AzStrategy` (`Bot` + `WinProb`). |
| `src/dqn/aznet_train.rs` | **new** — `AzSample`, `AzBatcher`/`permute_rows`, `build_az_samples`, `RecordingAz`, `play_training_game`, `benchmark_vs_ga`, `self_play_train`, `train_with_epochs`. |
| `src/dqn/mod.rs` | export `aznet`, `aznet_train`; expose shared helpers as needed (`row_progress` is private — re-derive or `pub(crate)` it). |
| `src/dqn/pair_train.rs` | promote `td_diff_targets` to `pub(crate)` (only change; behavior-preserving). |
| `src/main.rs` | `BotType::Aznet`/`AznetSearch`; `Commands::AznetTrain`. |
| `docs/EXPERIMENTS.md` | Phase 20 stub (pre-registration) before the run; results after. |

No changes to `pair.rs`, `state.rs`, `game.rs`, `search.rs`, `sim.rs`, or the
web crate.

---

## 11. Open knobs (resolved during implementation, not blocking)

- Exact encoder widths (`32→16`) and trunk widths (`128→64`) — start as specified
  (~27k); adjust only on observed under/overfit.
- **LayerNorm between trunk layers** — held in reserve; add only if training
  shows instability/divergence (the no-norm recipe is the default, per §4).
- Whether `row_progress` is re-derived in `aznet.rs` or promoted `pub(crate)`.
- Epochs per iteration (start at the pair net's tuned value; revisit on valid
  loss).
- Number of self-play iterations and games/iter (match the pair net's plain
  recipe so the recorded-baseline comparison is as close as the recipe allows).
