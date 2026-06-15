# aznet Memory Fix — store compact states, expand in the batcher

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Cut `aznet` replay-buffer memory ~20× so the pre-registered `-g 20000` recipe runs without OOM, by storing the two compact board `State`s in each training sample and expanding them to the 233-float input vector inside the batcher (instead of storing the expanded vector).

**Why:** `AzSample` currently stores `[f32; 233]` (940 B). The OOM (after ~4 iterations at `-g 20000`) and the data-starved `-g 5000` plateau (58.3%, below the ~59.2% baseline) both trace to this: a 233-float sample is ~5× the pair net's, the 3-iteration replay buffer plus transient copies multiply it, and sample count grows as play improves. Storing `(our: State, opp: State, value, final_diff)` is ~48 B (~20× smaller), making aznet *more* memory-efficient than the pair net while changing nothing about the math.

**Scope:** `src/dqn/aznet_train.rs` (AzSample, build_az_samples, AzBatcher, tests), a one-word `PartialEq` add to `src/state.rs`, and a `split_off` tweak in `train_with_epochs`. No change to `aznet.rs`, the model, the loss, ranking, CLI, or training dynamics. `az_features`/`permute_rows` are unchanged — they just run at batch time now.

**Key facts (verified):**
- `burn::data::dataset::InMemDataset::new` requires only `I: Clone + Send + Sync` — **no serde**. The `az_features_serde` adapter + `Serialize`/`Deserialize` derive on `AzSample` are removable.
- `State` and `Row` are `#[derive(Debug, Clone, Copy)]` (Row also `PartialEq`); `State` is all-`Copy` primitives ⇒ auto `Send + Sync`. `State` does **not** derive `PartialEq` yet (tests need it).
- `build_az_samples` already builds a transient `feats` Vec for the μ forward pass (`az_batch_forward`); that stays (transient), only the *stored* representation changes.

Run all tests with: `cargo test --features dqn`.

---

## Task 1: Add `PartialEq` to `State`

**Files:** Modify `src/state.rs:5`

- [ ] **Step 1: Add the derive**

Change `src/state.rs:5` from:
```rust
#[derive(Debug, Clone, Copy)]
pub struct State {
```
to:
```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct State {
```
(`Row` at line 11 already derives `PartialEq`; `strikes: u8` is `PartialEq`; so this compiles. It only adds a trait impl — nothing relies on `State` not being `PartialEq`.)

- [ ] **Step 2: Verify build + suite still green**

Run: `cargo build --features dqn` then `cargo test --features dqn 2>&1 | grep "test result"`
Expected: clean build; all existing tests still pass (94 lib + 2 bin).

- [ ] **Step 3: Commit**
```bash
git add src/state.rs
git commit -m "feat(state): derive PartialEq on State (enables aznet sample comparison)"
```

---

## Task 2: Store compact states in `AzSample`; expand in the batcher

**Files:** Modify `src/dqn/aznet_train.rs`

This task changes the sample type, the two producers/consumers (`build_az_samples`, `AzBatcher::batch`), removes the serde adapter, and updates the two tests that referenced `.features`. Do it as one cohesive change (the type change forces all sites at once), but follow the TDD order below: update the tests first to the new contract, watch them fail to compile, then make them pass.

- [ ] **Step 1: Rewrite the two tests that reference `.features` to the new state-based contract**

In `src/dqn/aznet_train.rs`, the `tests` module:

(a) In `build_az_samples_emits_negated_swapped_pairs`, replace the two feature-layout assertions:
```rust
            assert_eq!(fwd.features[0..BOARD_RAW], swp.features[BOARD_RAW..2 * BOARD_RAW]);
            assert_eq!(fwd.features[BOARD_RAW..2 * BOARD_RAW], swp.features[0..BOARD_RAW]);
```
with state-swap assertions:
```rust
            assert_eq!(fwd.our, swp.opp, "swap exchanges the two boards");
            assert_eq!(fwd.opp, swp.our, "swap exchanges the two boards");
```
(The `swp.value == -fwd.value` / `swp.final_diff == -fwd.final_diff` checks and the `samples[2].value`/`final_diff == 8.0` checks stay as-is — the value math is unchanged.)

(b) In `play_training_game_is_deterministic`, replace:
```rust
            assert_eq!(a.features, b.features);
```
with:
```rust
            assert_eq!(a.our, b.our);
            assert_eq!(a.opp, b.opp);
```
(The `a.value`/`a.final_diff` equality checks stay.)

- [ ] **Step 2: Add a new batcher-expansion test**

Append to the `tests` module (covers the new behavior: the batcher expands stored states to the 233-float input deterministically):
```rust
    #[test]
    fn batcher_expands_states_to_inputs_deterministically() {
        use burn::data::dataloader::batcher::Batcher;
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let batcher = AzBatcher::<MyBackend> { _phantom: std::marker::PhantomData };

        let mut a = State::default();
        a.apply_mark(Mark { row: 0, number: 5 });
        let b = State::default();
        let items = vec![
            AzSample { our: a, opp: b, value: 1.0, final_diff: 2.0 },
            AzSample { our: b, opp: a, value: -1.0, final_diff: -2.0 },
        ];

        let batch1 = batcher.batch(items.clone(), &device);
        let batch2 = batcher.batch(items.clone(), &device);
        // Shape: [2, AZ_FEATURES]. (`.shape().dims` is the repo idiom — see AzModel::forward.)
        assert_eq!(batch1.inputs.shape().dims, [2, AZ_FEATURES]);
        // Deterministic (seed derived from value.to_bits + batch_size).
        let v1 = batch1.inputs.into_data().to_vec::<f32>().unwrap();
        let v2 = batch2.inputs.into_data().to_vec::<f32>().unwrap();
        assert_eq!(v1, v2);
        // targets / final_diffs carried through.
        assert_eq!(batch1.targets.into_data().to_vec::<f32>().unwrap(), vec![1.0, -1.0]);
        assert_eq!(batch1.final_diffs.into_data().to_vec::<f32>().unwrap(), vec![2.0, -2.0]);
    }
```

- [ ] **Step 3: Run the tests; confirm they fail to compile**

Run: `cargo test --features dqn dqn::aznet_train:: -- --nocapture`
Expected: **compile errors** — `AzSample` has no `our`/`opp` fields yet.

- [ ] **Step 4: Change `AzSample` to store states; remove the serde adapter**

Replace the serde adapter module and the `AzSample` struct. Delete the entire `mod az_features_serde { ... }` block. Replace:
```rust
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct AzSample {
    #[serde(with = "az_features_serde")]
    pub features: [f32; AZ_FEATURES],
    pub value: f32,
    pub final_diff: f32,
}
```
with:
```rust
/// One training sample: the two boards (compact `State`s — expanded to the
/// 233-float input by [`AzBatcher`] at batch time, keeping the replay buffer
/// ~20x smaller than storing the expanded features), plus the TD(λ) target
/// (`value`) and the actual final-diff residual (`final_diff`).
#[derive(Clone, Copy, Debug)]
pub struct AzSample {
    pub our: State,
    pub opp: State,
    pub value: f32,
    pub final_diff: f32,
}
```
If `serde` is no longer referenced anywhere else in the file, also remove any now-unused `serde` import (there is none at module top in the current file — the derive was the only use; the `az_features_serde` module's `use serde::...` goes away with the module).

- [ ] **Step 5: Update `build_az_samples` to emit state-based samples**

The transient `feats`/`mus`/`cdiffs`/`g` computation stays (it feeds the μ forward + TD targets). Only the two `samples.push(...)` lines change. Replace:
```rust
        samples.push(AzSample { features: feats[t], value, final_diff: fdiff });
        samples.push(AzSample {
            features: az_features(&opps[0], our),
            value: -value,
            final_diff: -fdiff,
        });
```
with:
```rust
        samples.push(AzSample { our: *our, opp: opps[0], value, final_diff: fdiff });
        samples.push(AzSample { our: opps[0], opp: *our, value: -value, final_diff: -fdiff });
```
(`our` is `&State`, `opps[0]` is `State` (`Copy`). `feats` is still built above for the forward pass and dropped at function end — no longer stored.)

- [ ] **Step 6: Update `AzBatcher::batch` to expand states**

In the `Batcher` impl, replace the input-building closure. Change:
```rust
        let inputs: Vec<f32> = items
            .iter()
            .flat_map(|s| {
                let mut f = s.features;
                permute_rows(&mut f, rng.gen(), rng.gen(), rng.gen());
                f
            })
            .collect();
```
to:
```rust
        let inputs: Vec<f32> = items
            .iter()
            .flat_map(|s| {
                let mut f = az_features(&s.our, &s.opp);
                permute_rows(&mut f, rng.gen(), rng.gen(), rng.gen());
                f
            })
            .collect();
```
(The seed line `items[0].value.to_bits()` and the `targets`/`final_diffs` collection are unchanged. `az_features` is already imported in this file.)

- [ ] **Step 7: Run the tests; confirm pass**

Run: `cargo test --features dqn dqn::aznet_train:: -- --nocapture`
Expected: all `aznet_train` tests pass (permute_rows, build_az_samples [now state-based], play_training_game determinism, the new batcher-expansion test, and the self_play_train smoke).

- [ ] **Step 8: Commit**
```bash
git add src/dqn/aznet_train.rs
git commit -m "perf(aznet): store compact states in AzSample, expand in batcher (~20x less memory)"
```

---

## Task 3: Avoid the train/valid split double-copy in `train_with_epochs`

**Files:** Modify `src/dqn/aznet_train.rs` (`train_with_epochs`)

- [ ] **Step 1: Use `split_off` instead of two `.to_vec()` copies**

Replace:
```rust
    let split = (samples.len() * 9) / 10;
    let train_data = InMemDataset::new(samples[..split].to_vec());
    let valid_data = InMemDataset::new(samples[split..].to_vec());
```
with:
```rust
    let split = (samples.len() * 9) / 10;
    let mut samples = samples;
    let valid = samples.split_off(split);
    let train_data = InMemDataset::new(samples);
    let valid_data = InMemDataset::new(valid);
```
(`split_off` reuses the existing allocation for `train` and moves the tail into `valid` — no clone of the sample data. Behavior is identical: same split point, same order.)

- [ ] **Step 2: Run the smoke test + full suite**

Run: `cargo test --features dqn dqn::aznet_train::tests::self_play_train_smoke -- --nocapture` then `cargo test --features dqn 2>&1 | grep "test result"`
Expected: smoke passes; full suite green (94 lib + 2 bin, plus the new batcher test → 95 lib).

- [ ] **Step 3: Commit**
```bash
git add src/dqn/aznet_train.rs
git commit -m "perf(aznet): split_off for train/valid to avoid sample double-copy"
```

---

## Task 4: Record the memory fix + confounded-run note in EXPERIMENTS.md

**Files:** Modify `docs/EXPERIMENTS.md` (Phase 20 section)

- [ ] **Step 1: Add an implementation-note paragraph under Phase 20**

In the Phase 20 section of `docs/EXPERIMENTS.md`, immediately above the `### Results` line (keep the `_(to be filled after the run)_` placeholder), insert:
```markdown
### Implementation note — memory (2026-06-15)

The first run (`-i 40 -g 20000 -e 3 -b 200000 -c`) OOM'd after ~4 iterations:
each `AzSample` stored the expanded `[f32; 233]` input (940 B, ~5x the pair
net's sample), and sample count grows as play improves, so the 3-iteration
replay buffer plus transient copies exceeded RAM. A reduced `-g 5000` run
completed but **plateaued at 58.3%** (200k bench) — below the ~59.2% plain
baseline, **but confounded**: a larger-input/larger net was fed 4x less data
per iteration than the recipe. Fix: `AzSample` now stores the two compact
`State`s and the batcher expands them to the 233-float input (~20x less
memory), so the pre-registered `-g 20000` recipe runs. Adjudication waits on a
clean full-data run benched at 1M seed-42 (static + search).
```

- [ ] **Step 2: Commit**
```bash
git add docs/EXPERIMENTS.md
git commit -m "docs: Phase 20 note — OOM + data-starved 58.3% (confounded); memory fix"
```

---

## Final verification

- [ ] `cargo test --features dqn` — full suite green (95 lib + 2 bin).
- [ ] `cargo build --release --features dqn` — clean.
- [ ] Sanity: `cargo run --release --features dqn -- aznet-train -i 1 -g 200 -e 1` completes and writes a loadable `aznet_model/model`; then `rm -rf aznet_model` (do not commit the throwaway model).
- [ ] (User, separately) the real run `-i 40 -g 20000 -e 3 -b 200000 -c` should now stay within memory.
