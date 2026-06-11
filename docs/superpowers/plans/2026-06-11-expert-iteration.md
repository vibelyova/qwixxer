# Expert Iteration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `pair-train --search` generates self-play training games with the shipped K=128 search bot as every player's policy, so the value net distills search-improved play; the full training run itself is executed by the user on a separate machine.

**Architecture:** A `PairPolicy` enum (Static = today's behavior, Search = `SearchBot<PairStrategy>`) inside `RecordingPair`, with the ε-coin in front of search; the TD bootstrap moves to `ManualPairNet`; everything else in the training pipeline is untouched. Spec: `docs/superpowers/specs/2026-06-11-expert-iteration-design.md`.

**Tech Stack:** Rust; existing pair/search machinery. Branch: `expectimax` (verify with `git branch --show-current` before starting; never switch branches).

---

## File structure

```
Modify: src/dqn/pair_train.rs — build_pair_samples on ManualPairNet; PairPolicy; RecordingPair; play_training_game/self_play_train `search` param; tests
Modify: src/main.rs           — `--search` flag on PairTrain
Modify: README.md, docs/ARCHITECTURE.md — short notes (final task)
```

Conventions: `cargo fmt` before every commit; the full suite (currently 64 tests) must stay green after every task.

---

### Task 1: TD bootstrap on `ManualPairNet`

**Files:**
- Modify: `src/dqn/pair_train.rs`

- [ ] **Step 1: Change `build_pair_samples` to take the manual net**

Replace the signature and the one forward call (the rest of the body is unchanged):

```rust
fn build_pair_samples(
    net: &crate::dqn::pair::ManualPairNet,
    snapshots: &[Snapshot],
    our_final: f32,
    opp_finals: &[f32],
) -> Vec<PairSample> {
```

and inside, replace

```rust
        let mus: Vec<f32> = pair_batch_forward(model, device, &feats).into_iter().map(|(m, _)| m).collect();
```

with

```rust
        let mus: Vec<f32> = net.forward(&feats).into_iter().map(|(m, _)| m).collect();
```

Numeric note (expected, fine): targets shift within tolerance-level drift vs burn —
targets are estimates by definition; nothing asserts byte-equality of samples
against the burn path.

- [ ] **Step 2: Update the caller in `play_training_game`**

In `play_training_game`, the per-player sample collection currently passes
`(model, device, &snapshots, ...)`. Build the strategies first so the bootstrap
net is the same Arc the players use (per spec):

```rust
    let n = num_opponents + 1;
    let strategies: Vec<PairStrategy> = (0..n)
        .map(|_| PairStrategy::from_model(model.clone(), device.clone()))
        .collect();
    let boot_net = strategies[0].net.clone();
```

then construct players by iterating `strategies.into_iter().enumerate()` (replacing
the current per-loop `PairStrategy::from_model` construction), and at the bottom:

```rust
        all_samples.extend(build_pair_samples(&boot_net, &snapshots, finals[i], &opp_finals));
```

- [ ] **Step 3: Update the test `build_pair_samples_emits_negated_swapped_samples`**

```rust
        let model = PairModelConfig::new().init::<MyBackend>(&device);
        let net = crate::dqn::pair::ManualPairNet::from_model(&model);
        ...
        let samples = build_pair_samples(&net, &snapshots, 30.0, &[20.0]);
```

(The two hand-computed target asserts at the chain end — `value == 8.0`,
`final_diff == 8.0` — are anchored at `G_{n−1} = final_diff` and do not depend on
the bootstrap values; they must still pass exactly.)

- [ ] **Step 4: Run tests, commit**

Run: `cargo test --lib dqn::pair_train` — 4 tests pass. Then full `cargo test`.

```bash
cargo fmt && git add src/dqn/pair_train.rs && git commit -m "refactor(pair): TD bootstrap on ManualPairNet"
```

---

### Task 2: `PairPolicy` enum in `RecordingPair`

**Files:**
- Modify: `src/dqn/pair_train.rs`

- [ ] **Step 1: Add the policy enum and rewire `RecordingPair`**

Add the import `use crate::strategy::search::SearchBot;` to the top of the file.
Replace `RecordingPair`'s `bot: PairStrategy` field and add the enum just above
the struct:

```rust
/// Move-selection policy for training-game players: today's static net, or
/// the shipped search bot (expert iteration). The ε-coin in `RecordingPair`
/// fires BEFORE the policy, so exploring decisions never pay for (or get
/// polished by) search.
enum PairPolicy {
    Static(PairStrategy),
    Search(SearchBot<PairStrategy>),
}

impl PairPolicy {
    fn bot(&self) -> &PairStrategy {
        match self {
            PairPolicy::Static(b) => b,
            PairPolicy::Search(s) => &s.bot,
        }
    }

    fn active_phase1(&mut self, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
        match self {
            PairPolicy::Static(b) => crate::strategy::active_phase1_impl(&*b, state, opp_states, dice),
            PairPolicy::Search(s) => crate::strategy::Strategy::active_phase1(s, state, opp_states, dice),
        }
    }

    fn active_phase2(
        &mut self,
        state: &State,
        opp_states: &[State],
        dice: [u8; 6],
        has_marked: bool,
    ) -> Option<Mark> {
        match self {
            PairPolicy::Static(b) => crate::strategy::active_phase2_impl(&*b, state, opp_states, dice, has_marked),
            PairPolicy::Search(s) => crate::strategy::Strategy::active_phase2(s, state, opp_states, dice, has_marked),
        }
    }
}
```

NOTE: `active_phase1_impl`/`active_phase2_impl` are re-exported from
`crate::strategy` (check `src/strategy/mod.rs` — the trimmed re-export list still
carries the three `*_impl` names). `SearchBot`'s `Strategy` methods take `&mut
self`, hence the fully-qualified `Strategy::` calls on `s`.

In `RecordingPair`:

```rust
struct RecordingPair {
    policy: PairPolicy,
    epsilon: f32,
    rng: SmallRng,
    recorded: std::rc::Rc<std::cell::RefCell<Vec<Snapshot>>>,
}
```

Update its `Strategy` impl (only the non-ε branches and the passive call change):
- `active_phase1`: the ε-branch is untouched; the else-branch becomes
  `self.policy.active_phase1(state, opp_states, dice)`.
- `active_phase2`: ε-branch untouched; the else-branch becomes
  `self.policy.active_phase2(state, opp_states, dice, has_marked)`.
- `passive_phase1`: `crate::strategy::passive_phase1_impl(self.policy.bot(), state, opp_states, dice)`
  (search never applies to passive decisions; identical for both variants).

Also update the doc comment on `RecordingPair` to mention the two policies and
that recording cadence is identical in both.

- [ ] **Step 2: Fix the construction sites (still Static-only in this task)**

In `play_training_game`, where `RecordingPair { bot: ..., ... }` was constructed,
use `policy: PairPolicy::Static(strategy)` (with `strategy` from the Task 1
strategies vector).

- [ ] **Step 3: Add the dispatch-equivalence test**

```rust
    #[test]
    fn static_policy_dispatch_matches_impl_calls() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let model = PairModelConfig::new().init::<MyBackend>(&device);
        let bot = PairStrategy::from_model(model.clone(), device);
        let mut policy = PairPolicy::Static(PairStrategy::from_shared(bot.model.clone(), bot.device));

        let mut state = State::default();
        state.apply_mark(Mark { row: 1, number: 6 });
        let opps = [State::default()];
        let dice = [3, 4, 2, 3, 5, 1];

        assert_eq!(
            policy.active_phase1(&state, &opps, dice),
            crate::strategy::active_phase1_impl(&bot, &state, &opps, dice)
        );
        assert_eq!(
            policy.active_phase2(&state, &opps, dice, false),
            crate::strategy::active_phase2_impl(&bot, &state, &opps, dice, false)
        );
    }
```

- [ ] **Step 4: Run tests, commit**

Run: `cargo test --lib dqn::pair_train` (5 tests) then full `cargo test` — all green.

```bash
cargo fmt && git add src/dqn/pair_train.rs && git commit -m "feat(pair): PairPolicy enum — static or search move selection in training"
```

---

### Task 3: search-mode wiring + ε-short-circuit and determinism tests

**Files:**
- Modify: `src/dqn/pair_train.rs`

- [ ] **Step 1: Thread a `search: bool` through generation**

`play_training_game` gains the parameter after `num_opponents`:

```rust
fn play_training_game(
    model: &PairModel<MyBackend>,
    device: &burn::backend::ndarray::NdArrayDevice,
    num_opponents: usize,
    search: bool,
    epsilon: f32,
    seed: u64,
) -> (Vec<PairSample>, f32) {
```

and the player construction wraps each strategy accordingly:

```rust
    for (i, strategy) in strategies.into_iter().enumerate() {
        let policy = if search {
            PairPolicy::Search(SearchBot::new(strategy))
        } else {
            PairPolicy::Static(strategy)
        };
        ...
        players.push(Player::new(
            Box::new(RecordingPair {
                policy,
                epsilon: if i == 0 { epsilon } else { 0.0 },
                rng: SmallRng::seed_from_u64(seed.wrapping_add(100 + i as u64)),
                recorded: buf,
            }),
            Box::new(SmallRng::seed_from_u64(seed.wrapping_add(i as u64))),
        ));
    }
```

`self_play_train` gains `search: bool` after `artifact_dir` parameters — place it
right after `start_iteration` (last param) to minimize call-site churn — and
forwards it into `play_training_game`. Add a mode line once before the loop:

```rust
    if search {
        println!("Expert iteration: generation uses the search bot (K=128) for all players");
    }
```

- [ ] **Step 2: ε-short-circuit test**

With ε = 1.0 every active decision takes the random branch, so the wrapped
`SearchBot`'s `Strategy` methods must never run — assert via its stats hook:

```rust
    #[test]
    fn exploring_decisions_never_invoke_search() {
        use crate::game::{Game, Player};
        use crate::strategy::search::{SearchBot, SearchStats};
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let model = PairModelConfig::new().init::<MyBackend>(&device);

        let mut players = Vec::new();
        let mut stats = Vec::new();
        for i in 0..2u64 {
            let mut sb = SearchBot::new(PairStrategy::from_model(model.clone(), device));
            let st = std::rc::Rc::new(std::cell::RefCell::new(SearchStats::default()));
            sb.stats = Some(st.clone());
            stats.push(st);
            players.push(Player::new(
                Box::new(RecordingPair {
                    policy: PairPolicy::Search(sb),
                    epsilon: 1.0, // always explore
                    rng: SmallRng::seed_from_u64(900 + i),
                    recorded: std::rc::Rc::new(std::cell::RefCell::new(Vec::new())),
                }),
                Box::new(SmallRng::seed_from_u64(910 + i)),
            ));
        }
        let mut game = Game::new(players);
        game.play();

        for st in &stats {
            assert_eq!(
                st.borrow().active_decisions,
                0,
                "search ran despite epsilon=1.0 — the ε-coin must fire first"
            );
        }
    }
```

- [ ] **Step 3: generation-determinism test**

```rust
    #[test]
    fn search_generation_is_deterministic() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let model = PairModelConfig::new().init::<MyBackend>(&device);

        let run = || play_training_game(&model, &device, 1, true, 0.07, 4242);
        let (s1, f1) = run();
        let (s2, f2) = run();
        assert_eq!(f1, f2);
        assert_eq!(s1.len(), s2.len());
        for (a, b) in s1.iter().zip(&s2) {
            assert_eq!(a.features, b.features);
            assert_eq!(a.value, b.value);
            assert_eq!(a.final_diff, b.final_diff);
        }
    }
```

- [ ] **Step 4: Run tests, commit**

Run: `cargo test --lib dqn::pair_train` (7 tests; the search ones take a few
seconds — they run real searched games on a random net) then full `cargo test`.

```bash
cargo fmt && git add src/dqn/pair_train.rs && git commit -m "feat(pair): search-on expert generation behind a flag"
```

---

### Task 4: CLI flag, smoke test, docs, handoff

**Files:**
- Modify: `src/main.rs`
- Modify: `README.md`, `docs/ARCHITECTURE.md`

- [ ] **Step 1: `--search` on PairTrain**

In the `PairTrain` variant, after `start_iteration`:

```rust
        /// Generate games with the search bot as every player's policy
        /// (expert iteration)
        #[arg(long)]
        search: bool,
```

Destructure it in `main()`'s match arm and forward:

```rust
        }) => dqn::pair_train::self_play_train("pair_model", iterations, games, epochs, bench, checkpoints, start_iteration, search),
```

(adjust to the actual parameter position chosen in Task 3 — `search` is the last
parameter).

- [ ] **Step 2: End-to-end smoke (NOT the full run)**

```bash
cargo test 2>&1 | grep "test result"            # all green
cargo build --release
time ./target/release/qwixxer pair-train --search -i 1 -g 60 -b 200
```

Expected: the "Expert iteration:" mode line prints; one iteration completes with
sample counts and a winrate; note the wall time (60 search-generated games — the
per-iteration generation estimate for 5k games is ~83x this). DO NOT run a full
training — the user executes that on a separate machine.

- [ ] **Step 3: Docs**

`README.md` usage block, after the pair-train line:

```bash
cargo run --release -- pair-train --search -g 5000 -e 5 -b 500000 -c --start-iteration 20   # Expert iteration
```

`docs/ARCHITECTURE.md`, append to the Pair-Search section:

```markdown
**Expert iteration** (`pair-train --search`): training games are generated with
the search bot as every player's policy (the e-coin fires before search, so
exploring moves skip it); recording, TD targets, and losses are unchanged — the
net distills search-improved trajectories. Recommended run (executed on a
dedicated machine): 5k games/iter, 5 epochs, 500k-game per-iteration static
benchmark (the primary metric; paired SE ~0.05%). Design doc:
docs/superpowers/specs/2026-06-11-expert-iteration-design.md.
```

- [ ] **Step 4: Commit + handoff message**

```bash
cargo fmt && git add src/main.rs README.md docs/ARCHITECTURE.md && git commit -m "feat(pair): --search flag — expert iteration generation"
```

Report the handoff recipe verbatim in the final summary:

```bash
rm -f pair_model/iter-*.mpk
cargo run --release -- pair-train --search -g 5000 -e 5 -b 500000 -c --start-iteration 20
```

with the reading guide: primary curve = winrate column of
`pair_model/training_scores.csv` (500k fixed-set static bench; old peak 59.02 at
iter 19 was measured at 100k — read the new run's first iterations as the
re-baseline); success = clearly exceeding the re-baselined warm-start level by
≥ +0.5% at some checkpoint; then run `bench pair ga -n 100000` (vs 59.2%) and
`bench pair-search ga -n 50000` (vs 60.1%) on the selected checkpoint and record
Phase 14 in EXPERIMENTS.md (positive or negative).

---

## Execution notes

- Branch `expectimax`; verify before starting and after every subagent.
- Task 1 changes `build_pair_samples`' signature — Tasks 1–3 all touch
  `pair_train.rs` sequentially; no parallel execution.
- The search tests in Task 3 run real rollouts on a random-init net; if they are
  slow (> ~30 s), reduce to a 1v1 game only — do not weaken the assertions.
- Sample-target drift vs the burn bootstrap (Task 1) is expected and harmless;
  nothing in the suite asserts byte-equality of training samples across backends.
- The full expert-iteration training run is the USER's job on a separate machine.
  The implementation ends at the 60-game smoke.
