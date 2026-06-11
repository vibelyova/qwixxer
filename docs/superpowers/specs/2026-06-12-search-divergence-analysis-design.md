# Search-Divergence Analysis — Design

**Goal:** Find out *when and why* decision-time search (pair-search) overrides the
static pair bot, in a form a human can read — and, if a pattern is strong enough,
encode it as a meta-rule so the static bot captures part of search's +2.3–2.5%
head-to-head edge (EXPERIMENTS.md Phase 13) at static speed. Fallback deliverable:
a categorized, example-backed understanding of the disagreement cases.

**Precedent:** the GA-vs-MCTS divergence study (EXPERIMENTS.md, 17.4% divergence;
"#1 pattern: GA picks a Double where MCTS prefers a Single") produced exactly this
kind of encodable insight.

**Known rates (Phase 13, gated production config):** ~16 eligible decisions/game,
close gate fires on ~51% of eligible, search overrides static on ~6.6% of searched
decisions, ~1.3 disagreements/game, 100% of realized disagreements inside fired
gates.

## Decisions made during brainstorming

- **Distribution: static-bot trajectories, search as shadow oracle.** Play
  continues with the static move at every decision; search runs as an observer.
  This asks exactly the question a static meta-rule answers: "at states the static
  bot actually reaches, when is its choice wrong?" (The alternative — logging
  inside production search benches — samples search's own trajectory through a
  gate-filtered window; rejected for rule mining.)
- **Log agreements too, not just disagreements.** Agreements are the control
  group; without them "feature X predicts disagreement" cannot be separated from
  "feature X is common."
- **High-K relabel pass** (user choice): disagreements at K=128 can be sampling
  noise; each one is replayed offline at K=2048 and labeled
  `search_right` / `static_right` / `coinflip`.
- **Analysis in Python** (user choice): venv with pandas/scikit-learn/matplotlib;
  Rust only dumps and relabels events.
- **Opponent / format:** 1v1 vs GA, mirroring the headline bench (`bench ga pair`).
  Multiplayer variants are out of scope for v1.
- **Experiment lives in `examples/`** (user choice): no production refactor for a
  one-shot experiment. `src/` gets only minimal, non-invasive enablers; the
  pipeline is trivially deletable when done.

## Architecture

```
divergence run ──► events.jsonl ──► divergence relabel ──► relabeled.jsonl
(10k games,         (1 line per        (K=2048 replay of        │
 static play,        eligible           disagreements +         ▼
 shadow search)      decision +         agreement sample)   analysis/ (Python)
                     game summaries)                        load → mine → examples
                                                                │
                                                                ▼
                                                     EXPERIMENTS.md findings
                                                     → (maybe) Rust meta-rule
                                                     → 500k bench validation
```

### 1. Production-tree changes (the complete list)

1. `strategy/mod.rs`: `mod bot_impl;` → `pub mod bot_impl;` (visibility only; its
   `Decision`, `active_phase1_choices`, `active_phase2_choices`, `eval_decision`
   are already `pub` items).
2. `State::from_parts(strikes: u8, rows: [(u8, Option<u8>); 4]) -> State` — ~10
   lines; `(total, free)` per row plus strikes fully determines a `State`, but
   `rows` is private, so relabel needs a constructor. Doubles as a test helper.
3. `Cargo.toml`: `serde` (with derive) and `serde_json` under
   `[dev-dependencies]` — visible to examples, absent from the production binary.

Nothing else. No `SearchBot` refactor; `K_SAMPLES` stays a const. Everything else
the pipeline needs is already public: `strategy::sim` (`SimGame`,
`BatchedRollouts`), `strategy::search` (`SearchBot`, `context_seed`,
`sample_player_seed`, `gates`-equivalent consts `GATE_MARGIN`/`K_CANDIDATES`/
`K_SAMPLES`, `phi`), `game::{Game, Player}`, `bot::DNA::load_weights` +
`default_genes` (GA), `PairStrategy`, and `State`'s getters.

### 2. `examples/divergence.rs` — collection (`run` mode)

```
cargo run --release --example divergence -- run -n 10000 --seed 0 --out events.jsonl
```

- Game setup copied from `run_bench` (~40 lines): GA vs shadow-PAIR, rotating
  seats, paired per-seat dice streams, rayon-parallel.
- `ShadowPair` (in the example) implements `Strategy`. At each eligible active
  decision (≥2 distinct candidates, not meta-forced — visible directly from the
  `Decision` enum):
  1. Builds the candidate list with static values via `bot_impl` choices +
     `evaluate_batch`, replicating phase 1's collapse-to-distinct-marks (~15
     lines), sorted by value desc. Static pick = index 0.
  2. Obtains search's pick by calling the **production**
     `SearchBot::active_phase{1,2}` with `force=true` on the same context —
     logged disagreements are ground truth by construction, not a
     reimplementation's opinion. (Static values are computed twice, once here and
     once inside `SearchBot`; negligible next to rollout cost.)
  3. Recomputes the gate flags (close: `static_gap < GATE_MARGIN`; endgame: ~5
     duplicated lines) as *features*.
  4. Appends an `Event` to its internal `Vec` and **returns the static move**.
- Forced and single-candidate decisions are not logged. Passive decisions pass
  through to the static bot (search never applied there).
- After each game the driver drains the shadow's events, stamps game id and
  outcome linkage; the main thread writes JSONL.

### 3. Event schema (JSONL)

One line per eligible decision:

```json
{"t":"d","game":123,"turn":17,"phase":1,"has_marked":null,
 "dice":[3,4,2,2,5,1],
 "our":{"strikes":1,"rows":[[3,7],[0,2],[2,9],[0,null]]},
 "opps":[{"strikes":0,"rows":[[2,5],[1,4],[0,12],[3,6]]}],
 "gate_close":true,"gate_endgame":false,"static_gap":0.04,
 "cands":[{"mark":[0,8],"v":1.23},{"mark":null,"v":1.19}],
 "search_mark":null,"search_pick":1,"seed":1234567890,"disagree":true}
```

One line per game:

```json
{"t":"g","game":123,"pair_seat":0,"scores":[61,58],"pair_won":true}
```

- `rows` is `[total, free]` per row (`free: null` = locked) — exactly the
  information `State::from_parts` needs. Serialization structs live in the
  example (no serde derives on `State`).
- `mark: null` means skip (phase 1) / skip-or-strike (phase 2, disambiguated by
  `has_marked`).
- `cands` lists **all** distinct choices with static values. `search_pick` is the
  index of search's returned move in `cands` (matched by mark equality, not by
  position); `disagree = search_pick != 0`.
- K=128 rollout scores are **not** logged — `SearchBot` doesn't expose its
  internal scores and we are not refactoring it. The relabel pass supplies
  strictly better gap estimates for every event the analysis labels.
- Volume estimate: ~16 events/game × 10k games ≈ 160k events ≈ ~100 MB. Collection
  cost ≈ force-searching every eligible decision ≈ 2× the gated bench cost,
  ~15 min wall on 8 cores.

### 4. `examples/divergence.rs` — relabel mode

```
cargo run --release --example divergence -- relabel --input events.jsonl --out relabeled.jsonl -k 2048 --agree-sample 0.1 --seed 1
```

- Selects all `disagree` events plus a seeded random `--agree-sample` fraction of
  agreement events (control: does search still agree at high K?).
- The example carries its own copy of search's entries-construction (deterministic
  turn completion) and rollout-scoring loop (~100 lines), parameterized by K and
  keeping per-sample scores for paired statistics. CRN streams extend the original
  ones via `sample_player_seed`: samples `0..128` are bit-identical to what
  production search saw.
- **Drift guards, per event:**
  1. Recomputed `context_seed` must equal the logged `seed` (hard error).
  2. The duplicated machinery, run at K=128, must reproduce the logged
     `search_mark` (hard error). This fences in the duplication: if `search.rs`
     changes behavior, relabel fails loudly instead of silently diverging.
- Output: the input line plus `hk_scores`, `hk_gap_mean` (search-pick minus
  static-pick win prob, per-sample paired), `hk_gap_se`, and
  `verdict ∈ {search_right, static_right, coinflip}` via `z = gap_mean/gap_se`,
  |z| > 2 → confident.
- Cost: ~25k events at 16× rollout cost ≈ 20–30 min.

### 5. Python analysis (analysis/)

- `analysis/.venv` (git-ignored), `analysis/requirements.txt`: pandas,
  scikit-learn, matplotlib. Claude installs and runs these.
- `load.py`: JSONL → tidy DataFrame (one row per event, joined to its game
  outcome) + engineered features:
  - **Position:** turn, total marks (both sides), score diff, strikes (both),
    blanks, locked/lockable rows, per-row progress, gate flags, static gap,
    eligible-candidate count.
  - **Move-pair (static pick vs search pick):** kind (mark/skip/strike), row,
    number, jump size (blanks created), immediate points delta, makes-lock,
    row progress and marks-in-row before, distance to terminal, same-row flag.
- `mine.py`: disagreement-rate cross-tabs over single features; move-kind
  transition matrix (static kind × search kind, like the GA-vs-MCTS table);
  shallow decision tree + permutation feature importance discriminating
  `search_right` disagreements from agreements; prints candidate rules with
  coverage/precision on a held-out split.
- `examples.py`: renders the highest-|z| disagreements as ASCII boards with both
  moves annotated, grouped by mined cluster — the "humanly-detectable" check.
- Findings written up as a new EXPERIMENTS.md phase.

### 6. Rule validation loop (conditional on finding a pattern)

1. Encode the candidate rule in Python; measure on held-out events: coverage of
   `search_right` disagreements, precision (rule's flip matches search's pick),
   false-fire rate on agreements.
2. If it survives: implement as a meta-rule in the static pair bot's decision
   path, then validate with the standard yardsticks — 500k-game bench vs GA
   (paired SE ≈ 0.05%) and head-to-head vs pair-search.
3. The shadow harness itself doubles as a cheap intermediate check: re-run
   `divergence run` with the rule active; the disagreement rate at rule-matched
   decisions should drop toward zero.

## Error handling

- `divergence run` refuses `--out` overwrite without `--force-overwrite`.
- Relabel hard-fails on either drift guard (no silent divergence).
- Malformed JSONL lines in relabel/Python: fail loudly with line number.

## Testing

The pipeline is example code, so correctness checks live in the example's own
execution path (the drift guards) plus a minimal src-side test:

- **Unit (src):** `State::from_parts` roundtrip — for assorted states (including
  locked rows and strikes), `from_parts(s.strikes, zip(row_totals, row_free_values))`
  reproduces a state whose getters, `count_points`, and feature vectors match.
- **Example-level checks:** shadow determinism — `run` twice with the same seed
  on a small `-n` → byte-identical output; relabel of a fresh small run completes
  with zero drift-guard failures (this exercises seed reconstruction AND K=128
  pick reproduction end-to-end).
- **Smoke (end-to-end):** `run -n 100` → rates sanity-checked against Phase 13
  (~50% close-gate among eligible; ~5–8% disagreement among *gate-fired*
  decisions, the subset comparable to Phase 13's searched set); `load.py` ingests
  the output.

## Out of scope (v1)

- Multiplayer formats, non-GA opponents.
- Logging passive or forced decisions.
- Any training-loop integration (this is analysis tooling).
- Auto-tuning gates from the data (possible follow-up).
- Refactoring `SearchBot` / exposing its internal rollout scores.
