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

### 1. SearchBot refactor (src/strategy/search.rs)

The shared decision flow inside `active_phase1` / `active_phase2` — build
candidates, evaluate gates, build rollout entries, score, pick — is extracted into
a method returning a record:

```rust
/// Everything search knows about one eligible decision.
pub struct DecisionRecord {
    pub cands: Vec<Candidate>,        // all distinct choices, sorted by static value desc
    pub close: bool,
    pub endgame: bool,
    pub scores: Vec<f32>,             // mean rollout win prob per shortlisted candidate
    pub pick: usize,                  // search's choice (index into cands)
    pub seed: u64,                    // context seed (replay check)
}
```

- The production `Strategy` impl is unchanged in behavior: same gating, same
  stats, plays `cands[pick]`.
- `K_SAMPLES` becomes a `k_samples: usize` field on `SearchBot` (default 128);
  the const stays as the default value. Existing tests must pass untouched.
- `Candidate` (mark, static value, post state) becomes `pub` within the crate as
  needed by the shadow driver.
- `search_pick` keeps per-sample scores internally; the relabel path needs
  per-sample top-2 pairing to compute a paired SE, so the scoring loop returns
  per-entry per-sample scores to its caller (aggregation moves one level up).
  Production behavior (mean comparison, tie keeps static) is unchanged.

### 2. Shadow driver + `divergence run` subcommand (src/main.rs + new module)

```
qwixxer divergence run -n 10000 --seed 0 --out events.jsonl
```

- Game setup copied from `run_bench`: GA vs shadow-PAIR, rotating seats, paired
  per-seat dice streams, rayon-parallel.
- `ShadowSearch` is a `Strategy` wrapping `SearchBot<PairStrategy>` with
  `force=true`: at each eligible active decision it obtains a `DecisionRecord`,
  appends an `Event` to an internal `Vec`, and **returns the static move**
  (`cands[0]`). Forced and single-candidate decisions are not logged. Passive
  decisions pass through (search never applied there).
- After each game the driver drains the shadow's events, stamps game id / seat /
  final outcome linkage, and the collected per-game vectors are written as JSONL
  by the main thread (no locking inside games).

### 3. Event schema (JSONL)

One line per eligible decision:

```json
{"t":"d","game":123,"turn":17,"phase":1,"has_marked":null,
 "dice":[3,4,2,2,5,1],
 "our":{"strikes":1,"rows":[[3,7],[0,2],[2,9],[0,null]]},
 "opps":[{"strikes":0,"rows":[[2,5],[1,4],[0,12],[3,6]]}],
 "gate_close":true,"gate_endgame":false,"static_gap":0.04,
 "cands":[{"mark":[0,8],"v":1.23},{"mark":null,"v":1.19}],
 "scores":[0.512,0.534],
 "search_pick":1,"seed":1234567890,"disagree":true}
```

One line per game:

```json
{"t":"g","game":123,"pair_seat":0,"scores":[61,58],"pair_won":true}
```

- `rows` is `[total, free]` per row (`free: null` = locked) — exactly `State`'s
  fields. `State` and `Row` get `serde::{Serialize, Deserialize}` derives
  (behind the existing `serde` feature); `serde_json` is added as a dependency
  alongside `serde` (dqn feature).
- `mark: null` means skip (phase 1) / skip-or-strike (phase 2, disambiguated by
  `has_marked`).
- `cands` lists **all** distinct choices with static values; `scores` covers only
  the searched shortlist (top-`K_CANDIDATES`, currently 2). `search_pick` indexes
  into `cands`; static pick is by construction index 0; `disagree = search_pick != 0`.
- Volume estimate: ~16 events/game × 10k games ≈ 160k events ≈ ~100 MB, ~15 min
  wall (force-searching every eligible decision ≈ 2× the gated bench cost).

### 4. `divergence relabel` subcommand

```
qwixxer divergence relabel --input events.jsonl --out relabeled.jsonl -k 2048 --agree-sample 0.1 --seed 1
```

- Selects all `disagree` events plus a seeded random `--agree-sample` fraction of
  agreement events (control: does search still agree at high K?).
- For each event: deserialize states, rebuild the decision context with the same
  bot, assert the recomputed `context_seed` equals the logged `seed` (hard error
  on mismatch — schema/logic drift detector), and re-run rollouts at `-k` samples.
  CRN streams extend the original ones: samples `0..128` are bit-identical to the
  collection run.
- Output: the input line plus `hk_scores`, `hk_gap_mean` (search-pick minus
  static-pick, per-sample paired), `hk_gap_se`, and
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
- Relabel hard-fails on context-seed mismatch (no silent drift).
- Malformed JSONL lines in relabel/Python: fail loudly with line number.

## Testing

- **Unit (Rust):** State serde roundtrip (including locked rows); shadow
  determinism — same seed twice → byte-identical event vectors; relabel on a
  fresh tiny run reproduces `scores` at k=128 exactly; `k_samples`
  parameterization leaves all existing search tests green.
- **Smoke (end-to-end):** `divergence run -n 100` → rates sanity-checked against
  Phase 13 (~50% close-gate among eligible; ~5–8% disagreement among
  *gate-fired* decisions, the subset comparable to Phase 13's searched set);
  relabel of that file completes with 0 seed mismatches; `load.py` ingests it.

## Out of scope (v1)

- Multiplayer formats, non-GA opponents.
- Logging passive or forced decisions.
- Any training-loop integration (this is analysis tooling).
- Auto-tuning gates from the data (possible follow-up).
