# Decision-Time Search ("pair-search" bot) — Design

**Goal:** A search-augmented bot that, at gated active-turn decisions, replaces the
static value ranking with truncated greedy rollouts bootstrapped by the value net.
This targets exactly where a static evaluator is weakest: sharp game-end boundaries
(lock races, strike endgames) become *exact* outcomes inside the search horizon.

**Why now:** The pair network reached statistical parity with the old DQN (59.2% vs
GA; docs/EXPERIMENTS.md Phase 12), strong evidence that static evaluation is
saturated. Search is the lever that doesn't depend on squeezing more out of V.

**Scope:** Search at **both active phases** (phase 1 and phase 2). Passive decisions
stay static (v2 candidate). Inference-only — training is untouched. New bot type
`pair-search` coexists with `pair`/`dqn`, sharing `pair_model/`.

**Baselines to beat (seed-42 paired set):** pair = 59.2% vs GA; head-to-head
pair-search vs pair > 50% with significant paired CI is the primary acceptance signal.

---

## Architecture overview

```
SearchBot<B: WinProb>          (src/dqn/search.rs — generic over the value bot)
  ├─ implements Strategy directly (search needs turn context Bot doesn't carry)
  ├─ candidates from shared *_choices() pipelines   (bot_impl refactor)
  ├─ symmetry dedup → top K_CANDIDATES by static value → gate check
  ├─ BatchedSim: lockstep simulation of K_SAMPLES futures per candidate
  └─ leaf scoring: exact outcome if game ended, else WinProb at horizon
```

### Constants (tunable, top of search.rs)

| Const | Default | Meaning |
|---|---|---|
| `K_CANDIDATES` | 2 | candidates searched (after symmetry dedup, by static value) |
| `K_SAMPLES` | 64 | sampled futures per candidate, CRN dice shared across candidates |
| `HORIZON_TURNS` | num_players | full turns simulated after completing the current turn (one round) |
| `GATE_MARGIN` | 0.15 (calibrate) | top-2 static z-gap below which search triggers |

## The `WinProb` capability trait

Averaging across sampled futures requires a calibrated win *probability* — rank
scores can't be mixed with exact 1/0 outcomes from sims that reach game over.

```rust
pub trait WinProb: Bot {
    /// P(we end ahead of the leading opponent), per candidate state.
    fn win_prob_batch(&self, candidates: &[State], opp_states: &[State]) -> Vec<f32>;
}
```

v1 implements `WinProb` for `PairStrategy`: `Φ((current_diff + μ) / σ)` vs the
leading opponent, batched (one forward). `Φ` via the Abramowitz–Stegun erf
approximation (unit-tested against known values). `DqnStrategy` (has μ/σ for both
sides) and GA (logistic of instinct gap, uncalibrated) can implement it later —
the search machinery itself is generic over `WinProb`.

## Batched multi-group evaluation on `Bot`

```rust
// In trait Bot — default works for every existing Bot unchanged:
fn evaluate_batch_multi(&self, groups: &[(&[State], &[State])]) -> Vec<Vec<f32>> {
    groups.iter().map(|(c, o)| self.evaluate_batch(c, o)).collect()
}
```

`PairStrategy` overrides it: concatenate all groups' pair-feature rows into ONE
forward pass (each row is independent — `pair_features(cand, leader_of_group, opps_of_group)`),
then split results by group offsets. This is what makes lockstep simulation cheap:
~6 mega-forwards (hundreds of rows) per searched decision instead of ~768 small ones.
`PairStrategy::evaluate_batch` is then re-expressed as the 1-group case.

## bot_impl refactor: expose choices, keep meta-rules single-source

Split the two active pipelines into pure-logic candidate generation + evaluation,
so `SearchBot` and the blanket `Bot → Strategy` impl share one source of truth:

```rust
pub(crate) enum Decision {
    Forced(Option<Mark>),                       // winning game-end / safe lock — search never overrides
    Choices(Vec<(Option<Mark>, State)>),        // filtered + pruned, ready to evaluate
}

pub(crate) fn active_phase1_choices(bot, state, opp_states, dice) -> (Decision, Vec<State> /* sim_opp */);
pub(crate) fn active_phase2_choices(bot, state, opp_states, dice, has_marked) -> Decision;
```

The blanket impl becomes: get choices → `evaluate_batch` → argmax (behavior
identical to today). `SearchBot` becomes: get choices → static eval → dedup/gate →
search or fall back to static argmax. For phase 1, `Choices` entries are collapsed
to **distinct phase-1 marks** before search (the simulation re-decides phase 2
itself, so plans differing only in their phase-2 part are the same search candidate).

Note `active_phase1_choices` still runs `simulate_opp_phase1` internally (it needs
sim_opp for its winning/losing filters); the returned sim_opp is reused for static
evaluation.

**Regression gate:** this refactor touches shared code paths (`strategy/mod.rs`,
`bot_impl.rs`). After it, `bench pair ga -n 10000`, `bench dqn ga -n 10000`, and
`bench ga opportunist -n 10000` (default seed) must produce byte-identical output
to before the refactor.

## Symmetry dedup

Two candidates are strategically identical iff some color permutation maps one
post-state to the other **while fixing every opponent board** — the same 8-element
group as the training augmentation (red↔yellow, green↔blue, asc-pair↔desc-pair;
the pair swap is a true game symmetry because `P(sum=s) = P(sum=14−s)`).

- New helper `State::permuted(swap_ry: bool, swap_gb: bool, swap_pairs: bool) -> State`
  in state.rs. Under the pair swap, row contents move to the opposite-direction row
  with marked numbers mapped `x → 14−x` (free pointer likewise; totals/strikes
  unchanged; locked stays locked).
- Canonical form of a position = the lexicographically smallest serialization of
  `(our_state, opp_states)` over the 8 joint permutations. Candidates with equal
  canonical forms are deduped (keep the first); dedup runs BEFORE the top-K cut so
  a symmetric twin cannot consume the search budget.
- Property test: `board_features(s.permuted(a,b,c))` equals the batcher's
  `permute_colors(board_features(s), a,b,c)` (per-board block), for random states.

## The simulator (`BatchedSim`)

A dedicated lockstep simulator over plain `State`s — `Game` can't be entered
mid-turn and owns its players. One instance advances ALL live sims
(`K_CANDIDATES × K_SAMPLES`) phase-by-phase:

1. **Entry.** Phase-2 search enters at "current turn complete, next player rolls".
   Phase-1 search enters mid-turn: compute every opponent's phase-1 response
   against the pre-phase-1 snapshots (simultaneity, as in `Game::play`), apply
   them together with our candidate phase-1 mark, propagate locks, game-over
   check, then our phase-2 (re-decided by the sim), locks, game-over check — this
   deterministic completion is shared by all samples of a candidate and computed once.
2. **Turn loop** (replicates `Game::play` semantics exactly): roll active player's
   dice from the sim's dice stream → all players' phase-1 decisions against
   pre-phase-1 snapshots → apply simultaneously → propagate locks → game-over
   check → active phase 2 → locks → game-over check → advance active player.
3. **Decisions inside sims** use the shared pure-logic choices + one entry in the
   phase's `evaluate_batch_multi` mega-batch (segmented argmax distributes results).
   **Lite phase 1 for simulated players:** plans are evaluated against *current*
   opponent states with no nested opponent-simulation (full fidelity would recurse
   sim_opp evals inside every simulated turn for ~3× cost; sims are approximations).
4. **Termination.** A sim that reaches game over drops out of the lockstep batch
   and records its exact outcome. Surviving sims stop after `HORIZON_TURNS` and are
   scored by `win_prob_batch` (one batch over all survivors).

**Scoring.** Sample score = exact outcome (1.0 if our points beat every opponent,
0.5 top-tie, 0.0 loss) when ended, else `WinProb` at the horizon vs the leader.
Candidate score = mean over its K_SAMPLES. Highest mean wins; ties (incl. "all
samples identical") fall back to static order.

**Determinism + CRN.** Dice-stream seeds are a deterministic hash (SplitMix64 over
the byte serialization) of the decision context `(our_state, opp_states, dice)`,
plus the sample index. Same K dice streams for every candidate (CRN — candidate
ranking differences reflect decisions, not luck), and the whole bot is
stateless-deterministic, so paired benches stay byte-reproducible.

**Equivalence test.** Replay N seeded 2p/3p/4p games through both `Game::play` and
the simulator's turn loop (same dice) and assert identical final states. For this
test the simulator runs decisions in a **full-fidelity mode** (the same blanket-impl
functions `Game`'s Bot players use, including sim_opp in phase 1) so that any state
divergence isolates a *mechanics* bug (phases, lock propagation, game-over order).
The lite phase-1 mode is a deliberate policy deviation used only inside search
rollouts; it shares the same mechanics code path and is covered by unit tests.

## Gating

Search triggers at an active decision iff there are ≥2 deduped candidates AND:

- top-2 static z-gap < `GATE_MARGIN`, OR
- endgame proximity: any player has ≥1 locked row, any player has 3 strikes, or
  any shortlisted candidate locks a row.

`Forced` decisions (winning game-end, safe lock) bypass search entirely. Everything
else falls back to the static argmax — `pair-search` with gates that never fire
plays identically to `pair`.

**Calibration diagnostic** (`examples/search_calibration.rs`): play N games with
search forced ON at every active decision, logging per decision: top-2 z-gap,
whether each gate would have fired, static choice vs search choice, and sims-ended
fraction. Output: gap-distribution summary + per-gate hit rate + disagreement rate
(overall and per gate). We tune `GATE_MARGIN` from this before the big benchmarks
— the goal is gates that capture most disagreements while searching a minority of
decisions.

## CLI & benchmarking

- `BotType::PairSearch` (`bench pair-search ga` / `pair` / `dqn`), loads
  `pair_model/` like `pair`. `StrategyTemplates` shares the loaded model between
  them.
- Expected cost: ~1 ms per searched decision (post-batching); benches at 20–50k
  games are the workhorse. The paired head-to-head vs `pair` is unusually
  efficient: the two bots differ only at searched decisions, so most game pairs
  tie and the paired CI shrinks fast.

## Acceptance

1. `bench pair-search pair -n 50000` (seed 42): pair-search > 50% with the paired
   99% CI excluding 50%, confirmed on a second seed.
2. `bench pair-search ga -n 50000`: > 59.2% (the pair bot's level) — ideally
   toward the hypothesized 60–62% ceiling.
3. Refactor regression gate: byte-identical pre/post benches for pair, dqn, ga.
4. Equivalence test + full existing suite green.

If search shows no gain at any gate setting, that is itself a decisive data point:
record it in EXPERIMENTS.md and stop the search line (the remaining lever would be
expert iteration, which only makes sense if search beats static play).

## Testing

Unit: Φ approximation vs known values; `State::permuted` property test vs
`permute_colors`; canonical-form dedup (symmetric twins collapse, asymmetric
don't); seed-hash determinism (same context → same streams, candidate-independent);
gating predicate truth table; segmented `evaluate_batch_multi` override equals the
default loop's results (same model, same groups).

Integration: simulator-vs-`Game::play` equivalence over seeded games (2p/3p/4p);
`bench pair-search pair -n 200` smoke (runs, deterministic across two invocations);
the three byte-identical regression benches.

## Out of scope (v2+ candidates)

- Search on passive decisions
- Lockstep batching across *decisions* (process-wide); GPU backends
- `WinProb` for `DqnStrategy`/GA
- Expert iteration (train on search-improved play) — contingent on this working
- Deeper horizons / adaptive K
