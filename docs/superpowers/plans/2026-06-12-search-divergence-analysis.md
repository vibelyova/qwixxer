# Search-Divergence Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pipeline that logs every decision where pair-search disagrees with the static pair bot (on static-bot trajectories), relabels disagreements at K=2048 rollouts, and mines the labeled events in Python for humanly-encodable meta-rules.

**Architecture:** Static-pair-vs-GA games with search as a shadow oracle; one JSONL event per eligible decision. The pipeline lives in `examples/divergence.rs` (run + relabel modes); `src/` gets only visibility tweaks, a `State::from_parts` constructor, and dev-dependencies. Python analysis lives in `analysis/`.

**Tech Stack:** Rust (existing crate APIs: `SearchBot`, `bot_impl`, `SimGame`, `BatchedRollouts`), clap, serde/serde_json (dev-deps), rayon; Python venv with pandas/scikit-learn/matplotlib.

**Spec:** `docs/superpowers/specs/2026-06-12-search-divergence-analysis-design.md`

**Key crate facts** (verified):
- `Strategy` trait: `active_phase1(&mut self, &State, &[State], [u8; 6]) -> Option<Mark>`, `active_phase2(..., has_marked: bool)`, `passive_phase1(..., active_player: usize)`. Implementing `Bot` gives a blanket `Strategy` impl.
- `bot_impl` items are `pub(crate)`: `Decision::{Forced, Choices}`, `argmax`, `eval_decision`, `active_phase1_choices(bot, state, opps, dice) -> (Decision, Vec<State>)` (second element = simulated post-phase1 opponents), `active_phase2_choices(state, opps, dice, has_marked) -> Decision`.
- `search.rs` pub items: `SearchBot { bot, force, stats }`, `WinProb`, `context_seed`, `sample_player_seed`, consts `K_CANDIDATES=2`, `K_SAMPLES=128`, `GATE_MARGIN=0.15`, `HORIZON_ROUNDS=1`. Its `gates()`, `Candidate`, `splitmix64` are private — the example carries copies.
- `sim.rs`: `SimGame { pub states, pub active, pub rngs, pub over }`, `SimGame::{game_over, propagate_locks, opp_view, outcome, n}`, `BatchedRollouts::{new, all_over, step_turn}` with `pub sims`.
- `State` (all pub): `strikes` field, `row_totals() -> [u8;4]`, `row_free_values() -> [Option<u8>;4]`, `locked() -> [bool;4]`, `count_locked() -> u8`, `count_points() -> isize`, `apply_mark`, `apply_strike`, `would_end_game`, `would_lock_row`. Rows 0,1 ascend (terminal 12), rows 2,3 descend (terminal 2). `(total, free)` per row + strikes fully determines a state; locking adds +1 to `total` (lock bonus). `Mark { pub row: usize, pub number: u8 }`.
- `PairStrategy { pub model, pub device, pub net }`, `::load("pair_model")`, `::from_shared(model, device)`. GA: `bot::DNA::load_weights("champion.txt", Arc::new(bot::default_genes()))`, `DNA: Clone + Bot`.
- Bench seat/dice pattern (main.rs): games come in rotation pairs sharing per-seat dice streams via `seat_dice_seed(base, pair, seat) = splitmix64(base).wrapping_add(pair*8 + seat)`.

---

### Task 1: src enablers — `State::from_parts`, bot_impl visibility, dev-deps

**Files:**
- Modify: `src/state.rs` (next to `from_marks`, ~line 152)
- Modify: `src/strategy/mod.rs:1`
- Modify: `src/strategy/bot_impl.rs` (visibility on 5 items)
- Modify: `Cargo.toml` (`[dev-dependencies]`)

- [ ] **Step 1: Write the failing test** — append to the `#[cfg(test)] mod tests` at the bottom of `src/state.rs` (create the module if absent; check first — `grep -n "mod tests" src/state.rs`):

```rust
#[test]
fn from_parts_roundtrip() {
    // Mid-game state with marks in two rows and a strike.
    let mut mid = State::default();
    for n in [2u8, 3, 5, 7, 10] {
        mid.apply_mark(Mark { row: 0, number: n });
    }
    for n in [12u8, 9] {
        mid.apply_mark(Mark { row: 3, number: n });
    }
    mid.apply_strike();
    // State with a locked row (5 marks then the terminal).
    let mut locked = State::default();
    for n in [2u8, 3, 4, 5, 6, 12] {
        locked.apply_mark(Mark { row: 1, number: n });
    }
    for orig in [State::default(), mid, locked] {
        let totals = orig.row_totals();
        let frees = orig.row_free_values();
        let rebuilt = State::from_parts(orig.strikes, core::array::from_fn(|i| (totals[i], frees[i])));
        assert_eq!(rebuilt.row_totals(), orig.row_totals());
        assert_eq!(rebuilt.row_free_values(), orig.row_free_values());
        assert_eq!(rebuilt.locked(), orig.locked());
        assert_eq!(rebuilt.strikes, orig.strikes);
        assert_eq!(rebuilt.count_points(), orig.count_points());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test from_parts_roundtrip 2>&1 | tail -5`
Expected: compile error — `no function or associated item named `from_parts``

- [ ] **Step 3: Implement `from_parts`** in `src/state.rs`, right after `from_marks`:

```rust
    /// Reconstruct a State from its observable parts: per-row `(total, free)`
    /// — `free == None` means locked, `total` includes the lock bonus — plus
    /// strikes. Rows 0,1 are ascending, 2,3 descending, as in `default`.
    /// Analysis/test helper: `(total, free)` fully determines row behavior.
    pub fn from_parts(strikes: u8, parts: [(u8, Option<u8>); 4]) -> State {
        let mut s = State::default();
        s.strikes = strikes;
        for (i, &(total, free)) in parts.iter().enumerate() {
            s.rows[i].total = total;
            s.rows[i].free = free;
        }
        s
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test from_parts_roundtrip 2>&1 | tail -3`
Expected: `test result: ok. 1 passed`

- [ ] **Step 5: Visibility changes.** In `src/strategy/mod.rs` line 1: `mod bot_impl;` → `pub mod bot_impl;`. In `src/strategy/bot_impl.rs`, change `pub(crate)` → `pub` on exactly these five items (leave `mark_choices`, `phase1_plan_choices`, `passive_phase1_choices`, `*_impl` as `pub(crate)`):
  - `pub enum Decision`
  - `pub fn argmax`
  - `pub fn eval_decision`
  - `pub fn active_phase2_choices`
  - `pub fn active_phase1_choices`

  Note: `Decision`'s doc comment mentions it's the pure-logic half of the pipeline; append one line: `/// Pub for analysis examples (see examples/divergence.rs).`

- [ ] **Step 6: Dev-dependencies.** In `Cargo.toml`, extend `[dev-dependencies]`:

```toml
[dev-dependencies]
matrixmultiply = "0.3"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

- [ ] **Step 7: Full build + tests**

Run: `cargo build --release 2>&1 | tail -3 && cargo test --release 2>&1 | grep "test result"`
Expected: clean build, all suites `ok` (67 tests at time of writing)

- [ ] **Step 8: Commit**

```bash
git add src/state.rs src/strategy/mod.rs src/strategy/bot_impl.rs Cargo.toml Cargo.lock
git commit -m "feat: minimal enablers for divergence analysis example

State::from_parts, pub bot_impl decision plumbing, serde dev-deps."
```

---

### Task 2: example skeleton — CLI, JSONL schema, shared helpers

**Files:**
- Create: `examples/divergence.rs`

- [ ] **Step 1: Create `examples/divergence.rs`** with the CLI, event types, and helpers (run/relabel bodies stubbed with `todo!()` for now):

```rust
//! Search-divergence analysis pipeline (experiment tooling, deletable).
//!
//! `run`: static-pair vs GA games; search shadows every eligible decision
//! (force=true), one JSONL event each; play continues with the static move.
//! `relabel`: replays disagreements (+ an agreement sample) at high K with
//! per-sample paired stats and a verdict.
//!
//! Spec: docs/superpowers/specs/2026-06-12-search-divergence-analysis-design.md

use clap::{Parser, Subcommand};
use qwixxer::bot::{default_genes, DNA};
use qwixxer::dqn::pair::PairStrategy;
use qwixxer::game::{Game, Player};
use qwixxer::state::{Mark, State};
use qwixxer::strategy::bot_impl::{active_phase1_choices, active_phase2_choices, argmax, eval_decision, Decision};
use qwixxer::strategy::search::{
    context_seed, sample_player_seed, SearchBot, WinProb, GATE_MARGIN, HORIZON_ROUNDS, K_CANDIDATES, K_SAMPLES,
};
use qwixxer::strategy::sim::{BatchedRollouts, SimGame};
use qwixxer::strategy::{Bot, Strategy};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::io::Write;
use std::rc::Rc;
use std::sync::Arc;

#[derive(Parser)]
#[command(about = "Search-vs-static divergence data pipeline")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Play static-pair vs GA, shadow-search every eligible decision, write JSONL.
    Run {
        /// Number of games (rounded up to a rotation pair)
        #[arg(short)]
        n: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long)]
        out: String,
        #[arg(long)]
        force_overwrite: bool,
    },
    /// Replay disagreements (+ agreement sample) at high K; append verdicts.
    Relabel {
        #[arg(long)]
        input: String,
        #[arg(long)]
        out: String,
        #[arg(short, default_value_t = 2048)]
        k: usize,
        /// Fraction of agreement events to relabel as controls
        #[arg(long, default_value_t = 0.1)]
        agree_sample: f64,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        #[arg(long)]
        force_overwrite: bool,
    },
}

// ---- JSONL schema ----

#[derive(Serialize, Deserialize, Clone)]
struct StateJson {
    strikes: u8,
    /// Per-row (total, free); free == null means locked. Exactly
    /// `State::from_parts`'s input.
    rows: [(u8, Option<u8>); 4],
}

impl StateJson {
    fn of(s: &State) -> Self {
        let t = s.row_totals();
        let f = s.row_free_values();
        StateJson {
            strikes: s.strikes,
            rows: core::array::from_fn(|i| (t[i], f[i])),
        }
    }
    fn to_state(&self) -> State {
        State::from_parts(self.strikes, self.rows)
    }
}

#[derive(Serialize, Deserialize, Clone)]
struct CandJson {
    /// (row, number); null = skip (phase 1) / skip-or-strike (phase 2).
    mark: Option<(usize, u8)>,
    /// Static value (bot's evaluate units).
    v: f32,
}

/// One eligible decision ("t":"d").
#[derive(Serialize, Deserialize, Clone)]
struct DecisionEvent {
    t: String,
    game: usize,
    /// Our active-turn counter within the game (phase 1 and 2 share it).
    turn: u32,
    phase: u8,
    has_marked: Option<bool>,
    dice: [u8; 6],
    our: StateJson,
    opps: Vec<StateJson>,
    gate_close: bool,
    gate_endgame: bool,
    static_gap: f32,
    our_points: isize,
    opp_points: isize,
    /// All distinct choices, sorted by static value desc.
    cands: Vec<CandJson>,
    /// Index of the production static bot's move in `cands` (argmax
    /// semantics; usually 0, can differ on exact value ties).
    static_pick: usize,
    search_mark: Option<(usize, u8)>,
    search_pick: usize,
    seed: u64,
    disagree: bool,
}

/// One game summary ("t":"g").
#[derive(Serialize, Deserialize)]
struct GameEvent {
    t: String,
    game: usize,
    pair_seat: usize,
    scores: Vec<isize>,
    pair_won: bool,
}

fn mark_json(m: Option<Mark>) -> Option<(usize, u8)> {
    m.map(|m| (m.row, m.number))
}

// ---- Shared helpers (mirrors of search.rs internals; relabel's K=128
// pick-reproduction guard fails loudly if these drift from search.rs) ----

#[derive(Clone)]
struct Cand {
    mark: Option<Mark>,
    value: f32,
    post: State,
}

/// Collapse phase-1 plans to distinct phase-1 marks, each keeping its best
/// plan's value and end-state; sorted by value desc. Mirrors
/// `SearchBot::active_phase1`.
fn collapse_plans(plans: &[(Option<Mark>, State)], values: &[f32]) -> Vec<Cand> {
    let mut cands: Vec<Cand> = Vec::new();
    for (i, (m, s)) in plans.iter().enumerate() {
        match cands.iter_mut().find(|c| c.mark == *m) {
            Some(c) if values[i] > c.value => {
                c.value = values[i];
                c.post = *s;
            }
            Some(_) => {}
            None => cands.push(Cand {
                mark: *m,
                value: values[i],
                post: *s,
            }),
        }
    }
    cands.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap());
    cands
}

/// Gate predicate, recomputed as *features*. Mirrors `search.rs::gates`.
fn gate_flags(cands: &[Cand], our: &State, opps: &[State]) -> (bool, bool) {
    let close = cands[0].value - cands[1].value < GATE_MARGIN;
    let endgame = our.count_locked() >= 1
        || opps.iter().any(|s| s.count_locked() >= 1)
        || our.strikes >= 3
        || opps.iter().any(|s| s.strikes >= 3)
        || cands
            .iter()
            .take(K_CANDIDATES)
            .any(|c| c.post.count_locked() > our.count_locked());
    (close, endgame)
}

/// Mirrors main.rs (private there).
fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn seat_dice_seed(base: u64, pair: usize, seat: usize) -> u64 {
    splitmix64(base).wrapping_add(pair as u64 * 8 + seat as u64)
}

fn refuse_overwrite(path: &str, force: bool) {
    if std::path::Path::new(path).exists() && !force {
        eprintln!("{path} exists; pass --force-overwrite to replace it");
        std::process::exit(1);
    }
}

fn main() {
    match Cli::parse().cmd {
        Cmd::Run {
            n,
            seed,
            out,
            force_overwrite,
        } => {
            refuse_overwrite(&out, force_overwrite);
            todo!("run mode (Task 3)");
        }
        Cmd::Relabel {
            input,
            out,
            k,
            agree_sample,
            seed,
            force_overwrite,
        } => {
            refuse_overwrite(&out, force_overwrite);
            todo!("relabel mode (Task 4)");
        }
    }
}
```

- [ ] **Step 2: Verify it compiles** (unused-import warnings are expected at this stage; they disappear in Tasks 3–4)

Run: `cargo build --release --example divergence 2>&1 | grep -E "^error" | head; echo "exit: $?"`
Expected: no `error` lines

- [ ] **Step 3: Commit**

```bash
git add examples/divergence.rs
git commit -m "feat(divergence): example skeleton — CLI, JSONL schema, search.rs mirror helpers"
```

---

### Task 3: ShadowPair + run mode

**Files:**
- Modify: `examples/divergence.rs`
- Modify: `.gitignore`

- [ ] **Step 1: Add `ShadowPair`** (above `main`):

```rust
// ---- Shadow strategy: plays static, logs what search would have done ----

struct ShadowPair {
    static_bot: PairStrategy,
    search: SearchBot<PairStrategy>,
    events: Rc<RefCell<Vec<DecisionEvent>>>,
    turn: u32,
}

impl std::fmt::Debug for ShadowPair {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "ShadowPair")
    }
}

impl ShadowPair {
    fn new(template: &PairStrategy, events: Rc<RefCell<Vec<DecisionEvent>>>) -> Self {
        let mut search = SearchBot::new(PairStrategy::from_shared(template.model.clone(), template.device));
        search.force = true;
        ShadowPair {
            static_bot: PairStrategy::from_shared(template.model.clone(), template.device),
            search,
            events,
            turn: 0,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn log(
        &self,
        phase: u8,
        has_marked: Option<bool>,
        state: &State,
        opps: &[State],
        dice: [u8; 6],
        cands: &[Cand],
        static_mark: Option<Mark>,
        search_mark: Option<Mark>,
    ) {
        let find = |m: Option<Mark>, what: &str| {
            cands
                .iter()
                .position(|c| c.mark == m)
                .unwrap_or_else(|| panic!("{what} move {m:?} not among candidates — shadow/SearchBot drift"))
        };
        let static_pick = find(static_mark, "static");
        let search_pick = find(search_mark, "search");
        let (close, endgame) = gate_flags(cands, state, opps);
        self.events.borrow_mut().push(DecisionEvent {
            t: "d".into(),
            game: 0, // stamped by the driver after the game
            turn: self.turn,
            phase,
            has_marked,
            dice,
            our: StateJson::of(state),
            opps: opps.iter().map(StateJson::of).collect(),
            gate_close: close,
            gate_endgame: endgame,
            static_gap: cands[0].value - cands[1].value,
            our_points: state.count_points(),
            opp_points: opps.iter().map(|s| s.count_points()).max().unwrap(),
            cands: cands
                .iter()
                .map(|c| CandJson {
                    mark: mark_json(c.mark),
                    v: c.value,
                })
                .collect(),
            static_pick,
            search_mark: mark_json(search_mark),
            search_pick,
            seed: context_seed(state, opps, dice),
            disagree: search_pick != static_pick,
        });
    }
}

impl Strategy for ShadowPair {
    fn active_phase1(&mut self, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
        self.turn += 1;
        let (decision, sim_opp) = active_phase1_choices(&self.static_bot, state, opp_states, dice);
        let plans = match decision {
            Decision::Forced(m) => return m,
            Decision::Choices(c) => c,
        };
        let states: Vec<State> = plans.iter().map(|(_, s)| *s).collect();
        let values = self.static_bot.evaluate_batch(&states, &sim_opp);
        // Production-identical static play: eval_decision == argmax over plans.
        let static_mark = plans[argmax(&values)].0;
        let cands = collapse_plans(&plans, &values);
        if cands.len() < 2 || opp_states.is_empty() {
            return static_mark;
        }
        let search_mark = self.search.active_phase1(state, opp_states, dice);
        self.log(1, None, state, opp_states, dice, &cands, static_mark, search_mark);
        static_mark
    }

    fn active_phase2(&mut self, state: &State, opp_states: &[State], dice: [u8; 6], has_marked: bool) -> Option<Mark> {
        let choices = match active_phase2_choices(state, opp_states, dice, has_marked) {
            Decision::Forced(m) => return m,
            Decision::Choices(c) => c,
        };
        let states: Vec<State> = choices.iter().map(|(_, s)| *s).collect();
        let values = self.static_bot.evaluate_batch(&states, opp_states);
        let static_mark = choices[argmax(&values)].0;
        let mut cands: Vec<Cand> = choices
            .iter()
            .zip(&values)
            .map(|((m, s), &v)| Cand {
                mark: *m,
                value: v,
                post: *s,
            })
            .collect();
        cands.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap());
        if cands.len() < 2 || opp_states.is_empty() {
            return static_mark;
        }
        let search_mark = self.search.active_phase2(state, opp_states, dice, has_marked);
        self.log(2, Some(has_marked), state, opp_states, dice, &cands, static_mark, search_mark);
        static_mark
    }

    fn passive_phase1(
        &mut self,
        state: &State,
        opp_states: &[State],
        dice: [u8; 6],
        active_player: usize,
    ) -> Option<Mark> {
        // Search never applies passively; static pass-through.
        self.static_bot.passive_phase1(state, opp_states, dice, active_player)
    }
}
```

- [ ] **Step 2: Add the run driver** (above `main`; replace the `todo!("run mode (Task 3)")` arm with `cmd_run(n, seed, &out)`):

```rust
// ---- run mode ----

fn play_one(pair_template: &PairStrategy, champion: &DNA, game_idx: usize, base_seed: u64) -> (GameEvent, Vec<DecisionEvent>) {
    let pairing = game_idx / 2;
    let rotation = game_idx % 2;
    // Mirrors run_bench: seat j hosts bot (j + 2 - rotation) % 2, bots = [GA, PAIR].
    let pair_seat = (1 + rotation) % 2;
    let events = Rc::new(RefCell::new(Vec::new()));
    let players: Vec<Player> = (0..2)
        .map(|j| {
            let dice = Box::new(SmallRng::seed_from_u64(seat_dice_seed(base_seed, pairing, j)));
            let strategy: Box<dyn Strategy> = if j == pair_seat {
                Box::new(ShadowPair::new(pair_template, events.clone()))
            } else {
                Box::new(champion.clone())
            };
            Player::new(strategy, dice)
        })
        .collect();
    let mut game = Game::new(players);
    game.play();
    let scores: Vec<isize> = game.players.iter().map(|p| p.state.count_points()).collect();
    drop(game); // release the ShadowPair's Rc clone
    let max = *scores.iter().max().unwrap();
    let unique_winner = scores.iter().filter(|&&s| s == max).count() == 1;
    let mut evs = Rc::try_unwrap(events).expect("events Rc still shared").into_inner();
    for e in &mut evs {
        e.game = game_idx;
    }
    let pair_won = scores[pair_seat] == max && unique_winner;
    (
        GameEvent {
            t: "g".into(),
            game: game_idx,
            pair_seat,
            scores,
            pair_won,
        },
        evs,
    )
}

fn cmd_run(n: usize, seed: u64, out: &str) {
    use rayon::prelude::*;
    let num_games = n.div_ceil(2) * 2;
    eprintln!("divergence run: {num_games} games, seed {seed} -> {out}");
    let results: Vec<(GameEvent, Vec<DecisionEvent>)> = (0..num_games)
        .into_par_iter()
        .map_init(
            || {
                (
                    PairStrategy::load("pair_model"),
                    DNA::load_weights("champion.txt", Arc::new(default_genes()))
                        .expect("champion.txt missing — run `train ga` first"),
                )
            },
            |(pair, champ), i| play_one(pair, champ, i, seed),
        )
        .collect();

    let mut f = std::io::BufWriter::new(std::fs::File::create(out).unwrap());
    for (g, evs) in &results {
        for e in evs {
            writeln!(f, "{}", serde_json::to_string(e).unwrap()).unwrap();
        }
        writeln!(f, "{}", serde_json::to_string(g).unwrap()).unwrap();
    }
    f.flush().unwrap();

    let all: Vec<&DecisionEvent> = results.iter().flat_map(|(_, e)| e).collect();
    let eligible = all.len();
    let gated = all.iter().filter(|e| e.gate_close || e.gate_endgame).count();
    let close = all.iter().filter(|e| e.gate_close).count();
    let dis = all.iter().filter(|e| e.disagree).count();
    let dis_gated = all.iter().filter(|e| e.disagree && (e.gate_close || e.gate_endgame)).count();
    let wins = results.iter().filter(|(g, _)| g.pair_won).count();
    println!("{num_games} games ({:.1}% pair wins), {eligible} eligible decisions", wins as f64 / num_games as f64 * 100.0);
    println!(
        "gates: close {:.1}%, any {:.1}% of eligible",
        close as f64 / eligible as f64 * 100.0,
        gated as f64 / eligible as f64 * 100.0
    );
    println!(
        "disagreements: {dis} ({:.2}% of eligible, {:.2}% of gate-fired; {dis_gated} inside gates)",
        dis as f64 / eligible as f64 * 100.0,
        dis as f64 / gated as f64 * 100.0
    );
}
```

- [ ] **Step 3: Build and smoke-run**

Run: `cargo build --release --example divergence 2>&1 | grep -E "^(error|warning)" | head`
Expected: empty (no errors; unused-import warnings should now be gone except relabel-only imports — those go away in Task 4; ignore `sample_player_seed`/`BatchedRollouts`-class warnings for now)

Run: `./target/release/examples/divergence run -n 50 --seed 0 --out /tmp/div50.jsonl && head -c 400 /tmp/div50.jsonl && echo && wc -l /tmp/div50.jsonl`
Expected: summary block printing; first line is a `"t":"d"` JSON object with all schema fields; line count ≈ 50 games × ~17 (16 decisions + 1 game line)

- [ ] **Step 4: Determinism check**

Run: `./target/release/examples/divergence run -n 50 --seed 0 --out /tmp/div50b.jsonl && cmp /tmp/div50.jsonl /tmp/div50b.jsonl && echo IDENTICAL`
Expected: `IDENTICAL`

- [ ] **Step 5: Overwrite refusal check**

Run: `./target/release/examples/divergence run -n 2 --seed 0 --out /tmp/div50.jsonl; echo "exit $?"`
Expected: `exit 1` with the "pass --force-overwrite" message

- [ ] **Step 6: gitignore the data files and venv**

Append to `.gitignore`:

```
*.jsonl
analysis/.venv/
```

- [ ] **Step 7: Commit**

```bash
git add examples/divergence.rs .gitignore
git commit -m "feat(divergence): run mode — shadowed static-vs-GA games to JSONL"
```

---

### Task 4: relabel mode

**Files:**
- Modify: `examples/divergence.rs`

- [ ] **Step 1: Add rollout scoring + entry construction** (mirrors of `search.rs::search_pick` and its entry closures, parameterized by K and keeping per-sample values):

```rust
// ---- relabel mode ----

/// Deterministic completion of our phase-1 turn per shortlisted candidate.
/// Mirrors the entries closure in SearchBot::active_phase1.
fn phase1_entries(
    bot: &PairStrategy,
    state: &State,
    sim_opp: &[State],
    dice: [u8; 6],
    shortlist: &[Cand],
) -> Vec<(Vec<State>, bool)> {
    shortlist
        .iter()
        .map(|c| {
            let mut our = *state;
            if let Some(m) = c.mark {
                our.apply_mark(m);
            }
            let mut all: Vec<State> = std::iter::once(our).chain(sim_opp.iter().copied()).collect();
            SimGame::propagate_locks(&mut all);
            if SimGame::game_over(&all) {
                return (all, true);
            }
            let view = SimGame::opp_view(&all, 0);
            let d = active_phase2_choices(&all[0], &view, dice, c.mark.is_some());
            match eval_decision(bot, d, &view) {
                Some(m) => all[0].apply_mark(m),
                None if c.mark.is_none() => all[0].apply_strike(),
                None => {}
            }
            SimGame::propagate_locks(&mut all);
            let ended = SimGame::game_over(&all);
            (all, ended)
        })
        .collect()
}

/// Mirrors the entries closure in SearchBot::active_phase2.
fn phase2_entries(opps: &[State], shortlist: &[Cand]) -> Vec<(Vec<State>, bool)> {
    shortlist
        .iter()
        .map(|c| {
            let mut all: Vec<State> = std::iter::once(c.post).chain(opps.iter().copied()).collect();
            SimGame::propagate_locks(&mut all);
            let ended = SimGame::game_over(&all);
            (all, ended)
        })
        .collect()
}

/// Per-entry rollout scores, one per sample (ended entries: one deterministic
/// outcome). Mirrors SearchBot::search_pick but keeps per-sample values and
/// takes K as a parameter. CRN streams extend production's: samples
/// 0..K_SAMPLES are bit-identical to what search saw at collection time.
fn rollout_scores(bot: &PairStrategy, entries: &[(Vec<State>, bool)], seed: u64, k: usize) -> Vec<Vec<f32>> {
    let n = entries[0].0.len();
    let mut sims: Vec<SimGame> = Vec::new();
    let mut sim_owner: Vec<usize> = Vec::new();
    for (ei, (states, ended)) in entries.iter().enumerate() {
        if *ended {
            continue;
        }
        for s in 0..k {
            sims.push(SimGame {
                states: states.clone(),
                active: 1 % n,
                rngs: (0..n)
                    .map(|p| SmallRng::seed_from_u64(sample_player_seed(seed, s, p)))
                    .collect(),
                over: false,
            });
            sim_owner.push(ei);
        }
    }
    let mut driver = BatchedRollouts::new(bot, sims);
    for _ in 0..(HORIZON_ROUNDS * n) {
        if driver.all_over() {
            break;
        }
        driver.step_turn();
    }
    let survivor_idx: Vec<usize> = (0..driver.sims.len()).filter(|&i| !driver.sims[i].over).collect();
    let survivor_views: Vec<Vec<State>> = survivor_idx
        .iter()
        .map(|&i| SimGame::opp_view(&driver.sims[i].states, 0))
        .collect();
    let groups: Vec<(&State, &[State])> = survivor_idx
        .iter()
        .zip(&survivor_views)
        .map(|(&i, v)| (&driver.sims[i].states[0], v.as_slice()))
        .collect();
    let probs = bot.win_prob_multi(&groups);
    let mut prob_iter = probs.into_iter();
    let mut out: Vec<Vec<f32>> = entries
        .iter()
        .map(|(states, ended)| {
            if *ended {
                vec![SimGame::outcome(states)]
            } else {
                Vec::with_capacity(k)
            }
        })
        .collect();
    for (i, sim) in driver.sims.iter().enumerate() {
        let p = if sim.over {
            SimGame::outcome(&sim.states)
        } else {
            prob_iter.next().unwrap()
        };
        out[sim_owner[i]].push(p);
    }
    out
}

/// Production pick semantics: highest mean wins, ties keep the lower index.
/// `limit` truncates each entry's samples (sum order matches production, so
/// limit = K_SAMPLES reproduces the collection-time pick bit-for-bit).
fn pick_by_mean(per_sample: &[Vec<f32>], limit: usize) -> usize {
    let mean = |v: &Vec<f32>| {
        let m = v.len().min(limit);
        v[..m].iter().sum::<f32>() / m as f32
    };
    let mut best = 0;
    let mut best_score = mean(&per_sample[0]);
    for (ei, v) in per_sample.iter().enumerate().skip(1) {
        let s = mean(v);
        if s > best_score {
            best = ei;
            best_score = s;
        }
    }
    best
}

/// Paired (mean, SE) of b − a; a length-1 side broadcasts (ended entry's
/// deterministic outcome). n == 1 → SE 0.
fn paired_stats(a: &[f32], b: &[f32]) -> (f32, f32) {
    let n = a.len().max(b.len());
    let get = |v: &[f32], i: usize| (if v.len() == 1 { v[0] } else { v[i] }) as f64;
    let diffs: Vec<f64> = (0..n).map(|i| get(b, i) - get(a, i)).collect();
    let mean = diffs.iter().sum::<f64>() / n as f64;
    if n == 1 {
        return (mean as f32, 0.0);
    }
    let var = diffs.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    (mean as f32, (var / n as f64).sqrt() as f32)
}
```

- [ ] **Step 2: Add the per-event relabel + the driver** (replace the `todo!("relabel mode (Task 4)")` arm with `cmd_relabel(&input, &out, k, agree_sample, seed)`):

```rust
struct HighK {
    scores: (f32, f32),
    gap_mean: f32,
    gap_se: f32,
    verdict: &'static str,
}

/// Rebuild the logged decision, run high-K rollouts, verdict on cand1 vs
/// cand0. Hard-fails on any drift from the collection run.
fn relabel_event(ev: &DecisionEvent, bot: &PairStrategy, k: usize, lineno: usize) -> HighK {
    let our = ev.our.to_state();
    let opps: Vec<State> = ev.opps.iter().map(|o| o.to_state()).collect();
    let seed = context_seed(&our, &opps, ev.dice);
    assert_eq!(seed, ev.seed, "line {lineno}: context seed mismatch — schema/serialization drift");

    let (cands, entries) = match ev.phase {
        1 => {
            let (decision, sim_opp) = active_phase1_choices(bot, &our, &opps, ev.dice);
            let plans = match decision {
                Decision::Choices(c) => c,
                Decision::Forced(_) => panic!("line {lineno}: logged decision is now meta-forced — drift"),
            };
            let states: Vec<State> = plans.iter().map(|(_, s)| *s).collect();
            let values = bot.evaluate_batch(&states, &sim_opp);
            let cands = collapse_plans(&plans, &values);
            let entries = phase1_entries(bot, &our, &sim_opp, ev.dice, &cands[..cands.len().min(K_CANDIDATES)]);
            (cands, entries)
        }
        2 => {
            let choices = match active_phase2_choices(&our, &opps, ev.dice, ev.has_marked.unwrap()) {
                Decision::Choices(c) => c,
                Decision::Forced(_) => panic!("line {lineno}: logged decision is now meta-forced — drift"),
            };
            let states: Vec<State> = choices.iter().map(|(_, s)| *s).collect();
            let values = bot.evaluate_batch(&states, &opps);
            let mut cands: Vec<Cand> = choices
                .iter()
                .zip(&values)
                .map(|((m, s), &v)| Cand {
                    mark: *m,
                    value: v,
                    post: *s,
                })
                .collect();
            cands.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap());
            let entries = phase2_entries(&opps, &cands[..cands.len().min(K_CANDIDATES)]);
            (cands, entries)
        }
        p => panic!("line {lineno}: bad phase {p}"),
    };

    // Guard: rebuilt candidates must match the log.
    assert_eq!(cands.len(), ev.cands.len(), "line {lineno}: candidate count drift");
    for (c, l) in cands.iter().zip(&ev.cands) {
        assert_eq!(mark_json(c.mark), l.mark, "line {lineno}: candidate order drift");
        assert!((c.value - l.v).abs() < 1e-4, "line {lineno}: static value drift");
    }

    let per_sample = rollout_scores(bot, &entries, seed, k);

    // Guard: first K_SAMPLES samples must reproduce the logged search pick.
    let pick128 = pick_by_mean(&per_sample, K_SAMPLES);
    assert_eq!(
        mark_json(cands[pick128].mark),
        ev.search_mark,
        "line {lineno}: K=128 pick not reproduced — search.rs/example drift"
    );

    let full = |v: &Vec<f32>| v.iter().sum::<f32>() / v.len() as f32;
    let (gap, se) = paired_stats(&per_sample[0], &per_sample[1]);
    let z = if se > 0.0 {
        gap / se
    } else {
        match gap.partial_cmp(&0.0).unwrap() {
            std::cmp::Ordering::Greater => f32::INFINITY,
            std::cmp::Ordering::Less => f32::NEG_INFINITY,
            std::cmp::Ordering::Equal => 0.0,
        }
    };
    // flip: cands[1] confidently better than cands[0]; keep: the reverse.
    // (search_right in Python: flip & search_pick==1, or keep & search_pick==0.)
    let verdict = if z > 2.0 {
        "flip"
    } else if z < -2.0 {
        "keep"
    } else {
        "coinflip"
    };
    HighK {
        scores: (full(&per_sample[0]), full(&per_sample[1])),
        gap_mean: gap,
        gap_se: se,
        verdict,
    }
}

fn cmd_relabel(input: &str, out: &str, k: usize, agree_sample: f64, seed: u64) {
    use rayon::prelude::*;
    assert!(k >= K_SAMPLES, "-k must be >= {K_SAMPLES} (pick-reproduction guard needs the first {K_SAMPLES} samples)");
    let text = std::fs::read_to_string(input).expect("cannot read input");
    let lines: Vec<&str> = text.lines().collect();

    // Sequential selection pass (deterministic given the file + seed).
    let mut rng = SmallRng::seed_from_u64(seed);
    let parsed: Vec<(usize, Option<DecisionEvent>, bool)> = lines
        .iter()
        .enumerate()
        .map(|(i, line)| {
            let v: serde_json::Value =
                serde_json::from_str(line).unwrap_or_else(|e| panic!("line {}: bad JSON: {e}", i + 1));
            if v["t"] != "d" {
                return (i, None, false);
            }
            let ev: DecisionEvent =
                serde_json::from_value(v).unwrap_or_else(|e| panic!("line {}: bad event: {e}", i + 1));
            let selected = ev.disagree || rng.gen_bool(agree_sample);
            (i, Some(ev), selected)
        })
        .collect();

    let todo: Vec<(usize, &DecisionEvent)> = parsed
        .iter()
        .filter_map(|(i, ev, sel)| ev.as_ref().filter(|_| *sel).map(|e| (*i, e)))
        .collect();
    eprintln!("relabeling {} of {} events at K={k}", todo.len(), parsed.iter().filter(|(_, e, _)| e.is_some()).count());

    let results: Vec<(usize, HighK)> = todo
        .par_iter()
        .map_init(
            || PairStrategy::load("pair_model"),
            |bot, (i, ev)| (*i, relabel_event(ev, bot, k, *i + 1)),
        )
        .collect();
    let by_line: std::collections::HashMap<usize, HighK> = results.into_iter().collect();

    let mut f = std::io::BufWriter::new(std::fs::File::create(out).unwrap());
    let mut counts = std::collections::HashMap::new();
    for (i, line) in lines.iter().enumerate() {
        match by_line.get(&i) {
            None => writeln!(f, "{line}").unwrap(),
            Some(hk) => {
                let mut v: serde_json::Value = serde_json::from_str(line).unwrap();
                v["hk_k"] = k.into();
                v["hk_scores"] = serde_json::json!([hk.scores.0, hk.scores.1]);
                v["hk_gap_mean"] = hk.gap_mean.into();
                v["hk_gap_se"] = hk.gap_se.into();
                v["verdict"] = hk.verdict.into();
                writeln!(f, "{}", serde_json::to_string(&v).unwrap()).unwrap();
                *counts.entry(hk.verdict).or_insert(0u32) += 1;
            }
        }
    }
    f.flush().unwrap();
    println!("verdicts: {counts:?}");
}
```

- [ ] **Step 3: Build, relabel the smoke file, check the guards hold**

Run: `cargo build --release --example divergence 2>&1 | grep -E "^(error|warning)" | head`
Expected: empty

Run: `./target/release/examples/divergence relabel --input /tmp/div50.jsonl --out /tmp/div50.relabeled.jsonl -k 256 --agree-sample 0.05 && wc -l /tmp/div50.jsonl /tmp/div50.relabeled.jsonl`
Expected: completes with **no panics** (this exercises both drift guards end-to-end on every selected event); `verdicts: {...}` printed; both files have the same line count

- [ ] **Step 4: Spot-check an augmented line**

Run: `grep '"verdict"' /tmp/div50.relabeled.jsonl | head -1`
Expected: a `"t":"d"` line carrying `hk_k`, `hk_scores`, `hk_gap_mean`, `hk_gap_se`, `verdict`

- [ ] **Step 5: Commit**

```bash
git add examples/divergence.rs
git commit -m "feat(divergence): relabel mode — high-K paired replay with drift guards"
```

---

### Task 5: smoke validation against Phase 13 rates

**Files:** none (validation checkpoint)

- [ ] **Step 1: 500-game run**

Run: `./target/release/examples/divergence run -n 500 --seed 7 --out /tmp/div500.jsonl`
Expected (compare with EXPERIMENTS.md Phase 13): close gate ≈ 45–60% of eligible; disagreement rate of *gate-fired* decisions ≈ 4–9%; pair win rate ≈ 55–65% vs GA. If any number is wildly off (e.g. disagreements > 20% or 0%), STOP and debug before collecting 10k games — most likely a shadow/SearchBot mismatch.

- [ ] **Step 2: Relabel at full K on the 500-game file** (also a perf rehearsal)

Run: `time ./target/release/examples/divergence relabel --input /tmp/div500.jsonl --out /tmp/div500.relabeled.jsonl -k 2048 --agree-sample 0.1`
Expected: no panics; note wall time — ~1.2k selected events should take roughly 1–2 min on 8 cores; extrapolate to the 10k-game run (~20× more events) and flag if the full pass would exceed ~45 min

- [ ] **Step 3: Sanity-check verdict split**

Run: `grep -o '"verdict":"[a-z]*"' /tmp/div500.relabeled.jsonl | sort | uniq -c`
Expected: all three verdicts present; among *disagreements* expect a substantial `coinflip` share (K=128 noise) and a meaningful `flip` share (search confirmed). Record the numbers in the task notes for the findings write-up.

---

### Task 6: Python environment + load.py

**Files:**
- Create: `analysis/requirements.txt`
- Create: `analysis/load.py`

- [ ] **Step 1: Create the venv and requirements**

```bash
mkdir -p analysis
printf 'pandas\nscikit-learn\nmatplotlib\n' > analysis/requirements.txt
python3 -m venv analysis/.venv
analysis/.venv/bin/pip install -q -U pip
analysis/.venv/bin/pip install -q -r analysis/requirements.txt
analysis/.venv/bin/python -c "import pandas, sklearn, matplotlib; print('ok')"
```

Expected: `ok`

- [ ] **Step 2: Write `analysis/load.py`**

```python
"""Load divergence JSONL into a tidy DataFrame with engineered features.

Usage: load.py <events.jsonl>   (prints a summary; import load() elsewhere)
"""
import json
import sys

import pandas as pd

ROW_ASC = (True, True, False, False)
TERMINAL = (12, 12, 2, 2)


def tri(t):
    return t * (t + 1) // 2


def points(state):
    return sum(tri(t) for t, _ in state["rows"]) - 5 * state["strikes"]


def move_feats(prefix, mark, state, phase, has_marked):
    """Features of one candidate move from the pre-move state."""
    if mark is None:
        kind = "skip" if phase == 1 or has_marked else "strike"
        return {
            f"{prefix}_kind": kind,
            f"{prefix}_row": -1,
            f"{prefix}_jump": -1,
            f"{prefix}_points": -5 if kind == "strike" else 0,
            f"{prefix}_locks": False,
            f"{prefix}_to_terminal": -1,
            f"{prefix}_row_total": -1,
        }
    row, number = mark
    total, free = state["rows"][row]
    asc = ROW_ASC[row]
    jump = (number - free) if asc else (free - number)  # numbers skipped over
    locks = number == TERMINAL[row]
    return {
        f"{prefix}_kind": "mark",
        f"{prefix}_row": row,
        f"{prefix}_jump": jump,
        f"{prefix}_points": (total + 2) if locks else (total + 1),
        f"{prefix}_locks": locks,
        f"{prefix}_to_terminal": (TERMINAL[row] - number) if asc else (number - TERMINAL[row]),
        f"{prefix}_row_total": total,
    }


def load(path):
    decisions, games = [], {}
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                sys.exit(f"{path}:{lineno}: bad JSON: {e}")
            if obj["t"] == "g":
                games[obj["game"]] = obj
            else:
                decisions.append(obj)
    rows = []
    for ev in decisions:
        our, opp = ev["our"], ev["opps"][0]
        static_mark = ev["cands"][ev["static_pick"]]["mark"]
        r = {
            "game": ev["game"],
            "turn": ev["turn"],
            "phase": ev["phase"],
            "has_marked": ev["has_marked"],
            "gate_close": ev["gate_close"],
            "gate_endgame": ev["gate_endgame"],
            "static_gap": ev["static_gap"],
            "n_cands": len(ev["cands"]),
            "disagree": ev["disagree"],
            "our_points": ev["our_points"],
            "opp_points": ev["opp_points"],
            "cdiff": ev["our_points"] - ev["opp_points"],
            "our_strikes": our["strikes"],
            "opp_strikes": opp["strikes"],
            "our_marks": sum(t for t, _ in our["rows"]),
            "opp_marks": sum(t for t, _ in opp["rows"]),
            "our_locked": sum(f is None for _, f in our["rows"]),
            "opp_locked": sum(f is None for _, f in opp["rows"]),
            "v_static": ev["cands"][ev["static_pick"]]["v"],
            "v_search": ev["cands"][ev["search_pick"]]["v"],
            "static_pick": ev["static_pick"],
            "search_pick": ev["search_pick"],
            "hk_gap_mean": ev.get("hk_gap_mean"),
            "hk_gap_se": ev.get("hk_gap_se"),
            "verdict": ev.get("verdict"),
        }
        r.update(move_feats("static", static_mark, our, ev["phase"], ev["has_marked"]))
        r.update(move_feats("search", ev["search_mark"], our, ev["phase"], ev["has_marked"]))
        g = games.get(ev["game"])
        r["game_won"] = g["pair_won"] if g else None
        rows.append(r)
    df = pd.DataFrame(rows)
    # search_right: relabeled disagreement where high-K confirms search's side.
    # verdict 'flip' favors cands[1], 'keep' favors cands[0].
    df["search_right"] = df["disagree"] & (
        ((df["verdict"] == "flip") & (df["search_pick"] == 1))
        | ((df["verdict"] == "keep") & (df["search_pick"] == 0))
    )
    return df


if __name__ == "__main__":
    df = load(sys.argv[1])
    print(f"{len(df)} events from {df.game.nunique()} games")
    print(f"disagree: {df.disagree.mean():.2%}")
    print("verdicts among disagreements:")
    print(df[df.disagree].verdict.value_counts(dropna=False))
    print(f"search_right: {df.search_right.sum()}")
```

- [ ] **Step 3: Smoke-ingest the relabeled 500-game file**

Run: `analysis/.venv/bin/python analysis/load.py /tmp/div500.relabeled.jsonl`
Expected: event count matches Rust's summary; disagree % matches; verdict counts match the Rust `verdicts:` line

- [ ] **Step 4: Commit**

```bash
git add analysis/requirements.txt analysis/load.py
git commit -m "feat(divergence): analysis env + JSONL loader with engineered features"
```

---

### Task 7: mine.py + examples.py

**Files:**
- Create: `analysis/mine.py`
- Create: `analysis/examples.py`

- [ ] **Step 1: Write `analysis/mine.py`**

```python
"""Cross-tabs and decision-tree mining over relabeled divergence events.

Usage: mine.py <relabeled.jsonl>
"""
import sys

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

from load import load

pd.set_option("display.width", 200)


def main(path):
    df = load(path)
    print(
        f"{len(df)} events, {df.disagree.sum()} disagreements ({df.disagree.mean():.2%}), "
        f"{df.search_right.sum()} confirmed search_right"
    )

    print("\n== Disagreement rate by single features ==")
    for col in [
        "phase", "gate_close", "gate_endgame", "our_strikes", "opp_strikes",
        "our_locked", "opp_locked", "static_kind",
    ]:
        print(df.groupby(col)["disagree"].agg(["mean", "count"]).to_string(), "\n")

    print("== Move-kind transition matrix: all disagreements ==")
    d = df[df.disagree]
    print(pd.crosstab(d.static_kind, d.search_kind, margins=True))
    print("\n== ...confirmed (search_right) only ==")
    c = df[df.search_right]
    print(pd.crosstab(c.static_kind, c.search_kind, margins=True))
    mm = c[(c.static_kind == "mark") & (c.search_kind == "mark")]
    print("\n== Row transition among confirmed mark->mark ==")
    print(pd.crosstab(mm.static_row, mm.search_row, margins=True))
    print("\n== Jump-size shift among confirmed mark->mark ==")
    print((mm.search_jump - mm.static_jump).describe())

    # Tree over every event that has a verdict (relabeled agreements act as
    # controls; refuted/coinflip disagreements as hard negatives).
    lab = df[df.verdict.notna()].copy()
    feats = [
        "phase", "turn", "static_gap", "cdiff", "our_strikes", "opp_strikes",
        "our_marks", "opp_marks", "our_locked", "opp_locked", "n_cands",
        "static_jump", "static_points", "static_row_total", "static_to_terminal",
        "search_jump", "search_points", "search_row_total", "search_to_terminal",
    ]
    X = pd.get_dummies(lab[feats + ["static_kind", "search_kind"]], columns=["static_kind", "search_kind"])
    y = lab["search_right"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    tree = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=0)
    tree.fit(Xtr, ytr)
    print(f"\n== Decision tree (held-out acc {tree.score(Xte, yte):.3f}, base rate {1 - yte.mean():.3f}) ==")
    print(export_text(tree, feature_names=list(X.columns)))
    imp = permutation_importance(tree, Xte, yte, n_repeats=10, random_state=0)
    print("== Permutation importance (top 10) ==")
    for v, name in sorted(zip(imp.importances_mean, X.columns), reverse=True)[:10]:
        print(f"  {name:<30} {v:.4f}")


if __name__ == "__main__":
    main(sys.argv[1])
```

- [ ] **Step 2: Write `analysis/examples.py`**

```python
"""Render the most confident disagreements as ASCII boards for eyeballing.

Usage: examples.py <relabeled.jsonl> [top_n]
"""
import json
import math
import sys

COLORS = ["R", "Y", "G", "B"]


def render_row(i, total, free):
    asc = i < 2
    nums = range(2, 13) if asc else range(12, 1, -1)
    if free is None:
        cells = " ".join("##" for _ in nums)
        return f"  {COLORS[i]} [{cells}]  marks={total} LOCKED"
    open_ = (lambda n: n >= free) if asc else (lambda n: n <= free)
    cells = " ".join(f"{n:>2}" if open_(n) else " ." for n in nums)
    return f"  {COLORS[i]} [{cells}]  marks={total} free={free}"


def render_state(label, s):
    print(f"{label}: strikes={s['strikes']}")
    for i, (t, f) in enumerate(s["rows"]):
        print(render_row(i, t, f))


def fmt_mark(m, phase, has_marked):
    if m is None:
        return "skip" if phase == 1 or has_marked else "STRIKE"
    return f"{COLORS[m[0]]}{m[1]}"


def z_of(e):
    if e["hk_gap_se"] > 0:
        return e["hk_gap_mean"] / e["hk_gap_se"]
    return math.copysign(math.inf, e["hk_gap_mean"]) if e["hk_gap_mean"] else 0.0


def main(path, top_n=15):
    evs = []
    with open(path) as fh:
        for line in fh:
            e = json.loads(line)
            if e["t"] == "d" and e.get("verdict") and e["disagree"]:
                evs.append(e)
    evs.sort(key=lambda e: -abs(z_of(e)))
    for e in evs[:top_n]:
        print("=" * 78)
        print(
            f"game {e['game']} turn {e['turn']} phase {e['phase']} dice {e['dice']} "
            f"| verdict {e['verdict']} z={z_of(e):+.1f} hk_gap={e['hk_gap_mean']:+.4f}"
        )
        render_state("OUR", e["our"])
        render_state("OPP", e["opps"][0])
        st = e["cands"][e["static_pick"]]
        print(f"  static: {fmt_mark(st['mark'], e['phase'], e['has_marked'])} (v={st['v']:+.3f})")
        print(f"  search: {fmt_mark(e['search_mark'], e['phase'], e['has_marked'])}")
        print(f"  cands: {[(fmt_mark(c['mark'], e['phase'], e['has_marked']), round(c['v'], 3)) for c in e['cands'][:5]]}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 15)
```

- [ ] **Step 3: Smoke both on the 500-game file**

Run: `analysis/.venv/bin/python analysis/mine.py /tmp/div500.relabeled.jsonl 2>&1 | head -40`
Expected: tables print; tree fits without errors (small-sample results are noise at this size — that's fine, this validates plumbing only)

Run: `analysis/.venv/bin/python analysis/examples.py /tmp/div500.relabeled.jsonl 3`
Expected: three rendered positions with boards, moves, and z-scores

- [ ] **Step 4: Commit**

```bash
git add analysis/mine.py analysis/examples.py
git commit -m "feat(divergence): mining + example-rendering analysis scripts"
```

---

### Task 8: full collection, relabel, analysis, findings

**Files:**
- Create: `divergence-10k.jsonl`, `divergence-10k.relabeled.jsonl` (data, git-ignored)
- Modify: `docs/EXPERIMENTS.md` (new phase section)

- [ ] **Step 1: Full collection run** (~15 min)

Run: `time ./target/release/examples/divergence run -n 10000 --seed 0 --out divergence-10k.jsonl`
Expected: ~160k eligible decisions, ~10–13k disagreements; rates in line with Task 5

- [ ] **Step 2: Full relabel** (~20–45 min)

Run: `time ./target/release/examples/divergence relabel --input divergence-10k.jsonl --out divergence-10k.relabeled.jsonl -k 2048 --agree-sample 0.1 --seed 1`
Expected: no drift-guard panics across the full set; verdict counts printed

- [ ] **Step 3: Run the analysis**

```bash
analysis/.venv/bin/python analysis/load.py divergence-10k.relabeled.jsonl
analysis/.venv/bin/python analysis/mine.py divergence-10k.relabeled.jsonl | tee /tmp/mine-output.txt
analysis/.venv/bin/python analysis/examples.py divergence-10k.relabeled.jsonl 25
```

- [ ] **Step 4: Interpret.** This step is exploratory by design. Work the outputs in this order: (1) what fraction of disagreements survive high-K (`flip`-confirmed)? If most are `coinflip`, search's edge is diffuse sampling advantage and a crisp meta-rule is unlikely — that itself is a finding. (2) Do the transition matrices show a dominant pattern (e.g. static marks / search skips, or systematic jump-size differences)? (3) Does the tree find a high-precision leaf with non-trivial coverage? Follow up ad hoc in the venv (the DataFrame has everything); add derived features to `load.py` as hypotheses emerge, and eyeball `examples.py` output for each candidate pattern.

- [ ] **Step 5: Write the findings** as a new phase section in `docs/EXPERIMENTS.md`, following the existing format (numbered phase, header, what was run, numbers, conclusions). Must include: collection/relabel configuration and rates, verdict split, the transition matrices' headline, tree/importance summary, 2–3 rendered example positions (abridged), and an explicit go/no-go on encoding a meta-rule (with the candidate rule's held-out coverage/precision if go).

- [ ] **Step 6: Commit**

```bash
git add docs/EXPERIMENTS.md
git commit -m "docs: search-divergence analysis findings"
```

**If a rule is found** (go decision in Step 5): that's a new mini-cycle — brainstorm/spec the meta-rule placement in `bot_impl`, validate per the spec's section 6 (Python held-out first, then 500k bench vs GA + head-to-head vs pair-search, plus a `divergence run` re-check). Not planned here.

---

## Self-review notes

- Spec coverage: spec §1 → Task 1; §2 (schema) → Task 2; §2 (shadow/run) → Task 3; §4 (relabel + guards) → Task 4; smoke/testing section → Tasks 1, 3, 4, 5; §5 (Python) → Tasks 6–7; §6 (validation loop) → Task 8 hand-off note. Error handling: overwrite refusal (Task 3 Step 5), drift guards (Task 4), JSONL line numbers in both Rust (`panic!` with lineno) and Python (`sys.exit` with lineno).
- The `verdict` naming differs from the spec's `search_right/static_right` (events where `static_pick != 0` on value ties would make those names ambiguous): the relabeler emits candidate-relative `flip/keep/coinflip`, and `load.py` derives the spec's `search_right` from verdict + `search_pick`. Documented in both places.
- Type consistency: `Cand`/`CandJson`/`DecisionEvent` field names match across Tasks 2–4; `load.py` reads exactly the serialized names; `seat_dice_seed`, `collapse_plans`, `gate_flags`, `rollout_scores`, `pick_by_mean`, `paired_stats` are each defined once and referenced by those names.
