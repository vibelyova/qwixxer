# Safe-Lock Adjudication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine when the "safe lock" meta-rule (`find_safe_lock`: force a non-game-ending lock whenever available) is wrong, by logging every rule firing on production trajectories and adjudicating the forced lock against alternatives with full-game CRN-paired rollouts.

**Architecture:** Two new modes (`lock-run`, `lock-adjudicate`) in the existing `examples/divergence.rs`, reusing its StateJson/seeding/sim machinery. Zero src/ changes (everything needed is already public). The one new mirror surface — the rule pipelines minus the lock force — is fenced by a collection-time guard: production's actual decision must equal what the mirror predicts the rule forces, on every event.

**Tech Stack:** Rust (existing example + crate APIs), Python (existing `analysis/` venv).

**Spec:** `docs/superpowers/specs/2026-06-12-safe-lock-adjudication-design.md`

**Key facts** (verified against current sources):
- `examples/divergence.rs` (767 lines) already has: `Cli`/`Cmd` (clap, lines ~31-67), `StateJson` (`of`/`to_state`), `CandJson { mark, v }`, `GameEvent`, `mark_json`, `Cand { mark, value, post }`, `collapse_plans`, `splitmix64`, `seat_dice_seed`, `refuse_overwrite`, `ShadowPair`, `play_one`, `cmd_run`, `phase1_entries(bot, state, sim_opp, dice, shortlist)`, `phase2_entries(opps, shortlist)`, `rollout_scores(bot, entries, seed, k)` (truncated+bootstrap — do NOT modify), `paired_stats(a, b) -> (mean, se)` (mean of b−a, len-1 broadcast), `relabel_event`, `cmd_relabel`, `main`.
- `src/strategy/bot_impl.rs` internals to mirror (all private or `pub(crate)`; pub items: `Decision`, `argmax`, `eval_decision`, `active_phase1_choices`, `active_phase2_choices`):
  - `prune_dominated` (lines 5-32), `opp_best_phase1_score` (34-52), `find_safe_lock` (55-70, max-points among safe locks).
  - `mark_choices` (108-168): **lock force FIRST**, then winning-end force, losing-end filter, empty/single collapses, `prune_dominated`, empty/single collapses again.
  - `phase1_plan_choices` (240-316): plan enumeration (strike, color singles, white singles, doubles), winning-end force, losing-end retain (strike re-push if empty), **then a lock scan that forces the FIRST safe-locking phase-1 mark in plan order** (NOT max-points), then `prune_dominated`, collapse to `(p1, end_state)`.
  - Passive phase 1 = `mark_choices(state, generate_white_moves(white_sum), *state, opp_best_phase1_score(opps, white_sum))`.
- `State: PartialOrd` is pub (domination); `would_lock_row`, `would_end_game`, `generate_white_moves`, `generate_color_moves`, `count_points`, `apply_mark`, `apply_strike` all pub. `Mark { pub row, pub number }`.
- `SimGame { pub states, active, rngs, over }`, `BatchedRollouts::{new, all_over, step_turn}`, `SimGame::{opp_view, propagate_locks, game_over, outcome}` all pub. `sample_player_seed`, `context_seed` pub.
- Games are 1v1 vs GA: `opps.len() == 1`, and in passive phase 1 the single opponent IS the active player.

---

### Task 1: rule-pipeline mirrors + shared candidate builder + schema + CLI

**Files:**
- Modify: `examples/divergence.rs`

- [ ] **Step 1: Add CLI variants.** In `enum Cmd`, after `Relabel { .. }`:

```rust
    /// Play static-pair vs GA (rule ON); log every safe-lock force as JSONL.
    LockRun {
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
    /// Adjudicate logged lock events: full-game CRN rollouts, lock vs alternatives.
    LockAdjudicate {
        #[arg(long)]
        input: String,
        #[arg(long)]
        out: String,
        #[arg(short, default_value_t = 2048)]
        k: usize,
        #[arg(long)]
        force_overwrite: bool,
    },
```

In `main`, add the two match arms (stubs for now):

```rust
        Cmd::LockRun {
            n,
            seed,
            out,
            force_overwrite,
        } => {
            refuse_overwrite(&out, force_overwrite);
            cmd_lock_run(n, seed, &out);
        }
        Cmd::LockAdjudicate {
            input,
            out,
            k,
            force_overwrite,
        } => {
            refuse_overwrite(&out, force_overwrite);
            cmd_lock_adjudicate(&input, &out, k);
        }
```

(Define placeholder fns so this task compiles: `fn cmd_lock_run(_n: usize, _seed: u64, _out: &str) { todo!("Task 2") }` and `fn cmd_lock_adjudicate(_input: &str, _out: &str, _k: usize) { todo!("Task 3") }` — replaced in Tasks 2-3.)

- [ ] **Step 2: Add the LockEvent schema** (near `DecisionEvent`):

```rust
/// One safe-lock rule firing ("t":"l").
#[derive(Serialize, Deserialize, Clone)]
struct LockEvent {
    t: String,
    game: usize,
    /// Our active-turn counter (pp1 events carry the count at the time of
    /// the opponent's roll — a stage proxy, not our turn).
    turn: u32,
    /// "ap1" | "ap2" | "pp1".
    ctx: String,
    has_marked: Option<bool>,
    dice: [u8; 6],
    our: StateJson,
    opps: Vec<StateJson>,
    our_points: isize,
    opp_points: isize,
    /// The mark production forces.
    lock_mark: (usize, u8),
    /// Rule-free candidates, sorted desc by static value.
    cands: Vec<CandJson>,
    lock_idx: usize,
    /// Best candidate that is not a safe lock (skip/strike count as non-lock
    /// — deferral is a legitimate alternative).
    alt_idx: usize,
    /// Best safe lock other than the forced one, if any.
    alt2_idx: Option<usize>,
    n_safe_locks: usize,
    /// True when the rule-free pipeline itself returned Forced (e.g. a
    /// winning game-end the production lock force preempted) — the event
    /// then has exactly two candidates: lock and that forced alternative.
    rule_free_forced: bool,
    seed: u64,
}
```

- [ ] **Step 3: Add the bot_impl mirrors** (new section after the existing `gate_flags`; copy semantics EXACTLY from `src/strategy/bot_impl.rs` — read it side by side):

```rust
// ---- Rule-pipeline mirrors (bot_impl.rs internals; the lock-run
// equivalence guard fails loudly if these drift from production) ----

/// Mirrors bot_impl::prune_dominated.
fn prune_dominated<T>(items: &mut Vec<T>, state_of: impl Fn(&T) -> &State) {
    let n = items.len();
    let mut dominated = vec![false; n];
    for i in 0..n {
        if dominated[i] {
            continue;
        }
        for j in (i + 1)..n {
            if dominated[j] {
                continue;
            }
            match state_of(&items[i]).partial_cmp(state_of(&items[j])) {
                Some(std::cmp::Ordering::Greater) => dominated[j] = true,
                Some(std::cmp::Ordering::Less) => {
                    dominated[i] = true;
                    break;
                }
                _ => {}
            }
        }
    }
    let mut idx = 0;
    items.retain(|_| {
        let keep = !dominated[idx];
        idx += 1;
        keep
    });
}

/// Mirrors bot_impl::opp_best_phase1_score.
fn opp_best_phase1_score(opp_states: &[State], white_sum: u8) -> isize {
    opp_states
        .iter()
        .map(|opp| {
            let base = opp.count_points();
            opp.generate_white_moves(white_sum)
                .iter()
                .map(|&m| {
                    let mut s = *opp;
                    s.apply_mark(m);
                    s.count_points()
                })
                .max()
                .unwrap_or(base)
                .max(base)
        })
        .max()
        .unwrap_or(0)
}

/// Mirrors bot_impl::find_safe_lock (max-points among non-ending locks).
fn find_safe_lock(state: &State, marks: &[Mark]) -> Option<Mark> {
    marks
        .iter()
        .copied()
        .filter(|&m| state.would_lock_row(m))
        .filter(|&m| {
            let mut s = *state;
            s.apply_mark(m);
            !s.would_end_game()
        })
        .max_by_key(|&m| {
            let mut s = *state;
            s.apply_mark(m);
            s.count_points()
        })
}

/// Mirrors bot_impl::mark_choices with the find_safe_lock force REMOVED
/// (production forces it before anything else). Everything downstream —
/// winning-end force, losing-end filter, collapses, domination pruning —
/// is byte-faithful.
fn mark_choices_nolock(state: &State, marks: &[Mark], baseline: State, opp_best: isize) -> Decision {
    if marks.is_empty() {
        return Decision::Forced(None);
    }
    let mark_states: Vec<State> = marks
        .iter()
        .map(|&m| {
            let mut s = *state;
            s.apply_mark(m);
            s
        })
        .collect();
    if let Some((mark, _)) = mark_states
        .iter()
        .enumerate()
        .map(|(i, &s)| (Some(marks[i]), s))
        .chain(std::iter::once((None, baseline)))
        .filter(|(_, post)| post.would_end_game() && post.count_points() > opp_best)
        .max_by_key(|(_, post)| post.count_points())
    {
        return Decision::Forced(mark);
    }
    let mut cands: Vec<(Option<Mark>, State)> = mark_states
        .iter()
        .enumerate()
        .map(|(i, &s)| (Some(marks[i]), s))
        .chain(std::iter::once((None, baseline)))
        .filter(|(_, s)| !(s.would_end_game() && s.count_points() < opp_best))
        .collect();
    if cands.is_empty() {
        return Decision::Forced(None);
    }
    if cands.len() == 1 {
        return Decision::Forced(cands[0].0);
    }
    prune_dominated(&mut cands, |(_, s)| s);
    if cands.is_empty() {
        return Decision::Forced(None);
    }
    if cands.len() == 1 {
        return Decision::Forced(cands[0].0);
    }
    Decision::Choices(cands)
}

/// Mirrors bot_impl::phase1_plan_choices, returning BOTH what production's
/// safe-lock scan would force and the rule-free decision. The scan runs at
/// the exact pipeline point production runs it (after the losing-end retain,
/// BEFORE pruning) and uses production's semantics: the FIRST safe-locking
/// phase-1 mark in plan order — not max-points like find_safe_lock.
fn phase1_plans_mirror(state: &State, comparison_opps: &[State], dice: [u8; 6]) -> (Option<Mark>, Decision) {
    let white_sum = dice[0] + dice[1];
    let opp_best = comparison_opps.iter().map(|s| s.count_points()).max().unwrap_or(0);
    let white_marks = state.generate_white_moves(white_sum);
    let color_marks = state.generate_color_moves(dice);

    let mut plans: Vec<(Option<Mark>, Option<Mark>, State)> = Vec::new();
    {
        let mut s = *state;
        s.apply_strike();
        plans.push((None, None, s));
    }
    for &cm in &color_marks {
        let mut s = *state;
        s.apply_mark(cm);
        plans.push((None, Some(cm), s));
    }
    for &wm in &white_marks {
        let mut s = *state;
        s.apply_mark(wm);
        plans.push((Some(wm), None, s));
    }
    for &wm in &white_marks {
        let mut post_white = *state;
        post_white.apply_mark(wm);
        for &cm in &post_white.generate_color_moves(dice) {
            let mut s = post_white;
            s.apply_mark(cm);
            plans.push((Some(wm), Some(cm), s));
        }
    }

    if plans.is_empty() {
        return (None, Decision::Forced(None));
    }
    let winning = plans
        .iter()
        .enumerate()
        .filter(|(_, (_, _, post))| post.would_end_game() && post.count_points() > opp_best)
        .max_by_key(|(_, (_, _, post))| post.count_points());
    if let Some((i, _)) = winning {
        return (None, Decision::Forced(plans[i].0));
    }
    plans.retain(|(_, _, post)| !(post.would_end_game() && post.count_points() < opp_best));
    if plans.is_empty() {
        let mut s = *state;
        s.apply_strike();
        plans.push((None, None, s));
    }

    // Production's lock scan happens HERE.
    let mut scan_lock = None;
    for (phase1, _, _) in &plans {
        if let Some(m) = phase1 {
            if state.would_lock_row(*m) && {
                let mut s = *state;
                s.apply_mark(*m);
                !s.would_end_game()
            } {
                scan_lock = Some(*m);
                break;
            }
        }
    }

    prune_dominated(&mut plans, |(_, _, s)| s);
    (
        scan_lock,
        Decision::Choices(plans.into_iter().map(|(p1, _, s)| (p1, s)).collect()),
    )
}
```

- [ ] **Step 4: Add the shared candidate builder** — used by BOTH collection and adjudication so the two passes cannot drift from each other:

```rust
/// Is this candidate's mark a safe lock from `state`?
fn is_safe_lock(state: &State, mark: Option<Mark>) -> bool {
    match mark {
        Some(m) if state.would_lock_row(m) => {
            let mut s = *state;
            s.apply_mark(m);
            !s.would_end_game()
        }
        _ => false,
    }
}

/// Built once per firing: rule-free candidates (sorted desc by static value)
/// plus the comparison indices. Returns None when there is no real decision
/// to adjudicate (fewer than 2 rule-free options, or the rule-free pipeline
/// forces the lock itself).
struct LockCands {
    cands: Vec<Cand>,
    lock_idx: usize,
    alt_idx: usize,
    alt2_idx: Option<usize>,
    n_safe_locks: usize,
    rule_free_forced: bool,
}

fn build_lock_cands(
    bot: &PairStrategy,
    state: &State,
    eval_opps: &[State],
    baseline: State,
    rule_free: Decision,
    lock: Mark,
    collapse: bool,
) -> Option<LockCands> {
    let (cands, rule_free_forced) = match rule_free {
        Decision::Choices(plans) => {
            let states: Vec<State> = plans.iter().map(|(_, s)| *s).collect();
            let values = bot.evaluate_batch(&states, eval_opps);
            let cands = if collapse {
                collapse_plans(&plans, &values)
            } else {
                let mut c: Vec<Cand> = plans
                    .iter()
                    .zip(&values)
                    .map(|((m, s), &v)| Cand {
                        mark: *m,
                        value: v,
                        post: *s,
                    })
                    .collect();
                c.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap());
                c
            };
            (cands, false)
        }
        Decision::Forced(alt) => {
            // Rule-free pipeline forces something on its own. If it's the
            // lock there is nothing to compare; otherwise adjudicate the
            // 2-candidate decision {forced alternative, lock}.
            if alt == Some(lock) {
                return None;
            }
            let mk = |m: Option<Mark>| match m {
                Some(m) => {
                    let mut s = *state;
                    s.apply_mark(m);
                    s
                }
                // None's post is the ctx-correct baseline (strike for ap2 has_marked=false).
                None => baseline,
            };
            let states = [mk(alt), mk(Some(lock))];
            let values = bot.evaluate_batch(&states, eval_opps);
            let mut c: Vec<Cand> = [(alt, states[0], values[0]), (Some(lock), states[1], values[1])]
                .into_iter()
                .map(|(m, s, v)| Cand {
                    mark: m,
                    value: v,
                    post: s,
                })
                .collect();
            c.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap());
            (c, true)
        }
    };
    if cands.len() < 2 {
        return None;
    }
    let lock_idx = cands.iter().position(|c| c.mark == Some(lock))?;
    let safe: Vec<usize> = (0..cands.len())
        .filter(|&i| is_safe_lock(state, cands[i].mark))
        .collect();
    let alt_idx = (0..cands.len()).find(|i| !safe.contains(i))?;
    let alt2_idx = safe.iter().copied().find(|&i| i != lock_idx);
    Some(LockCands {
        n_safe_locks: safe.len(),
        cands,
        lock_idx,
        alt_idx,
        alt2_idx,
        rule_free_forced,
    })
}
```

Note the two `?` early-returns: if the lock itself was pruned out of the rule-free set (`position` fails) or every candidate is a safe lock (`find` fails), the event is skipped — both are pathological and fine to drop, but they must not panic.

- [ ] **Step 5: Build.** `cargo build --release --example divergence 2>&1 | grep -E "^error" | head` → no errors (dead-code warnings for the new fns are expected until Tasks 2-3).

- [ ] **Step 6: Commit.**

```bash
git add examples/divergence.rs
git commit -m "feat(lockcheck): rule-pipeline mirrors, LockEvent schema, CLI stubs"
```

---

### Task 2: LockShadowPair + lock-run mode

**Files:**
- Modify: `examples/divergence.rs`

- [ ] **Step 1: Add `LockShadowPair`** (after `ShadowPair`'s impl):

```rust
// ---- Lock shadow: plays production moves (rule ON), logs rule firings ----

struct LockShadowPair {
    static_bot: PairStrategy,
    events: Rc<RefCell<Vec<LockEvent>>>,
    turn: u32,
}

impl std::fmt::Debug for LockShadowPair {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "LockShadowPair")
    }
}

impl LockShadowPair {
    fn new(template: &PairStrategy, events: Rc<RefCell<Vec<LockEvent>>>) -> Self {
        LockShadowPair {
            static_bot: PairStrategy::from_shared(template.model.clone(), template.device),
            events,
            turn: 0,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn log(
        &self,
        ctx: &str,
        has_marked: Option<bool>,
        state: &State,
        opps: &[State],
        dice: [u8; 6],
        lock: Mark,
        lc: LockCands,
    ) {
        self.events.borrow_mut().push(LockEvent {
            t: "l".into(),
            game: 0, // stamped by the driver
            turn: self.turn,
            ctx: ctx.into(),
            has_marked,
            dice,
            our: StateJson::of(state),
            opps: opps.iter().map(StateJson::of).collect(),
            our_points: state.count_points(),
            opp_points: opps.iter().map(|s| s.count_points()).max().unwrap(),
            lock_mark: (lock.row, lock.number),
            cands: lc
                .cands
                .iter()
                .map(|c| CandJson {
                    mark: mark_json(c.mark),
                    v: c.value,
                })
                .collect(),
            lock_idx: lc.lock_idx,
            alt_idx: lc.alt_idx,
            alt2_idx: lc.alt2_idx,
            n_safe_locks: lc.n_safe_locks,
            rule_free_forced: lc.rule_free_forced,
            seed: context_seed(state, opps, dice),
        });
    }
}

impl Strategy for LockShadowPair {
    fn active_phase1(&mut self, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
        self.turn += 1;
        let (decision, sim_opp) = active_phase1_choices(&self.static_bot, state, opp_states, dice);
        let forced = match &decision {
            Decision::Forced(m) => Some(*m),
            Decision::Choices(_) => None,
        };
        let prod_move = eval_decision(&self.static_bot, decision, &sim_opp);

        let (scan_lock, rule_free) = phase1_plans_mirror(state, &sim_opp, dice);
        if let Some(lock) = scan_lock {
            // Equivalence guard: production must have forced exactly this lock.
            assert_eq!(
                forced,
                Some(Some(lock)),
                "ap1 lock-force mismatch (mirror drift): mirror {lock:?}, production {forced:?}"
            );
            if let Some(lc) = build_lock_cands(&self.static_bot, state, &sim_opp, *state, rule_free, lock, true) {
                self.log("ap1", None, state, opp_states, dice, lock, lc);
            }
        }
        prod_move
    }

    fn active_phase2(&mut self, state: &State, opp_states: &[State], dice: [u8; 6], has_marked: bool) -> Option<Mark> {
        let decision = active_phase2_choices(state, opp_states, dice, has_marked);
        let forced = match &decision {
            Decision::Forced(m) => Some(*m),
            Decision::Choices(_) => None,
        };
        let prod_move = eval_decision(&self.static_bot, decision, opp_states);

        let marks = state.generate_color_moves(dice);
        if let Some(lock) = find_safe_lock(state, &marks) {
            assert_eq!(
                forced,
                Some(Some(lock)),
                "ap2 lock-force mismatch (mirror drift): mirror {lock:?}, production {forced:?}"
            );
            let baseline = if has_marked {
                *state
            } else {
                let mut s = *state;
                s.apply_strike();
                s
            };
            let opp_best = opp_states.iter().map(|s| s.count_points()).max().unwrap_or(0);
            let rule_free = mark_choices_nolock(state, &marks, baseline, opp_best);
            if let Some(lc) = build_lock_cands(&self.static_bot, state, opp_states, baseline, rule_free, lock, false) {
                self.log("ap2", Some(has_marked), state, opp_states, dice, lock, lc);
            }
        }
        prod_move
    }

    fn passive_phase1(
        &mut self,
        state: &State,
        opp_states: &[State],
        dice: [u8; 6],
        active_player: usize,
    ) -> Option<Mark> {
        let prod_move = self.static_bot.passive_phase1(state, opp_states, dice, active_player);
        let white_sum = dice[0] + dice[1];
        let marks = state.generate_white_moves(white_sum);
        if let Some(lock) = find_safe_lock(state, &marks) {
            assert_eq!(
                prod_move,
                Some(lock),
                "pp1 lock-force mismatch (mirror drift): mirror {lock:?}, production {prod_move:?}"
            );
            let opp_best = opp_best_phase1_score(opp_states, white_sum);
            let rule_free = mark_choices_nolock(state, &marks, *state, opp_best);
            if let Some(lc) = build_lock_cands(&self.static_bot, state, opp_states, *state, rule_free, lock, false) {
                self.log("pp1", None, state, opp_states, dice, lock, lc);
            }
        }
        prod_move
    }
}
```

Note on ap1: `prod_move` is computed via the same `eval_decision` path production uses, so played moves are production-identical; when the rule fires the decision is `Forced` and `eval_decision` just returns it.

- [ ] **Step 2: Add the driver** (clone of `play_one`/`cmd_run` adapted to `LockEvent`; replace the Task-1 `cmd_lock_run` stub):

```rust
fn lock_play_one(
    pair_template: &PairStrategy,
    champion: &DNA,
    game_idx: usize,
    base_seed: u64,
) -> (GameEvent, Vec<LockEvent>) {
    let pairing = game_idx / 2;
    let rotation = game_idx % 2;
    let pair_seat = (1 + rotation) % 2;
    let events = Rc::new(RefCell::new(Vec::new()));
    let players: Vec<Player> = (0..2)
        .map(|j| {
            let dice = Box::new(SmallRng::seed_from_u64(seat_dice_seed(base_seed, pairing, j)));
            let strategy: Box<dyn Strategy> = if j == pair_seat {
                Box::new(LockShadowPair::new(pair_template, events.clone()))
            } else {
                Box::new(champion.clone())
            };
            Player::new(strategy, dice)
        })
        .collect();
    let mut game = Game::new(players);
    game.play();
    let scores: Vec<isize> = game.players.iter().map(|p| p.state.count_points()).collect();
    let our_locks = game.players[pair_seat].state.count_locked();
    drop(game);
    let max = *scores.iter().max().unwrap();
    let unique_winner = scores.iter().filter(|&&s| s == max).count() == 1;
    let mut evs = Rc::try_unwrap(events)
        .unwrap_or_else(|_| panic!("events Rc still shared"))
        .into_inner();
    for e in &mut evs {
        e.game = game_idx;
    }
    let pair_won = scores[pair_seat] == max && unique_winner;
    // our_locks rides along in the summary via a metadata trick: stash it in
    // the GameEvent? No — keep GameEvent stable; reconciliation is computed
    // from firings + final states in the summary below instead.
    let _ = our_locks;
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

fn cmd_lock_run(n: usize, seed: u64, out: &str) {
    use rayon::prelude::*;
    let num_games = n.div_ceil(2) * 2;
    eprintln!("lock-run: {num_games} games, seed {seed} -> {out}");
    let results: Vec<(GameEvent, Vec<LockEvent>)> = (0..num_games)
        .into_par_iter()
        .map_init(
            || {
                (
                    PairStrategy::load("pair_model"),
                    DNA::load_weights("champion.txt", Arc::new(default_genes()))
                        .expect("champion.txt missing — run `train ga` first"),
                )
            },
            |(pair, champ), i| lock_play_one(pair, champ, i, seed),
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

    let all: Vec<&LockEvent> = results.iter().flat_map(|(_, e)| e).collect();
    let per_ctx = |c: &str| all.iter().filter(|e| e.ctx == c).count();
    println!(
        "{num_games} games, {} lock events ({:.2}/game): ap1 {} / ap2 {} / pp1 {}",
        all.len(),
        all.len() as f64 / num_games as f64,
        per_ctx("ap1"),
        per_ctx("ap2"),
        per_ctx("pp1")
    );
    println!(
        "multi-lock states: {}, rule-free-forced: {}",
        all.iter().filter(|e| e.n_safe_locks > 1).count(),
        all.iter().filter(|e| e.rule_free_forced).count()
    );
}
```

(Drop the `our_locks` stub lines from `lock_play_one` if unused — keep the driver minimal; the spec's lock reconciliation is a reported sanity check done in the analysis script, not here.)

- [ ] **Step 3: Build + smoke.**

`cargo build --release --example divergence 2>&1 | grep -E "^error" | head` → empty.

`./target/release/examples/divergence lock-run -n 50 --seed 0 --out /tmp/lock50.jsonl && head -c 500 /tmp/lock50.jsonl && echo && wc -l /tmp/lock50.jsonl`
Expected: zero guard panics (the equivalence guard runs on EVERY firing — this is the mirror's correctness proof); ~0.3-1.2 events/game; all three ctx values present (pp1 may be the largest bucket); first line a `"t":"l"` object with all schema fields.

- [ ] **Step 4: Determinism.** Run twice with the same seed, `cmp` → identical.

- [ ] **Step 5: Sanity-inspect one event** (`head -1`, pipe through `python3 -m json.tool`): `cands` sorted desc by `v`; `lock_idx` points at `lock_mark`; `cands[alt_idx]` is not a safe lock; if `alt2_idx` present, it differs from `lock_idx`.

- [ ] **Step 6: Commit.**

```bash
git add examples/divergence.rs
git commit -m "feat(lockcheck): lock-run mode — production play with rule-firing log"
```

---

### Task 3: lock-adjudicate mode

**Files:**
- Modify: `examples/divergence.rs`

- [ ] **Step 1: Add the passive entry builder and full-game scorer:**

```rust
/// pp1 entries: our passive mark applied, then the active player's turn
/// (1v1: opps[0]) completed deterministically from their perspective via the
/// same public pipeline. Approximation: the real opponent decided phase 1
/// simultaneously with us, not after seeing our mark — but the completion is
/// identical across compared candidates, so CRN-paired gaps remain valid.
fn passive_entries(
    bot: &PairStrategy,
    state: &State,
    opps: &[State],
    dice: [u8; 6],
    shortlist: &[&Cand],
) -> Vec<(Vec<State>, bool)> {
    shortlist
        .iter()
        .map(|c| {
            let mut our = *state;
            if let Some(m) = c.mark {
                our.apply_mark(m);
            }
            let mut all: Vec<State> = std::iter::once(our).chain(opps.iter().copied()).collect();
            SimGame::propagate_locks(&mut all);
            if SimGame::game_over(&all) {
                return (all, true);
            }
            // Active player's phase 1 (index 1).
            let view = SimGame::opp_view(&all, 1);
            let (d, _) = active_phase1_choices(bot, &all[1], &view, dice);
            let p1 = eval_decision(bot, d, &view);
            if let Some(m) = p1 {
                all[1].apply_mark(m);
            }
            SimGame::propagate_locks(&mut all);
            if SimGame::game_over(&all) {
                return (all, true);
            }
            // Active player's phase 2.
            let view = SimGame::opp_view(&all, 1);
            let d = active_phase2_choices(&all[1], &view, dice, p1.is_some());
            match eval_decision(bot, d, &view) {
                Some(m) => all[1].apply_mark(m),
                None if p1.is_none() => all[1].apply_strike(),
                None => {}
            }
            SimGame::propagate_locks(&mut all);
            let ended = SimGame::game_over(&all);
            (all, ended)
        })
        .collect()
}

/// Full-game rollout scores: like rollout_scores but rolls every sim to
/// completion and scores exact outcomes only (no truncation, no win-prob
/// bootstrap). `first_active` = player to act first in the rollout
/// (ap1/ap2: 1 % n — the player after us; pp1: 0 — us, after the active
/// player's completed turn).
fn rollout_scores_full(
    bot: &PairStrategy,
    entries: &[(Vec<State>, bool)],
    seed: u64,
    k: usize,
    first_active: usize,
) -> Vec<Vec<f32>> {
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
                active: first_active,
                rngs: (0..n)
                    .map(|p| SmallRng::seed_from_u64(sample_player_seed(seed, s, p)))
                    .collect(),
                over: false,
            });
            sim_owner.push(ei);
        }
    }
    let mut driver = BatchedRollouts::new(bot, sims);
    let mut turns = 0;
    while !driver.all_over() {
        driver.step_turn();
        turns += 1;
        assert!(turns <= 200, "rollout exceeded 200 turns — game-end invariant violated");
    }
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
        out[sim_owner[i]].push(SimGame::outcome(&sim.states));
    }
    out
}
```

- [ ] **Step 2: Add the per-event adjudication + driver** (replace the Task-1 `cmd_lock_adjudicate` stub):

```rust
struct PairAdj {
    gap_mean: f32,
    gap_se: f32,
    z: f32,
    verdict: &'static str,
}

fn pair_adj(lock_scores: &[f32], alt_scores: &[f32]) -> PairAdj {
    // Oriented alternative − lock: positive = the rule is wrong.
    let (gap, se) = paired_stats(lock_scores, alt_scores);
    let z = if se > 0.0 {
        gap / se
    } else {
        match gap.partial_cmp(&0.0).unwrap() {
            std::cmp::Ordering::Greater => f32::INFINITY,
            std::cmp::Ordering::Less => f32::NEG_INFINITY,
            std::cmp::Ordering::Equal => 0.0,
        }
    };
    let verdict = if z > 2.0 {
        "lock_wrong"
    } else if z < -2.0 {
        "lock_right"
    } else {
        "coinflip"
    };
    PairAdj {
        gap_mean: gap,
        gap_se: se,
        z,
        verdict,
    }
}

/// Rebuild the logged firing, run full-game rollouts, adjudicate lock vs
/// alternatives. Hard-fails on drift from the collection run.
fn adjudicate_event(ev: &LockEvent, bot: &PairStrategy, k: usize, lineno: usize) -> (PairAdj, Option<PairAdj>) {
    let our = ev.our.to_state();
    let opps: Vec<State> = ev.opps.iter().map(|o| o.to_state()).collect();
    let seed = context_seed(&our, &opps, ev.dice);
    assert_eq!(seed, ev.seed, "line {lineno}: context seed mismatch — schema drift");
    let lock = Mark {
        row: ev.lock_mark.0,
        number: ev.lock_mark.1,
    };

    let mut sim_opp_holder: Vec<State> = Vec::new();
    let lc = match ev.ctx.as_str() {
        "ap1" => {
            let (_, sim_opp) = active_phase1_choices(bot, &our, &opps, ev.dice);
            sim_opp_holder = sim_opp;
            let (scan_lock, rule_free) = phase1_plans_mirror(&our, &sim_opp_holder, ev.dice);
            assert_eq!(scan_lock, Some(lock), "line {lineno}: ap1 lock rebuild mismatch");
            build_lock_cands(bot, &our, &sim_opp_holder, our, rule_free, lock, true)
        }
        "ap2" => {
            let marks = our.generate_color_moves(ev.dice);
            assert_eq!(find_safe_lock(&our, &marks), Some(lock), "line {lineno}: ap2 lock rebuild mismatch");
            let baseline = if ev.has_marked.unwrap() {
                our
            } else {
                let mut s = our;
                s.apply_strike();
                s
            };
            let opp_best = opps.iter().map(|s| s.count_points()).max().unwrap_or(0);
            let rule_free = mark_choices_nolock(&our, &marks, baseline, opp_best);
            build_lock_cands(bot, &our, &opps, baseline, rule_free, lock, false)
        }
        "pp1" => {
            let white_sum = ev.dice[0] + ev.dice[1];
            let marks = our.generate_white_moves(white_sum);
            assert_eq!(find_safe_lock(&our, &marks), Some(lock), "line {lineno}: pp1 lock rebuild mismatch");
            let opp_best = opp_best_phase1_score(&opps, white_sum);
            let rule_free = mark_choices_nolock(&our, &marks, our, opp_best);
            build_lock_cands(bot, &our, &opps, our, rule_free, lock, false)
        }
        c => panic!("line {lineno}: bad ctx {c}"),
    }
    .unwrap_or_else(|| panic!("line {lineno}: candidate rebuild produced no decision — drift"));

    // Guard: rebuilt candidates and indices must match the log.
    assert_eq!(lc.cands.len(), ev.cands.len(), "line {lineno}: candidate count drift");
    for (c, l) in lc.cands.iter().zip(&ev.cands) {
        assert_eq!(mark_json(c.mark), l.mark, "line {lineno}: candidate order drift");
        assert!((c.value - l.v).abs() < 1e-4, "line {lineno}: static value drift");
    }
    assert_eq!(
        (lc.lock_idx, lc.alt_idx, lc.alt2_idx),
        (ev.lock_idx, ev.alt_idx, ev.alt2_idx),
        "line {lineno}: comparison index drift"
    );

    // Entries for [lock, alt, alt2?] in that order.
    let mut compared: Vec<&Cand> = vec![&lc.cands[lc.lock_idx], &lc.cands[lc.alt_idx]];
    if let Some(i2) = lc.alt2_idx {
        compared.push(&lc.cands[i2]);
    }
    let (entries, first_active) = match ev.ctx.as_str() {
        "ap1" => {
            let owned: Vec<Cand> = compared.iter().map(|c| (*c).clone()).collect();
            (
                phase1_entries(bot, &our, &sim_opp_holder, ev.dice, &owned),
                1 % (1 + opps.len()),
            )
        }
        "ap2" => {
            let owned: Vec<Cand> = compared.iter().map(|c| (*c).clone()).collect();
            (phase2_entries(&opps, &owned), 1 % (1 + opps.len()))
        }
        "pp1" => (passive_entries(bot, &our, &opps, ev.dice, &compared), 0),
        _ => unreachable!(),
    };

    let scores = rollout_scores_full(bot, &entries, seed, k, first_active);
    let alt = pair_adj(&scores[0], &scores[1]);
    let alt2 = lc.alt2_idx.map(|_| pair_adj(&scores[0], &scores[2]));
    (alt, alt2)
}
```

(`phase1_entries`/`phase2_entries` take `&[Cand]` — hence the `owned` copies; `passive_entries` was written for `&[&Cand]`. Adjust `passive_entries`'s signature to `&[Cand]` instead and clone uniformly if that reads cleaner — implementer's choice, note which.)

```rust
fn cmd_lock_adjudicate(input: &str, out: &str, k: usize) {
    use rayon::prelude::*;
    let text = std::fs::read_to_string(input).expect("cannot read input");
    let lines: Vec<&str> = text.lines().collect();
    let events: Vec<(usize, LockEvent)> = lines
        .iter()
        .enumerate()
        .filter_map(|(i, line)| {
            let v: serde_json::Value =
                serde_json::from_str(line).unwrap_or_else(|e| panic!("line {}: bad JSON: {e}", i + 1));
            if v["t"] != "l" {
                return None;
            }
            Some((
                i,
                serde_json::from_value(v).unwrap_or_else(|e| panic!("line {}: bad event: {e}", i + 1)),
            ))
        })
        .collect();
    eprintln!("adjudicating {} lock events at K={k} (full-game)", events.len());

    let results: Vec<(usize, PairAdj, Option<PairAdj>)> = events
        .par_iter()
        .map_init(
            || PairStrategy::load("pair_model"),
            |bot, (i, ev)| {
                let (a, a2) = adjudicate_event(ev, bot, k, *i + 1);
                (*i, a, a2)
            },
        )
        .collect();
    let by_line: std::collections::HashMap<usize, (PairAdj, Option<PairAdj>)> =
        results.into_iter().map(|(i, a, a2)| (i, (a, a2))).collect();

    let mut f = std::io::BufWriter::new(std::fs::File::create(out).unwrap());
    let mut counts = std::collections::HashMap::new();
    for (i, line) in lines.iter().enumerate() {
        match by_line.get(&i) {
            None => writeln!(f, "{line}").unwrap(),
            Some((a, a2)) => {
                let mut v: serde_json::Value = serde_json::from_str(line).unwrap();
                v["adj_k"] = k.into();
                v["alt_gap_mean"] = a.gap_mean.into();
                v["alt_gap_se"] = a.gap_se.into();
                v["alt_z"] = a.z.into();
                v["alt_verdict"] = a.verdict.into();
                if let Some(a2) = a2 {
                    v["alt2_gap_mean"] = a2.gap_mean.into();
                    v["alt2_gap_se"] = a2.gap_se.into();
                    v["alt2_z"] = a2.z.into();
                    v["alt2_verdict"] = a2.verdict.into();
                }
                writeln!(f, "{}", serde_json::to_string(&v).unwrap()).unwrap();
                *counts.entry(a.verdict).or_insert(0u32) += 1;
            }
        }
    }
    f.flush().unwrap();
    println!("alt verdicts: {counts:?}");
}
```

One serde wrinkle: `f32::INFINITY` is not representable in JSON — `serde_json` serializes it as `null`. That is acceptable (`alt_z: null` with `alt_gap_se == 0` is unambiguous; Python reads it as None/NaN), but the implementer must confirm the writer doesn't panic on it (`serde_json::Value::from(f32::INFINITY)` → `Null`, no panic). Verify with a quick check; document in the analysis loader.

- [ ] **Step 3: Build + smoke.**

`cargo build --release --example divergence 2>&1 | grep -E "^(error|warning)" | head` → empty.

`./target/release/examples/divergence lock-adjudicate --input /tmp/lock50.jsonl --out /tmp/lock50.adj.jsonl -k 256 && wc -l /tmp/lock50.jsonl /tmp/lock50.adj.jsonl`
Expected: zero guard panics (rebuild + seed + index guards run on every event); 200-turn cap never hit; verdict counts printed; line counts equal; non-event lines byte-identical.

- [ ] **Step 4: Determinism.** Adjudicate twice, `cmp` outputs → identical.

- [ ] **Step 5: Commit.**

```bash
git add examples/divergence.rs
git commit -m "feat(lockcheck): lock-adjudicate mode — full-game CRN rollouts with drift guards"
```

---

### Task 4: 500-game rehearsal (validation checkpoint, no code)

- [ ] **Step 1:** `./target/release/examples/divergence lock-run -n 500 --seed 7 --out /tmp/lock500.jsonl` — record events/game (expect ~0.3-1.2), ctx split, multi-lock and rule-free-forced counts. Zero guard panics.
- [ ] **Step 2:** `time ./target/release/examples/divergence lock-adjudicate --input /tmp/lock500.jsonl --out /tmp/lock500.adj.jsonl -k 2048` — zero panics; record wall time; extrapolate ×20 for the 10k run and flag if > ~40 min.
- [ ] **Step 3:** Verdict preview: `grep -o '"alt_verdict":"[a-z_]*"' /tmp/lock500.adj.jsonl | sort | uniq -c`. Record the split; any `lock_wrong` at all is already informative. Also count `alt2_verdict` occurrences.

---

### Task 5: analysis/lock_analysis.py

**Files:**
- Create: `analysis/lock_analysis.py`

- [ ] **Step 1: Write the script:**

```python
"""Adjudicated safe-lock events: verdict splits, conditioning, cost, examples.

Usage: lock_analysis.py <adjudicated.jsonl> [top_n]
"""
import json
import math
import sys

import pandas as pd

from examples import fmt_mark, render_state

pd.set_option("display.width", 200)


def z_or_inf(gap, se, z):
    """alt_z is null in JSON when se==0 (inf not representable); reconstruct."""
    if z is not None:
        return z
    if se == 0:
        return math.copysign(math.inf, gap) if gap else 0.0
    return gap / se


def load_lock(path):
    rows, raw, games = [], [], {}
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            try:
                e = json.loads(line)
            except json.JSONDecodeError as err:
                sys.exit(f"{path}:{lineno}: bad JSON: {err}")
            if e["t"] == "g":
                games[e["game"]] = e
                continue
            if e["t"] != "l":
                continue
            try:
                our, opp = e["our"], e["opps"][0]
                z = z_or_inf(e["alt_gap_mean"], e["alt_gap_se"], e.get("alt_z"))
                r = {
                    "game": e["game"],
                    "turn": e["turn"],
                    "ctx": e["ctx"],
                    "our_points": e["our_points"],
                    "opp_points": e["opp_points"],
                    "cdiff": e["our_points"] - e["opp_points"],
                    "our_marks": sum(t for t, _ in our["rows"]),
                    "opp_marks": sum(t for t, _ in opp["rows"]),
                    "locks_on_board": sum(f is None for _, f in our["rows"])
                    + sum(f is None for _, f in opp["rows"]),
                    "our_strikes": our["strikes"],
                    "n_cands": len(e["cands"]),
                    "n_safe_locks": e["n_safe_locks"],
                    "rule_free_forced": e["rule_free_forced"],
                    "lock_row": e["lock_mark"][0],
                    "v_lock": e["cands"][e["lock_idx"]]["v"],
                    "v_alt": e["cands"][e["alt_idx"]]["v"],
                    "alt_is_defer": e["cands"][e["alt_idx"]]["mark"] is None,
                    "alt_gap": e["alt_gap_mean"],
                    "alt_se": e["alt_gap_se"],
                    "alt_z": z,
                    "alt_verdict": e["alt_verdict"],
                    "alt2_gap": e.get("alt2_gap_mean"),
                    "alt2_verdict": e.get("alt2_verdict"),
                }
            except KeyError as err:
                sys.exit(f"{path}:{lineno}: missing field {err}")
            rows.append(r)
            raw.append(e)
    return pd.DataFrame(rows), raw, games


def main(path, top_n=10):
    df, raw, games = load_lock(path)
    n_games = len(games)
    print(f"{len(df)} adjudicated lock firings from {n_games} games ({len(df) / n_games:.2f}/game)")

    print("\n== Verdict split (lock vs best non-lock; positive gap = rule wrong) ==")
    print(df.alt_verdict.value_counts().to_string())
    print("\n== ...by context ==")
    print(pd.crosstab(df.ctx, df.alt_verdict, margins=True))

    print("\n== lock_wrong rate by score situation (cdiff bins) ==")
    df["cdiff_bin"] = pd.cut(df.cdiff, [-200, -10, -1, 0, 9, 200], labels=["<=-10", "-9..-1", "0", "1..9", ">=10"])
    print(df.groupby("cdiff_bin", observed=True).alt_verdict.value_counts(normalize=True).unstack(fill_value=0).to_string())
    print(df.groupby("cdiff_bin", observed=True).size().to_string())

    print("\n== lock_wrong rate by stage / board ==")
    for col in ["locks_on_board", "n_safe_locks", "rule_free_forced", "alt_is_defer", "lock_row"]:
        sub = df.groupby(col).alt_verdict.value_counts(normalize=True).unstack(fill_value=0)
        sub["n"] = df.groupby(col).size()
        print(sub.to_string(), "\n")

    wrong = df[df.alt_verdict == "lock_wrong"]
    print(f"== Cost: {len(wrong)} lock_wrong events, "
          f"sum gap {wrong.alt_gap.sum():.3f} win-prob pts over {n_games} games "
          f"= {wrong.alt_gap.sum() / n_games:.5f} wpp/game ==")

    if df.alt2_verdict.notna().any():
        print("\n== Runner-up lock (wrong-lock-chosen check) ==")
        print(df.alt2_verdict.value_counts(dropna=True).to_string())

    print(f"\n== Top {top_n} lock_wrong positions by z ==")
    wrong_raw = [e for e in raw if e["alt_verdict"] == "lock_wrong"]
    wrong_raw.sort(key=lambda e: -z_or_inf(e["alt_gap_mean"], e["alt_gap_se"], e.get("alt_z")))
    for e in wrong_raw[:top_n]:
        print("=" * 78)
        print(
            f"game {e['game']} turn {e['turn']} ctx {e['ctx']} dice {e['dice']} "
            f"| gap {e['alt_gap_mean']:+.4f} z={z_or_inf(e['alt_gap_mean'], e['alt_gap_se'], e.get('alt_z')):+.1f}"
        )
        render_state("OUR", e["our"])
        render_state("OPP", e["opps"][0])
        phase = 1 if e["ctx"] in ("ap1", "pp1") else 2
        lk = e["cands"][e["lock_idx"]]
        alt = e["cands"][e["alt_idx"]]
        print(f"  forced lock: {fmt_mark(lk['mark'], phase, e['has_marked'])} (v={lk['v']:+.3f})")
        print(f"  better alt:  {fmt_mark(alt['mark'], phase, e['has_marked'])} (v={alt['v']:+.3f})")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10)
```

- [ ] **Step 2: Smoke** on the 500-game file: `cd /root/qwixx && analysis/.venv/bin/python analysis/lock_analysis.py /tmp/lock500.adj.jsonl 3` — tables print, renders work, no exceptions; firing-per-game and verdict counts match the Rust summaries from Task 4.

- [ ] **Step 3: Commit.**

```bash
git add analysis/lock_analysis.py
git commit -m "feat(lockcheck): lock-event analysis script"
```

---

### Task 6: full run + findings

- [ ] **Step 1:** `time ./target/release/examples/divergence lock-run -n 10000 --seed 0 --out lock-events.jsonl` (~6 min; expect ~5-10k events).
- [ ] **Step 2:** `time ./target/release/examples/divergence lock-adjudicate --input lock-events.jsonl --out lock-events.adj.jsonl -k 2048` (zero panics or STOP).
- [ ] **Step 3:** `analysis/.venv/bin/python analysis/lock_analysis.py lock-events.adj.jsonl 15 > /tmp/lock-analysis.txt; less /tmp/lock-analysis.txt` plus ad-hoc follow-ups in the venv as patterns emerge (especially: does `lock_wrong` concentrate at `cdiff < 0`? Is `alt_is_defer` the dominant better-alternative? Any `alt2_verdict == "lock_wrong"`-analog showing the wrong lock chosen?).
- [ ] **Step 4:** Write the findings as the next EXPERIMENTS.md phase (read the tail; next number after Phase 15). Must include: setup recap + configuration + rates; verdict split overall/per-ctx; the cost number (wpp/game) with its per-event distribution; conditioning results against the three suspected failure modes from the spec; 2-3 rendered examples; estimator notes — full-game scoring (no truncation/bootstrap), **no CRN-selection caveat this time** (the rule is unconditional, no K=128 pick selected the events), but the **rollout-policy circularity** (the net drives rollout moves and was trained rule-on, likely flattering the rule) and the **pp1 sequential-completion approximation** must both be stated; explicit verdict on the rule (keep as-is / conditional variant worth A/B-ing per spec §5 / wrong rarely-but-costly) with next steps.
- [ ] **Step 5:** Commit: `git add docs/EXPERIMENTS.md && git commit -m "docs: safe-lock adjudication findings"`. Do NOT commit jsonl files (git-ignored).

---

## Self-review notes

- Spec coverage: §1 (no src changes) → all code in the example; §2 collection (detection, rule-free mirrors, equivalence guard, schema, volumes) → Tasks 1-2; §3 adjudication (rebuild guards, entries incl. pp1 approximation, full-game scorer with cap, verdict orientation, no-CRN-caveat note) → Task 3; §4 analysis → Task 5; testing section → Tasks 2-4 smokes (guards exercised on every event, determinism both modes, scale rehearsal); §5 conditional A/B → explicitly out of scope, referenced in Task 6 findings; error handling (overwrite refusal via existing `refuse_overwrite`, panics with line/game context, rollout cap) → Tasks 1-3.
- Type consistency: `LockCands`/`LockEvent` field names match between builder, logger, and adjudication asserts; `pair_adj` orientation (alternative − lock) matches the analysis script's "positive gap = rule wrong" and the spec.
- Known wrinkles called out inline: `f32::INFINITY` → JSON null (handled in `z_or_inf`), `phase1_entries`/`passive_entries` slice-type mismatch (implementer normalizes), `adjudicate_event` sim_opp binding (full correct version provided).
