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
    let mut evs = Rc::try_unwrap(events)
        .unwrap_or_else(|_| panic!("events Rc still shared"))
        .into_inner();
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
    eprintln!(
        "relabeling {} of {} events at K={k}",
        todo.len(),
        parsed.iter().filter(|(_, e, _)| e.is_some()).count()
    );

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

fn main() {
    match Cli::parse().cmd {
        Cmd::Run {
            n,
            seed,
            out,
            force_overwrite,
        } => {
            refuse_overwrite(&out, force_overwrite);
            cmd_run(n, seed, &out);
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
            cmd_relabel(&input, &out, k, agree_sample, seed);
        }
    }
}
