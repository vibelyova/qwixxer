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
            todo!("relabel mode (Task 4)");
        }
    }
}
