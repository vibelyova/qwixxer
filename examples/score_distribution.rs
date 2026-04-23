//! Dump the distribution of final scores reached from a single mid-game state.
//! Plays a real DQN-vs-GA game up to a configurable turn, snapshots both
//! players' states, then runs K rollouts from that snapshot and writes every
//! final score to a CSV.
//!
//! Used to visually check how Normal/skewed the final-score distribution is
//! for a representative Qwixx state.
//!
//! Run:    cargo run --release --example score_distribution
//! Output: /tmp/score_distribution.csv  (one column: final_score)

use qwixxer::bot::{self, DNA};
use qwixxer::dqn::{MyBackend, QwixxModel, QwixxModelConfig, DqnStrategy};
use qwixxer::game::{Game, Player};
use qwixxer::state::{Mark, State};
use qwixxer::strategy::Strategy;
use burn::module::Module;
use burn::record::CompactRecorder;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::cell::RefCell;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::rc::Rc;
use std::sync::Arc;

/// How many turns to play before snapshotting (mid-game).
const SNAPSHOT_AT_TURN: u32 = 8;
/// Rollouts from the snapshot.
const N_ROLLOUTS: usize = 20_000;

/// Strategy wrapper that records both our and opp state at a target turn via
/// a shared cell.
struct Snapshotter {
    inner: DqnStrategy,
    snapshot: Rc<RefCell<Option<(State, State, u32)>>>, // (our, opp, turn)
    target_turn: u32,
    turn: u32,
}

impl std::fmt::Debug for Snapshotter {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Snapshotter")
    }
}

impl Strategy for Snapshotter {
    fn active_phase1(&mut self, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
        if self.turn == self.target_turn && self.snapshot.borrow().is_none() {
            if let Some(opp) = opp_states.first() {
                *self.snapshot.borrow_mut() = Some((*state, *opp, self.turn));
            }
        }
        let mark = self.inner.active_phase1(state, opp_states, dice);
        self.turn += 1;
        mark
    }

    fn active_phase2(&mut self, state: &State, opp_states: &[State], dice: [u8; 6], has_marked: bool) -> Option<Mark> {
        self.inner.active_phase2(state, opp_states, dice, has_marked)
    }

    fn passive_phase1(&mut self, state: &State, opp_states: &[State], dice: [u8; 6], active_player: usize) -> Option<Mark> {
        self.inner.passive_phase1(state, opp_states, dice, active_player)
    }
}

fn main() {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    let model: QwixxModel<MyBackend> = QwixxModelConfig::new()
        .init::<MyBackend>(&device)
        .load_file("dqn_model/model", &CompactRecorder::new(), &device)
        .expect("load dqn model");

    let genes = Arc::new(bot::default_genes());
    let champion = DNA::load_weights("champion.txt", genes).expect("load champion");

    // Play a game up to SNAPSHOT_AT_TURN so we get a realistic mid-game state.
    let snapshot_cell: Rc<RefCell<Option<(State, State, u32)>>> = Rc::new(RefCell::new(None));
    let dqn = DqnStrategy::from_model(model.clone(), device);
    let snap = Snapshotter {
        inner: dqn,
        snapshot: Rc::clone(&snapshot_cell),
        target_turn: SNAPSHOT_AT_TURN,
        turn: 0,
    };
    let players = vec![
        Player::new(Box::new(snap), Box::new(SmallRng::seed_from_u64(42))),
        Player::new(
            Box::new(champion.clone()),
            Box::new(SmallRng::seed_from_u64(43)),
        ),
    ];
    let mut game = Game::new(players);
    game.play();

    let (our_start, opp_start, captured_turn) = snapshot_cell
        .borrow()
        .clone()
        .expect("game ended before snapshot turn — try a smaller SNAPSHOT_AT_TURN");
    eprintln!(
        "Snapshot at our turn {captured_turn}: our_score={}, opp_score={}",
        our_start.count_points(),
        opp_start.count_points()
    );
    eprintln!(
        "Running {N_ROLLOUTS} rollouts from this state (DQN as player 0 vs GA)..."
    );

    // Parallel rollouts: clone model+champion+device per worker (Tensor !Sync).
    let workers: Vec<_> = (0..N_ROLLOUTS)
        .map(|_| (model.clone(), champion.clone(), device))
        .collect();
    let scores: Vec<isize> = workers
        .into_par_iter()
        .enumerate()
        .map(|(i, (m, c, d))| {
            let dqn = DqnStrategy::from_model(m, d);
            let players = vec![
                Player::new_with_state(
                    Box::new(dqn),
                    Box::new(SmallRng::seed_from_u64(10_000 + i as u64)),
                    our_start,
                ),
                Player::new_with_state(
                    Box::new(c),
                    Box::new(SmallRng::seed_from_u64(20_000 + i as u64)),
                    opp_start,
                ),
            ];
            // Snapshot was taken just before our `active_phase1` executed — so the
            // active player's state is still `our_start` pre-move, but we've
            // already rolled dice. For a clean rollout, start with opponent's
            // turn (start_turn = 1) so the game continues with fresh dice.
            let mut game = Game::new_from_turn(players, 1);
            game.play();
            game.players[0].state.count_points()
        })
        .collect();

    let output_path = "/tmp/score_distribution.csv";
    let mut out = BufWriter::new(File::create(output_path).unwrap());
    writeln!(out, "final_score").unwrap();
    for s in &scores {
        writeln!(out, "{s}").unwrap();
    }
    out.flush().unwrap();

    // Quick stats.
    let n = scores.len() as f64;
    let mean = scores.iter().sum::<isize>() as f64 / n;
    let var = scores
        .iter()
        .map(|&s| (s as f64 - mean).powi(2))
        .sum::<f64>()
        / (n - 1.0);
    let std = var.sqrt();
    eprintln!(
        "wrote {output_path} — mean={mean:.2}, std={std:.2}, n={}",
        scores.len()
    );
}
