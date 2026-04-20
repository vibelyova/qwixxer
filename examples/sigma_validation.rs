//! Diagnostic: validate the DQN's σ predictions against empirical σ from MC
//! rollouts. For a sample of game decisions, compare (μ_dqn, σ_dqn) per
//! candidate to (μ_empirical, σ_empirical) estimated from K rollouts of
//! DQN-plays-player-0 vs GA-plays-player-1 starting from the candidate's
//! post-move state.
//!
//! Run:    cargo run --release --example sigma_validation
//! Output: /tmp/sigma_validation.csv
//!
//! Tunables: see consts below.

use qwixxer::bot::{self, DNA};
use qwixxer::dqn::{
    build_opponent_context_for, state_features, win_rank_score, MyBackend, OpponentContext,
    QwixxModel, QwixxModelConfig, DqnStrategy,
};
use qwixxer::game::{Game, Player};
use qwixxer::state::{MetaDecision, Move, State};
use qwixxer::strategy::Strategy;
use burn::module::Module;
use burn::record::CompactRecorder;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::cell::RefCell;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::rc::Rc;
use std::sync::Arc;

const N_GAMES: usize = 3;
const SAMPLE_PROB: f32 = 0.4;
const ROLLOUTS_PER_CAND: usize = 10_000;

type Log = Rc<RefCell<BufWriter<File>>>;

/// Run `n` rollouts from (our_start, opp_start) with DQN (shared model) as player 0
/// and GA (champion) as player 1. Returns (empirical_mean, empirical_std).
fn empirical_moments(
    model: &QwixxModel<MyBackend>,
    champion: &DNA,
    device: &burn::backend::ndarray::NdArrayDevice,
    our_start: State,
    opp_start: State,
    n: usize,
    seed: u64,
) -> (f32, f32) {
    // Pre-clone model+champion+device per rollout so the closure takes owned
    // values only — Burn's tensors aren't Sync, so rayon needs owned captures.
    let workers: Vec<_> = (0..n)
        .map(|_| (model.clone(), champion.clone(), *device))
        .collect();
    let scores: Vec<isize> = workers
        .into_par_iter()
        .enumerate()
        .map(|(i, (m, c, d))| {
            let dqn = DqnStrategy::from_model(m, d);
            let players = vec![
                Player::new_with_state(
                    Box::new(dqn),
                    Box::new(SmallRng::seed_from_u64(seed.wrapping_add(i as u64))),
                    our_start,
                ),
                Player::new_with_state(
                    Box::new(c),
                    Box::new(SmallRng::seed_from_u64(
                        seed.wrapping_add((n as u64).wrapping_add(i as u64)),
                    )),
                    opp_start,
                ),
            ];
            // We just made our active move; it's opp's turn next.
            let mut game = Game::new_from_turn(players, 1);
            game.play();
            game.players[0].state.count_points()
        })
        .collect();
    let n_f = scores.len() as f32;
    let mean = scores.iter().sum::<isize>() as f32 / n_f;
    let var = scores.iter().map(|&s| (s as f32 - mean).powi(2)).sum::<f32>() / (n_f - 1.0).max(1.0);
    (mean, var.sqrt())
}

/// Instrumented DQN that samples decisions and runs MC rollouts to compare
/// DQN's (μ, σ) predictions to empirical estimates.
struct ValidateDqn {
    model: QwixxModel<MyBackend>,
    device: burn::backend::ndarray::NdArrayDevice,
    context: OpponentContext,
    leader_state: Option<State>,
    non_leader_states: Vec<State>,
    champion: DNA,
    log: Log,
    game_idx: usize,
    turn: u32,
    rng: SmallRng,
    sampled_count: usize,
}

impl std::fmt::Debug for ValidateDqn {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "ValidateDqn")
    }
}

impl ValidateDqn {
    fn leader_prediction(&self, our_state: &State) -> (f32, f32) {
        let Some(leader) = self.leader_state else {
            return (0.0, 0.0);
        };
        let leader_ctx = build_opponent_context_for(
            leader.count_points(),
            our_state,
            &self.non_leader_states,
        );
        let features = state_features(&leader, &leader_ctx);
        self.model.evaluate_state(&features, &self.device)
    }
}

impl Strategy for ValidateDqn {
    fn your_move(&mut self, state: &State, dice: [u8; 6]) -> Move {
        let moves = match state.apply_meta_rules(dice, self.context.score_gap_to_leader) {
            MetaDecision::Forced(mov) => {
                self.turn += 1;
                return mov;
            }
            MetaDecision::Choices(moves) => moves,
        };

        // DQN predictions per candidate.
        let preds: Vec<(Move, f32, f32, State)> = moves
            .iter()
            .map(|&m| {
                let mut s = *state;
                s.apply_move(m);
                let (mean, log_var) = self.model.evaluate_state(
                    &state_features(&s, &self.context),
                    &self.device,
                );
                (m, mean, log_var, s)
            })
            .collect();

        // Sample a subset of decisions for rollout validation.
        let should_sample = moves.len() >= 2
            && self.leader_state.is_some()
            && self.rng.gen::<f32>() < SAMPLE_PROB;

        if should_sample {
            let opp_start = self.leader_state.unwrap();
            let decision_idx = self.sampled_count;
            for (cand_idx, (m, dqn_mean, dqn_log_var, post_state)) in preds.iter().enumerate() {
                let seed = (self.game_idx as u64) * 1_000_000
                    + (self.turn as u64) * 1_000
                    + cand_idx as u64;
                let (emp_mean, emp_std) = empirical_moments(
                    &self.model,
                    &self.champion,
                    &self.device,
                    *post_state,
                    opp_start,
                    ROLLOUTS_PER_CAND,
                    seed,
                );
                let dqn_sigma = (0.5 * dqn_log_var).exp();
                writeln!(
                    self.log.borrow_mut(),
                    "{},{},{},{},{},{:.4},{:.4},{:.4},{:.4},{}",
                    self.game_idx,
                    self.turn,
                    decision_idx,
                    cand_idx,
                    format_move(m),
                    dqn_mean,
                    dqn_sigma,
                    emp_mean,
                    emp_std,
                    ROLLOUTS_PER_CAND,
                )
                .unwrap();
            }
            self.sampled_count += 1;
            eprintln!(
                "  game {} turn {} → sampled {} candidates ({}  total)",
                self.game_idx,
                self.turn,
                preds.len(),
                self.sampled_count,
            );
        }

        // Pick move via the standard ranking (don't perturb real gameplay).
        let (opp_mean, opp_log_var) = self.leader_prediction(state);
        let best = preds
            .iter()
            .max_by(|a, b| {
                win_rank_score(a.1, a.2, opp_mean, opp_log_var)
                    .partial_cmp(&win_rank_score(b.1, b.2, opp_mean, opp_log_var))
                    .unwrap()
            })
            .unwrap()
            .0;
        self.turn += 1;
        best
    }

    fn opponents_move(&mut self, state: &State, number: u8, locked: [bool; 4]) -> Option<Move> {
        // Delegate to a plain DqnStrategy path — we only instrument active turns.
        let moves = state.generate_opponent_moves(number);
        if let Some(mov) = state.find_smart_lock(&moves, self.context.score_gap_to_leader) {
            return Some(mov);
        }
        if moves.is_empty() {
            return None;
        }
        let (opp_mean, opp_log_var) = self.leader_prediction(state);

        let mut eval_states: Vec<State> = moves
            .iter()
            .map(|&m| {
                let mut s = *state;
                s.apply_move(m);
                s.lock(locked);
                s
            })
            .collect();
        let mut skip_state = *state;
        skip_state.lock(locked);
        eval_states.push(skip_state);

        let values: Vec<(f32, f32)> = eval_states
            .iter()
            .map(|s| self.model.evaluate_state(&state_features(s, &self.context), &self.device))
            .collect();
        let skip_rank = win_rank_score(values.last().unwrap().0, values.last().unwrap().1, opp_mean, opp_log_var);
        let (best_idx, best_rank) = values[..moves.len()]
            .iter()
            .enumerate()
            .map(|(i, v)| (i, win_rank_score(v.0, v.1, opp_mean, opp_log_var)))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        if best_rank > skip_rank {
            Some(moves[best_idx])
        } else {
            None
        }
    }

    fn observe_opponents(&mut self, our_score: isize, opponents: &[State]) {
        if opponents.is_empty() {
            self.context = OpponentContext::default();
            self.leader_state = None;
            self.non_leader_states.clear();
            return;
        }
        let leader_idx = (0..opponents.len())
            .max_by_key(|&i| opponents[i].count_points())
            .unwrap();
        let leader = opponents[leader_idx];
        let non_leader: Vec<State> = opponents
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != leader_idx)
            .map(|(_, s)| *s)
            .collect();
        self.context = build_opponent_context_for(our_score, &leader, &non_leader);
        self.leader_state = Some(leader);
        self.non_leader_states = non_leader;
    }
}

fn format_move(mov: &Move) -> String {
    match mov {
        Move::Strike => "Strike".to_string(),
        Move::Single(m) => format!("S({},{})", m.row, m.number),
        Move::Double(m1, m2) => format!("D({},{}/{},{})", m1.row, m1.number, m2.row, m2.number),
    }
}

fn main() {
    let output_path = "/tmp/sigma_validation.csv";
    let file = File::create(output_path).expect("open output file");
    let log: Log = Rc::new(RefCell::new(BufWriter::new(file)));
    writeln!(
        log.borrow_mut(),
        "game,turn,decision,cand_idx,move,dqn_mean,dqn_sigma,emp_mean,emp_sigma,n_rollouts"
    )
    .unwrap();

    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    let model: QwixxModel<MyBackend> = QwixxModelConfig::new()
        .init::<MyBackend>(&device)
        .load_file("dqn_model/model", &CompactRecorder::new(), &device)
        .expect("load dqn model");

    let genes = Arc::new(bot::default_genes());
    let champion = DNA::load_weights("champion.txt", genes).expect("load champion");

    eprintln!(
        "Running {} games, sample_prob={}, rollouts_per_candidate={}",
        N_GAMES, SAMPLE_PROB, ROLLOUTS_PER_CAND
    );
    for game_idx in 0..N_GAMES {
        let validator = ValidateDqn {
            model: model.clone(),
            device: device.clone(),
            context: OpponentContext::default(),
            leader_state: None,
            non_leader_states: Vec::new(),
            champion: champion.clone(),
            log: Rc::clone(&log),
            game_idx,
            turn: 0,
            rng: SmallRng::seed_from_u64(1000 + game_idx as u64),
            sampled_count: 0,
        };
        let players = vec![
            Player::new(
                Box::new(validator),
                Box::new(SmallRng::seed_from_u64(100 + game_idx as u64)),
            ),
            Player::new(
                Box::new(champion.clone()),
                Box::new(SmallRng::seed_from_u64(200 + game_idx as u64)),
            ),
        ];
        let mut game = Game::new(players);
        game.play();
        let scores: Vec<isize> = game.players.iter().map(|p| p.state.count_points()).collect();
        eprintln!("game {game_idx}: DQN {} — GA {}", scores[0], scores[1]);
    }

    log.borrow_mut().flush().unwrap();
    eprintln!("wrote {output_path}");
}
