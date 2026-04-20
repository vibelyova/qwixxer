//! Diagnostic: play games with the DQN and log per-candidate (μ, σ²) predictions
//! at every decision point. Used to check whether the variance head learned a
//! state-sensitive σ or collapsed to a population average.
//!
//! Run:   cargo run --release --example sigma_probe
//! Output: /tmp/sigma_probe.csv

use qwixxer::bot::{self, DNA};
use qwixxer::dqn::{
    build_opponent_context_for, state_features, win_rank_score, MyBackend, OpponentContext,
    QwixxModel, QwixxModelConfig, NUM_FEATURES,
};
use qwixxer::game::{Game, Player};
use qwixxer::state::{MetaDecision, Move, State};
use qwixxer::strategy::Strategy;
use burn::module::Module;
use burn::record::CompactRecorder;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::cell::RefCell;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::rc::Rc;
use std::sync::Arc;

type Log = Rc<RefCell<BufWriter<File>>>;

/// Instrumented DQN strategy that replicates `DqnStrategy`'s decision logic and
/// writes one row per candidate to the log at every active and passive turn.
struct ProbeDqn {
    model: QwixxModel<MyBackend>,
    device: burn::backend::ndarray::NdArrayDevice,
    context: OpponentContext,
    leader_state: Option<State>,
    non_leader_states: Vec<State>,
    log: Log,
    game_idx: usize,
    turn: u32,
}

impl std::fmt::Debug for ProbeDqn {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "ProbeDqn")
    }
}

impl ProbeDqn {
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

    fn log_candidate(
        &self,
        phase: &str,
        n_cand: usize,
        cand_idx: usize,
        mean: f32,
        log_var: f32,
        rank: f32,
        chosen: bool,
        our_score: isize,
        opp_score: isize,
        opp_mean: f32,
        opp_log_var: f32,
    ) {
        let sigma = (0.5 * log_var).exp();
        let opp_sigma = (0.5 * opp_log_var).exp();
        writeln!(
            self.log.borrow_mut(),
            "{},{},{},{},{},{:.4},{:.4},{:.4},{:.4},{},{},{},{:.4},{:.4}",
            self.game_idx,
            self.turn,
            phase,
            n_cand,
            cand_idx,
            mean,
            log_var,
            sigma,
            rank,
            chosen as u8,
            our_score,
            opp_score,
            opp_mean,
            opp_sigma,
        )
        .unwrap();
    }
}

impl Strategy for ProbeDqn {
    fn your_move(&mut self, state: &State, dice: [u8; 6]) -> Move {
        let our_score = state.count_points();
        let opp_score = our_score - self.context.score_gap_to_leader;

        let moves = match state.apply_meta_rules(dice, self.context.score_gap_to_leader) {
            MetaDecision::Forced(mov) => {
                // Still log a single-candidate row for forced moves so the CSV is complete.
                let (opp_mean, opp_log_var) = self.leader_prediction(state);
                let mut s = *state;
                s.apply_move(mov);
                let features = state_features(&s, &self.context);
                let (mean, log_var) = self.model.evaluate_state(&features, &self.device);
                let rank = win_rank_score(mean, log_var, opp_mean, opp_log_var);
                self.log_candidate(
                    "active_forced", 1, 0, mean, log_var, rank, true, our_score, opp_score,
                    opp_mean, opp_log_var,
                );
                self.turn += 1;
                return mov;
            }
            MetaDecision::Choices(moves) => moves,
        };

        let (opp_mean, opp_log_var) = self.leader_prediction(state);

        let per: Vec<(usize, Move, f32, f32, f32)> = moves
            .iter()
            .enumerate()
            .map(|(i, &m)| {
                let mut s = *state;
                s.apply_move(m);
                let features = state_features(&s, &self.context);
                let (mean, log_var) = self.model.evaluate_state(&features, &self.device);
                let rank = win_rank_score(mean, log_var, opp_mean, opp_log_var);
                (i, m, mean, log_var, rank)
            })
            .collect();

        let chosen_idx = per
            .iter()
            .max_by(|a, b| a.4.partial_cmp(&b.4).unwrap())
            .unwrap()
            .0;

        for (i, _m, mean, log_var, rank) in &per {
            self.log_candidate(
                "active",
                per.len(),
                *i,
                *mean,
                *log_var,
                *rank,
                *i == chosen_idx,
                our_score,
                opp_score,
                opp_mean,
                opp_log_var,
            );
        }
        self.turn += 1;
        per[chosen_idx].1
    }

    fn opponents_move(&mut self, state: &State, number: u8, locked: [bool; 4]) -> Option<Move> {
        let our_score = state.count_points();
        let opp_score = our_score - self.context.score_gap_to_leader;
        let moves = state.generate_opponent_moves(number);

        if let Some(mov) = state.find_smart_lock(&moves, self.context.score_gap_to_leader) {
            let (opp_mean, opp_log_var) = self.leader_prediction(state);
            let mut s = *state;
            s.apply_move(mov);
            s.lock(locked);
            let features = state_features(&s, &self.context);
            let (mean, log_var) = self.model.evaluate_state(&features, &self.device);
            let rank = win_rank_score(mean, log_var, opp_mean, opp_log_var);
            self.log_candidate(
                "passive_smartlock",
                1,
                0,
                mean,
                log_var,
                rank,
                true,
                our_score,
                opp_score,
                opp_mean,
                opp_log_var,
            );
            return Some(mov);
        }

        if moves.is_empty() {
            return None;
        }

        let (opp_mean, opp_log_var) = self.leader_prediction(state);

        // Candidates + skip.
        let mut per: Vec<(usize, Option<Move>, f32, f32, f32)> = moves
            .iter()
            .enumerate()
            .map(|(i, &m)| {
                let mut s = *state;
                s.apply_move(m);
                s.lock(locked);
                let features = state_features(&s, &self.context);
                let (mean, log_var) = self.model.evaluate_state(&features, &self.device);
                let rank = win_rank_score(mean, log_var, opp_mean, opp_log_var);
                (i, Some(m), mean, log_var, rank)
            })
            .collect();
        let mut skip_state = *state;
        skip_state.lock(locked);
        let skip_features = state_features(&skip_state, &self.context);
        let (skip_mean, skip_log_var) = self.model.evaluate_state(&skip_features, &self.device);
        let skip_rank = win_rank_score(skip_mean, skip_log_var, opp_mean, opp_log_var);
        per.push((moves.len(), None, skip_mean, skip_log_var, skip_rank));

        let chosen_idx = per
            .iter()
            .max_by(|a, b| a.4.partial_cmp(&b.4).unwrap())
            .unwrap()
            .0;

        for (i, _m, mean, log_var, rank) in &per {
            self.log_candidate(
                "passive",
                per.len(),
                *i,
                *mean,
                *log_var,
                *rank,
                *i == chosen_idx,
                our_score,
                opp_score,
                opp_mean,
                opp_log_var,
            );
        }

        per[chosen_idx].1
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

fn main() {
    let output_path = "/tmp/sigma_probe.csv";
    let file = File::create(output_path).expect("open output file");
    let log: Log = Rc::new(RefCell::new(BufWriter::new(file)));
    writeln!(
        log.borrow_mut(),
        "game,turn,phase,n_cand,cand_idx,mean,log_var,sigma,rank,chosen,our_score,opp_score,opp_mean,opp_sigma"
    )
    .unwrap();

    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    let model: QwixxModel<MyBackend> = QwixxModelConfig::new()
        .init::<MyBackend>(&device)
        .load_file("dqn_model/model", &CompactRecorder::new(), &device)
        .expect("load dqn model");

    let genes = Arc::new(bot::default_genes());
    let champion = DNA::load_weights("champion.txt", genes).expect("load champion");

    let num_games = 25;
    for game_idx in 0..num_games {
        let probe = ProbeDqn {
            model: model.clone(),
            device: device.clone(),
            context: OpponentContext::default(),
            leader_state: None,
            non_leader_states: Vec::new(),
            log: Rc::clone(&log),
            game_idx,
            turn: 0,
        };
        let players = vec![
            Player::new(
                Box::new(probe),
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
