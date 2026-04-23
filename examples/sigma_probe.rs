//! Diagnostic: play games with the DQN and log per-candidate (μ, σ²) predictions
//! at every decision point. Used to check whether the variance head learned a
//! state-sensitive σ or collapsed to a population average.
//!
//! Run:   cargo run --release --example sigma_probe
//! Output: /tmp/sigma_probe.csv

use qwixxer::bot::{self, DNA};
use qwixxer::dqn::{
    build_opponent_context_for, state_features, win_rank_score, MyBackend, OpponentContext,
    QwixxModel, QwixxModelConfig,
};
use qwixxer::game::{Game, Player};
use qwixxer::state::{Mark, State};
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
    /// Build opponent context from opp_states for our score.
    fn build_context(our_score: isize, opp_states: &[State]) -> (OpponentContext, Option<State>, Vec<State>) {
        if opp_states.is_empty() {
            return (OpponentContext::default(), None, Vec::new());
        }
        let leader_idx = (0..opp_states.len())
            .max_by_key(|&i| opp_states[i].count_points())
            .unwrap();
        let leader = opp_states[leader_idx];
        let non_leader: Vec<State> = opp_states
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != leader_idx)
            .map(|(_, s)| *s)
            .collect();
        let context = build_opponent_context_for(our_score, &leader, &non_leader);
        (context, Some(leader), non_leader)
    }

    fn leader_prediction(&self, our_state: &State, leader_state: Option<State>, non_leader_states: &[State]) -> (f32, f32) {
        let Some(leader) = leader_state else {
            return (0.0, 0.0);
        };
        let leader_ctx = build_opponent_context_for(
            leader.count_points(),
            our_state,
            non_leader_states,
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
    fn active_phase1(&mut self, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
        let our_score = state.count_points();
        let (context, leader_state, non_leader_states) = Self::build_context(our_score, opp_states);
        let opp_score = our_score - context.score_gap_to_leader;

        let white_sum = dice[0] + dice[1];
        let marks = state.generate_white_moves(white_sum);

        if marks.is_empty() {
            self.turn += 1;
            return None;
        }

        let (opp_mean, opp_log_var) = self.leader_prediction(state, leader_state, &non_leader_states);

        // Evaluate each white mark + skip
        let mut per: Vec<(usize, Option<Mark>, f32, f32, f32)> = marks
            .iter()
            .enumerate()
            .map(|(i, &m)| {
                let mut s = *state;
                s.apply_mark(m);
                let features = state_features(&s, &context);
                let (mean, log_var) = self.model.evaluate_state(&features, &self.device);
                let rank = win_rank_score(mean, log_var, opp_mean, opp_log_var);
                (i, Some(m), mean, log_var, rank)
            })
            .collect();
        // skip option
        let skip_features = state_features(state, &context);
        let (skip_mean, skip_log_var) = self.model.evaluate_state(&skip_features, &self.device);
        let skip_rank = win_rank_score(skip_mean, skip_log_var, opp_mean, opp_log_var);
        per.push((marks.len(), None, skip_mean, skip_log_var, skip_rank));

        let chosen_idx = per
            .iter()
            .max_by(|a, b| a.4.partial_cmp(&b.4).unwrap())
            .unwrap()
            .0;

        for (i, _m, mean, log_var, rank) in &per {
            self.log_candidate(
                "active_phase1",
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

    fn active_phase2(&mut self, state: &State, opp_states: &[State], dice: [u8; 6], has_marked: bool) -> Option<Mark> {
        let our_score = state.count_points();
        let (context, leader_state, non_leader_states) = Self::build_context(our_score, opp_states);
        let opp_score = our_score - context.score_gap_to_leader;

        let marks = state.generate_color_moves(dice);

        if marks.is_empty() {
            self.turn += 1;
            return None;
        }

        let (opp_mean, opp_log_var) = self.leader_prediction(state, leader_state, &non_leader_states);

        // The "no mark" state is a strike if we haven't marked yet in phase1
        let no_mark_state = if has_marked {
            *state
        } else {
            let mut s = *state;
            s.apply_strike();
            s
        };

        let mut per: Vec<(usize, Option<Mark>, f32, f32, f32)> = marks
            .iter()
            .enumerate()
            .map(|(i, &m)| {
                let mut s = *state;
                s.apply_mark(m);
                let features = state_features(&s, &context);
                let (mean, log_var) = self.model.evaluate_state(&features, &self.device);
                let rank = win_rank_score(mean, log_var, opp_mean, opp_log_var);
                (i, Some(m), mean, log_var, rank)
            })
            .collect();
        // skip/strike option
        let skip_features = state_features(&no_mark_state, &context);
        let (skip_mean, skip_log_var) = self.model.evaluate_state(&skip_features, &self.device);
        let skip_rank = win_rank_score(skip_mean, skip_log_var, opp_mean, opp_log_var);
        per.push((marks.len(), None, skip_mean, skip_log_var, skip_rank));

        let chosen_idx = per
            .iter()
            .max_by(|a, b| a.4.partial_cmp(&b.4).unwrap())
            .unwrap()
            .0;

        for (i, _m, mean, log_var, rank) in &per {
            self.log_candidate(
                "active_phase2",
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

    fn passive_phase1(&mut self, state: &State, opp_states: &[State], dice: [u8; 6], _active_player: usize) -> Option<Mark> {
        let our_score = state.count_points();
        let (context, leader_state, non_leader_states) = Self::build_context(our_score, opp_states);
        let opp_score = our_score - context.score_gap_to_leader;

        let white_sum = dice[0] + dice[1];
        let marks = state.generate_white_moves(white_sum);

        if marks.is_empty() {
            return None;
        }

        let (opp_mean, opp_log_var) = self.leader_prediction(state, leader_state, &non_leader_states);

        // Candidates + skip
        let mut per: Vec<(usize, Option<Mark>, f32, f32, f32)> = marks
            .iter()
            .enumerate()
            .map(|(i, &m)| {
                let mut s = *state;
                s.apply_mark(m);
                let features = state_features(&s, &context);
                let (mean, log_var) = self.model.evaluate_state(&features, &self.device);
                let rank = win_rank_score(mean, log_var, opp_mean, opp_log_var);
                (i, Some(m), mean, log_var, rank)
            })
            .collect();
        let skip_features = state_features(state, &context);
        let (skip_mean, skip_log_var) = self.model.evaluate_state(&skip_features, &self.device);
        let skip_rank = win_rank_score(skip_mean, skip_log_var, opp_mean, opp_log_var);
        per.push((marks.len(), None, skip_mean, skip_log_var, skip_rank));

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
