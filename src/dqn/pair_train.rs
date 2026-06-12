//! Training for the pairwise differential network: per-opponent TD(λ) chains,
//! board-swap sample doubling, decoupled μ/σ loss, pure self-play loop.
//!
//! Only compiled with the `dqn` feature, mirroring `dqn::train`.

use crate::bot::{self, DNA};
use crate::dqn::pair::{pair_features, PairModel, PairModelConfig, PairStrategy, BOARD_FEATURES, PAIR_FEATURES};
use crate::dqn::{MyBackend, LOG_VAR_MAX, LOG_VAR_MIN, TRAIN_SEED};
use crate::state::{Mark, State};
use crate::strategy::bot_impl::{
    active_phase1_choices, active_phase2_choices, eval_decision, find_safe_lock, mark_choices_with,
    opp_best_phase1_score, passive_phase1_impl, phase1_plan_choices_with, Decision,
};
use crate::strategy::search::{
    context_seed, gates, phase1_entry, phase2_entry, rollout_final_states, SearchBot,
};
use crate::strategy::sim::SimGame;
use crate::strategy::{Bot, Strategy};
use burn::{
    backend::Autodiff,
    data::{
        dataloader::{batcher::Batcher, DataLoaderBuilder},
        dataset::InMemDataset,
    },
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{metric::LossMetric, InferenceStep, Learner, RegressionOutput, SupervisedTraining, TrainOutput, TrainStep},
};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rayon::prelude::*;
use std::sync::Arc;

type MyAutodiffBackend = Autodiff<MyBackend>;

const LAMBDA: f32 = 0.8;

/// Distillation configuration (all off ⇒ behavior identical to before).
#[derive(Clone, Copy)]
pub struct DistillCfg {
    pub enabled: bool,
    /// Full-game rollouts per candidate.
    pub k: usize,
    /// Samples emitted per (candidate, opponent) pair at GATED decisions.
    pub m_gated: usize,
    /// Samples emitted per (candidate, opponent) pair at LOCK firings.
    /// Weighted separately from the gated pool: lock firings are ~15x rarer
    /// (~0.55/game vs ~8/game), so a shared m drowns the lock signal —
    /// Phase 18 measured the lock pool at ~1% of the buffer and saw no
    /// lock-calibration effect (lock-ab edge unchanged).
    pub m_lock: usize,
    /// Probability of declining a forced safe lock during generation
    /// (player 0 only, like uniform ε).
    pub epsilon_lock: f32,
}

impl DistillCfg {
    /// The disabled config used by every non-distilling caller.
    pub fn off() -> Self {
        DistillCfg {
            enabled: false,
            k: 32,
            m_gated: 2,
            m_lock: 16,
            epsilon_lock: 0.0,
        }
    }
}

/// Absolute-space TD(λ) targets over one diff chain.
///
/// `mus[t]` is the model's *future-space* μ_diff at step t (used for the
/// bootstrap at t+1), `cdiffs[t]` the current point differential. Returns
/// `G_t` in absolute-diff space:
///
/// ```text
/// G_{n−1} = final_diff
/// G_t     = (1−λ)·(μ_{t+1} + cdiff_{t+1}) + λ·G_{t+1}
/// ```
fn td_diff_targets(mus: &[f32], cdiffs: &[f32], final_diff: f32, lambda: f32) -> Vec<f32> {
    let n = cdiffs.len();
    let mut g = vec![0.0f32; n];
    g[n - 1] = final_diff;
    for t in (0..n - 1).rev() {
        g[t] = (1.0 - lambda) * (mus[t + 1] + cdiffs[t + 1]) + lambda * g[t + 1];
    }
    g
}

// ---- Training samples / loss / batcher ----

/// serde adapter for `[f32; PAIR_FEATURES]` (length 45 exceeds serde's built-in
/// array impls). Round-trips through a `Vec<f32>`.
mod pair_features_serde {
    use super::PAIR_FEATURES;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(arr: &[f32; PAIR_FEATURES], s: S) -> Result<S::Ok, S::Error> {
        arr.as_slice().serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[f32; PAIR_FEATURES], D::Error> {
        let v = Vec::<f32>::deserialize(d)?;
        <[f32; PAIR_FEATURES]>::try_from(v.as_slice()).map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct PairSample {
    // serde's built-in array impls stop at length 32; PAIR_FEATURES is 45, so
    // (de)serialize via a slice helper. (train.rs's TrainingSample escapes this
    // because NUM_FEATURES = 25.)
    #[serde(with = "pair_features_serde")]
    pub features: [f32; PAIR_FEATURES],
    /// TD(λ) target in future-diff space: `G_t − current_diff_t`.
    pub value: f32,
    /// Actual final diff in future space: `final_diff − current_diff_t`.
    /// σ's residual reference (unbiased, bypasses TD smoothing).
    pub final_diff: f32,
}

impl<B: Backend> PairModel<B> {
    /// Decoupled μ/σ loss on the differential — same recipe as the old net:
    /// μ: MSE toward the TD(λ) target; σ: Gaussian NLL against the actual
    /// final-diff residual with μ detached.
    pub fn forward_step(&self, batch: PairBatch<B>) -> RegressionOutput<B> {
        let output = self.forward(batch.inputs);
        let mean = output.clone().narrow(1, 0, 1);
        let log_var = output.narrow(1, 1, 1).clamp(LOG_VAR_MIN, LOG_VAR_MAX);

        let targets = batch.targets.clone().unsqueeze_dim(1);
        let final_diffs = batch.final_diffs.clone().unsqueeze_dim(1);

        let mu_residual = targets.clone() - mean.clone();
        let mu_loss = (mu_residual.clone() * mu_residual).mean();

        let mean_detached = mean.clone().detach();
        let sigma_residual = final_diffs - mean_detached;
        let sigma_sq = sigma_residual.clone() * sigma_residual;
        let inv_var = log_var.clone().neg().exp();
        let sigma_nll = log_var + sigma_sq * inv_var;
        let sigma_loss = sigma_nll.mean().mul_scalar(0.5);

        let loss = mu_loss + sigma_loss;

        RegressionOutput {
            loss,
            output: mean,
            targets,
        }
    }
}

impl<B: AutodiffBackend> TrainStep for PairModel<B> {
    type Input = PairBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, batch: PairBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_step(batch);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for PairModel<B> {
    type Input = PairBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, batch: PairBatch<B>) -> RegressionOutput<B> {
        self.forward_step(batch)
    }
}

#[derive(Clone)]
pub struct PairBatcher<B: Backend> {
    _phantom: std::marker::PhantomData<B>,
}

#[derive(Clone, Debug)]
pub struct PairBatch<B: Backend> {
    pub inputs: Tensor<B, 2>,
    pub targets: Tensor<B, 1>,
    pub final_diffs: Tensor<B, 1>,
}

/// Color-permutation augmentation applied identically to BOTH board blocks.
/// Per-row groups within each block: progress [0–3], marks [4–7],
/// locked [8–11], wprob [12–15]; pair-level features 40–44 are untouched
/// (they are row-permutation invariant).
fn permute_colors(f: &mut [f32; PAIR_FEATURES], swap_ry: bool, swap_gb: bool, swap_pairs: bool) {
    for block in [0, BOARD_FEATURES] {
        if swap_ry {
            for base in [0, 4, 8, 12] {
                f.swap(block + base, block + base + 1);
            }
        }
        if swap_gb {
            for base in [2, 6, 10, 14] {
                f.swap(block + base, block + base + 1);
            }
        }
        if swap_pairs {
            for base in [0, 4, 8, 12] {
                f.swap(block + base, block + base + 2);
                f.swap(block + base + 1, block + base + 3);
            }
        }
    }
}

impl<B: Backend> Batcher<B, PairSample, PairBatch<B>> for PairBatcher<B> {
    fn batch(&self, items: Vec<PairSample>, device: &B::Device) -> PairBatch<B> {
        let batch_size = items.len();
        let batch_seed = TRAIN_SEED
            .wrapping_add(items[0].value.to_bits() as u64)
            .wrapping_add(batch_size as u64);
        let mut rng = SmallRng::seed_from_u64(batch_seed);

        let inputs: Vec<f32> = items
            .iter()
            .flat_map(|s| {
                let mut f = s.features;
                permute_colors(&mut f, rng.gen(), rng.gen(), rng.gen());
                f
            })
            .collect();
        let targets: Vec<f32> = items.iter().map(|s| s.value).collect();
        let final_diffs: Vec<f32> = items.iter().map(|s| s.final_diff).collect();

        let inputs = Tensor::<B, 1>::from_floats(inputs.as_slice(), device).reshape([batch_size, PAIR_FEATURES]);
        let targets = Tensor::<B, 1>::from_floats(targets.as_slice(), device);
        let final_diffs = Tensor::<B, 1>::from_floats(final_diffs.as_slice(), device);

        PairBatch {
            inputs,
            targets,
            final_diffs,
        }
    }
}

// ---- Self-play recording ----

/// One recorded decision: our post-decision state + all opponents' states at
/// that moment (turn-ordered relative to us, constant order per game).
type Snapshot = (State, Vec<State>);

/// Move-selection policy for training-game players: today's static net, or
/// the shipped search bot (expert iteration). The ε-coin in `RecordingPair`
/// fires BEFORE the policy, so exploring decisions never pay for (or get
/// polished by) search.
enum PairPolicy {
    Static(PairStrategy),
    Search(SearchBot<PairStrategy>),
}

impl PairPolicy {
    fn bot(&self) -> &PairStrategy {
        match self {
            PairPolicy::Static(b) => b,
            PairPolicy::Search(s) => &s.bot,
        }
    }

    fn active_phase1(&mut self, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
        match self {
            PairPolicy::Static(b) => crate::strategy::active_phase1_impl(&*b, state, opp_states, dice),
            PairPolicy::Search(s) => crate::strategy::Strategy::active_phase1(s, state, opp_states, dice),
        }
    }

    fn active_phase2(&mut self, state: &State, opp_states: &[State], dice: [u8; 6], has_marked: bool) -> Option<Mark> {
        match self {
            PairPolicy::Static(b) => crate::strategy::active_phase2_impl(&*b, state, opp_states, dice, has_marked),
            PairPolicy::Search(s) => crate::strategy::Strategy::active_phase2(s, state, opp_states, dice, has_marked),
        }
    }
}

/// Pair-bot wrapper used during self-play training. ε-greedy on active
/// decisions, greedy on passive; records snapshots for chain building.
/// Move selection runs through a [`PairPolicy`] — either the static net or
/// the search bot; the recording cadence is identical in both. Recording
/// cadence: active turns once after phase 2 (post-turn state); passive turns
/// after every real decision — including skips, which the old recorder
/// dropped (skip afterstates are evaluated at inference, so they belong in
/// the training distribution).
/// A decision worth distilling, captured during play; rollouts happen
/// post-game.
enum DistillCtx {
    /// Gated (close/endgame) active decision: top-2 candidates by static value.
    /// `sim_opp` is the evaluation context (ap1: simulated post-phase1 opps;
    /// ap2: current opps). `cands` = (mark, post/end state) sorted desc by value.
    Gated {
        phase: u8,
        state: State,
        opps: Vec<State>,
        sim_opp: Vec<State>,
        dice: [u8; 6],
        cands: Vec<(Option<Mark>, State)>,
    },
    /// Safe-lock firing: the rule-free candidate list (lock + alternatives);
    /// the lock / best-non-lock / runner-up-lock trio is selected at emission.
    Lock {
        ctx: LockCtx,
        state: State,
        opps: Vec<State>,
        sim_opp: Vec<State>,  // ap1 only; empty otherwise
        dice: [u8; 6],
        active_player: usize, // pp1 only (opp-relative index of the active player); 0 otherwise
        cands: Vec<(Option<Mark>, State)>,
        lock: Mark,
    },
}

#[derive(Clone, Copy, PartialEq)]
enum LockCtx {
    Ap1,
    Ap2,
    Pp1,
}

struct RecordingPair {
    policy: PairPolicy,
    epsilon: f32,
    rng: SmallRng,
    recorded: std::rc::Rc<std::cell::RefCell<Vec<Snapshot>>>,
    /// Player-0-only distill capture buffer (None ⇒ no capture).
    distill: Option<std::rc::Rc<std::cell::RefCell<Vec<DistillCtx>>>>,
    /// Probability of declining a forced safe lock (player 0 only).
    epsilon_lock: f32,
}

impl std::fmt::Debug for RecordingPair {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "RecordingPair")
    }
}

/// Collapse phase-1 plans to distinct phase-1 marks (best static value per
/// mark wins, keeping that plan's end-state), then sort descending by value.
/// Mirrors the collapse loop in `SearchBot::active_phase1`.
fn collapse_plans(plans: &[(Option<Mark>, State)], values: &[f32]) -> Vec<(Option<Mark>, State, f32)> {
    let mut cands: Vec<(Option<Mark>, State, f32)> = Vec::new();
    for (i, (m, s)) in plans.iter().enumerate() {
        match cands.iter_mut().find(|c| c.0 == *m) {
            Some(c) if values[i] > c.2 => {
                c.1 = *s;
                c.2 = values[i];
            }
            Some(_) => {}
            None => cands.push((*m, *s, values[i])),
        }
    }
    cands.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    cands
}

/// Pair each choice with its static value and sort descending (no collapse;
/// for ap2/pp1 where each choice is already a distinct mark).
fn value_sort(choices: &[(Option<Mark>, State)], values: &[f32]) -> Vec<(Option<Mark>, State, f32)> {
    let mut cands: Vec<(Option<Mark>, State, f32)> = choices
        .iter()
        .zip(values)
        .map(|((m, s), &v)| (*m, *s, v))
        .collect();
    cands.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    cands
}

impl RecordingPair {
    /// True with probability `epsilon_lock` — decline a forced safe lock.
    fn decline_lock(&mut self) -> bool {
        self.epsilon_lock > 0.0 && self.rng.gen::<f32>() < self.epsilon_lock
    }
}

impl Strategy for RecordingPair {
    fn active_phase1(&mut self, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
        // ε-uniform branch bypasses both search and distill capture (as today).
        if self.rng.gen::<f32>() < self.epsilon {
            let white_marks = state.generate_white_moves(dice[0] + dice[1]);
            if white_marks.is_empty() {
                return None;
            }
            let idx = self.rng.gen_range(0..=white_marks.len());
            return if idx < white_marks.len() {
                Some(white_marks[idx])
            } else {
                None
            };
        }

        // Distill detection (player 0 only). Uses the static pipeline directly
        // so capture is independent of the configured move policy. On a lock
        // firing, may stage a rule-free candidate set for ε-decline (resolved
        // after the immutable `bot` borrow ends).
        let mut decline_rf: Option<(Vec<(Option<Mark>, State)>, Vec<State>)> = None;
        if self.distill.is_some() {
            let bot = self.policy.bot();
            let (decision, sim_opp) = active_phase1_choices(bot, state, opp_states, dice);
            match decision {
                Decision::Choices(plans) => {
                    // Gated capture: collapse to distinct phase-1 marks, top-2.
                    let states: Vec<State> = plans.iter().map(|(_, s)| *s).collect();
                    let values = bot.evaluate_batch(&states, &sim_opp);
                    let cands = collapse_plans(&plans, &values);
                    if cands.len() >= 2 {
                        let value_posts: Vec<(f32, State)> = cands.iter().map(|c| (c.2, c.1)).collect();
                        let (close, endgame) = gates(&value_posts, state, opp_states);
                        if (close || endgame) && !opp_states.is_empty() {
                            self.distill.as_ref().unwrap().borrow_mut().push(DistillCtx::Gated {
                                phase: 1,
                                state: *state,
                                opps: opp_states.to_vec(),
                                sim_opp: sim_opp.clone(),
                                dice,
                                cands: cands[..2].iter().map(|c| (c.0, c.1)).collect(),
                            });
                        }
                    }
                }
                Decision::Forced(Some(m)) => {
                    // Lock firing: production forces a safe lock that doesn't end
                    // the game. Capture the rule-free candidate set for emission.
                    let locks_safe = state.would_lock_row(m) && {
                        let mut s = *state;
                        s.apply_mark(m);
                        !s.would_end_game()
                    };
                    if locks_safe && !opp_states.is_empty() {
                        if let Decision::Choices(plans) = phase1_plan_choices_with(state, &sim_opp, dice, false) {
                            let states: Vec<State> = plans.iter().map(|(_, s)| *s).collect();
                            let values = bot.evaluate_batch(&states, &sim_opp);
                            let cands = collapse_plans(&plans, &values);
                            // Need the lock plus at least one distinct alternative.
                            if cands.len() >= 2 && cands.iter().any(|c| c.0 == Some(m)) {
                                let rf: Vec<(Option<Mark>, State)> = cands.iter().map(|c| (c.0, c.1)).collect();
                                self.distill.as_ref().unwrap().borrow_mut().push(DistillCtx::Lock {
                                    ctx: LockCtx::Ap1,
                                    state: *state,
                                    opps: opp_states.to_vec(),
                                    sim_opp: sim_opp.clone(),
                                    dice,
                                    active_player: 0,
                                    cands: rf.clone(),
                                    lock: m,
                                });
                                decline_rf = Some((rf, sim_opp));
                            }
                        }
                    }
                }
                Decision::Forced(None) => {}
            }
        }

        // ε-decline: at a captured firing, play the value-best rule-free move.
        if let Some((rf, sim_opp)) = decline_rf {
            if self.decline_lock() {
                let bot = self.policy.bot();
                return eval_decision(bot, Decision::Choices(rf), &sim_opp);
            }
        }

        self.policy.active_phase1(state, opp_states, dice)
    }

    fn active_phase2(&mut self, state: &State, opp_states: &[State], dice: [u8; 6], has_marked: bool) -> Option<Mark> {
        let marks = state.generate_color_moves(dice);
        let no_mark_state = if has_marked {
            *state
        } else {
            let mut s = *state;
            s.apply_strike();
            s
        };

        if marks.is_empty() {
            self.recorded.borrow_mut().push((no_mark_state, opp_states.to_vec()));
            return None;
        }

        // ε-uniform branch: bypass distill capture (matches search bypass).
        if self.rng.gen::<f32>() < self.epsilon {
            let idx = self.rng.gen_range(0..=marks.len());
            let mark = if idx < marks.len() { Some(marks[idx]) } else { None };
            let chosen_state = match mark {
                Some(m) => {
                    let mut s = *state;
                    s.apply_mark(m);
                    s
                }
                None => no_mark_state,
            };
            self.recorded.borrow_mut().push((chosen_state, opp_states.to_vec()));
            return mark;
        }

        // Distill detection (player 0 only); sim_opp = current opps for ap2.
        let mut decline_rf: Option<Vec<(Option<Mark>, State)>> = None;
        if self.distill.is_some() {
            let bot = self.policy.bot();
            match active_phase2_choices(state, opp_states, dice, has_marked) {
                Decision::Choices(choices) => {
                    let states: Vec<State> = choices.iter().map(|(_, s)| *s).collect();
                    let values = bot.evaluate_batch(&states, opp_states);
                    let cands = value_sort(&choices, &values);
                    if cands.len() >= 2 {
                        let value_posts: Vec<(f32, State)> = cands.iter().map(|c| (c.2, c.1)).collect();
                        let (close, endgame) = gates(&value_posts, state, opp_states);
                        if (close || endgame) && !opp_states.is_empty() {
                            self.distill.as_ref().unwrap().borrow_mut().push(DistillCtx::Gated {
                                phase: 2,
                                state: *state,
                                opps: opp_states.to_vec(),
                                sim_opp: opp_states.to_vec(),
                                dice,
                                cands: cands[..2].iter().map(|c| (c.0, c.1)).collect(),
                            });
                        }
                    }
                }
                Decision::Forced(Some(m)) => {
                    let is_lock = find_safe_lock(state, &marks) == Some(m);
                    if is_lock && !opp_states.is_empty() {
                        let opp_best = opp_states.iter().map(|s| s.count_points()).max().unwrap_or(0);
                        if let Decision::Choices(choices) =
                            mark_choices_with(state, &marks, no_mark_state, opp_best, false)
                        {
                            let states: Vec<State> = choices.iter().map(|(_, s)| *s).collect();
                            let values = bot.evaluate_batch(&states, opp_states);
                            let cands = value_sort(&choices, &values);
                            if cands.len() >= 2 && cands.iter().any(|c| c.0 == Some(m)) {
                                let rf: Vec<(Option<Mark>, State)> = cands.iter().map(|c| (c.0, c.1)).collect();
                                self.distill.as_ref().unwrap().borrow_mut().push(DistillCtx::Lock {
                                    ctx: LockCtx::Ap2,
                                    state: *state,
                                    opps: opp_states.to_vec(),
                                    sim_opp: Vec::new(),
                                    dice,
                                    active_player: 0,
                                    cands: rf.clone(),
                                    lock: m,
                                });
                                decline_rf = Some(rf);
                            }
                        }
                    }
                }
                Decision::Forced(None) => {}
            }
        }

        // ε-decline resolved after the `bot` borrow ends.
        let forced_override: Option<Option<Mark>> = match decline_rf {
            Some(rf) if self.decline_lock() => {
                let bot = self.policy.bot();
                Some(eval_decision(bot, Decision::Choices(rf), opp_states))
            }
            _ => None,
        };

        let mark = match forced_override {
            Some(m) => m,
            None => self.policy.active_phase2(state, opp_states, dice, has_marked),
        };

        let chosen_state = match mark {
            Some(m) => {
                let mut s = *state;
                s.apply_mark(m);
                s
            }
            None => no_mark_state,
        };
        self.recorded.borrow_mut().push((chosen_state, opp_states.to_vec()));
        mark
    }

    fn passive_phase1(
        &mut self,
        state: &State,
        opp_states: &[State],
        dice: [u8; 6],
        active_player: usize,
    ) -> Option<Mark> {
        let white_sum = dice[0] + dice[1];
        let marks = state.generate_white_moves(white_sum);
        if marks.is_empty() {
            return None;
        }

        let mut mark = passive_phase1_impl(self.policy.bot(), state, opp_states, dice);

        // Distill detection (player 0 only). No Gated capture on passive
        // decisions (search never gated them); Lock firings only.
        let mut decline_rf: Option<Vec<(Option<Mark>, State)>> = None;
        if self.distill.is_some() {
            let bot = self.policy.bot();
            let lock = find_safe_lock(state, &marks);
            if let Some(lock) = lock.filter(|_| !opp_states.is_empty()) {
                debug_assert_eq!(mark, Some(lock), "production must force the detected safe lock");
                let opp_best = opp_best_phase1_score(opp_states, white_sum);
                if let Decision::Choices(choices) = mark_choices_with(state, &marks, *state, opp_best, false) {
                    let states: Vec<State> = choices.iter().map(|(_, s)| *s).collect();
                    let values = bot.evaluate_batch(&states, opp_states);
                    let cands = value_sort(&choices, &values);
                    if cands.len() >= 2 && cands.iter().any(|c| c.0 == Some(lock)) {
                        let rf: Vec<(Option<Mark>, State)> = cands.iter().map(|c| (c.0, c.1)).collect();
                        self.distill.as_ref().unwrap().borrow_mut().push(DistillCtx::Lock {
                            ctx: LockCtx::Pp1,
                            state: *state,
                            opps: opp_states.to_vec(),
                            sim_opp: Vec::new(),
                            dice,
                            active_player,
                            cands: rf.clone(),
                            lock,
                        });
                        decline_rf = Some(rf);
                    }
                }
            }
        }

        // ε-decline resolved after the `bot` borrow ends.
        if let Some(rf) = decline_rf {
            if self.decline_lock() {
                let bot = self.policy.bot();
                mark = eval_decision(bot, Decision::Choices(rf), opp_states);
            }
        }

        let post = match mark {
            Some(m) => {
                let mut s = *state;
                s.apply_mark(m);
                s
            }
            None => *state, // record skips too
        };
        self.recorded.borrow_mut().push((post, opp_states.to_vec()));
        mark
    }
}

/// Is `mark` a safe lock from `state` (locks a row without ending the game)?
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

/// Complete the WHOLE current turn deterministically for a passive-phase-1
/// candidate, in the entry's `[us, opps...]` frame.
///
/// Frame mapping (verified against `Game::play`): the game passes player i's
/// opponents as `opp_states = [pre_phase1[(i+1)%n], ..]` (turn-ordered from i)
/// and `active_player` = `(active + n − i) % n − 1` = the index of the active
/// player *within that opp_states array* (relative-to-receiver, 0-based into
/// opps), NOT an absolute seat. So in the entry `[us, opps...]`, the active
/// player sits at entry index `active_player + 1`, and the player who acts
/// first in the rollout (the seat after the active one) is at entry index
/// `(active_player + 2) % n` in our-relative frame.
///
/// Approximation (identical across candidates, which is all that matters for
/// the contrast): the real phase-1 was simultaneous; here our candidate mark
/// is applied first, then every OTHER passive player's phase 1, then the
/// active player's phase 1 + phase 2, with lock propagation between steps.
fn pp1_entry(
    bot: &PairStrategy,
    state: &State,
    opps: &[State],
    dice: [u8; 6],
    active_player: usize,
    mark: Option<Mark>,
) -> (Vec<State>, bool) {
    let n = opps.len() + 1;
    // entry index of the active player (us = 0, opps start at 1).
    let active_idx = active_player + 1;

    let mut all: Vec<State> = std::iter::once(*state).chain(opps.iter().copied()).collect();
    if let Some(m) = mark {
        all[0].apply_mark(m);
    }
    SimGame::propagate_locks(&mut all);
    if SimGame::game_over(&all) {
        return (all, true);
    }

    // Other passive players' (simultaneous) phase 1 — every entry seat that is
    // neither us (0) nor the active player. We apply sequentially with lock
    // propagation; this is the same approximation search makes for sim_opp.
    for j in 1..n {
        if j == active_idx {
            continue;
        }
        let view = SimGame::opp_view(&all, j);
        if let Some(m) = passive_phase1_impl(bot, &all[j], &view, dice) {
            all[j].apply_mark(m);
        }
    }
    SimGame::propagate_locks(&mut all);
    if SimGame::game_over(&all) {
        return (all, true);
    }

    // Active player's phase 1.
    let view = SimGame::opp_view(&all, active_idx);
    let (d, _) = active_phase1_choices(bot, &all[active_idx], &view, dice);
    let p1 = eval_decision(bot, d, &view);
    if let Some(m) = p1 {
        all[active_idx].apply_mark(m);
    }
    SimGame::propagate_locks(&mut all);
    if SimGame::game_over(&all) {
        return (all, true);
    }

    // Active player's phase 2.
    let view = SimGame::opp_view(&all, active_idx);
    let d = active_phase2_choices(&all[active_idx], &view, dice, p1.is_some());
    match eval_decision(bot, d, &view) {
        Some(m) => all[active_idx].apply_mark(m),
        None if p1.is_none() => all[active_idx].apply_strike(),
        None => {}
    }
    SimGame::propagate_locks(&mut all);
    let ended = SimGame::game_over(&all);
    (all, ended)
}

/// Emit distill samples for one game's captured contexts. Per candidate:
/// complete the turn deterministically (entries), run K full-game CRN
/// rollouts, and emit `m_gated`/`m_lock` samples per opponent pairing —
/// value = mean future-diff, final_diff = an individual rollout's future-diff
/// — plus the swap-doubled negated sample, matching `build_pair_samples`'
/// conventions. Returns `(samples, lock_pool_sample_count)`.
fn build_distill_samples(
    boot: &PairStrategy,
    ctxs: &[DistillCtx],
    cfg: DistillCfg,
    seed: u64,
) -> (Vec<PairSample>, usize) {
    let mut samples = Vec::new();
    let mut lock_samples = 0usize;

    for ctx in ctxs {
        let (m, is_lock_ctx) = match ctx {
            DistillCtx::Gated { .. } => (cfg.m_gated, false),
            DistillCtx::Lock { .. } => (cfg.m_lock, true),
        };
        let before = samples.len();
        // (compared candidate marks+posts, entries, first_active, rollout seed)
        let (compared, entries, first_active, roll_seed): (
            Vec<(Option<Mark>, State)>,
            Vec<(Vec<State>, bool)>,
            usize,
            u64,
        ) = match ctx {
            DistillCtx::Gated {
                phase,
                state,
                opps,
                sim_opp,
                dice,
                cands,
            } => {
                let n = opps.len() + 1;
                let entries: Vec<(Vec<State>, bool)> = if *phase == 1 {
                    cands
                        .iter()
                        .map(|(m, _)| phase1_entry(boot, state, sim_opp, *dice, *m))
                        .collect()
                } else {
                    cands.iter().map(|(_, post)| phase2_entry(*post, opps)).collect()
                };
                (cands.clone(), entries, 1 % n, context_seed(state, opps, *dice))
            }
            DistillCtx::Lock {
                ctx: lctx,
                state,
                opps,
                sim_opp,
                dice,
                active_player,
                cands,
                lock,
            } => {
                // Re-evaluate + sort the rule-free candidates at emission time,
                // then select the lock / best-non-lock / runner-up-lock trio
                // (Phase 16 `build_lock_cands` semantics).
                let eval_opps: &[State] = match lctx {
                    LockCtx::Ap1 => sim_opp,
                    _ => opps,
                };
                let states: Vec<State> = cands.iter().map(|(_, s)| *s).collect();
                let values = boot.evaluate_batch(&states, eval_opps);
                let sorted = value_sort(cands, &values);

                let lock_i = sorted.iter().position(|c| c.0 == Some(*lock));
                let lock_i = match lock_i {
                    Some(i) => i,
                    None => continue, // lock pruned out — skip
                };
                let alt_i = (0..sorted.len()).find(|&i| !is_safe_lock(state, sorted[i].0));
                let alt_i = match alt_i {
                    Some(i) => i,
                    None => continue, // all candidates are safe locks — skip
                };
                let alt2_i = (0..sorted.len()).find(|&i| i != lock_i && is_safe_lock(state, sorted[i].0));

                let mut compared: Vec<(Option<Mark>, State)> =
                    vec![(sorted[lock_i].0, sorted[lock_i].1), (sorted[alt_i].0, sorted[alt_i].1)];
                if let Some(i2) = alt2_i {
                    compared.push((sorted[i2].0, sorted[i2].1));
                }

                let (entries, first_active, n) = match lctx {
                    LockCtx::Ap1 => {
                        let n = opps.len() + 1;
                        let e: Vec<(Vec<State>, bool)> = compared
                            .iter()
                            .map(|(m, _)| phase1_entry(boot, state, sim_opp, *dice, *m))
                            .collect();
                        (e, 1 % n, n)
                    }
                    LockCtx::Ap2 => {
                        let n = opps.len() + 1;
                        let e: Vec<(Vec<State>, bool)> =
                            compared.iter().map(|(_, post)| phase2_entry(*post, opps)).collect();
                        (e, 1 % n, n)
                    }
                    LockCtx::Pp1 => {
                        let n = opps.len() + 1;
                        let e: Vec<(Vec<State>, bool)> = compared
                            .iter()
                            .map(|(m, _)| pp1_entry(boot, state, opps, *dice, *active_player, *m))
                            .collect();
                        // first_active = seat after the active player, our-relative.
                        ((e), (active_player + 2) % n, n)
                    }
                };
                // `n` is intentionally unused here: it is bound per match arm only to
                // compute first_active; the outer tuple takes first_active directly.
                let _ = n;
                (compared, entries, first_active, context_seed(state, opps, *dice))
            }
        };

        if entries.is_empty() {
            continue;
        }
        let finals = rollout_final_states(boot, &entries, roll_seed, cfg.k, first_active);

        for (entry, fin) in entries.iter().zip(&finals) {
            let our = entry.0[0];
            let e_opps: Vec<State> = entry.0[1..].to_vec();
            let num_opps = e_opps.len();
            let ended = entry.1;

            for k in 0..num_opps {
                let cdiff = (our.count_points() - e_opps[k].count_points()) as f32;
                // future-diffs per rollout sample
                let diffs: Vec<f32> = fin
                    .iter()
                    .map(|f| ((f[0].count_points() - f[1 + k].count_points()) as f32) - cdiff)
                    .collect();
                if diffs.is_empty() {
                    continue;
                }
                let value = diffs.iter().sum::<f32>() / diffs.len() as f32;

                let feats = pair_features(&our, &e_opps[k], &e_opps);
                let mut swapped_opps: Vec<State> = Vec::with_capacity(num_opps);
                swapped_opps.push(our);
                swapped_opps.extend(e_opps.iter().enumerate().filter(|(j, _)| *j != k).map(|(_, s)| *s));
                let swap_feats = pair_features(&e_opps[k], &our, &swapped_opps);

                if ended {
                    // Deterministic single outcome: value == final_diff.
                    let fdiff = diffs[0];
                    samples.push(PairSample {
                        features: feats,
                        value: fdiff,
                        final_diff: fdiff,
                    });
                    samples.push(PairSample {
                        features: swap_feats,
                        value: -fdiff,
                        final_diff: -fdiff,
                    });
                } else {
                    for r in 0..m {
                        let fdiff = diffs[r % diffs.len()];
                        samples.push(PairSample {
                            features: feats,
                            value,
                            final_diff: fdiff,
                        });
                        samples.push(PairSample {
                            features: swap_feats,
                            value: -value,
                            final_diff: -fdiff,
                        });
                    }
                }
            }
        }
        // `compared` is intentionally unused after building entries: candidate
        // identity is implicit in entry order; sample emission reads entry states.
        let _ = compared;
        if is_lock_ctx {
            lock_samples += samples.len() - before;
        }
    }

    // `seed` is intentionally unused: rollouts reseed per-ctx via context_seed
    // (roll_seed) so each decision's CRN rollouts are deterministic on its own
    // board, independent of the per-game seed.
    let _ = seed;
    (samples, lock_samples)
}

/// Build training samples from one player's trajectory: one TD(λ) chain per
/// opponent, every sample emitted in both board orders (swap doubling) with
/// pair-level features recomputed exactly from the swapped perspective and
/// targets negated.
fn build_pair_samples(
    net: &crate::dqn::pair::ManualPairNet,
    snapshots: &[Snapshot],
    our_final: f32,
    opp_finals: &[f32],
) -> Vec<PairSample> {
    let mut samples = Vec::new();
    if snapshots.is_empty() {
        return samples;
    }
    let num_opps = snapshots[0].1.len();
    debug_assert_eq!(num_opps, opp_finals.len());

    for k in 0..num_opps {
        let final_diff = our_final - opp_finals[k];
        let feats: Vec<[f32; PAIR_FEATURES]> = snapshots
            .iter()
            .map(|(our, opps)| pair_features(our, &opps[k], opps))
            .collect();
        let cdiffs: Vec<f32> = snapshots
            .iter()
            .map(|(our, opps)| (our.count_points() - opps[k].count_points()) as f32)
            .collect();
        let mus: Vec<f32> = net.forward(&feats).into_iter().map(|(m, _)| m).collect();
        let g = td_diff_targets(&mus, &cdiffs, final_diff, LAMBDA);

        for (t, (our, opps)) in snapshots.iter().enumerate() {
            let value = g[t] - cdiffs[t];
            let fdiff = final_diff - cdiffs[t];
            samples.push(PairSample {
                features: feats[t],
                value,
                final_diff: fdiff,
            });

            // Swapped sample: the paired opponent's perspective. Their
            // opponents are us plus the remaining opponents.
            let mut swapped_opps: Vec<State> = Vec::with_capacity(num_opps);
            swapped_opps.push(*our);
            swapped_opps.extend(opps.iter().enumerate().filter(|(j, _)| *j != k).map(|(_, s)| *s));
            samples.push(PairSample {
                features: pair_features(&opps[k], our, &swapped_opps),
                value: -value,
                final_diff: -fdiff,
            });
        }
    }
    samples
}

// ---- Self-play training loop ----

/// Play one pure-self-play training game; every player is a recording pair
/// bot (player 0 explores with ε, the rest are greedy). Returns all samples
/// plus player 0's final score.
fn play_training_game(
    model: &PairModel<MyBackend>,
    device: &burn::backend::ndarray::NdArrayDevice,
    num_opponents: usize,
    search: bool,
    epsilon: f32,
    seed: u64,
    distill: DistillCfg,
) -> (Vec<PairSample>, f32, usize, usize) {
    use crate::game::{Game, Player};

    let n = num_opponents + 1;
    let strategies: Vec<PairStrategy> = (0..n)
        .map(|_| PairStrategy::from_model(model.clone(), device.clone()))
        .collect();
    let boot_net = strategies[0].net.clone();

    // Player-0-only distill capture buffer (shared with that RecordingPair).
    let distill_buf = if distill.enabled {
        Some(std::rc::Rc::new(std::cell::RefCell::new(Vec::new())))
    } else {
        None
    };

    let mut buffers = Vec::with_capacity(n);
    let mut players: Vec<Player> = Vec::with_capacity(n);
    for (i, strategy) in strategies.into_iter().enumerate() {
        let policy = if search {
            PairPolicy::Search(SearchBot::new(strategy))
        } else {
            PairPolicy::Static(strategy)
        };
        let buf = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        buffers.push(std::rc::Rc::clone(&buf));
        players.push(Player::new(
            Box::new(RecordingPair {
                policy,
                epsilon: if i == 0 { epsilon } else { 0.0 },
                rng: SmallRng::seed_from_u64(seed.wrapping_add(100 + i as u64)),
                recorded: buf,
                distill: if i == 0 { distill_buf.clone() } else { None },
                epsilon_lock: if i == 0 { distill.epsilon_lock } else { 0.0 },
            }),
            Box::new(SmallRng::seed_from_u64(seed.wrapping_add(i as u64))),
        ));
    }

    let mut game = Game::new(players);
    game.play();

    let finals: Vec<f32> = game.players.iter().map(|p| p.state.count_points() as f32).collect();

    let mut all_samples = Vec::new();
    for (i, buf) in buffers.iter().enumerate() {
        let snapshots = std::mem::take(&mut *buf.borrow_mut());
        // Opponent k of player i is player (i + 1 + k) % n — matches the
        // turn-ordered opp_states the game loop passes to strategies.
        let opp_finals: Vec<f32> = (1..n).map(|off| finals[(i + off) % n]).collect();
        all_samples.extend(build_pair_samples(&boot_net, &snapshots, finals[i], &opp_finals));
    }

    // Post-game distill emission for player 0's captured contexts.
    let mut distill_count = 0;
    let mut lock_count = 0;
    if let Some(buf) = &distill_buf {
        let ctxs = std::mem::take(&mut *buf.borrow_mut());
        if !ctxs.is_empty() {
            // The boot net (player 0's net) drives rollouts and value re-eval.
            let boot = PairStrategy::from_model(model.clone(), device.clone());
            let (ds, locks) = build_distill_samples(&boot, &ctxs, distill, seed);
            distill_count = ds.len();
            lock_count = locks;
            all_samples.extend(ds);
        }
    }

    (all_samples, finals[0], distill_count, lock_count)
}

/// Same fixed-game-set paired benchmark as `train::benchmark_vs_ga`, for the
/// pair bot. Seat-swapped pairs share per-seat dice streams.
const BENCH_SEED: u64 = 0xB54C;

fn benchmark_vs_ga(artifact_dir: &str, champion: &DNA, num_games: usize) -> f64 {
    use crate::game::{Game, Player};

    let wins: u32 = (0..num_games)
        .into_par_iter()
        .map_init(
            || PairStrategy::load(artifact_dir),
            |template, i| {
                let bot = PairStrategy::from_shared(template.model.clone(), template.device.clone());
                let pair = (i / 2) as u64;
                let rotation = i % 2;
                let seat_dice = |seat: u64| Box::new(SmallRng::seed_from_u64(BENCH_SEED.wrapping_add(pair * 2 + seat)));
                let players: Vec<Player> = if rotation == 0 {
                    vec![
                        Player::new(Box::new(bot), seat_dice(0)),
                        Player::new(Box::new(champion.clone()), seat_dice(1)),
                    ]
                } else {
                    vec![
                        Player::new(Box::new(champion.clone()), seat_dice(0)),
                        Player::new(Box::new(bot), seat_dice(1)),
                    ]
                };
                let mut game = Game::new(players);
                game.play();
                let scores: Vec<isize> = game.players.iter().map(|p| p.state.count_points()).collect();
                let idx = rotation;
                if scores[idx] > scores[1 - idx] {
                    1u32
                } else {
                    0u32
                }
            },
        )
        .sum();
    wins as f64 / num_games as f64
}

pub fn self_play_train(
    artifact_dir: &str,
    num_iterations: usize,
    games_per_iteration: usize,
    epochs_per_iteration: usize,
    bench_games: usize,
    checkpoints: bool,
    start_iteration: usize,
    search: bool,
    distill: DistillCfg,
) {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    MyBackend::seed(&device, TRAIN_SEED);
    std::fs::create_dir_all(artifact_dir).ok();
    let buffer_iterations = 3;
    let mut replay_buffer: std::collections::VecDeque<Vec<PairSample>> = std::collections::VecDeque::new();

    let scores_log_path = format!("{artifact_dir}/training_scores.csv");
    // Fresh runs reset the log; resumed runs (--start-iteration) keep an
    // existing one but must still create it when absent — the appends below
    // would otherwise fail silently for the whole run.
    if start_iteration == 0 || !std::path::Path::new(&scores_log_path).exists() {
        std::fs::write(&scores_log_path, "iteration,avg_score,winrate\n").ok();
    }

    let genes = Arc::new(bot::default_genes());
    let champion = DNA::load_weights("champion.txt", genes).expect("No champion.txt");
    let start_time = std::time::Instant::now();

    let mut iteration_stats: Vec<(usize, f32, Option<f64>)> = Vec::new();

    let mut model: PairModel<MyBackend> = PairModelConfig::new()
        .init::<MyBackend>(&device)
        .load_file(format!("{artifact_dir}/model"), &CompactRecorder::new(), &device)
        .unwrap_or_else(|_| {
            println!("  No pretrained pair model, starting fresh");
            PairModelConfig::new().init::<MyBackend>(&device)
        });

    if search {
        println!(
            "Expert iteration: generation uses the search bot (K={}) for all players",
            crate::strategy::search::K_SAMPLES
        );
    }
    if distill.enabled {
        println!(
            "Distillation: emitting rollout-value targets (K={}, m_gated={}, m_lock={}, epsilon_lock={:.3})",
            distill.k, distill.m_gated, distill.m_lock, distill.epsilon_lock
        );
    }

    for iteration in 0..num_iterations {
        let global_iter = start_iteration + iteration;
        let epsilon = (0.2 * (0.95f32).powi(global_iter as i32)).max(0.07);
        println!("\n=== Pair iteration {} (epsilon={epsilon:.3}) ===", global_iter + 1);

        let games_each = games_per_iteration / 3;
        // Pure self-play thirds: 1v1, 3p, 4p.
        let game_configs: Vec<usize> = [1usize, 2, 3]
            .iter()
            .flat_map(|&num_opps| std::iter::repeat(num_opps).take(games_each))
            .collect();

        let models: Vec<PairModel<MyBackend>> = (0..game_configs.len()).map(|_| model.clone()).collect();

        let game_results: Vec<(Vec<PairSample>, f32, usize, usize)> = game_configs
            .into_par_iter()
            .zip(models.into_par_iter())
            .enumerate()
            .map(|(game_idx, (num_opps, thread_model))| {
                let seed = TRAIN_SEED.wrapping_add((iteration * games_per_iteration + game_idx) as u64);
                play_training_game(&thread_model, &device, num_opps, search, epsilon, seed, distill)
            })
            .collect();

        let game_scores: Vec<f32> = game_results.iter().map(|(_, score, _, _)| *score).collect();
        let distill_samples: usize = game_results.iter().map(|(_, _, d, _)| *d).sum();
        let lock_pool_samples: usize = game_results.iter().map(|(_, _, _, l)| *l).sum();
        let new_samples: Vec<PairSample> = game_results.into_iter().flat_map(|(s, _, _, _)| s).collect();
        let avg_score = if game_scores.is_empty() {
            0.0
        } else {
            game_scores.iter().sum::<f32>() / game_scores.len() as f32
        };

        replay_buffer.push_back(new_samples);
        if replay_buffer.len() > buffer_iterations {
            replay_buffer.pop_front();
        }

        let all_samples: Vec<PairSample> = replay_buffer.iter().flatten().copied().collect();
        if distill.enabled {
            println!(
                "  Generated {} new samples (avg score: {avg_score:.1}), of which {} distill ({} lock-pool), replay buffer: {} total",
                replay_buffer.back().unwrap().len(),
                distill_samples,
                lock_pool_samples,
                all_samples.len()
            );
        } else {
            println!(
                "  Generated {} new samples (avg score: {avg_score:.1}), replay buffer: {} total",
                replay_buffer.back().unwrap().len(),
                all_samples.len()
            );
        }

        model = train_with_epochs(all_samples, artifact_dir, epochs_per_iteration, 4e-4);

        if checkpoints {
            let src = format!("{artifact_dir}/model.mpk");
            let dst = format!("{artifact_dir}/iter-{}.mpk", global_iter + 1);
            if let Err(e) = std::fs::copy(&src, &dst) {
                eprintln!("  Failed to save iter-{} checkpoint: {e}", global_iter + 1);
            }
        }

        let elapsed = start_time.elapsed().as_secs();
        let (mins, secs) = (elapsed / 60, elapsed % 60);

        use std::io::Write;
        let winrate_opt = if bench_games > 0 {
            let winrate = benchmark_vs_ga(artifact_dir, &champion, bench_games);
            println!(
                "  Iteration {:>3}: avg score {:.1}, winrate {:.1}%, elapsed {}m{}s",
                global_iter + 1,
                avg_score,
                winrate * 100.0,
                mins,
                secs,
            );
            if let Ok(mut f) = std::fs::OpenOptions::new().append(true).create(true).open(&scores_log_path) {
                writeln!(f, "{},{avg_score:.2},{:.2}", global_iter + 1, winrate * 100.0).ok();
            }
            Some(winrate)
        } else {
            println!(
                "  Iteration {:>3}: avg score {:.1}, elapsed {}m{}s",
                global_iter + 1,
                avg_score,
                mins,
                secs,
            );
            if let Ok(mut f) = std::fs::OpenOptions::new().append(true).create(true).open(&scores_log_path) {
                writeln!(f, "{},{avg_score:.2}", global_iter + 1).ok();
            }
            None
        };

        iteration_stats.push((global_iter + 1, avg_score, winrate_opt));
    }

    println!("\nPair self-play training complete. Model saved to {artifact_dir}/model");

    if bench_games > 0 {
        let best_wr = iteration_stats
            .iter()
            .filter_map(|&(i, s, w)| w.map(|w| (i, s, w)))
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
        let best_score = iteration_stats.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        if let Some((i, s, w)) = best_wr {
            println!("  Best winrate:    iter {i:>3}: {:.1}% (avg score {s:.1})", w * 100.0);
        }
        if let Some(&(i, s, w)) = best_score {
            let wr_str = w.map(|w| format!("{:.1}%", w * 100.0)).unwrap_or_else(|| "-".into());
            println!("  Best avg score:  iter {i:>3}: {s:.1} (winrate {wr_str})");
        }
    }
}

fn train_with_epochs(samples: Vec<PairSample>, artifact_dir: &str, num_epochs: usize, lr: f64) -> PairModel<MyBackend> {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;

    let split = (samples.len() * 9) / 10;
    let train_data = InMemDataset::new(samples[..split].to_vec());
    let valid_data = InMemDataset::new(samples[split..].to_vec());

    let model: PairModel<MyAutodiffBackend> = PairModelConfig::new()
        .init::<MyAutodiffBackend>(&device)
        .load_file(format!("{artifact_dir}/model"), &CompactRecorder::new(), &device)
        .unwrap_or_else(|_| PairModelConfig::new().init::<MyAutodiffBackend>(&device));

    let batcher_train = PairBatcher::<MyAutodiffBackend> {
        _phantom: std::marker::PhantomData,
    };
    let batcher_valid = PairBatcher::<MyBackend> {
        _phantom: std::marker::PhantomData,
    };

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(1024)
        .shuffle(TRAIN_SEED)
        .build(train_data);

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(1024)
        .shuffle(TRAIN_SEED)
        .build(valid_data);

    let ckpt_dir = format!("{artifact_dir}/ckpt");
    std::fs::remove_dir_all(&ckpt_dir).ok();
    std::fs::create_dir_all(&ckpt_dir).ok();

    let training = SupervisedTraining::new(&ckpt_dir, dataloader_train, dataloader_valid)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .num_epochs(num_epochs)
        .summary();

    let result = training.launch(Learner::new(model, AdamConfig::new().init(), lr));

    result
        .model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Failed to save pair model");

    std::fs::remove_dir_all(&ckpt_dir).ok();

    PairModelConfig::new()
        .init::<MyBackend>(&device)
        .load_file(format!("{artifact_dir}/model"), &CompactRecorder::new(), &device)
        .expect("Failed to reload pair model for inference")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn td_diff_targets_hand_computed() {
        // 3 steps, λ = 0.8.
        let mus = [1.0, 2.0, 3.0];
        let cdiffs = [0.0, 5.0, 10.0];
        let g = td_diff_targets(&mus, &cdiffs, 20.0, 0.8);
        // G2 = 20
        // G1 = 0.2·(μ2 + cd2) + 0.8·G2 = 0.2·13 + 16    = 18.6
        // G0 = 0.2·(μ1 + cd1) + 0.8·G1 = 0.2·7 + 14.88  = 16.28
        assert!((g[2] - 20.0).abs() < 1e-5);
        assert!((g[1] - 18.6).abs() < 1e-5);
        assert!((g[0] - 16.28).abs() < 1e-5);
    }

    #[test]
    fn permute_colors_swaps_both_blocks_and_preserves_pair_features() {
        // Distinct values everywhere so swaps are observable.
        let mut f = [0.0f32; PAIR_FEATURES];
        for (i, v) in f.iter_mut().enumerate() {
            *v = i as f32;
        }
        let orig = f;

        permute_colors(&mut f, true, false, false); // red <-> yellow
        for block in [0, BOARD_FEATURES] {
            for base in [0usize, 4, 8, 12] {
                assert_eq!(f[block + base], orig[block + base + 1]);
                assert_eq!(f[block + base + 1], orig[block + base]);
            }
            // green/blue untouched
            for base in [2usize, 6, 10, 14] {
                assert_eq!(f[block + base], orig[block + base]);
            }
            // per-board aggregates untouched
            assert_eq!(f[block + 16..block + 20], orig[block + 16..block + 20]);
        }
        // Pair-level untouched.
        assert_eq!(f[40..45], orig[40..45]);

        // Applying the same swap again restores the original (involution).
        permute_colors(&mut f, true, false, false);
        assert_eq!(f, orig);
    }

    #[test]
    fn permute_colors_covers_gb_and_asc_desc_paths() {
        // Distinct values everywhere so swaps are observable.
        let mut f = [0.0f32; PAIR_FEATURES];
        for (i, v) in f.iter_mut().enumerate() {
            *v = i as f32;
        }
        let orig = f;

        // green <-> blue
        permute_colors(&mut f, false, true, false);
        for block in [0, BOARD_FEATURES] {
            for base in [2usize, 6, 10, 14] {
                assert_eq!(f[block + base], orig[block + base + 1]);
                assert_eq!(f[block + base + 1], orig[block + base]);
            }
            // red/yellow untouched
            for base in [0usize, 4, 8, 12] {
                assert_eq!(f[block + base], orig[block + base]);
            }
            // per-board aggregates untouched
            assert_eq!(f[block + 16..block + 20], orig[block + 16..block + 20]);
        }
        // Pair-level untouched.
        assert_eq!(f[40..45], orig[40..45]);
        // Involution.
        permute_colors(&mut f, false, true, false);
        assert_eq!(f, orig);

        // asc <-> desc pairs (two swaps per base).
        f = orig;
        permute_colors(&mut f, false, false, true);
        for block in [0, BOARD_FEATURES] {
            for base in [0usize, 4, 8, 12] {
                assert_eq!(f[block + base], orig[block + base + 2]);
                assert_eq!(f[block + base + 1], orig[block + base + 3]);
                assert_eq!(f[block + base + 2], orig[block + base]);
                assert_eq!(f[block + base + 3], orig[block + base + 1]);
            }
            // per-board aggregates untouched
            assert_eq!(f[block + 16..block + 20], orig[block + 16..block + 20]);
        }
        // Pair-level untouched.
        assert_eq!(f[40..45], orig[40..45]);
        // Involution.
        permute_colors(&mut f, false, false, true);
        assert_eq!(f, orig);
    }

    #[test]
    fn static_policy_dispatch_matches_impl_calls() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let model = PairModelConfig::new().init::<MyBackend>(&device);
        let bot = PairStrategy::from_model(model.clone(), device);
        let mut policy = PairPolicy::Static(PairStrategy::from_shared(bot.model.clone(), bot.device));

        let mut state = State::default();
        state.apply_mark(Mark { row: 1, number: 6 });
        let opps = [State::default()];
        let dice = [3, 4, 2, 3, 5, 1];

        assert_eq!(
            policy.active_phase1(&state, &opps, dice),
            crate::strategy::active_phase1_impl(&bot, &state, &opps, dice)
        );
        assert_eq!(
            policy.active_phase2(&state, &opps, dice, false),
            crate::strategy::active_phase2_impl(&bot, &state, &opps, dice, false)
        );
    }

    #[test]
    fn build_pair_samples_emits_negated_swapped_samples() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let model = PairModelConfig::new().init::<MyBackend>(&device);
        let net = crate::dqn::pair::ManualPairNet::from_model(&model);

        // 2-step 1v1 trajectory with asymmetric boards.
        let mut our1 = State::default();
        our1.apply_mark(crate::state::Mark { row: 0, number: 5 });
        let opp1 = State::default();
        let mut our2 = our1;
        our2.apply_mark(crate::state::Mark { row: 0, number: 7 });
        let mut opp2 = State::default();
        opp2.apply_mark(crate::state::Mark { row: 2, number: 10 });

        let snapshots = vec![(our1, vec![opp1]), (our2, vec![opp2])];
        let samples = build_pair_samples(&net, &snapshots, 30.0, &[20.0]);

        // 2 steps × 1 opponent × 2 orders.
        assert_eq!(samples.len(), 4);
        for pair in samples.chunks(2) {
            let (fwd, swp) = (&pair[0], &pair[1]);
            assert_eq!(swp.value, -fwd.value);
            assert_eq!(swp.final_diff, -fwd.final_diff);
            // Board blocks exchanged.
            assert_eq!(
                fwd.features[..BOARD_FEATURES],
                swp.features[BOARD_FEATURES..2 * BOARD_FEATURES]
            );
            assert_eq!(
                fwd.features[BOARD_FEATURES..2 * BOARD_FEATURES],
                swp.features[..BOARD_FEATURES]
            );
            // cdiff input negated (within clamp range here).
            assert!((fwd.features[40] + swp.features[40]).abs() < 1e-6);
        }
        // Last forward sample's targets: G_{n−1} = final_diff = 10;
        // cdiff at t=1: our 3 pts (2 marks) − opp 1 pt (1 mark) = 2.
        let last_fwd = &samples[2];
        assert!((last_fwd.value - 8.0).abs() < 1e-5);
        assert!((last_fwd.final_diff - 8.0).abs() < 1e-5);
    }

    #[test]
    fn exploring_decisions_never_invoke_search() {
        use crate::game::{Game, Player};
        use crate::strategy::search::{SearchBot, SearchStats};
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let model = PairModelConfig::new().init::<MyBackend>(&device);

        let mut players = Vec::new();
        let mut stats = Vec::new();
        for i in 0..2u64 {
            let mut sb = SearchBot::new(PairStrategy::from_model(model.clone(), device));
            let st = std::rc::Rc::new(std::cell::RefCell::new(SearchStats::default()));
            sb.stats = Some(st.clone());
            stats.push(st);
            players.push(Player::new(
                Box::new(RecordingPair {
                    policy: PairPolicy::Search(sb),
                    epsilon: 1.0, // always explore
                    rng: SmallRng::seed_from_u64(900 + i),
                    recorded: std::rc::Rc::new(std::cell::RefCell::new(Vec::new())),
                    distill: None,
                    epsilon_lock: 0.0,
                }),
                Box::new(SmallRng::seed_from_u64(910 + i)),
            ));
        }
        let mut game = Game::new(players);
        game.play();

        for st in &stats {
            assert_eq!(
                st.borrow().active_decisions,
                0,
                "search ran despite epsilon=1.0 — the ε-coin must fire first"
            );
        }
    }

    #[test]
    fn search_generation_is_deterministic() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let model = PairModelConfig::new().init::<MyBackend>(&device);

        let run = || play_training_game(&model, &device, 1, true, 0.07, 4242, DistillCfg::off());
        let (s1, f1, _, _) = run();
        let (s2, f2, _, _) = run();
        assert_eq!(f1, f2);
        assert_eq!(s1.len(), s2.len());
        for (a, b) in s1.iter().zip(&s2) {
            assert_eq!(a.features, b.features);
            assert_eq!(a.value, b.value);
            assert_eq!(a.final_diff, b.final_diff);
        }
    }

    fn distill_boot() -> PairStrategy {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        PairStrategy::from_model(PairModelConfig::new().init::<MyBackend>(&device), device)
    }

    /// Red 2..6 marked: white sum 12 (Mark{0,12}) is a safe lock (first lock,
    /// game does not end).
    fn lockable() -> State {
        let mut s = State::default();
        for n in 2..=6 {
            s.apply_mark(Mark { row: 0, number: n });
        }
        s
    }

    #[test]
    fn distill_emission_counts_and_semantics() {
        let boot = distill_boot();
        let cfg = DistillCfg {
            enabled: true,
            k: 8,
            m_gated: 4,
            m_lock: 16,
            epsilon_lock: 0.0,
        };

        // A phase-2 gated ctx, 1 opponent, two distinct (non-ended) candidates.
        let our = State::default();
        let opp = State::default();
        let mut post_a = our;
        post_a.apply_mark(Mark { row: 0, number: 5 });
        let mut post_b = our;
        post_b.apply_mark(Mark { row: 1, number: 6 });
        let dice = [3, 2, 4, 5, 1, 6];
        let ctx = DistillCtx::Gated {
            phase: 2,
            state: our,
            opps: vec![opp],
            sim_opp: vec![opp],
            dice,
            cands: vec![(Some(Mark { row: 0, number: 5 }), post_a), (Some(Mark { row: 1, number: 6 }), post_b)],
        };

        let (samples, lock_samples) = build_distill_samples(&boot, std::slice::from_ref(&ctx), cfg, 0);
        assert_eq!(lock_samples, 0, "a gated ctx contributes no lock-pool samples");

        // Count = #candidates(2) × #opponents(1) × m(4) × 2 (swap) = 16.
        assert_eq!(samples.len(), 16, "emission count must match candidates×opp×m×swap");

        // Layout (per emission loop): for each candidate, for each opponent, for
        // each r in 0..m: [forward, swap]. So pairs of (fwd, swap).
        for pair in samples.chunks(2) {
            let (fwd, swp) = (&pair[0], &pair[1]);
            assert!(fwd.value.is_finite() && fwd.final_diff.is_finite());
            assert_eq!(swp.value, -fwd.value, "swap value must be the negation");
            assert_eq!(swp.final_diff, -fwd.final_diff, "swap final_diff must be the negation");
            assert_eq!(
                fwd.features[..BOARD_FEATURES],
                swp.features[BOARD_FEATURES..2 * BOARD_FEATURES],
                "swap exchanges board blocks"
            );
        }

        // Within each candidate's m forward samples: identical value, and the
        // final_diffs reproduce the first m rollout future-diffs exactly.
        let entries = [phase2_entry(post_a, &[opp]), phase2_entry(post_b, &[opp])];
        let finals = rollout_final_states(&boot, &entries, context_seed(&our, &[opp], dice), cfg.k, 1 % 2);
        for (ci, (entry, fin)) in entries.iter().zip(&finals).enumerate() {
            let e_our = entry.0[0];
            let e_opp = entry.0[1];
            let cdiff = (e_our.count_points() - e_opp.count_points()) as f32;
            let diffs: Vec<f32> = fin
                .iter()
                .map(|f| ((f[0].count_points() - f[1].count_points()) as f32) - cdiff)
                .collect();
            let exp_value = diffs.iter().sum::<f32>() / diffs.len() as f32;

            // candidate ci's forward samples are at indices ci*m*2, +2, +4, ...
            let base = ci * cfg.m_gated * 2;
            let fwds: Vec<&PairSample> = (0..cfg.m_gated).map(|r| &samples[base + r * 2]).collect();
            for f in &fwds {
                assert!((f.value - exp_value).abs() < 1e-5, "all m samples share the mean value");
            }
            for (r, f) in fwds.iter().enumerate() {
                assert!((f.final_diff - diffs[r % diffs.len()]).abs() < 1e-5, "final_diff = individual rollout diff");
            }
            // The rollouts produce spread (not a degenerate single value).
            assert!(
                fwds.iter().any(|f| (f.final_diff - exp_value).abs() > 1e-6),
                "final_diffs must vary around the mean (σ reference)"
            );
        }
    }

    #[test]
    fn lock_ctx_uses_m_lock_and_is_counted() {
        let boot = distill_boot();
        let cfg = DistillCfg {
            enabled: true,
            k: 8,
            m_gated: 2,
            m_lock: 5,
            epsilon_lock: 0.0,
        };

        // ap2 firing on the lockable state: red die pair gives red 12.
        let state = lockable();
        let opp = State::default();
        let dice = [6u8, 6, 6, 1, 1, 1];
        let marks = state.generate_color_moves(dice);
        let lock = crate::strategy::bot_impl::find_safe_lock(&state, &marks).expect("fixture must have a safe lock");
        let rule_free =
            match crate::strategy::bot_impl::mark_choices_with(&state, &marks, state, 0, false) {
                crate::strategy::bot_impl::Decision::Choices(c) => c,
                _ => panic!("rule-free fixture must yield choices"),
            };
        let ctx = DistillCtx::Lock {
            ctx: LockCtx::Ap2,
            state,
            opps: vec![opp],
            sim_opp: vec![],
            dice,
            active_player: 0,
            cands: rule_free,
            lock,
        };

        let (samples, lock_samples) = build_distill_samples(&boot, std::slice::from_ref(&ctx), cfg, 0);
        // Compared set = lock + best non-lock (no runner-up lock on this
        // board); neither entry ends the game (first lock). Count =
        // 2 cands × 1 opp × m_lock(5) × 2 swap = 20 — m_gated(2) must NOT
        // apply here.
        assert_eq!(samples.len(), 20, "lock ctx must emit with m_lock");
        assert_eq!(lock_samples, samples.len(), "all samples from a lock ctx count as lock-pool");
    }

    #[test]
    fn epsilon_lock_declines_only_at_firings() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let model = PairModelConfig::new().init::<MyBackend>(&device);

        // Build a RecordingPair with a distill buffer and a chosen epsilon_lock.
        let make = |eps_lock: f32| RecordingPair {
            policy: PairPolicy::Static(PairStrategy::from_model(model.clone(), device)),
            epsilon: 0.0,
            rng: SmallRng::seed_from_u64(12345),
            recorded: std::rc::Rc::new(std::cell::RefCell::new(Vec::new())),
            distill: Some(std::rc::Rc::new(std::cell::RefCell::new(Vec::new()))),
            epsilon_lock: eps_lock,
        };

        let s = lockable();
        let opps = [State::default()];
        let dice = [6, 6, 1, 1, 1, 1]; // white sum 12 completes the red lock

        // At a firing: epsilon_lock=0.0 plays the forced lock; 1.0 instead
        // value-selects over the rule-free candidate set (A/B-variant
        // semantics) — equal to eval_decision over that set.
        let lock = Mark { row: 0, number: 12 };
        let mut keep = make(0.0);
        assert_eq!(keep.passive_phase1(&s, &opps, dice, 0), Some(lock), "no decline ⇒ forced lock");

        let boot = PairStrategy::from_model(model.clone(), device);
        let white_sum = dice[0] + dice[1];
        let marks = s.generate_white_moves(white_sum);
        let opp_best = opp_best_phase1_score(&opps, white_sum);
        let rule_free = mark_choices_with(&s, &marks, s, opp_best, false);
        let expected_decline = eval_decision(&boot, rule_free, &opps);

        let mut decline = make(1.0);
        let played = decline.passive_phase1(&s, &opps, dice, 0);
        assert_eq!(
            played, expected_decline,
            "epsilon_lock=1.0 ⇒ value-select over the rule-free set, not the forced lock"
        );

        // Non-firing decision: identical move under either setting, and no
        // decline perturbation.
        let fresh = State::default();
        let nf_dice = [3, 4, 2, 3, 5, 1];
        let mut a = make(0.0);
        let mut b = make(1.0);
        assert_eq!(
            a.passive_phase1(&fresh, &opps, nf_dice, 0),
            b.passive_phase1(&fresh, &opps, nf_dice, 0),
            "non-firing decisions are unaffected by epsilon_lock"
        );
    }

    #[test]
    fn distill_off_is_bit_identical() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let model = PairModelConfig::new().init::<MyBackend>(&device);

        // Disabled distill must reproduce identical samples run-to-run (and the
        // capture path is fully bypassed). The existing suite staying green is
        // the true pre-change equivalence; this pins determinism.
        let run = || play_training_game(&model, &device, 1, false, 0.1, 7777, DistillCfg::off());
        let (s1, f1, d1, _) = run();
        let (s2, f2, d2, _) = run();
        assert_eq!(d1, 0, "disabled distill emits zero distill samples");
        assert_eq!(d2, 0);
        assert_eq!(f1, f2);
        assert_eq!(s1.len(), s2.len());
        for (a, b) in s1.iter().zip(&s2) {
            assert_eq!(a.features, b.features);
            assert_eq!(a.value, b.value);
            assert_eq!(a.final_diff, b.final_diff);
        }

        // And: a disabled-cfg game produces the SAME sample stream as one with
        // epsilon_lock=0 but enabled=false — i.e. the enabled flag gates all of
        // it. (Run with enabled but k=m=0 would change nothing only if no ctx;
        // we assert the off path equals a second off path above.)
        let (s3, _, _, _) = play_training_game(&model, &device, 2, false, 0.1, 7777, DistillCfg::off());
        let (s4, _, _, _) = play_training_game(&model, &device, 2, false, 0.1, 7777, DistillCfg::off());
        assert_eq!(s3.len(), s4.len());
    }
}
