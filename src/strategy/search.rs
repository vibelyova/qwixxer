//! Decision-time search: gated truncated rollouts bootstrapped by a win
//! probability. Generic over any [`WinProb`] bot.
//!
//! Design: docs/superpowers/specs/2026-06-10-pair-search-design.md

use super::bot_impl::{active_phase1_choices, active_phase2_choices, eval_decision, passive_phase1_impl, Decision};
use super::sim::{BatchedRollouts, SimGame};
use super::Bot;
use super::Strategy;
use crate::state::{Mark, State};
use rand::rngs::SmallRng;
use rand::SeedableRng;

/// Candidates searched per decision (top by static value).
pub const K_CANDIDATES: usize = 2;
/// Sampled futures per candidate (CRN: dice shared across candidates).
pub const K_SAMPLES: usize = 128;
/// Top-2 static value gap (bot's evaluate units) below which search triggers.
pub const GATE_MARGIN: f32 = 0.15;
/// Full turns simulated after completing the current turn, as a multiple of
/// the player count (1 = one full round).
pub const HORIZON_ROUNDS: usize = 1;

/// Capability for search leaf scoring: a calibrated win probability, batched
/// over independent (our_state, opp_states) groups in one inference call.
pub trait WinProb: Bot {
    /// P(group's player finishes ahead of its leading opponent), one per group.
    fn win_prob_multi(&self, groups: &[(&State, &[State])]) -> Vec<f32>;
}

/// Standard normal CDF via the Abramowitz–Stegun erf approximation
/// (7.1.26, |error| < 1.5e-7).
pub fn phi(z: f32) -> f32 {
    0.5 * (1.0 + erf(z as f64 / std::f64::consts::SQRT_2)) as f32
}

fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn absorb_state(mut h: u64, s: &State) -> u64 {
    let totals = s.row_totals();
    let frees = s.row_free_values();
    for i in 0..4 {
        // (total, free) fully determines a row; locked == free.is_none().
        h = splitmix64(h ^ (((totals[i] as u64) << 8) | frees[i].map_or(0xFF, |f| f as u64)));
    }
    splitmix64(h ^ s.strikes as u64)
}

/// Deterministic seed from the decision context. Candidate-independent by
/// construction (the candidate is not hashed), so all candidates of one
/// decision share dice streams (common random numbers), and identical
/// contexts replay identically (reproducible benches).
pub fn context_seed(our: &State, opps: &[State], dice: [u8; 6]) -> u64 {
    let mut h = absorb_state(0x9E3779B97F4A7C15, our);
    for o in opps {
        h = absorb_state(h, o);
    }
    let packed = dice.iter().fold(0u64, |acc, &d| (acc << 8) | d as u64);
    splitmix64(h ^ packed)
}

/// Per-player dice-stream seed for one rollout sample.
pub fn sample_player_seed(decision_seed: u64, sample: usize, player: usize) -> u64 {
    splitmix64(decision_seed ^ ((sample as u64) << 16) ^ player as u64)
}

/// Why search fired (for diagnostics).
#[derive(Default, Debug, Clone)]
pub struct SearchStats {
    pub active_decisions: u32,
    /// Decisions that reached the gate (>= 2 candidates, not meta-forced).
    pub eligible: u32,
    pub gate_close: u32,
    pub gate_endgame: u32,
    pub searched: u32,
    /// Search picked a different move than static evaluation.
    pub disagreements: u32,
    /// ...of which a gate (close OR endgame) had fired.
    pub disagreements_gated: u32,
    /// Top-2 static value gaps at eligible decisions.
    pub gaps: Vec<f32>,
}

pub struct SearchBot<B: WinProb> {
    pub bot: B,
    /// Search every eligible decision regardless of gates (diagnostics).
    pub force: bool,
    pub stats: Option<std::rc::Rc<std::cell::RefCell<SearchStats>>>,
}

impl<B: WinProb> SearchBot<B> {
    pub fn new(bot: B) -> Self {
        Self {
            bot,
            force: false,
            stats: None,
        }
    }
}

impl<B: WinProb> std::fmt::Debug for SearchBot<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "SearchBot({:?})", self.bot)
    }
}

/// A shortlisted candidate: the move, its static value, its post-move state.
struct Candidate {
    mark: Option<Mark>,
    value: f32,
    post: State,
}

/// Gate predicate. `our` is the pre-decision state (lock comparisons are
/// relative to it).
fn gates(cands: &[Candidate], our: &State, opps: &[State]) -> (bool, bool) {
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

impl<B: WinProb> SearchBot<B> {
    /// Shared post-static-eval flow: search the top-K entries, pick.
    /// `entries[i]` = (full player states after candidate i's deterministic
    /// completion of the current turn, game-already-over flag). Returns the
    /// index of the winning candidate. Takes `&self` (stats use RefCell) so
    /// `self.bot` can be borrowed by the rollout driver without fuss.
    fn search_pick(&self, cands: &[Candidate], entries: Vec<(Vec<State>, bool)>, seed: u64) -> usize {
        debug_assert_eq!(entries.len(), cands.len().min(K_CANDIDATES));
        let n = entries[0].0.len();

        // Build rollouts: K_SAMPLES per non-ended entry, CRN dice across entries.
        let mut sims: Vec<SimGame> = Vec::new();
        let mut sim_owner: Vec<usize> = Vec::new(); // entry index per sim
        for (ei, (states, ended)) in entries.iter().enumerate() {
            if *ended {
                continue;
            }
            for k in 0..K_SAMPLES {
                sims.push(SimGame {
                    states: states.clone(),
                    active: 1 % n,
                    rngs: (0..n)
                        .map(|p| SmallRng::seed_from_u64(sample_player_seed(seed, k, p)))
                        .collect(),
                    over: false,
                });
                sim_owner.push(ei);
            }
        }

        let mut driver = BatchedRollouts::new(&self.bot, sims);
        for _ in 0..(HORIZON_ROUNDS * n) {
            if driver.all_over() {
                break;
            }
            driver.step_turn();
        }

        // Score: exact outcome for finished sims, win prob for survivors
        // (one batched call over all survivors of all entries).
        let mut scores: Vec<f32> = vec![0.0; entries.len()];
        let mut counts: Vec<u32> = vec![0; entries.len()];
        let survivor_idx: Vec<usize> = (0..driver.sims.len()).filter(|&i| !driver.sims[i].over).collect();
        let survivor_views: Vec<Vec<State>> = survivor_idx
            .iter()
            .map(|&i| SimGame::opp_view(&driver.sims[i].states, 0))
            .collect();
        let groups: Vec<(&State, &[State])> = survivor_idx
            .iter()
            .zip(&survivor_views)
            .map(|(&i, view)| (&driver.sims[i].states[0], view.as_slice()))
            .collect();
        let probs = self.bot.win_prob_multi(&groups);
        let mut prob_iter = probs.into_iter();
        for (i, sim) in driver.sims.iter().enumerate() {
            let p = if sim.over {
                SimGame::outcome(&sim.states)
            } else {
                prob_iter.next().unwrap()
            };
            scores[sim_owner[i]] += p;
            counts[sim_owner[i]] += 1;
        }
        for (ei, (states, ended)) in entries.iter().enumerate() {
            if *ended {
                scores[ei] = SimGame::outcome(states);
                counts[ei] = 1;
            }
        }

        // Highest mean wins; ties keep the static (lower-index) candidate.
        let mut best = 0;
        let mut best_score = scores[0] / counts[0] as f32;
        for ei in 1..entries.len() {
            let s = scores[ei] / counts[ei] as f32;
            if s > best_score {
                best = ei;
                best_score = s;
            }
        }
        best
    }

    /// Gate bookkeeping. Returns true if search should run. Never searches
    /// with no opponents (solo is unsupported; the simulator's outcome and
    /// leader logic require at least one opponent).
    fn gate_and_record(&self, cands: &[Candidate], our: &State, opps: &[State]) -> Option<(bool, bool)> {
        if opps.is_empty() {
            return None;
        }
        if let Some(stats) = &self.stats {
            stats.borrow_mut().eligible += 1;
            stats.borrow_mut().gaps.push(cands[0].value - cands[1].value);
        }
        let (close, endgame) = gates(cands, our, opps);
        if let Some(stats) = &self.stats {
            let mut s = stats.borrow_mut();
            if close {
                s.gate_close += 1;
            }
            if endgame {
                s.gate_endgame += 1;
            }
        }
        if self.force || close || endgame {
            Some((close, endgame))
        } else {
            None
        }
    }

    fn record_search_result(&self, gated: bool, disagreed: bool) {
        if let Some(stats) = &self.stats {
            let mut s = stats.borrow_mut();
            s.searched += 1;
            if disagreed {
                s.disagreements += 1;
                if gated {
                    s.disagreements_gated += 1;
                }
            }
        }
    }
}

impl<B: WinProb> Strategy for SearchBot<B> {
    fn active_phase1(&mut self, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
        if let Some(stats) = &self.stats {
            stats.borrow_mut().active_decisions += 1;
        }
        let (decision, sim_opp) = active_phase1_choices(&self.bot, state, opp_states, dice);
        let plans = match decision {
            Decision::Forced(m) => return m,
            Decision::Choices(c) => c,
        };
        let states: Vec<State> = plans.iter().map(|(_, s)| *s).collect();
        let values = self.bot.evaluate_batch(&states, &sim_opp);

        // Collapse plans to distinct phase-1 marks: each keeps its best plan's
        // static value (for shortlisting) and that plan's end-state (for the
        // lock gate). The rollout re-decides phase 2 itself.
        let mut cands: Vec<Candidate> = Vec::new();
        for (i, (p1, _)) in plans.iter().enumerate() {
            match cands.iter_mut().find(|c| c.mark == *p1) {
                Some(c) if values[i] > c.value => {
                    c.value = values[i];
                    c.post = states[i];
                }
                Some(_) => {}
                None => cands.push(Candidate {
                    mark: *p1,
                    value: values[i],
                    post: states[i],
                }),
            }
        }
        cands.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap());

        let Some((close, endgame)) = (if cands.len() < 2 {
            None
        } else {
            self.gate_and_record(&cands, state, opp_states)
        }) else {
            return cands[0].mark;
        };

        // Entry per shortlisted phase-1 mark: opponents' (shared, simultaneous)
        // phase-1 marks are sim_opp; complete OUR turn deterministically.
        let shortlist = &cands[..cands.len().min(K_CANDIDATES)];
        let entries: Vec<(Vec<State>, bool)> = shortlist
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
                // Phase 2, re-decided exactly like real play would follow up.
                let view = SimGame::opp_view(&all, 0);
                let d = active_phase2_choices(&all[0], &view, dice, c.mark.is_some());
                match eval_decision(&self.bot, d, &view) {
                    Some(m) => all[0].apply_mark(m),
                    None if c.mark.is_none() => all[0].apply_strike(),
                    None => {}
                }
                SimGame::propagate_locks(&mut all);
                let ended = SimGame::game_over(&all);
                (all, ended)
            })
            .collect();

        let seed = context_seed(state, opp_states, dice);
        let pick = self.search_pick(shortlist, entries, seed);
        self.record_search_result(close || endgame, pick != 0);
        shortlist[pick].mark
    }

    fn active_phase2(&mut self, state: &State, opp_states: &[State], dice: [u8; 6], has_marked: bool) -> Option<Mark> {
        if let Some(stats) = &self.stats {
            stats.borrow_mut().active_decisions += 1;
        }
        let choices = match active_phase2_choices(state, opp_states, dice, has_marked) {
            Decision::Forced(m) => return m,
            Decision::Choices(c) => c,
        };
        let states: Vec<State> = choices.iter().map(|(_, s)| *s).collect();
        let values = self.bot.evaluate_batch(&states, opp_states);
        let mut cands: Vec<Candidate> = choices
            .iter()
            .zip(&values)
            .map(|((m, s), &v)| Candidate {
                mark: *m,
                value: v,
                post: *s,
            })
            .collect();
        cands.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap());

        let Some((close, endgame)) = (if cands.len() < 2 {
            None
        } else {
            self.gate_and_record(&cands, state, opp_states)
        }) else {
            return cands[0].mark;
        };

        // Entry per candidate: post state + lock propagation; turn is then over.
        let shortlist = &cands[..cands.len().min(K_CANDIDATES)];
        let entries: Vec<(Vec<State>, bool)> = shortlist
            .iter()
            .map(|c| {
                let mut all: Vec<State> = std::iter::once(c.post).chain(opp_states.iter().copied()).collect();
                SimGame::propagate_locks(&mut all);
                let ended = SimGame::game_over(&all);
                (all, ended)
            })
            .collect();

        let seed = context_seed(state, opp_states, dice);
        let pick = self.search_pick(shortlist, entries, seed);
        self.record_search_result(close || endgame, pick != 0);
        shortlist[pick].mark
    }

    fn passive_phase1(
        &mut self,
        state: &State,
        opp_states: &[State],
        dice: [u8; 6],
        _active_player: usize,
    ) -> Option<Mark> {
        passive_phase1_impl(&self.bot, state, opp_states, dice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dqn::pair::{PairModelConfig, PairStrategy};
    use crate::dqn::MyBackend;
    use crate::state::Mark;
    use crate::strategy::Strategy;

    fn test_bot() -> PairStrategy {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        PairStrategy::from_model(PairModelConfig::new().init::<MyBackend>(&device), device)
    }

    #[test]
    fn phi_matches_known_values() {
        assert!((phi(0.0) - 0.5).abs() < 1e-6);
        assert!((phi(1.0) - 0.8413).abs() < 1e-3);
        assert!((phi(1.96) - 0.9750).abs() < 1e-3);
        assert!((phi(-1.0) - (1.0 - phi(1.0))).abs() < 1e-6);
        assert!(phi(10.0) > 0.9999);
        assert!(phi(-10.0) < 0.0001);
    }

    #[test]
    fn context_seed_is_deterministic_and_sensitive() {
        let a = State::default();
        let mut b = State::default();
        b.apply_mark(Mark { row: 0, number: 5 });
        let dice = [1, 2, 3, 4, 5, 6];

        assert_eq!(context_seed(&a, &[b], dice), context_seed(&a, &[b], dice));
        assert_ne!(context_seed(&a, &[b], dice), context_seed(&b, &[a], dice));
        assert_ne!(context_seed(&a, &[b], dice), context_seed(&a, &[b], [2, 1, 3, 4, 5, 6]));
        // Sample/player streams are distinct.
        let s = context_seed(&a, &[b], dice);
        assert_ne!(sample_player_seed(s, 0, 0), sample_player_seed(s, 0, 1));
        assert_ne!(sample_player_seed(s, 0, 0), sample_player_seed(s, 1, 0));
    }

    #[test]
    fn search_decisions_are_deterministic() {
        // SearchBot holds no RNG state: repeating the same decision context on
        // the same instance must reproduce the same move (context-hashed dice).
        let mut bot = SearchBot::new(test_bot());
        bot.force = true; // exercise the search path regardless of gates
        let state = State::default();
        let opps = [State::default()];
        let dice = [3, 4, 2, 3, 5, 1];
        let a = bot.active_phase1(&state, &opps, dice);
        let b = bot.active_phase1(&state, &opps, dice);
        assert_eq!(a, b);
        let c = bot.active_phase2(&state, &opps, dice, false);
        let d = bot.active_phase2(&state, &opps, dice, false);
        assert_eq!(c, d);
    }

    #[test]
    fn gate_predicate_truth_table() {
        let mk = |value, post: State| Candidate {
            mark: None,
            value,
            post,
        };
        let fresh = State::default();
        let mut locked_one = State::default();
        for n in 2..=6 {
            locked_one.apply_mark(Mark { row: 0, number: n });
        }
        locked_one.apply_mark(Mark { row: 0, number: 12 }); // locks red

        // Close gap, no endgame: close gate only.
        let cands = [mk(1.00, fresh), mk(0.95, fresh)];
        assert_eq!(gates(&cands, &fresh, &[fresh]), (true, false));
        // Wide gap, no endgame: nothing fires.
        let cands = [mk(1.00, fresh), mk(0.50, fresh)];
        assert_eq!(gates(&cands, &fresh, &[fresh]), (false, false));
        // Wide gap, opponent has a locked row: endgame fires.
        assert_eq!(gates(&cands, &fresh, &[locked_one]), (false, true));
        // Wide gap, our candidate would lock: endgame fires.
        let cands = [mk(1.00, locked_one), mk(0.50, fresh)];
        assert_eq!(gates(&cands, &fresh, &[fresh]), (false, true));
        // Wide gap, we sit at 3 strikes: endgame fires.
        let mut striked = State::default();
        for _ in 0..3 {
            striked.apply_strike();
        }
        let cands = [mk(1.00, fresh), mk(0.50, fresh)];
        assert_eq!(gates(&cands, &striked, &[fresh]), (false, true));
    }

    #[test]
    fn forced_decisions_bypass_search() {
        // A state with a safe lock available must return the lock without
        // search: 5 marks in red, white sum 12 completes it (first lock,
        // doesn't end the game).
        let mut bot = SearchBot::new(test_bot());
        bot.force = true;
        let mut state = State::default();
        for n in 2..=6 {
            state.apply_mark(Mark { row: 0, number: n });
        }
        let opps = [State::default()];
        let dice = [6, 6, 1, 1, 1, 1]; // white sum 12
        let m = bot.active_phase1(&state, &opps, dice);
        assert_eq!(m, Some(Mark { row: 0, number: 12 }));
    }

    #[test]
    fn fallback_when_gates_closed() {
        // Ungated decisions must be deterministic and legal (static path).
        let mut bot = SearchBot::new(test_bot());
        let state = State::default();
        let opps = [State::default()];
        let m1 = bot.active_phase2(&state, &opps, [3, 4, 2, 3, 5, 1], true);
        let m2 = bot.active_phase2(&state, &opps, [3, 4, 2, 3, 5, 1], true);
        assert_eq!(m1, m2);
    }

    #[test]
    fn stats_are_recorded() {
        let stats = std::rc::Rc::new(std::cell::RefCell::new(SearchStats::default()));
        let mut bot = SearchBot::new(test_bot());
        bot.force = true;
        bot.stats = Some(stats.clone());
        let state = State::default();
        let opps = [State::default()];
        bot.active_phase1(&state, &opps, [3, 4, 2, 3, 5, 1]);
        let s = stats.borrow();
        assert_eq!(s.active_decisions, 1);
        assert!(s.eligible <= 1);
        assert_eq!(s.gaps.len() as u32, s.eligible);
    }
}
