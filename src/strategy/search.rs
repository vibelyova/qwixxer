//! Decision-time search: gated truncated rollouts bootstrapped by a win
//! probability. Generic over any [`WinProb`] bot.
//!
//! Design: docs/superpowers/specs/2026-06-10-pair-search-design.md

use super::Bot;
use crate::state::State;

/// Candidates searched per decision (top by static value).
pub const K_CANDIDATES: usize = 2;
/// Sampled futures per candidate (CRN: dice shared across candidates).
pub const K_SAMPLES: usize = 64;
/// Top-2 static value gap (bot's evaluate units) below which search triggers.
pub const GATE_MARGIN: f32 = 0.15;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::Mark;

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
}
