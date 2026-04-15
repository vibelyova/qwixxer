use crate::game::{Game, Player};
use crate::state::Move;
use crate::state::State;
use crate::strategy::Strategy;

use rand::distributions::WeightedIndex;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::sync::Arc;

use rand::{prelude::*, rngs::SmallRng, Rng};

pub type GeneFn = fn(&State) -> f64;

pub const GENE_NAMES: [&str; 4] = [
    "weighted_prob",
    "strikes",
    "concentration",
    "blanks",
];

pub fn default_genes() -> Vec<GeneFn> {
    vec![
        // weighted_probability: sum of P(rolling free) * (total + 1) per active row
        |state| {
            state
                .row_free_values()
                .iter()
                .zip(state.row_totals().iter())
                .map(|(&free, &total)| {
                    let Some(f) = free else { return 0.0 };
                    let ways = 6.0 - (7.0f64 - f as f64).abs();
                    ways / 36.0 * (total as f64 + 1.0)
                })
                .sum()
        },
        // strikes
        |state| state.strikes as f64,
        // concentration: sum of row totals squared
        |state| {
            state
                .row_totals()
                .iter()
                .map(|&t| (t as f64) * (t as f64))
                .sum()
        },
        // blanks: skipped positions
        |state| state.blanks() as f64,
    ]
}

#[derive(Clone, Debug)]
pub struct DNA {
    weights: Vec<f64>,
    genes: Arc<Vec<GeneFn>>,
    score_gap: isize,
}

impl DNA {
    /// If any move locks a row, return it immediately.
    fn find_locking_move(state: &State, moves: &[Move], score_gap: isize) -> Option<Move> {
        let current_locked = state.count_locked();
        moves.iter().copied().find(|&mov| {
            let mut new_state = *state;
            new_state.apply_move(mov);
            if new_state.count_locked() <= current_locked { return false; }
            if new_state.count_locked() >= 2 {
                let opp_score = state.count_points() - score_gap;
                return new_state.count_points() > opp_score;
            }
            true
        })
    }
}

impl Strategy for DNA {
    fn your_move(&mut self, state: &State, dice: [u8; 6]) -> Move {
        let moves = state.generate_moves(dice);

        if let Some(mov) = Self::find_locking_move(state, &moves, self.score_gap) {
            return mov;
        }

        let mut moves = moves;
        moves.push(Move::Strike);

        moves
            .into_iter()
            .map(|mov| {
                let mut new_state = *state;
                new_state.apply_move(mov);
                (self.instinct(&new_state), mov)
            })
            .max_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap())
            .unwrap()
            .1
    }

    fn opponents_move(&mut self, state: &State, number: u8, locked: [bool; 4]) -> Option<Move> {
        let moves = state.generate_opponent_moves(number);

        if let Some(mov) = Self::find_locking_move(state, &moves, self.score_gap) {
            return Some(mov);
        }

        let mut states: Vec<_> = moves
            .into_iter()
            .map(|mov| {
                let mut new_state = *state;
                new_state.apply_move(mov);
                (new_state, Some(mov))
            })
            .collect();
        states.push((*state, None));

        for (state, _) in states.iter_mut() {
            state.lock(locked);
        }

        *states
            .iter()
            .map(|(state, mov)| (self.instinct(state), mov))
            .max_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap())
            .unwrap()
            .1
    }

    fn observe_opponents(&mut self, our_score: isize, opponents: &[State]) {
        let max_opp_score = opponents.iter().map(|s| s.count_points()).max().unwrap_or(0);
        self.score_gap = our_score - max_opp_score;
    }
}

impl DNA {
    pub fn new_random(genes: Arc<Vec<GeneFn>>, rng: &mut impl Rng) -> Self {
        let weights: Vec<f64> = (0..genes.len()).map(|_| rng.gen_range(-1.0..=1.0)).collect();
        DNA { weights, genes, score_gap: 0 }.normalize()
    }

    fn normalize(mut self) -> Self {
        let norm = self.weights.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            self.weights.iter_mut().for_each(|x| *x /= norm);
        }
        self
    }

    pub fn crossover(&self, other: &Self, rng: &mut impl Rng) -> Self {
        debug_assert!(self.weights.len() == other.weights.len());
        let weights: Vec<f64> = self
            .weights
            .iter()
            .zip(other.weights.iter())
            .map(|(a, b)| if rng.gen_bool(0.5) { *a } else { *b })
            .collect();
        DNA {
            weights,
            genes: Arc::clone(&self.genes),
            score_gap: 0,
        }
        .normalize()
    }

    pub fn mutate(mut self, rate: f64, rng: &mut impl Rng) -> Self {
        for x in self.weights.iter_mut() {
            if rng.gen_range(0.0..=1.0) < rate {
                *x += rng.gen_range(-0.1..=0.1);
            }
        }
        self.normalize()
    }

    pub fn print_weights(&self) {
        let pairs: Vec<String> = GENE_NAMES
            .iter()
            .zip(self.weights.iter())
            .map(|(name, w)| format!("{name}:{w:+.3}"))
            .collect();
        println!("[{}]", pairs.join("  "));
    }

    pub fn save_weights(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        for (name, w) in GENE_NAMES.iter().zip(self.weights.iter()) {
            writeln!(f, "{name} {w}")?;
        }
        Ok(())
    }

    pub fn load_weights(path: &str, genes: Arc<Vec<GeneFn>>) -> std::io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let weights: Vec<f64> = content
            .lines()
            .map(|line| {
                line.split_whitespace()
                    .last()
                    .unwrap()
                    .parse()
                    .unwrap()
            })
            .collect();
        assert_eq!(weights.len(), genes.len(), "Weight count mismatch");
        Ok(DNA { weights, genes, score_gap: 0 })
    }

    pub fn load_weights_from_bytes(bytes: &[u8], genes: Arc<Vec<GeneFn>>) -> Self {
        let content = std::str::from_utf8(bytes).expect("Invalid UTF-8");
        let weights: Vec<f64> = content
            .lines()
            .map(|line| line.split_whitespace().last().unwrap().parse().unwrap())
            .collect();
        assert_eq!(weights.len(), genes.len(), "Weight count mismatch");
        DNA { weights, genes, score_gap: 0 }
    }

    pub fn instinct(&self, state: &State) -> f64 {
        self.genes
            .iter()
            .zip(self.weights.iter())
            .map(|(gene, weight)| gene(state) * weight)
            .sum()
    }

    /// Return individual gene contributions: (gene_name, raw_value, weight, weighted_value)
    pub fn gene_contributions(&self, state: &State) -> Vec<(&'static str, f64, f64, f64)> {
        self.genes
            .iter()
            .zip(self.weights.iter())
            .enumerate()
            .map(|(i, (gene, &weight))| {
                let raw = gene(state);
                (GENE_NAMES[i], raw, weight, raw * weight)
            })
            .collect()
    }
}

#[allow(dead_code)]
pub struct Population {
    genes: Arc<Vec<GeneFn>>,
    dna: Vec<DNA>,
    rng: SmallRng,
}

impl Population {
    #[allow(dead_code)]
    pub fn single(dna: DNA) -> Self {
        Self {
            genes: Arc::clone(&dna.genes),
            dna: vec![dna],
            rng: SmallRng::from_entropy(),
        }
    }

    pub fn new(size: usize, genes: Vec<GeneFn>, seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let genes = Arc::new(genes);
        let dna: Vec<DNA> = (0..size)
            .map(|_| DNA::new_random(Arc::clone(&genes), &mut rng))
            .collect();
        Population { genes, dna, rng }
    }

    fn rank_generation(dna: &[DNA], seed: u64) -> Vec<f32> {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut indices: Vec<usize> = (0..dna.len()).collect();
        indices.shuffle(&mut rng);

        // Pad to multiple of 4
        while indices.len() % 4 != 0 {
            indices.push(rng.gen_range(0..dna.len()));
        }

        let mut score = vec![0.0f32; dna.len()];

        for group in indices.chunks(4) {
            let players: Vec<Player> = group
                .iter()
                .map(|&i| {
                    Player::new(
                        Box::new(dna[i].clone()) as Box<dyn Strategy>,
                        Box::new(SmallRng::from_rng(&mut rng).unwrap()),
                    )
                })
                .collect();

            let mut game = Game::new(players);
            game.play();

            let points: Vec<isize> = game
                .players
                .into_iter()
                .map(|p| p.state.count_points())
                .collect();
            let max_points = *points.iter().max().unwrap();

            for (j, &di) in group.iter().enumerate() {
                let p = points[j];
                let normalized = if max_points <= 0 {
                    0.25
                } else {
                    (p as f32).max(0.0) / max_points as f32
                };
                score[di] += normalized;
            }
        }

        score
    }

    pub fn next_generation(&mut self) {
        const NUMBER_OF_SIMULATIONS: usize = 1000;

        // Pre-generate seeds
        let seeds: Vec<u64> = (0..NUMBER_OF_SIMULATIONS)
            .map(|_| self.rng.gen())
            .collect();

        // Run simulations (parallel when available)
        let dna = &self.dna;

        #[cfg(feature = "parallel")]
        let global_rank = seeds
            .par_iter()
            .map(|&seed| Self::rank_generation(dna, seed))
            .reduce(
                || vec![0.0f32; dna.len()],
                |mut acc, rank| {
                    acc.iter_mut().zip(rank.iter()).for_each(|(a, b)| *a += b);
                    acc
                },
            );

        #[cfg(not(feature = "parallel"))]
        let global_rank = seeds
            .iter()
            .map(|&seed| Self::rank_generation(dna, seed))
            .fold(vec![0.0f32; dna.len()], |mut acc, rank| {
                acc.iter_mut().zip(rank.iter()).for_each(|(a, b)| *a += b);
                acc
            });

        let dist = WeightedIndex::new(&global_rank).expect("All weights are zero");

        // Elitism: carry the top individual forward unchanged
        let champion_idx = global_rank
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        let mut new_dna = vec![self.dna[champion_idx].clone()];

        for _ in 1..self.dna.len() {
            let a = dist.sample(&mut self.rng);
            let b = dist.sample(&mut self.rng);
            new_dna.push(
                self.dna[a]
                    .crossover(&self.dna[b], &mut self.rng)
                    .mutate(0.2, &mut self.rng),
            );
        }
        self.dna = new_dna;
    }

    pub fn evolve(&mut self, generations: usize) {
        for gen in 0..generations {
            println!("Generation #{gen}...");
            let now = std::time::Instant::now();
            self.next_generation();
            let best = &self.dna[0]; // elitism puts champion at index 0
            print!("  {:?} ", now.elapsed());
            best.print_weights();
        }
        println!("Evolution complete.");
        if let Err(e) = self.dna[0].save_weights("champion.txt") {
            eprintln!("Failed to save weights: {e}");
        } else {
            println!("Champion saved to champion.txt");
        }
    }

    #[allow(dead_code)]
    pub fn champion(&self, rank: &[f32]) -> DNA {
        self.dna
            .iter()
            .enumerate()
            .max_by(|(a, _), (b, _)| rank[*a].partial_cmp(&rank[*b]).unwrap())
            .unwrap()
            .1
            .clone()
    }

    pub fn current_champion(&self) -> &DNA {
        &self.dna[0]
    }
}
