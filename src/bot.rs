use crate::game::{Game, Player};
use crate::state::Move;
use crate::state::State;
use crate::strategy::{Conservative, Rusher, Strategy};
use itertools::Itertools;
use rand::distributions::WeightedIndex;
use rayon::prelude::*;
use std::sync::Arc;

use rand::{prelude::*, rngs::SmallRng, Rng};

pub type GeneFn = fn(&State) -> f64;

pub const GENE_NAMES: [&str; 5] = [
    "concentration",
    "blanks",
    "strikes",
    "lockable_rows",
    "probability",
];

pub fn default_genes() -> Vec<GeneFn> {
    vec![
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
        // strikes
        |state| state.strikes as f64,
        // lockable_rows: rows with 5+ marks (ready to lock)
        |state| state.lockable_rows() as f64,
        // probability: chance of useful opponent roll
        |state| state.probability() as f64,
    ]
}

#[derive(Clone, Debug)]
pub struct DNA {
    weights: Vec<f64>,
    genes: Arc<Vec<GeneFn>>,
}

impl DNA {
    /// If any move locks a row, return it immediately.
    fn find_locking_move(state: &State, moves: &[Move]) -> Option<Move> {
        let current_locked = state.count_locked();
        moves.iter().copied().find(|&mov| {
            let mut new_state = *state;
            new_state.apply_move(mov);
            new_state.count_locked() > current_locked
        })
    }
}

impl Strategy for DNA {
    fn your_move(&mut self, state: &State, dice: [u8; 6]) -> Move {
        let moves = state.generate_moves(dice);

        // Always lock if possible
        if let Some(mov) = Self::find_locking_move(state, &moves) {
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

        // Always lock if possible
        if let Some(mov) = Self::find_locking_move(state, &moves) {
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
}

impl DNA {
    pub fn new_random(genes: Arc<Vec<GeneFn>>, rng: &mut impl Rng) -> Self {
        let weights: Vec<f64> = (0..genes.len()).map(|_| rng.gen_range(-1.0..=1.0)).collect();
        DNA { weights, genes }.normalize()
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
        Ok(DNA { weights, genes })
    }

    pub fn instinct(&self, state: &State) -> f64 {
        self.genes
            .iter()
            .zip(self.weights.iter())
            .map(|(gene, weight)| gene(state) * weight)
            .sum()
    }
}

pub struct Population {
    genes: Arc<Vec<GeneFn>>,
    dna: Vec<DNA>,
    rng: SmallRng,
}

impl Population {
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

        let mut score = vec![0.0f32; dna.len()];

        // Each game: 2 DNA bots + 1 Conservative + 1 Rusher
        for pair in indices.chunks(2) {
            let dna_indices = if pair.len() == 2 {
                [pair[0], pair[1]]
            } else {
                [pair[0], rng.gen_range(0..dna.len())]
            };

            // Build players: 2 DNA bots + 1 Conservative + 1 Rusher
            // Track which positions the DNA bots land in after shuffle
            let mut slots: Vec<(Option<usize>, Box<dyn Strategy>)> = Vec::with_capacity(4);
            for &i in &dna_indices {
                slots.push((Some(i), Box::new(dna[i].clone()) as Box<dyn Strategy>));
            }
            slots.push((None, Box::<Conservative>::default()));
            slots.push((None, Box::<Rusher>::default()));

            // Shuffle seat order to eliminate position bias
            slots.shuffle(&mut rng);

            let dna_positions: Vec<(usize, usize)> = slots
                .iter()
                .enumerate()
                .filter_map(|(seat, (di, _))| di.map(|di| (seat, di)))
                .collect();

            let players: Vec<Player> = slots
                .into_iter()
                .map(|(_, strat)| {
                    Player::new(strat, Box::new(SmallRng::from_rng(&mut rng).unwrap()))
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

            // Only score the DNA bots at their shuffled positions
            for (seat, di) in &dna_positions {
                let p = points[*seat];
                let normalized = if max_points <= 0 {
                    0.25
                } else {
                    (p as f32).max(0.0) / max_points as f32
                };
                score[*di] += normalized;
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

        // Run simulations in parallel
        let dna = &self.dna;
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
