use crate::game::{Game, Player};
use crate::state::Move;
use crate::state::State;
use crate::strategy::Strategy;
use itertools::Itertools;
use rand::distributions::WeightedIndex;
use std::sync::Arc;

use rand::{prelude::*, rngs::SmallRng, Rng};

pub type GeneFn = fn(&State) -> f64;

#[derive(Clone, Debug)]
pub struct DNA {
    weights: Vec<f64>,
    genes: Arc<Vec<GeneFn>>,
}

impl Strategy for DNA {
    fn your_move(&mut self, state: &State, dice: [u8; 6]) -> Move {
        let mut moves = state.generate_moves(dice);
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

    pub fn rank_generation(&mut self, seed: u64) -> Vec<f32> {
        let mut indices: Vec<usize> = (0..self.dna.len()).collect();
        indices.shuffle(&mut self.rng);

        // Pad to multiple of 4 by repeating random picks
        while indices.len() % 4 != 0 {
            indices.push(self.rng.gen_range(0..self.dna.len()));
        }

        let mut score = vec![0.0f32; self.dna.len()];
        let mut game_rng = SmallRng::seed_from_u64(seed);

        indices
            .into_iter()
            .map(|i| (i, self.dna[i].clone()))
            .chunks(4)
            .into_iter()
            .for_each(|chunk| {
                let (indexes, players): (Vec<_>, Vec<_>) = chunk
                    .into_iter()
                    .map(|(i, bot)| {
                        let player = Player::new(
                            Box::new(bot) as Box<dyn Strategy>,
                            Box::new(SmallRng::from_rng(&mut game_rng).unwrap()),
                        );
                        (i, player)
                    })
                    .unzip();
                let mut game = Game::new(players);
                game.play();
                let points: Vec<isize> = game
                    .players
                    .into_iter()
                    .map(|p| p.state.count_points())
                    .collect();
                let max_points = *points.iter().max().unwrap();
                let normalized = if max_points <= 0 {
                    vec![1.0 / indexes.len() as f32; indexes.len()]
                } else {
                    points
                        .into_iter()
                        .map(|p| (p as f32).max(0.0) / (max_points as f32))
                        .collect_vec()
                };

                for (dna_index, sc) in indexes.into_iter().zip(normalized.into_iter()) {
                    score[dna_index] += sc;
                }
            });

        score
    }

    pub fn next_generation(&mut self) {
        let mut global_rank = vec![0.0f32; self.dna.len()];
        const NUMBER_OF_SIMULATIONS: usize = 500;
        for _ in 0..NUMBER_OF_SIMULATIONS {
            let seed = self.rng.gen();
            let rank = self.rank_generation(seed);
            global_rank
                .iter_mut()
                .zip(rank.iter())
                .for_each(|(a, b)| *a += b);
        }

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
            println!("Growing generation #{gen}");
            let now = std::time::Instant::now();
            self.next_generation();
            println!("Generation #{gen} complete in {:?}", now.elapsed());
        }
        println!("Evolution complete");
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
}
