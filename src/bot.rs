use crate::game::{Game, Player};
use crate::state::Move;
use crate::state::State;
use crate::strategy::Strategy;
use itertools::Itertools;
use rand::distributions::WeightedIndex;

use rand::{prelude::*, rngs::SmallRng, Rng};
static GENES: [Box<dyn Gene + Sync>; 0] = [];

impl Strategy for DNA {
    fn your_move(&mut self, state: &State, dice: [u8; 6]) -> Move {
        let mut moves = state.generate_moves(dice);
        moves.push(Move::Strike);

        let states = moves.into_iter().map(|mov| {
            let mut new_state = state.clone();
            new_state.apply_move(mov);
            (new_state, mov)
        });

        let mov = states
            .map(|(state, mov)| (self.instinct(&state), mov))
            .max_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap())
            .unwrap();

        mov.1
    }

    fn opponents_move(&mut self, state: &State, number: u8, locked: [bool; 4]) -> Option<Move> {
        let moves = state.generate_opponent_moves(number);

        let mut states = moves
            .into_iter()
            .map(|mov| {
                let mut new_state = state.clone();
                new_state.apply_move(mov);
                (new_state, Some(mov))
            })
            .collect::<Vec<_>>();
        states.push((*state, None));

        for (state, _) in states.iter_mut() {
            state.lock(locked);
        }

        let mov = states
            .iter()
            .map(|(state, mov)| (self.instinct(&state), mov))
            .max_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap())
            .unwrap();

        *mov.1
    }
}

pub trait Gene {
    fn evaluate(&self, state: &State) -> f64;
}

// TODO: use arrays to reduce allocation
#[derive(Clone, Debug, Default)]
pub struct DNA(pub Vec<f64>);

impl DNA {
    pub fn new_random(size: usize, rng: &mut impl Rng) -> Self {
        DNA((0..size).map(|_| rng.gen_range(-1.0..=1.0)).collect()).normalize()
    }

    pub fn normalize(mut self) -> Self {
        let norm = self.0.iter().map(|x| x * x).sum::<f64>().sqrt();
        self.0.iter_mut().for_each(|x| *x /= norm);
        self
    }

    pub fn crossover(&self, other: &Self) -> Self {
        debug_assert!(self.0.len() == other.0.len());
        let mut child = Vec::with_capacity(self.0.len());
        for (a, b) in self.0.iter().zip(other.0.iter()) {
            child.push(*a + *b);
        }
        DNA(child).normalize()
    }

    pub fn mutate(mut self, rate: f64, rng: &mut impl Rng) -> Self {
        for x in self.0.iter_mut() {
            if rng.gen_range(0.0..=1.0) < rate {
                *x += rng.gen_range(-0.1..=0.1);
            }
        }
        self.normalize()
    }

    pub fn instinct(&self, state: &State) -> f64 {
        GENES
            .iter()
            .zip(self.0.iter())
            .map(|(gene, weight)| gene.evaluate(state) * weight)
            .sum()
    }
}

pub struct Population {
    dna: Vec<DNA>,
    rng: SmallRng,
}

impl Population {
    pub fn single(dna: DNA) -> Self {
        assert_eq!(GENES.len(), dna.0.len());
        Self {
            dna: vec![dna],
            rng: SmallRng::from_entropy(),
        }
    }

    pub fn new(size: usize, genes: Vec<Box<dyn Gene + Sync>>, seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut dna = Vec::with_capacity(size);
        for _ in 0..size {
            dna.push(DNA::new_random(genes.len(), &mut rng));
        }
        Population { dna, rng }
    }

    pub fn rank_generation(&mut self, seed: u64) -> Vec<f32> {
        // TODO parallelize

        // shuffle indices
        let mut indices: Vec<usize> = (0..self.dna.len()).collect();
        indices.shuffle(&mut self.rng);

        let mut score = vec![0.0; self.dna.len()];

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
                            Box::new(SmallRng::from_entropy()),
                        );
                        (i, player)
                    })
                    .unzip();
                let mut game = Game::new(players);
                game.play();
                let points = game
                    .players
                    .into_iter()
                    .map(|p| p.state.count_points())
                    .collect_vec();
                let max_points = *points.iter().max().unwrap();
                let points = if max_points <= 0 {
                    vec![1.0 / points.len() as f32; points.len()]
                } else {
                    points
                        .into_iter()
                        .map(|p| (p as f32).max(0.0) / (max_points as f32))
                        .collect_vec()
                };

                for (dna_index, sc) in indexes.into_iter().zip(points.into_iter()) {
                    score[dna_index] += sc;
                }
            });

        score
    }

    pub fn next_generation(&mut self) {
        let mut new_dna = Vec::with_capacity(self.dna.len());
        // Do X rankings for each generation
        let mut global_rank = vec![0.0; self.dna.len()];
        const NUMBER_OF_SIMULATIONS: usize = 10_000;
        for _ in 0..NUMBER_OF_SIMULATIONS {
            // Seed is fixed for each generation
            let seed = self.rng.gen();
            let rank = self.rank_generation(seed);
            // log::info!(
            //     "Champion (score {}): {:?}",
            //     *rank.iter().max().unwrap(),
            //     self.champion(&rank)
            // );
            global_rank
                .iter_mut()
                .zip(rank.iter())
                .for_each(|(a, b)| *a += b);
        }
        // log::info!(
        //     "Champion (score {}): {:?}",
        //     *global_rank.iter().max().unwrap() / NUMBER_OF_SIMULATIONS as u32,
        //     self.champion(&global_rank)
        // );

        let dist = WeightedIndex::new(global_rank /* .into_iter().map(|x| x * x) */)
            .expect("This generation is shit");
        for _ in 0..self.dna.len() {
            let a = dist.sample(&mut self.rng);
            let b = dist.sample(&mut self.rng);
            // TODO: vanishing rate
            new_dna.push(
                self.dna[a]
                    .crossover(&self.dna[b])
                    .mutate(0.2, &mut self.rng),
            );
        }
        self.dna = new_dna;
    }

    pub fn evolve(&mut self, generations: usize) {
        for gen in 0..generations {
            // log::info!("Growing generation #{gen}");
            println!("Growing generation #{gen}");
            let now = std::time::Instant::now();
            self.next_generation();
            println!("Generation #{gen} complete in {:?}", now.elapsed());
            // log::info!("Generation grew up in {:?}", now.elapsed());
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
