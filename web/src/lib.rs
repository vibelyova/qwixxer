use qwixxer::bot::{self, DNA};
use qwixxer::state::{Move, State};
use qwixxer::strategy::{self, Strategy};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use std::sync::Arc;
use wasm_bindgen::prelude::*;

// Embedded champion weights for the GA bot
const CHAMPION_BYTES: &[u8] = include_bytes!("../../champion.txt");

const ROW_NAMES: [&str; 4] = ["RED", "YLW", "GRN", "BLU"];

// ---- Serializable view types ----

#[derive(Serialize)]
pub struct GameView {
    player: StateView,
    bot: StateView,
    dice: Vec<u8>,
    white_sum: u8,
    phase: String,
    available_moves: Vec<MoveView>,
    game_over: bool,
    message: String,
}

#[derive(Serialize)]
pub struct StateView {
    rows: Vec<RowView>,
    strikes: u8,
    score: isize,
}

#[derive(Serialize)]
pub struct RowView {
    numbers: Vec<u8>,
    free: Option<u8>,
    total: u8,
    locked: bool,
    ascending: bool,
    color: String,
    marks: Vec<bool>,
}

#[derive(Serialize)]
pub struct MoveView {
    index: usize,
    description: String,
    is_strike: bool,
    is_pass: bool,
}

// ---- Phase tracking ----

#[derive(Clone, Debug, PartialEq)]
#[allow(dead_code)]
enum Phase {
    BotActive,
    PlayerPassive,
    PlayerActive,
    BotPassive,
    GameOver,
}

impl Phase {
    fn as_str(&self) -> &'static str {
        match self {
            Phase::BotActive => "bot_active",
            Phase::PlayerPassive => "player_passive",
            Phase::PlayerActive => "player_active",
            Phase::BotPassive => "bot_passive",
            Phase::GameOver => "game_over",
        }
    }
}

// ---- WebGame state machine ----

#[wasm_bindgen]
pub struct WebGame {
    player_state: State,
    bot_state: State,
    bot_strategy: Box<dyn Strategy>,
    rng: SmallRng,
    phase: Phase,
    dice: [u8; 6],
    white_sum: u8,
    message: String,
    /// Moves available in the current phase for the player
    available_moves: Vec<Move>,
    /// Per-cell mark tracking: [row][number_index] = true if marked
    player_marks: [[bool; 11]; 4],
    bot_marks: [[bool; 11]; 4],
}

fn roll_dice(rng: &mut SmallRng) -> [u8; 6] {
    core::array::from_fn(|_| rng.gen_range(1..=6))
}

/// Convert a (row, number) into the index into the 11-element marks array.
fn mark_index(row: usize, number: u8) -> usize {
    if row < 2 {
        // ascending rows: 2..=12 maps to indices 0..=10
        (number - 2) as usize
    } else {
        // descending rows: 12..=2 maps to indices 0..=10
        (12 - number) as usize
    }
}

/// Record marks from a Move into the given marks array.
fn record_marks(marks: &mut [[bool; 11]; 4], mov: Move) {
    match mov {
        Move::Strike => {}
        Move::Single(m) => {
            marks[m.row][mark_index(m.row, m.number)] = true;
            // If locking (ascending row marking 12, or descending row marking 2),
            // the lock number itself also counts as a mark — already covered above.
            // The lock icon (last cell) is also marked.
            if (m.row < 2 && m.number == 12) || (m.row >= 2 && m.number == 2) {
                // Lock cell: the last index (10) is already set by the formula above.
            }
        }
        Move::Double(m1, m2) => {
            marks[m1.row][mark_index(m1.row, m1.number)] = true;
            marks[m2.row][mark_index(m2.row, m2.number)] = true;
        }
    }
}

// Embedded DQN model weights
const DQN_MODEL_BYTES: &[u8] = include_bytes!("../../dqn_model/model.mpk");

fn make_strategy(bot_type: &str) -> Box<dyn Strategy> {
    match bot_type {
        "ga" => {
            let genes = Arc::new(bot::default_genes());
            let champion = DNA::load_weights_from_bytes(CHAMPION_BYTES, genes);
            Box::new(champion)
        }
        "dqn" => Box::new(qwixxer::dqn::DqnStrategy::load_from_bytes(DQN_MODEL_BYTES)),
        "opportunist" => Box::<strategy::Opportunist>::default(),
        "conservative" => Box::<strategy::Conservative>::default(),
        "rusher" => Box::new(strategy::Rusher),
        "random" => Box::new(strategy::Random),
        _ => Box::<strategy::Opportunist>::default(),
    }
}

fn describe_move(mov: &Move) -> String {
    match mov {
        Move::Strike => "Strike".to_string(),
        Move::Single(mark) => {
            format!("{} {}", ROW_NAMES[mark.row], mark.number)
        }
        Move::Double(m1, m2) => {
            format!(
                "{} {} + {} {}",
                ROW_NAMES[m1.row], m1.number, ROW_NAMES[m2.row], m2.number
            )
        }
    }
}

fn build_state_view(state: &State, marks: &[[bool; 11]; 4]) -> StateView {
    let totals = state.row_totals();
    let free_values = state.row_free_values();
    let locked = state.locked();
    let colors = ["red", "yellow", "green", "blue"];

    let rows: Vec<RowView> = (0..4)
        .map(|i| {
            let ascending = i < 2;
            let numbers: Vec<u8> = if ascending {
                (2..=12).collect()
            } else {
                (2..=12).rev().collect()
            };
            RowView {
                numbers,
                free: free_values[i],
                total: totals[i],
                locked: locked[i],
                ascending,
                color: colors[i].to_string(),
                marks: marks[i].to_vec(),
            }
        })
        .collect();

    StateView {
        rows,
        strikes: state.strikes,
        score: state.count_points(),
    }
}

impl WebGame {
    fn is_game_over(&self) -> bool {
        // 4 strikes by either player
        if self.player_state.strikes >= 4 || self.bot_state.strikes >= 4 {
            return true;
        }
        // 2 rows locked (check the max across both players)
        let player_locked = self.player_state.count_locked();
        let bot_locked = self.bot_state.count_locked();
        let total_locked = player_locked.max(bot_locked);
        total_locked >= 2
    }

    fn check_game_over(&mut self) -> bool {
        if self.is_game_over() {
            self.phase = Phase::GameOver;
            let player_score = self.player_state.count_points();
            let bot_score = self.bot_state.count_points();
            self.message = if player_score > bot_score {
                format!("You win! {} to {}", player_score, bot_score)
            } else if bot_score > player_score {
                format!("Bot wins! {} to {}", bot_score, player_score)
            } else {
                format!("Tie game! Both scored {}", player_score)
            };
            self.available_moves.clear();
            true
        } else {
            false
        }
    }

    fn propagate_locks(&mut self) {
        let mut locked = self.player_state.locked();
        for (i, l) in self.bot_state.locked().iter().enumerate() {
            locked[i] |= *l;
        }
        self.player_state.lock(locked);
        self.bot_state.lock(locked);
    }

    /// Execute the bot's active turn: roll dice, pick a move.
    fn do_bot_active(&mut self) {
        self.dice = roll_dice(&mut self.rng);
        self.white_sum = self.dice[0] + self.dice[1];

        let mov = self.bot_strategy.your_move(&self.bot_state, self.dice);
        record_marks(&mut self.bot_marks, mov);
        self.bot_state.apply_move(mov);

        self.message = format!("Bot played: {}", describe_move(&mov));

        // DON'T propagate locks yet — player gets their passive move first.
        // Generate passive moves from player's current (pre-lock) state.
        self.phase = Phase::PlayerPassive;
        self.available_moves = self.player_state.generate_opponent_moves(self.white_sum);
        if self.available_moves.is_empty() {
            self.message
                .push_str(" | No moves for you on white sum.");
            // Propagate locks now (player had no moves anyway)
            self.propagate_locks();
            if self.is_game_over() {
                self.phase = Phase::GameOver;
            }
        }
    }

    /// Execute the bot's passive turn (during player's active turn).
    fn do_bot_passive(&mut self) {
        let locked = self.player_state.locked();
        let bot_mov =
            self.bot_strategy
                .opponents_move(&self.bot_state, self.white_sum, locked);
        if let Some(mov) = bot_mov {
            record_marks(&mut self.bot_marks, mov);
            self.bot_state.apply_move(mov);
        }
        // Apply any locks from player's active turn to bot
        self.bot_state.lock(locked);

        // Propagate locks after both turns of the round
        self.propagate_locks();
    }

    /// Begin the player's active turn: roll dice, compute moves.
    fn start_player_active(&mut self) {
        self.dice = roll_dice(&mut self.rng);
        self.white_sum = self.dice[0] + self.dice[1];

        self.phase = Phase::PlayerActive;
        let mut moves = self.player_state.generate_moves(self.dice);
        if moves.is_empty() {
            // Must strike
            self.available_moves = vec![Move::Strike];
            self.message = "Your turn! No valid moves - you must strike.".to_string();
        } else {
            moves.push(Move::Strike);
            self.available_moves = moves;
            self.message = "Your turn! Pick a move.".to_string();
        }
    }

    fn build_view(&self) -> GameView {
        let move_views: Vec<MoveView> = self
            .available_moves
            .iter()
            .enumerate()
            .map(|(i, mov)| MoveView {
                index: i,
                description: describe_move(mov),
                is_strike: matches!(mov, Move::Strike),
                is_pass: false,
            })
            .collect();

        GameView {
            player: build_state_view(&self.player_state, &self.player_marks),
            bot: build_state_view(&self.bot_state, &self.bot_marks),
            dice: self.dice.to_vec(),
            white_sum: self.white_sum,
            phase: self.phase.as_str().to_string(),
            available_moves: move_views,
            game_over: self.phase == Phase::GameOver,
            message: self.message.clone(),
        }
    }
}

#[wasm_bindgen]
impl WebGame {
    #[wasm_bindgen(constructor)]
    pub fn new(bot_type: &str) -> Self {
        let mut game = WebGame {
            player_state: State::default(),
            bot_state: State::default(),
            bot_strategy: make_strategy(bot_type),
            rng: SmallRng::from_entropy(),
            phase: Phase::BotActive,
            dice: [0; 6],
            white_sum: 0,
            message: String::new(),
            available_moves: Vec::new(),
            player_marks: [[false; 11]; 4],
            bot_marks: [[false; 11]; 4],
        };

        // Process first bot turn automatically
        game.do_bot_active();

        game
    }

    pub fn view(&self) -> String {
        serde_json::to_string(&self.build_view()).unwrap()
    }

    pub fn make_move(&mut self, index: usize) {
        match self.phase {
            Phase::PlayerPassive => {
                // Player chose to mark during bot's active turn (white sum)
                if index >= self.available_moves.len() {
                    return;
                }
                let mov = self.available_moves[index];
                record_marks(&mut self.player_marks, mov);
                self.player_state.apply_move(mov);

                // Apply locks from bot's turn
                let bot_locked = self.bot_state.locked();
                self.player_state.lock(bot_locked);
                self.propagate_locks();

                if self.check_game_over() {
                    return;
                }

                // Advance to player_active
                self.start_player_active();
            }
            Phase::PlayerActive => {
                // Player picks a move on their active turn
                if index >= self.available_moves.len() {
                    return;
                }
                let mov = self.available_moves[index];
                record_marks(&mut self.player_marks, mov);
                self.player_state.apply_move(mov);

                self.message = format!("You played: {}", describe_move(&mov));

                // Propagate locks from player's move
                self.propagate_locks();

                if self.check_game_over() {
                    return;
                }

                // Bot passive turn
                self.do_bot_passive();

                if self.check_game_over() {
                    return;
                }

                // Start next round: bot active
                self.do_bot_active();
            }
            _ => {}
        }
    }

    pub fn skip(&mut self) {
        match self.phase {
            Phase::PlayerPassive => {
                // Player skips on passive turn — no penalty
                self.propagate_locks();

                if self.check_game_over() {
                    return;
                }

                // Advance to player_active
                self.start_player_active();
            }
            _ => {}
        }
    }

    pub fn new_game(&mut self, bot_type: &str) {
        self.player_state = State::default();
        self.bot_state = State::default();
        self.bot_strategy = make_strategy(bot_type);
        self.rng = SmallRng::from_entropy();
        self.phase = Phase::BotActive;
        self.dice = [0; 6];
        self.white_sum = 0;
        self.message = String::new();
        self.available_moves = Vec::new();
        self.player_marks = [[false; 11]; 4];
        self.bot_marks = [[false; 11]; 4];

        // Process first bot turn automatically
        self.do_bot_active();
    }
}

// ---- State Explorer ----

#[derive(Serialize)]
struct ExplorerResult {
    ga_value: f64,
    dqn_value: f32,
    score: isize,
    blanks: u8,
    probability: f32,
    weighted_probability: f32,
    gene_breakdown: Vec<GeneContribution>,
}

#[derive(Serialize)]
struct GeneContribution {
    name: String,
    raw_value: f64,
    weight: f64,
    contribution: f64,
}

#[derive(serde::Deserialize)]
struct ExplorerInput {
    marks: [[bool; 11]; 4],
    strikes: u8,
    num_opponents: u8,
    max_opponent_strikes: u8,
    score_gap: isize,
}

#[wasm_bindgen]
pub struct StateExplorer {
    ga_champion: bot::DNA,
    dqn: qwixxer::dqn::DqnStrategy,
}

#[wasm_bindgen]
impl StateExplorer {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let genes = Arc::new(bot::default_genes());
        let ga_champion = bot::DNA::load_weights_from_bytes(CHAMPION_BYTES, genes);
        let dqn = qwixxer::dqn::DqnStrategy::load_from_bytes(DQN_MODEL_BYTES);

        StateExplorer {
            ga_champion,
            dqn,
        }
    }

    pub fn evaluate(&self, state_json: &str) -> String {
        let input: ExplorerInput = serde_json::from_str(state_json)
            .expect("Invalid state JSON");

        let state = State::from_marks(&input.marks, input.strikes);

        // GA evaluation
        let ga_value = self.ga_champion.instinct(&state);
        let gene_breakdown: Vec<GeneContribution> = self.ga_champion.gene_contributions(&state)
            .into_iter()
            .map(|(name, raw, weight, contribution)| GeneContribution {
                name: name.to_string(),
                raw_value: raw,
                weight,
                contribution,
            })
            .collect();

        // DQN evaluation
        let ctx = qwixxer::dqn::OpponentContext {
            num_opponents: input.num_opponents,
            max_opponent_strikes: input.max_opponent_strikes,
            score_gap_to_leader: input.score_gap,
        };
        let dqn_value = self.dqn.evaluate_with_context(&state, &ctx);

        // Compute weighted probability: sum of P(free) * (total+1) per row
        let totals = state.row_totals();
        let frees = state.row_free_values();
        let weighted_probability: f32 = frees.iter()
            .zip(totals.iter())
            .map(|(&free, &total)| {
                let Some(f) = free else { return 0.0 };
                let ways = 6.0 - (7.0f32 - f as f32).abs();
                ways / 36.0 * (total as f32 + 1.0)
            })
            .sum();

        let result = ExplorerResult {
            ga_value,
            dqn_value,
            score: state.count_points(),
            blanks: state.blanks(),
            probability: state.probability(),
            weighted_probability,
            gene_breakdown,
        };

        serde_json::to_string(&result).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn game_runs_to_completion() {
        let mut game = WebGame::new("opportunist");
        let mut turns = 0;
        loop {
            let view_json = game.view();
            let view: serde_json::Value = serde_json::from_str(&view_json).unwrap();
            if view["game_over"].as_bool().unwrap() {
                break;
            }

            let moves = view["available_moves"].as_array().unwrap();
            if !moves.is_empty() {
                game.make_move(0); // always pick first move
            } else {
                game.skip();
            }
            turns += 1;
            if turns > 200 {
                panic!("Game didn't end");
            }
        }
    }
}
