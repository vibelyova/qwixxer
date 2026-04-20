# Qwixxer Architecture

## Project Structure

Qwixxer is a Rust workspace with a native binary/library crate and a WASM web crate.

```
qwixxer/
  Cargo.toml          -- lib + bin crate, feature flags: dqn, burn, parallel
  src/
    lib.rs             -- Public module re-exports
    main.rs            -- CLI entry point (clap subcommands: play, bench, solo, evolve, dqn-train, dqn-selfplay)
    state.rs           -- Game state, rows, marks, moves, scoring, display
    game.rs            -- Game loop, players, DiceSource trait
    strategy/mod.rs    -- Strategy trait + Interactive, Conservative, Opportunist, Random
    bot.rs             -- Genetic algorithm: DNA, genes, Population, evolution
    dqn/
      mod.rs           -- DQN model, feature extraction, DqnStrategy, P(win) ranking helpers
      train.rs         -- TD(lambda) targets, decoupled mu/sigma losses, self-play loop
    mcts.rs            -- Monte Carlo tree search with pluggable rollout policy
  web/
    Cargo.toml         -- WASM crate (qwixxer-web), depends on qwixxer with burn feature only
    src/
      lib.rs           -- WebGame state machine, StateExplorer, wasm-bindgen exports
      ts/
        main.ts        -- Game entry point, render loop, event wiring
        board.ts       -- Scoresheet renderer (colored rows, marks, strikes, totals)
        dice.ts        -- Dice display with colored faces
        moves.ts       -- Move parsing, cell click handling, selection state machine
        types.ts       -- TypeScript interfaces (GameView, StateView, RowView, MoveView, etc.)
        explorer.ts    -- State Explorer page: interactive board + live GA/DQN evaluation
        style.css      -- Dark theme with Qwixx color palette
    index.html         -- Play page
    explorer.html      -- State Explorer page
    vite.config.ts     -- Multi-page Vite build, base path /qwixxer/ for GitHub Pages
```

### Feature Flags

| Feature | Description |
|---------|-------------|
| `dqn` | Full DQN training (burn with train support, serde, rayon) |
| `burn` | Burn inference only (model definition + forward pass, no training) |
| `parallel` | Rayon parallelism for benchmarks and GA training |

Default features are `dqn` + `parallel` for the native binary. The web crate uses only `burn` (no rayon, no training code) to keep the WASM binary small (~1 MB with wasm-opt).

---

## Game Engine

### Core Types (`state.rs`)

**`State`** -- A player's game state: 4 rows + a strike count (0-4).

**`Row`** (private) -- Per-row state with three fields:
- `ascending: bool` -- Red/Yellow ascend (2 to 12); Green/Blue descend (12 to 2)
- `total: u8` -- Number of marks in the row (0-12; locking adds a bonus mark)
- `free: Option<u8>` -- Next markable number, or `None` if locked

**`Mark`** -- A specific marking action: `{ row: usize, number: u8 }`.

**`Move`** -- One of three variants:
- `Strike` -- Take a penalty (-5 points)
- `Single(Mark)` -- Mark one number
- `Double(Mark, Mark)` -- Mark two numbers (white sum + color sum, active turn only)

### Move Generation

`State::generate_moves(dice: [u8; 6])` produces all legal active-player moves. The 6 dice are ordered `[W1, W2, R, Y, G, B]`. The method:
1. Computes the white sum (`W1 + W2`)
2. For each row where `can_mark(white_sum)` is true, generates a white-only `Single`
3. For each row `i` and each white die `w`, checks if `w + color[i]` is markable, generating color-only `Single` moves
4. Cross-products white moves with color moves on different rows for `Double` moves, plus same-row doubles where marking white first opens up a color mark
5. Deduplicates via `itertools::unique()`
6. Does NOT include `Strike` -- callers add it themselves

`State::generate_opponent_moves(number: u8)` produces only `Single` marks for the white sum. Passive players cannot use colored dice and can legally skip without taking a strike.

### Row Locking Rules

A row can be locked by marking its terminal number (12 for ascending, 2 for descending), but only if the player already has 5+ marks in that row. Locking adds a bonus mark (`total += 2`: one for the terminal number, one for the lock). A locked row has `free = None` and can no longer be marked by anyone.

### Scoring

Triangular number scoring: `n` marks = `n * (n + 1) / 2` points. Each strike subtracts 5. Maximum per row is 12 marks (including lock bonus) = 78 points.

### State Metrics

- `blanks()` -- Skipped positions across all unlocked rows (marks that were jumped over)
- `probability()` -- Probability that two random dice sum to any free value (for passive turns)
- `probability_n(n)` -- Probability that at least one of `n` opponents rolls a usable sum

### Game Loop (`game.rs`)

**`Player`** wraps a `Strategy`, a `State`, and a `DiceSource`. Each player has their own dice (supports manual input for humans, `SmallRng` for bots).

**`DiceSource`** trait with one method: `fn roll(&mut self) -> [u8; 6]`. Implemented for `SmallRng` (random) and `ManualDice` (stdin).

**`Game::play()`** runs the full game loop:

```
while !game_over():
    active_player rolls dice
    notify active_player of opponent states (observe_opponents)
    active_player picks a move (your_move) -> apply

    for each other player (in seat order):
        notify of opponent states (observe_opponents)
        passive player optionally marks white sum (opponents_move) -> apply
        accumulate locks from passive player's state

    propagate ALL accumulated locks to ALL players
    advance active_player to next seat
```

**Game-over conditions** (checked before each turn):
1. Any player has 4 strikes
2. Any player has 2+ locked rows

The game rotates the active player each turn. `Game::new_from_turn()` allows starting mid-game from a specific seat (used by MCTS rollouts).

---

## Strategy Trait (`strategy/mod.rs`)

```rust
pub trait Strategy: std::fmt::Debug {
    fn your_move(&mut self, state: &State, dice: [u8; 6]) -> Move;
    fn opponents_move(&mut self, state: &State, number: u8, locked: [bool; 4]) -> Option<Move>;
    fn is_interactive(&self) -> bool { false }
    fn observe_opponents(&mut self, _our_score: isize, _opponents: &[State]) {}
}
```

- `your_move` -- Called when this player is active. Receives the full 6-die roll. Must return a `Move` (including `Strike`).
- `opponents_move` -- Called when another player is active. Receives just the white sum and the current global lock state. Returns `Some(Move::Single(...))` to mark, or `None` to skip. No penalty for skipping.
- `is_interactive` -- Returns `true` for the human player (suppresses verbose logging).
- `observe_opponents` -- Called before each decision with the player's current score and all opponents' states. Default is a no-op; overridden by DQN to track opponent context.

---

## Strategies Implemented

### Hand-Crafted

**Random** -- Picks a uniformly random legal move. Baseline for benchmarking.

**Conservative** -- Minimizes new blanks. Only accepts moves that skip at most `max_new_blanks` (default 2) numbers. Among qualifying moves, picks the one creating the fewest blanks. Strikes if nothing qualifies.

**Opportunist** -- Probability-maximizing with blank discipline. Always locks when possible. Caps blanks (2 on active, 1 on passive), then picks the move maximizing `State::probability()` (the chance of getting a useful white sum on future opponent turns). The strongest hand-crafted strategy.

**Interactive** -- Terminal-based human player. Shows the board, dice, numbered move list, and reads input. Groups moves into "Single" and "Both" sections for readability.

### Genetic Algorithm Champion (`bot.rs`)

**DNA** -- A weight vector over 4 gene functions, evaluated as a linear combination:

| Gene | Function | Champion Weight |
|------|----------|----------------|
| `weighted_prob` | `sum(P(rolling free) * (total + 1))` per active row | +0.560 |
| `strikes` | Strike count | -0.826 |
| `concentration` | `sum(total^2)` across rows | +0.021 |
| `blanks` | Total skipped positions | -0.063 |

The `instinct(&self, state: &State) -> f64` method computes the weighted sum. Weights are L2-normalized.

**Meta-rules**: Before evaluating moves, `State::apply_meta_rules()` applies smart lock, smart strike, and move pruning (shared with DQN). The GA also tracks `score_gap` via `observe_opponents`.

**Move selection**: For active turns, evaluates `instinct()` on each resulting state (including Strike) and picks the maximum. For passive turns, also compares against the skip state (with locks applied) to decide whether marking is worthwhile.

### DQN Neural Net (`dqn/mod.rs`, `dqn/train.rs`)

**Architecture**: 25-input MLP with ReLU activations, two-head output.

```
Input (25) -> Linear(128) -> ReLU -> Linear(64) -> ReLU -> Linear(2)
                                                           |    |
                                                           mu   log sigma^2
```

Total parameters: `25*128 + 128 + 128*64 + 64 + 64*2 + 2 = 11,778`.

The model outputs two scalars per state: `mu(s)` (expected final score, TD value learner) and `log sigma^2(s)` (log-variance of the final-score distribution). At inference, `log sigma^2` is clamped to `[LOG_VAR_MIN, LOG_VAR_MAX] = [-5.0, 10.0]`.

**Move ranking via P(win)**: Instead of picking `argmax mu(s')`, the DQN ranks candidates with `win_rank_score(mu_us, log_var_us, mu_opp, log_var_opp) = (mu_us - mu_opp) / sqrt(sigma^2_us + sigma^2_opp)` — monotonic in the Gaussian-approximation probability `P(score_us > score_opp)`. The leading opponent's `(mu_opp, log_var_opp)` is obtained by a forward pass on the opponent's actual state, batched with all our candidates in one forward call.

**Per-candidate opponent context**: Each candidate post-move state rebuilds *both* its own `OpponentContext` (from the post-move our_score) and the opponent's view (from the opponent's score and our post-move state). This matters because context-derived features (score_gap, max_opp_strikes, max_opp_locks, max_opp_total_progress) change across candidates — different moves leave different gaps. Both our and opp feature vectors are batched together (`2N` rows) and evaluated in a single forward pass per decision.

Uses a HashMap cache keyed by quantized features for repeated states.

**Meta-rules** (`State::apply_meta_rules`): Before the model evaluates moves, shared strategic rules are applied:
- **Smart lock**: Always lock to win the game (2+ locked rows and ahead). Always lock the first row. Never lock into a game-ending loss.
- **Smart strike**: With 3 strikes, strike to end the game if ahead. Never strike into a loss (unless forced).
- **Move pruning**: Remove dominated singles (same row: keep closest to free pointer; cross-row: equal blanks + progress, prefer higher total marks).

These meta-rules are shared by DQN, GA, and training code via `State::apply_meta_rules()` and `State::find_smart_lock()`.

**Opponent awareness**: `observe_opponents()` updates an `OpponentContext` with `num_opponents`, `max_opponent_strikes`, and `score_gap_to_leader`, which feed into the last 3 input features.

### Feature Extraction (25 features)

| Index | Feature | Normalization |
|-------|---------|---------------|
| 0-3 | Row progress (how far the free pointer has advanced) | 0.0-1.0, locked=1.0 |
| 4-7 | Row mark counts | /11 |
| 8-11 | Row locked flags | 0 or 1 |
| 12-15 | Per-row weighted probability: `P(rolling free) * (total+1)` | /(6*11) |
| 16 | Our strikes | /3 (max in-play; 4 ends the game) |
| 17 | Our blanks | /40 |
| 18 | Number of opponents | /4 |
| 19 | Score gap to leader | /100, clamped to [-1, 1] |
| 20 | Max strikes across all players (ours + opponents) | /3 |
| 21 | Max locks across all players | capped at 1 (2 ends the game) |
| 22 | Max total row-progress across all players (game-duration proxy) | 0.0-1.0 |
| 23 | Aggregate probability over usable dice sums | 0.0-1.0 |
| 24 | Total lockable rows across all players | /8 |

Features 20-24 are game-end-proximity signals: they let the network predict how much of the game remains, which is the dominant driver of remaining-score variance.

### Monte Carlo Tree Search (`mcts.rs`)

**`MonteCarlo`** runs `N` rollout simulations per candidate move (default 500). For each move:
1. Apply the move to a copy of the current state
2. Simulate `N` complete games from that point using a rollout policy
3. Average the final scores across simulations

The rollout policy is pluggable via `RolloutFactory = Arc<dyn Fn() -> Box<dyn Strategy> + Send + Sync>`. The standard configuration uses the GA champion as the rollout policy.

**Parallelism**: Rollouts run in parallel via `rayon::par_iter` when the `parallel` feature is enabled.

**Passive turns**: Delegates to the rollout strategy's `opponents_move` for fast evaluation (no MC simulation on passive turns).

**Always-lock rule**: Same as GA and DQN.

---

## Training Pipeline

### Genetic Algorithm Evolution

**Population** -- A vector of `DNA` individuals, each with random initial weights.

**`next_generation()`**:
1. Run 1,000 4-player tournaments per generation (parallelized with rayon)
2. In each tournament, 4 randomly-selected DNA bots play a full game
3. Fitness = normalized score relative to the game's top scorer
4. Accumulate fitness across all tournaments
5. **Elitism**: The #1 individual carries forward unchanged
6. Remaining slots: fitness-proportional selection of two parents, uniform crossover (each weight randomly from either parent), then mutation (20% chance per weight of adding uniform noise in [-0.1, +0.1])
7. L2-normalize the weight vector

**Hyperparameters**: Population 100, 200 generations, 1,000 simulations/generation, mutation rate 0.2, elitism = 1.

### DQN Training

**Phase 1: MC-supervised pretraining** (`generate_training_data` + `train`):
1. Play 1,500 games using the GA champion
2. At each active-turn decision point, evaluate all candidate moves with 500 MC rollouts
3. Record `(features_after_move, mc_average_score)` tuples
4. Train the MLP for 50 epochs with MSE loss, Adam optimizer (lr=1e-3), batch size 1024

**Phase 2: Self-play RL** (`self_play_train`):
1. Can start from pretrained model or from scratch (random weights)
2. For each iteration (default 40, 20,000 games per iteration):
   a. Play games with epsilon-greedy exploration (epsilon decays from 0.2 to 0.07)
   b. 4 training configurations (5,000 games each): 3p vs 2 GA, 4p vs 3 GA, 3p vs GA + self-play copy, 4p vs 2 GA + self-play copy
   c. Record `(state_features, TD_target, final_score)` tuples at each decision point for player 0 (the DQN agent)
   d. Compute TD(lambda=0.8) targets: `G_t = (1-0.8) * V(s_{t+1}) + 0.8 * G_{t+1}`, with `G_{n-1} = final_score`
   e. Add new samples to a replay buffer (last 3 iterations retained, VecDeque)
   f. Retrain for 10 epochs on the full replay buffer (Adam lr=4e-4, batch 1024)
   g. Optionally benchmark against GA each iteration (`--bench N`)
   h. Save per-iteration checkpoint (iter-N.mpk)

**Decoupled loss (mu + sigma)**: The joint loss is `L = L_mu + alpha * L_sigma` with `alpha = 1.0` (see `SIGMA_LOSS_WEIGHT`).
- `L_mu = mean((G_t - mu)^2)` — MSE on the TD(lambda) target.
- `L_sigma = 0.5 * mean(log_var + (final_score - mu.detach())^2 * exp(-log_var))` — Gaussian NLL against the **final-score residual** (not the TD target). `mu.detach()` prevents sigma's gradient from corrupting mu training. Training sigma against TD residuals was tried first and failed (corr = -0.44 vs empirical MC sigma), because G_t has lower variance than the true final-score distribution; training against final-score residuals gives an unbiased estimator (corr = +0.91, see `examples/sigma_validation.rs`).

**Data augmentation**: The batcher randomly swaps symmetric row features (3 independent swaps, 8 permutations): red↔yellow, green↔blue, ascending↔descending pairs.

**Reproducibility**: All RNG is seeded from a global `TRAIN_SEED` constant. Per-game seeds are deterministic (`TRAIN_SEED + iteration * games + game_idx`). Burn's weight initialization is seeded. Training is reproducible across runs (identical training scores, model weights may differ slightly due to floating-point non-determinism in burn's optimizer).

**Training infrastructure**: Self-play games are parallelized with rayon. Models are pre-cloned (Arc-backed NdArray tensors make cloning cheap) instead of saving/loading from disk. LTO and `target-cpu=native` enabled for release builds.

**Key finding**: With P(win) ranking and 25-feature inputs, win rate peaks at **60.3% vs GA** around iterations 4-15 and then plateaus while avg score keeps climbing. The upper ceiling is structural — empirical Monte Carlo rollouts confirm that within-decision sigma variation across candidates is typically only 4-5%, so the P(win) ranker degenerates to argmax-mu at most decisions. Early stopping at the win-rate peak still produces the best checkpoint.

---

## Web UI Architecture

### WASM Crate (`web/src/lib.rs`)

**`WebGame`** is a `#[wasm_bindgen]` struct that manages a 2-player game (human vs bot) as a state machine with phases:

```
BotActive -> PlayerPassive -> PlayerActive -> BotPassive -> (BotActive or GameOver)
```

- `BotActive`: Bot rolls dice, picks a move. Transitions to `PlayerPassive`.
- `PlayerPassive`: Player may mark the white sum (or skip). Transitions to `PlayerActive`.
- `PlayerActive`: Player rolls dice, picks from full move set. Bot then takes its passive turn. Transitions back to `BotActive`.
- `GameOver`: Final scores displayed.

The struct tracks per-cell marks (`player_marks` and `bot_marks` as `[[bool; 11]; 4]`) separately from the `State` free pointer, so the UI can distinguish marked cells from skipped cells.

**Exported methods**: `new(bot_type)`, `view() -> JSON`, `make_move(index)`, `skip()`, `new_game(bot_type)`.

**Model embedding**: The GA champion weights (`champion.txt`) and DQN model (`dqn_model/model.mpk`) are embedded at compile time via `include_bytes!`. The DQN model loads via `NamedMpkBytesRecorder` with half-precision settings.

**`StateExplorer`** is a second `#[wasm_bindgen]` struct for the explorer page. It holds both a GA champion and a DQN model, and evaluates arbitrary board states constructed from a JSON-serialized marks array. Returns GA value, DQN value, score, blanks, probability, weighted probability, and per-gene breakdowns.

### TypeScript Frontend

Built with Vite (vanilla TypeScript, no framework). Multi-page configuration serves both `index.html` (game) and `explorer.html` (explorer).

**Board rendering** (`board.ts`): Programmatically constructs DOM elements for each row and cell. Cells are classified as `marked` (X overlay), `skipped` (dimmed), `available` (colored), or `lock-unavailable` (lock number without 5 marks). Footer shows clickable strike boxes and a detailed scoring breakdown.

**Move interaction** (`moves.ts`): Players click directly on scoresheet cells. The system:
1. Parses move descriptions (e.g., "RED 7 + GRN 10") into `MoveTarget` arrays
2. Highlights all cells that appear in any legal move
3. As cells are clicked, narrows compatible moves and updates highlights
4. Requires explicit Confirm button press (no auto-confirm)
5. Supports toggle deselection (click a selected cell to unselect)

**State Explorer** (`explorer.ts`): A standalone page where users can click any cell to toggle marks, click strike boxes, and adjust sliders for opponent count / max opponent strikes / score gap. Evaluations update live, showing both GA instinct value and DQN predicted value with a full gene-by-gene breakdown table.

---

## Benchmarking (`main.rs`)

The `bench` subcommand runs N games between 2+ bot types with seat rotation to cancel position bias.

**Seat rotation**: Game `i` rotates seats by `i % num_players`, so each bot plays from every position equally across the benchmark run.

**Parallelism**: When the `parallel` feature is enabled, games run on a rayon thread pool. Each game constructs its own strategies and RNG (no shared mutable state). Results are collected into a `Vec` and aggregated serially.

**Aggregation**: Tracks wins (outright, no ties), total points, and tie count per bot. Reports win rate percentage and average points. When multiple bots share a strategy, prints aggregate stats where ties between same-strategy bots count as wins for that strategy. For 1v1 and 2-strategy matchups, shows 99% confidence interval (SE = sqrt(p*(1-p)/N), z=2.576).

**Solo mode**: Runs single-player games for each strategy, reporting average/min/max scores.

**Scale**: 500,000 games between DQN and GA complete in ~32 seconds with rayon parallelization and DQN batch+cache optimizations.

---

## PWA Support

The web app is installable as a Progressive Web App with offline caching:
- `web/public/manifest.json` — app name, theme color, icons
- `web/public/sw.js` — service worker with stale-while-revalidate caching
- WASM binary optimized with `opt-level='z'`, `strip`, `panic='abort'`, and `wasm-opt -Oz` (~1 MB)
- Build script: `web/build.sh` runs wasm-pack + wasm-opt in one step
