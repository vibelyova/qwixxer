# qwixxer

A bot framework for the board game [Qwixx](https://en.wikipedia.org/wiki/Qwixx), with hand-crafted strategies, a genetic algorithm, a DQN value network, MCTS, and both terminal and web UIs.

**[Play in your browser](https://vibelyova.github.io/qwixxer/)** — no installation needed. Supports GA, DQN, and heuristic bots.

## Usage

The native binary uses `clap` subcommands:

```bash
# Play interactively (human vs bots in the terminal)
cargo run --release -- play dqn              # vs DQN
cargo run --release -- play ga opportunist   # vs 2 bots

# Benchmark strategies against each other (with seat rotation)
cargo run --release -- bench dqn ga -n 500000
cargo run --release -- bench dqn dqn ga ga -n 100000  # 2v2, aggregate stats by strategy

# Single-player score benchmark across all strategies
cargo run --release -- solo -n 10000

# Evolve the GA champion
cargo run --release -- evolve

# Train DQN
cargo run --release -- dqn-train          # MC-supervised pretraining
cargo run --release -- dqn-selfplay -i 40 -b 100000  # Self-play RL (40 iters, 100k bench/iter)
```

Model artifacts:
- `champion.txt` — GA weights (committed)
- `dqn_model/model.mpk` — DQN weights (committed)

## Strategies

| Strategy | Description | vs GA (1v1) |
|---|---|---|
| **DQN** | TD(λ) value network + Gaussian P(win) ranking + shared meta-rules | **~60.3%** |
| **GA Champion** | Genetically-evolved 4-weight heuristic + meta-rules | baseline |
| **MCTS** | Monte Carlo tree search with GA rollouts | ~GA-level |
| **Opportunist** | Max weighted probability, blank caps, always-lock | ~30% |
| **Conservative** | Skip at most 2 numbers per mark | ~25% |
| **Random** | Uniform random legal move | ~0% |

All strategies share the **meta-rules** (`State::apply_meta_rules`):
- **Smart lock**: lock to end a winning game; never lock into a loss
- **Smart strike**: strike to end a winning game with 3 strikes; never strike into a loss
- **Move pruning**: remove moves dominated by another (same-row closest-to-free, equal-progress higher-total, post-state dominance on doubles)

## DQN

A TD(λ) value network (not a true DQN — closer to TD-Gammon). The network predicts both `μ(s)` (expected final score) and `log σ²(s)` (log-variance of the final-score distribution). Move selection ranks candidates by P(win) under a Gaussian approximation: `(μ_us − μ_opp) / sqrt(σ²_us + σ²_opp)`, evaluated against the leading opponent's actual state.

**Architecture**: 25→128→64→2 MLP with ReLU, ~11k params. Two-head output (μ, log σ²).

**Features (25)**: per-row progress, marks, locked, weighted probability; our strikes, blanks; num opponents, score gap; **max-across-all-players strikes / locks / row-progress (game-end proximity); aggregate dice-sum probability; lockable-rows count**.

**Losses (decoupled)**: μ trained on TD(λ) target with MSE; σ trained on Gaussian NLL against `(final_score − μ.detach())²` residuals. Detaching μ prevents σ's loss from biasing μ training.

**Training pipeline** (`src/dqn/`):
1. Optional MC-supervised pretraining (1500 games × 500 rollouts each)
2. Self-play RL with TD(λ=0.8) over a 3-iteration replay buffer
3. Four opponent configs per iteration: 1v1 GA/self, 3p/4p vs GA, 3p/4p vs GA + self
4. Color-symmetry data augmentation (8 row-permutations)
5. Per-candidate opponent-context re-inference: each candidate re-builds both our *and* opponent's `OpponentContext` from the post-move score, so features like `score_gap` reflect the move under consideration
6. Reproducible via `TRAIN_SEED` (all RNG seeded; results are deterministic)
7. Per-iteration checkpoints (`iter-N.mpk`) + optional benchmark after each

**Known dynamics**: Win rate peaks around iterations 4-15 at ~60.0-60.3% then plateaus while avg score keeps climbing. The upper ceiling appears structural — Qwixx's shared-dice dynamics mean within-decision σ variation is only ~4%, so variance-aware ranking approaches argmax-μ in practice. See `docs/EXPERIMENTS.md` for σ-validation and distribution diagnostics.

## Genetic Algorithm

Evolves a 4-weight vector over these gene functions:

- `weighted_prob` — `sum(P(rolling free) * (total + 1))` across rows
- `strikes` — strike count
- `concentration` — `sum(total²)` across rows
- `blanks` — total skipped positions

**Training**: 100 individuals, 200 generations, 1,000 simulated 4-player games per generation, parallelized with rayon. Fitness = normalized score relative to the game's top scorer. Uniform crossover + mutation + L2-normalization.

## Project Structure

```
src/
  state.rs         Game state, moves, scoring, meta-rules, move pruning
  game.rs          Game loop, players, dice sources
  strategy/        Strategy trait + heuristic implementations
  bot.rs           GA: DNA, genes, Population, evolution
  dqn/             DQN (split): mod.rs (model + inference + P(win) ranking),
                   train.rs (TD targets, losses, self-play loop)
  mcts.rs          Monte Carlo tree search with pluggable rollout policy
  main.rs          CLI entry point (clap subcommands)
examples/
  sigma_probe.rs           Per-decision (μ, σ) trace for one game
  sigma_validation.rs      DQN (μ, σ) vs empirical MC moments (CSV)
  score_distribution.rs    Rollout distribution from a mid-game snapshot
web/
  src/lib.rs       WASM crate: WebGame state machine + StateExplorer
  src/ts/          TypeScript frontend (game + explorer pages)
  build.sh         wasm-pack + wasm-opt pipeline
  README.md        Build and run instructions
docs/
  ARCHITECTURE.md  Technical reference
  EXPERIMENTS.md   Chronological experiment log
```

## Feature Flags

| Feature | Purpose |
|---------|---------|
| `dqn` | Full DQN training (burn with train support, serde, rayon) — default |
| `burn` | DQN inference only (smaller WASM binary for the web crate) |
| `parallel` | Rayon parallelism for benchmarks and GA training — default |

## Web UI

Built with Vite + TypeScript + WASM. PWA-installable, offline-capable. See `web/README.md` for build instructions. GitHub Pages deployment is automated via `.github/workflows/deploy.yml`.

Two pages:
- **Game** (`index.html`): Play against any bot (human is player 2)
- **State Explorer** (`explorer.html`): Construct arbitrary states, see GA and DQN evaluations with per-gene breakdown

## Rules

Qwixx is a dice game for 2-5 players. Each turn, the active player rolls 6 dice (2 white + 4 colored). All players may mark the white sum in any row; the active player may additionally mark one white+color sum in the matching row. Numbers must be marked left-to-right (ascending for Red/Yellow, descending for Green/Blue). Locking a row requires 5+ marks plus marking the terminal number. The game ends when someone takes 4 strikes or 2 rows are locked. Scoring uses triangular numbers: n marks = n*(n+1)/2 points, minus 5 per strike.
