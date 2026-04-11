# qwixxer

A bot framework for the board game [Qwixx](https://en.wikipedia.org/wiki/Qwixx), featuring hand-crafted strategies, a genetic algorithm for evolving optimal play, a DQN neural network, and both terminal and web UIs.

**[Play in your browser](https://vibelyova.github.io/qwixxer/)** — no installation needed!

## Usage

```bash
# Play against the GA-trained champion bot
cargo run --release

# Train a new champion (population=100, 200 generations)
cargo run --release -- --train

# Benchmark strategies against each other
cargo run --release -- --bench
```

The champion's weights are saved to `champion.txt` after training. If the file exists, `cargo run` loads it automatically; otherwise it trains from scratch.

## Strategies

| Strategy | Description | Avg pts (3p) |
|---|---|---|
| **GA Bot** | Genetically evolved weighted heuristic + always-lock | ~67 |
| **Opportunist** | Maximize probability with blank caps + always-lock | ~54 |
| **Conservative** | Only mark if skipping at most 2 numbers | ~50 |
| **Rusher** | Concentrate marks, rush to lock rows | ~43 |
| **Random** | Pick a random legal move | ~10 |

## Genetic Algorithm

The GA evolves a weight vector over 4 state-evaluation genes:

- **weighted_prob** -- expected points per opponent roll: `P(free) * (total + 1)` per row
- **strikes** -- strike count
- **concentration** -- sum of row totals squared
- **blanks** -- skipped positions

Additionally, the bot always locks a row when the opportunity arises (hardcoded, not learned).

Training: 100 individuals, 200 generations, 1000 simulated 4-player games per generation, parallelized with rayon. The champion consistently discovers that **weighted probability** and **strike avoidance** are the dominant signals.

## Architecture

```
src/
  state.rs      -- Game state, rows, moves, scoring, display
  game.rs       -- Game loop, players, dice sources
  strategy/     -- Strategy trait + implementations (Interactive, Conservative, Rusher, Opportunist, Random)
  bot.rs        -- GA: DNA, genes, population, evolution
  main.rs       -- CLI: play / train / bench modes
```

## Rules

Qwixx is a dice game for 2-5 players. Each turn, the active player rolls 6 dice (2 white + 4 colored). All players may mark the white sum in any row; the active player may additionally mark one white+color sum in the matching row. Numbers must be marked left-to-right (ascending for Red/Yellow, descending for Green/Blue). Locking a row requires 5+ marks and gives a bonus cross. The game ends when someone takes 4 strikes or 2 rows are locked. Scoring uses triangular numbers: n marks = n*(n+1)/2 points, minus 5 per strike.
