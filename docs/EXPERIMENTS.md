# Qwixxer Experiment Log

A chronological record of everything we tried while building the Qwixx bot framework, including what worked, what failed, and why.

---

## Phase 1: Game Engine and Hand-Crafted Strategies

### Building the core engine

Started with `State`, `Row`, `Move`, `Mark` types and a basic game loop. The initial implementation had a critical **opponent double-move bug**: opponents were allowed to mark two rows per turn (using the full `generate_moves`), when the Qwixx rules only allow passive players to mark the white dice sum in a single row. Fixed by adding `generate_opponent_moves()` which only returns `Single` marks.

Also fixed: `DiceSource` was on `Game` (shared), moved to `Player` so each player can have their own dice source (manual for humans, RNG for bots).

### Conservative strategy

First real strategy. Only marks numbers that skip at most 2 positions. Picks the minimum-blank option. Beats Random ~95% of the time.

### Rusher strategy

Concentrate marks in few rows, race to lock. Scores states with `locked * 1000 + sum(total^2) - blanks - strikes*30`. Competitive with Conservative in 3-player games:

| Strategy | Win % (3p) |
|----------|-----------|
| Conservative | ~52% |
| Rusher | ~49% |
| Random | ~2% |

### Opportunist strategy

Combined Conservative's blank discipline with probability-maximizing move selection and always-lock. The insight: on passive turns, you want your free pointers on common 2d6 sums (6, 7, 8) to maximize free marks.

| Strategy | Win % (3p) | Avg Pts |
|----------|-----------|---------|
| Opportunist | 39.8% | 54.1 |
| Conservative | 35.9% | 52.7 |
| Rusher | 27.2% | 44.9 |

**Key insight**: Probability management is more important than concentration. Getting more passive-turn marks compounds across the game.

---

## Phase 2: Genetic Algorithm

### Gene design iterations

**Attempt 1: 5 orthogonal genes** -- `concentration`, `blanks`, `strikes`, `lockable_rows`, `probability`. Training DNA against itself produced mediocre results (~27-32% in 3-player vs Conservative ~42-45%).

**Attempt 2: Training against sparring partners** -- Added Conservative and Rusher as opponents in tournament games. Results were closer (GA 37.5% vs Conservative 39.1%) but still not dominant. The GA was learning to beat specific opponents rather than play well generally.

**Attempt 3: DNA-only training** -- Removed sparring partners. All 4 players in each tournament game are DNA bots competing against each other. This forces the GA to discover generally strong play. Combined with:
- Shuffled seat order (eliminate position bias)
- Always-lock rule (hardcoded, not learned)
- 1,000 simulations per generation (up from 500)

Result: **65.7%** win rate vs Conservative + Rusher.

**Attempt 4: Replace probability + lockable_rows with weighted_probability** -- New gene: `sum(P(rolling free) * (total + 1))` per row. Captures both the likelihood of getting a useful roll AND the payoff (more marks = more triangular-number points). Subsumes plain probability.

Final 4 genes: `weighted_prob`, `strikes`, `concentration`, `blanks`.

Result: **71.3%** win rate, 67.5 avg pts. Best yet.

### Hyperparameter experiments

| Config | Win % vs Conservative + Rusher |
|--------|-------------------------------|
| pop=100, gen=200, elite=1 (baseline) | **71%** |
| elite=3 | 67.6% |
| elite=5 | 68.5% |
| pop=200, gen=100 | 45.2% |
| pop=200, gen=200 | 65.9% |
| pop=50, gen=400 | 67.6% |

**Key insight**: More elitism hurts diversity. Larger populations need proportionally more generations. The baseline (pop=100, gen=200, elite=1) was already near-optimal.

### 2-player vs 4-player training

Tried switching to 2-player head-to-head training. Converged weights shifted: `weighted_prob` +0.73 (up from +0.56), `strikes` -0.68 (softer). Benchmark: 67.9% -- **worse** than 4-player training (71%). Reverted.

**Key insight**: 4-player training produces more robust play because it exposes the GA to more diverse opponent behaviors per game.

### Always-lock rule

Hardcoded: if any legal move locks a row, take it immediately. Evolution consistently rediscovered this, so we baked it in. Combined with original genes and 1,000 sims/gen: 67.4% (up from a failed 4-gene experiment at 16-19%).

### Final champion

Converged weights: `weighted_prob +0.560`, `strikes -0.826`, `concentration +0.021`, `blanks -0.063`.

Translation: **avoid strikes above all else**, then **maximize expected future marks** (weighted probability). Concentration and blanks have near-zero weight -- they're correlated with weighted probability and the GA figured out they're redundant.

Head-to-head benchmark (2-player, seat-rotated):

| Matchup | GA Win % |
|---------|---------|
| GA vs Opportunist | **70.2%** |
| GA vs GA (self-play, 500k games) | ~54% (slight edge to seat position) |

---

## Phase 3: Monte Carlo Tree Search

### Basic MCTS

500 simulations per move, GA champion as rollout policy. For each candidate move, simulate 500 complete games and average the final scores. Parallelized with rayon.

| Matchup (3-player) | Win % |
|--------------------|-------|
| MCTS | **47.4%** |
| GA | 41.2% |
| Third player | ~11% |

MCTS beats GA in 3-player because it can simulate the exact consequences of each move, while the GA relies on a linear heuristic. However, head-to-head results were closer -- the noise from 500 simulations means MCTS and GA often agree.

| Matchup | MCTS vs Opportunist |
|---------|-------------------|
| Win % | **73.5%** |

### MCTS divergence analysis

Played 100 games where GA makes moves but MCTS also evaluates each decision. Found **17.4% divergence rate**. The #1 pattern (176/379 disagreements): GA chooses a Double move when MCTS prefers a Single. The second mark damages future positioning (creates blanks, moves free pointer to a low-probability number) more than the immediate points help. #2 pattern: GA marks when MCTS would Strike (24/37 cases).

**Key insight**: The "double move problem" -- greedily taking both marks on an active turn is often worse than taking just the white sum mark. The GA's linear heuristic can't capture this interaction; MCTS discovers it through simulation.

### MC + DQN rollouts

Tried using the DQN neural net as the rollout policy instead of GA. Impractically slow -- each MCTS simulation requires many DQN forward passes for the rollout policy, and the neural net is much slower than the GA's linear `instinct()` function. Abandoned.

---

## Phase 4: DQN Neural Network

### MC-supervised pretraining

Generated training data by playing 500 games with the GA champion, evaluating each decision point with 200 MC simulations. This produced (state_features, mc_value) pairs. Trained a small MLP (18 inputs, 32-16-1 hidden layers, 1,153 params) for 50 epochs.

Initial DQN: **73.3% vs Opportunist** -- matched MCTS's accuracy with instant inference (no simulation needed at play time).

### Self-play reinforcement learning

Iteratively improved the net by playing against itself and GA opponents:
1. Play games with epsilon-greedy exploration
2. Compute TD(lambda=0.8) targets backwards through the trajectory
3. Retrain on a replay buffer (last 5 iterations)

Self-play avg score improved from 59.4 to 71.3 across 20 initial iterations. DQN: **75.5% vs Opportunist** after self-play (up from 73.3% MC-supervised).

### Feature engineering

**18-feature baseline**: 4x row progress, 4x mark counts, 4x locked flags, 2x aggregated metrics (strikes, blanks, probability, weighted_prob).

**21-feature expansion**: Added opponent-aware features:
- `num_opponents` (normalized /4)
- `max_opponent_strikes` (normalized /4)
- `score_gap_to_leader` (normalized /100, clamped [-1, 1])

Also split weighted probability into 4 per-row features (indices 12-15) instead of one aggregate, giving the net row-level detail.

21-feature DQN: **51% vs GA head-to-head** (first time beating GA). Score gap feature was critical (~46% without it).

### Architecture iterations

**18-32-16-1 (original)**: 1,153 params. Decent baseline.

**21-64-32-1 (final)**: 3,489 params. Bigger net + TD(lambda=0.8) targets: **52.6% vs GA, 73.5% vs Opportunist**.

### Focused 3-4 player training

Dropped MC pretraining entirely. Trained from random initialization using only 3-4 player games against GA champions and self-play copies. 4 configurations per iteration: 3p vs 2 GA, 4p vs 3 GA, 3p vs GA + self, 4p vs 2 GA + self.

| Iteration | vs Opportunist | vs GA (1v1) | vs GA (2v2) |
|-----------|---------------|-------------|-------------|
| From scratch, 3-4p | 77.1% | 53.2% | 42% |
| +80 iterations | **78.3%** | **56.4%** | **44.5%** |

### Batch normalization + target network

Added BatchNorm after each hidden layer and a target network (frozen copy updated every 5 iterations) to stabilize training.

Result: **49.5% vs GA** -- worse than the 56.4% baseline.

**Why it failed**: BatchNorm computes running statistics during training (batch size 256) but inference is batch-size-1 (evaluating one state at a time). The batch statistics were unreliable at inference time, degrading move quality. The target network didn't compensate.

Reverted to the simple architecture. Simple wins.

### Extended training (200 iterations)

Ran 200 iterations (up from 80). Avg score plateaued at 64-65 throughout. Benchmark slightly regressed (52.6% vs GA, was 56.4%). The value function had fully converged.

### Best DQN result

| Matchup | DQN Win % | Avg Pts |
|---------|----------|---------|
| DQN vs GA (1v1, 500k games) | **54.2%** | -- |
| DQN vs Opportunist | **78.3%** | -- |
| DQN vs GA (2v2) | 44.5% | -- |

**The 2v2 gap**: DQN performs worse in team settings. Hypothesis: DQN was trained primarily in configurations where it's the sole DQN player, so it hasn't learned to exploit situations where two DQN players could coordinate (or at least not interfere with each other's strategies).

---

## Phase 5: Policy Network and REINFORCE

### Policy network design

Instead of a value function (score states), tried a policy network that scores (state, move) pairs directly. 34 features: 20 state features + 4 weighted probability features (after move) + 10 move descriptors (strike/pass flags, row one-hots, blanks created, rows locked).

### MSE regression training

Trained the policy net to predict final score from (state, move) features. Played games with GA champion, recorded features and outcomes.

Result: **33% vs GA** -- weak. The net learned move-score correlations but couldn't distinguish good moves from bad ones in the same state.

### REINFORCE from scratch

Switched to REINFORCE policy gradient: increase probability of moves from above-average games, decrease for below-average. Custom training loop with per-game advantage weighting and a moving baseline.

Result: **0.3% vs GA** -- complete failure. Cold start problem: with random initial weights, the net plays randomly, generates only losing games, and the gradient signal is pure noise. REINFORCE needs a reasonable starting policy to generate meaningful training signal.

### REINFORCE with pretraining

First pretrained with MSE regression (2,000 games, 30 epochs), then switched to REINFORCE updates.

Result: **collapsed immediately**. The REINFORCE gradients were so noisy that they destroyed the pretrained weights within a few iterations. The policy net quickly degenerated back to near-random play.

**Key insight**: REINFORCE requires either (a) a very good initial policy or (b) a massive number of games per update to reduce gradient variance. For a game with Qwixx's branching factor, neither was practical. Value-function approaches (DQN) work much better because they can learn from individual state evaluations rather than full-game outcomes.

---

## Phase 6: Training Objective Experiments

Investigated whether the DQN should optimize for winning rather than score maximization.

### Win-blend (50/50 score + did_win)

Target: `0.5 * normalized_score + 0.5 * did_win`. Rationale: GA was evolved to win tournaments, DQN was trained on score -- this mismatch might explain why DQN has higher avg score but lower win rate in multiplayer.

Result: **52.3% vs GA** -- worse than the 54.2% score-only model.

### Pure win signal

Target: 0 for loss, 100 for win (binary).

Result: **26.5% vs GA** -- much worse. The binary signal is too sparse: most games are losses (playing against a strong opponent), so the net mostly sees "0" targets and can't learn useful gradients.

### Conclusion

**Score maximization is the best training objective.** In Qwixx, higher scores correlate strongly with winning. The extra signal density from continuous scores (vs binary win/loss) more than compensates for any misalignment between score and win probability. The GA's win-rate edge comes from its evaluation function being tuned to a population of opponents, not from optimizing a different objective.

---

## Phase 7: Other Experiments

### Curriculum randomization

50% of training games start from a random mid-game state (`State::random()` with random free pointers, totals, and strikes) instead of the default. Idea: expose the DQN to states not normally reached through its own play.

Result: **no improvement**. The randomly generated states were often unreachable or unrealistic (e.g., free pointer at 12 with 0 marks). The DQN wasted capacity learning to evaluate nonsensical positions.

### 1-step lookahead

Instead of evaluating the immediate post-move state, simulate all 11 possible opponent white sums, pick the best mark for each, and score the expected resulting state.

Result: 74.4% vs Conservative+Rusher (up from 71%). But this was **redundant with the weighted_probability gene**, which already captures expected future mark value. The GA's `instinct()` with weighted_prob produces the same move rankings as the explicit lookahead at a fraction of the cost. Reverted.

**Key insight**: Good features can substitute for explicit search. The weighted_probability gene is essentially a one-step lookahead compressed into a single feature.

### Phase-based weights (early/late game)

Separate weight vectors for early game (< 3 marks per row on average) and late game (>= 3). Idea: the optimal strategy might change as the game progresses.

Result: **65.3%** -- worse than single weights (71%). The search space doubled (8 parameters instead of 4) without the GA having enough generations to explore it. Furthermore, Qwixx strategy doesn't change much across phases: weighted probability and strike avoidance are dominant throughout.

### Non-linear genes (squared terms)

Added squared versions of weighted_prob and concentration as additional genes. Idea: allow the evaluation function to capture non-linear interactions.

Result: **worse**, with convergence issues. The squared terms made the fitness landscape less smooth, and the GA struggled to find good weight vectors in the larger search space.

### Separate active/passive weights

8-parameter DNA: 4 weights for active-turn evaluation, 4 weights for passive-turn evaluation. Idea: marking decisions should weight factors differently depending on whether you're active (full dice) or passive (white sum only).

Result: **worse**. 8 parameters is too many for the GA to optimize in 200 generations. The active/passive distinction didn't provide enough signal to justify the increased dimensionality.

---

## Phase 8: Performance Optimization

### DQN cache + batch evaluation

Two optimizations for DQN inference:
1. **HashMap cache**: If the same state features are evaluated twice (common in move generation), return the cached value. The DQN is deterministic, so caching is exact.
2. **Batch forward pass**: Evaluate all candidate moves in a single tensor operation instead of N separate forward passes.

Result: **2x speedup** (500-game benchmark: 16s vs 30s baseline).

### Rayon-parallelized benchmarks

Each game runs independently on its own thread with its own strategy instances and RNG. No shared mutable state. Results collected into a `Vec` and aggregated.

**500,000 DQN vs GA games complete in ~32 seconds** on a multi-core machine.

---

## Phase 9: Web UI

### WASM compilation

Split the DQN module: inference code (model definition, forward pass, DqnStrategy) compiles with just the `burn` feature; training code (data generation, self-play, Learner) gated behind `dqn` feature. This keeps the web crate free of rayon and burn-train dependencies.

Model embedding: champion weights (text) and DQN model (binary mpk) are `include_bytes!` at compile time. The DQN loads via `NamedMpkBytesRecorder` with half-precision settings.

**WASM binary size: 1.6 MB** (includes burn NdArray inference runtime).

### Board game-style scoresheet

The web UI mimics a physical Qwixx scoresheet:
- 4 colored rows (red, yellow, green, blue) with 11 numbered cells each
- Marked cells show an X overlay with the number faintly visible underneath
- Skipped cells are dimmed
- Lock cells (12/2) are grayed out until 5 marks are reached
- Clickable strike boxes in the footer
- Detailed scoring breakdown: per-row triangular scores + - strikes = total

Players interact by clicking directly on scoresheet cells. The system highlights valid targets, narrows options as selections are made, and requires explicit Confirm.

### State Explorer

A separate page (`explorer.html`) for analyzing bot evaluations:
- Click any cell to toggle marks on/off
- Click strike boxes to add/remove strikes
- Sliders for opponent count, max opponent strikes, and score gap
- Live evaluation updates showing: score, blanks, probability, weighted probability, GA instinct value, DQN predicted value, and a per-gene breakdown table

Useful for understanding why the bots make specific decisions and where GA and DQN disagree.

### Deployment

GitHub Pages via GitHub Actions. Vite builds the TypeScript + WASM frontend with base path `/qwixxer/`.

---

## Phase 10: Academic Comparison

### Blank's thesis strategies (Cal Poly, 2024)

Reimplemented both strategies from Joshua Blank's thesis "Qwixx Strategies Using Simulation and MCMC Methods":

**Score-based strategy**: Solo avg 87.7 (thesis reports 90.4). Our implementation is slightly lower, likely due to minor differences in skip-score handling.

**Race-to-lock strategy**: Solo avg 79.2 (thesis reports 76.6). Our implementation is slightly higher after fixing a fallback bug where the original version was striking excessively (3.35 strikes/game) when no Markov-allowed mark existed. After adding smallest-gap fallback, strikes dropped to 2.51/game.

### Solo benchmarks

| Strategy | Avg Solo Score |
|----------|---------------|
| Blank's score-based | **87.7** |
| Blank's race-to-lock | 79.2 |
| GA Champion | 71.0 |
| Conservative | 70.6 |
| Opportunist | 65.6 |
| Rusher | 44.7 |
| Random | 6.6 |

Blank's strategies dominate in solo play because they were designed for it. The score-based strategy in particular optimizes a solo-specific formula.

### Multiplayer: GA vs Blank

| Matchup | GA Win % | Blank Win % | GA Avg Pts | Blank Avg Pts |
|---------|---------|------------|-----------|--------------|
| GA vs Score-based (1v1) | **81.8%** | 17.2% | 66.5 | 44.7 |
| GA vs Score-based (2v2) | **83.2%** | 15.5% | 57.1 | 35.4 |
| GA vs Race-to-lock (1v1) | **~71%** | ~27.4% | -- | -- |

**Key insight**: Solo-optimized strategies collapse in multiplayer. Blank's score-based strategy achieves 87.7 in solo but only 44.7 avg points against the GA in 1v1. The GA's multiplayer awareness (through tournament-based evolution and opponent-turn decision making) gives it a massive edge.

The race-to-lock strategy fares better because its Markov chain approach naturally handles opponent interference (unexpected row locks), but it still loses convincingly.

### Comparison with optimal single-player

Bmhowe34's dynamic programming solution achieves ~115.5 points in optimal single-player Qwixx. This isn't directly comparable to our multiplayer results since the DP solution uses perfect information and exhaustive search over a single player's decision tree, while multiplayer introduces opponent actions, imperfect information, and strategic interaction.

---

## Summary of What Worked and What Didn't

### Techniques that worked

| Technique | Impact |
|-----------|--------|
| Weighted probability gene | Best single gene; captures future mark value |
| Always-lock rule (hardcoded) | Consistent improvement across all strategies |
| DNA-only 4-player training | More robust than sparring partner training |
| DQN with MC-supervised pretraining | Matched MCTS accuracy with instant inference |
| TD(lambda=0.8) self-play | Improved DQN from 73% to 78% vs Opportunist |
| Opponent-aware features (21 inputs) | First DQN to beat GA head-to-head |
| Batch + cache evaluation | 2x DQN inference speedup |
| Rayon parallelism | 500k games in 32 seconds |
| Score maximization as objective | Better than any win-based variant |

### Techniques that failed

| Technique | Why it failed |
|-----------|--------------|
| REINFORCE policy gradient | Cold start: random policy generates no useful signal |
| REINFORCE with pretrain | Noisy gradients destroyed pretrained weights |
| Batch normalization | Batch-size-1 inference produces unreliable statistics |
| Target network | Didn't compensate for BN problems |
| Win-blend training objective | Score signal is already aligned with winning |
| Pure win signal | Too sparse for gradient-based learning |
| Phase-based weights | Doubled search space without benefit |
| Non-linear genes | Worse convergence in noisier landscape |
| Separate active/passive weights | 8 params too many for 200 generations |
| Curriculum randomization | Random states were unrealistic |
| MC + DQN rollouts | Impractically slow |
| 2-player GA training | Less robust than 4-player |
| Extended DQN training (200 iters) | Fully converged at 80 iterations |

### Overall rankings (multiplayer)

At 500,000 games with seat rotation, the approximate hierarchy is:

```
DQN (~54% vs GA head-to-head)
 > GA Champion (~70% vs Opportunist)
   > MCTS (~73% vs Opportunist, ~tied with GA)
     > Opportunist
       > Conservative
         > Blank's Race-to-Lock
           > Blank's Score-Based (in multiplayer)
             > Rusher
               > Random
```

The DQN and GA are close enough that variance matters at smaller sample sizes. MCTS is approximately GA-level but much slower. All learned strategies dominate the hand-crafted ones.
