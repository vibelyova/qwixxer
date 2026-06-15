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

---

## Phase 5: Meta-Rules and Strategic Improvements

### Smart lock rule (+3.5% win rate)

The biggest single improvement. Instead of always locking when possible, the bot now checks:
- **Lock to win**: If locking ends the game (2+ locked rows) and we're ahead, force it.
- **Don't lock into a loss**: If locking ends the game and we're behind, don't do it.
- **First lock**: Always lock the first row (doesn't end the game, pure upside).

Result: DQN jumped from ~53.5% to ~57% vs GA. The GA bot also benefits from the same rule.

Experiments isolating the effect (200k games each):
| DQN lock | GA lock | DQN win % |
|----------|---------|-----------|
| forced | forced | 57.8% |
| forced | relaxed | 57.1% |
| relaxed | forced | 56.0% |
| relaxed | relaxed | 55.2% |

Forced first-lock helps DQN (+1.8%). Relaxing GA barely matters. The rule is kept as forced for both bots.

### Smart strike rule

With 3 strikes: intentionally strike to end the game if ahead. Never strike into a loss (unless forced). Marginal impact (~0.1%) since the scenario is rare.

### Move pruning

Prune dominated singles before model evaluation:
1. **Same row**: Keep only the mark closest to free pointer (fewer blanks, more future options).
2. **Cross-row**: Equal blanks + equal resulting progress → prefer the row with higher total marks (more marginal score from triangular scoring).

Theoretically correct (strict dominance), marginal benchmark impact (~0.1%).

### Extracted meta-rules (`State::apply_meta_rules`)

All strategic rules consolidated into `State::apply_meta_rules()` and `State::find_smart_lock()`. Returns `MetaDecision::Forced(move)` for smart lock/strike, or `MetaDecision::Choices(moves)` with pruned list. Shared by DQN, GA, and training code — eliminated ~60 lines of duplication.

### Network architecture experiments

| Architecture | Win % vs GA | Notes |
|-------------|------------|-------|
| 64→32 (3.5k params) | ~53.5% | Original, pre-meta-rules |
| 128→64 (11k params) | ~56% | Current, with meta-rules |
| 96→48 (7k params) | ~56% | No improvement over 128→64 |

Network size doesn't matter much — the bottleneck is the training objective (score vs winrate).

### Hyperparameter sweep

Many hyperparameters tested with no significant impact beyond meta-rules:

| Change | Result |
|--------|--------|
| Data augmentation (8 color permutations) | Neutral (53.5% with or without) |
| Batch size 256→1024→2048 | Neutral when LR scaled proportionally |
| LR decay (0.97 per iter) | Slightly worse |
| Epsilon floor 0.01 vs 0.05 vs 0.07 | Neutral |
| Replay buffer 3 vs 5 iterations | Neutral |
| No replay buffer (30k fresh games) | Worse (51.9%) |
| MC pretrain (1500×500 sims) | Same ceiling as from scratch |
| OpenBLAS | Slower for small matrices |
| 1v1 training configs | Worse (diluted multi-player training) |

### Score vs win rate divergence

**Key discovery**: Win rate peaks around iteration 9-10 (~58.8%) then declines to ~56% while avg score keeps climbing from 62 to 65+. The model optimizes for score (TD target = final game score), which diverges from winning after ~10 iterations.

| Iteration | Avg Score | Win Rate |
|-----------|-----------|----------|
| 5 | 55.7 | 54.8% |
| 9 | 62.2 | **58.8%** |
| 15 | 64.3 | 57.6% |
| 25 | 65.2 | 56.4% |

This explains why our "lucky" 57.2% model (from early experiments) couldn't be reproduced — it was trained for 40 iterations but with a seed that happened to plateau well. Systematic benchmarking per iteration reveals the true optimum is at ~10 iterations.

### Reproducible training

All RNG seeded from a global `TRAIN_SEED` constant. Verified: two runs produce byte-identical training scores. Model weights differ slightly (floating-point non-determinism in burn's optimizer) but benchmark identically.

### Techniques that failed (Phase 5)

| Technique | Why it failed |
|-----------|--------------|
| Relaxed lock rule for DQN | Model delays locking to chase score, hurting winrate |
| Score gap as training target | More noise than absolute score, model learns near-identity |
| Tabular TD (20M states) | Only 12.7% state coverage after 100M games, 31% vs GA |

### Overall rankings (multiplayer)

At 500,000 games with seat rotation and meta-rules enabled:

```
DQN at iteration 9 (~58.8% vs GA, 99% CI: 58.5-59.1%)
  > DQN at iteration 40 (~56% vs GA)
    > GA Champion (~70% vs Opportunist)
      > MCTS (~73% vs Opportunist, ~tied with GA)
        > Opportunist
          > Conservative
            > Blank's Race-to-Lock
              > Blank's Score-Based (in multiplayer)
                > Rusher
                  > Random
```

The DQN's advantage over GA comes primarily from meta-rules (smart lock/strike) rather than neural net evaluation.

---

## Phase 11: Variance-Aware Ranking (P(win) Formula)

Motivation: the score-maximizing DQN plateaus at ~58.8% vs GA even though it clearly beats GA on expected points. Hypothesis: `argmax μ(s')` is the wrong objective when you're behind — you want to *maximize the probability of finishing ahead*, which under a Gaussian approximation is monotonic in `(μ_us − μ_opp) / √(σ²_us + σ²_opp)`. This requires the net to predict σ as well as μ.

### Two-head model: μ + log σ²

Changed the output layer from `Linear(1)` to `Linear(2)`: first scalar is `μ`, second is `log σ²`. Clamped `log σ²` to `[-5.0, 10.0]` at inference.

Initial joint Gaussian NLL loss: `L = 0.5 * (log σ² + (G_t − μ)² / σ²)`. Trained end-to-end. Benchmark vs GA: no improvement, sometimes worse. Investigation followed.

### Why σ was wrong: TD target ≠ final-score distribution

Ran `examples/sigma_validation.rs` — for each decision in real games, compare DQN's `(μ, σ)` to empirical `(μ, σ)` from 10k Monte Carlo rollouts from the post-move state.

Result: **Pearson correlation(DQN σ, empirical σ) = −0.44**. The variance head was *anti-correlated* with truth.

Root cause: we were training σ against `(G_t − μ)²`, where `G_t` is the TD(λ=0.8) target. G_t is a smoothed estimate of expected return; it has systematically *lower variance* than the actual final-score distribution because TD smoothing averages across the tail of the trajectory. Training σ on G_t residuals gives a biased estimator of `Var(X_final | s)`.

### Fix: decoupled losses with final-score residuals

Split the loss:
- `L_μ = mean((G_t − μ)²)` — MSE on the TD target, unchanged.
- `L_σ = 0.5 * mean(log σ² + (final_score − μ.detach())² * exp(−log σ²))` — Gaussian NLL against **actual final-score residual**. `μ.detach()` critical: without it, σ's gradient corrupts μ training (raising μ reduces `L_σ` if σ is under-predicting).

Total: `L = L_μ + α * L_σ` with `α = 1.0`.

Re-validated: **Pearson(DQN σ, empirical σ) = +0.91**, DQN median σ = 20.6 vs empirical 19.8. The variance head now tracks empirical truth.

### P(win) ranking at decision time

`win_rank_score(μ_us, log_σ²_us, μ_opp, log_σ²_opp) = (μ_us − μ_opp) / √(σ²_us + σ²_opp)`.

Requires the opponent's μ/σ from a forward pass on the leading opponent's actual state. Batched efficiently: all our candidates + all their corresponding opp-views are packed into a single `2N`-row forward call per decision. A `rank_candidates_with_opp_context` helper in `dqn/mod.rs` does this.

### Per-candidate opponent context

Initial bug: we computed opp's features once per decision (using `self.context` captured in `observe_opponents`), but opp's context depends on our post-move state — `score_gap_to_leader`, `max_opp_strikes`, `max_opp_locks`, and `max_opp_total_progress` all change when we take different candidate moves. Fix: rebuild opp's `OpponentContext` per candidate via `build_opponent_context_for(opp_score, post_our_state, non_leader_states)`. Same fix applied to our own features (the pre-move our_score was being used against the post-move state).

### New game-end-proximity features (21 → 25)

While investigating σ, realized the network had no direct signal for how much game remains. Added four features, all normalized to `[0, 1]`:
- `max_player_strikes / 3` (our strikes vs opp strikes, whichever is higher)
- `max_player_locks` (capped at 1, since 2 ends the game)
- `max_player_total_progress` (sum of row-progress across 4 rows, max across players)
- `aggregate_weighted_probability` (across all usable dice sums)
- `total_lockable_rows / 8`

Replaced the old `max_opponent_strikes` feature with the symmetric "max across all players" version; split `score_gap` position slightly. Final count: **25 features**.

### Score-distribution normality check

Before committing to Gaussian P(win), ran `examples/score_distribution.rs`: rolls out 20,000 trajectories from a single mid-game snapshot. The resulting final-score distribution has skew +0.04, excess kurtosis −0.34 — well within "Normal enough" for the `Φ(...)` approximation.

### Results

| Change | Peak winrate vs GA |
|--------|--------------------|
| Baseline (21-feature, argmax μ) | 58.8% |
| 21-feature + P(win) + joint NLL | ~55-57% (σ anti-correlated) |
| 25-feature + P(win) + decoupled μ/σ loss | ~59.9% |
| + per-candidate opp context re-inference | **60.3%** |

Peak achieved at iterations 4-15 with warm-start from a prior self-play checkpoint. Winrate plateaus at ~59.7-60.0% from iteration 5 onward while avg score climbs from 63.4 → 68.9.

### Why the gain is small (and probably near the ceiling)

Empirical diagnostics from `sigma_validation`:
- Median σ is ~20 points across decisions.
- **Within-decision σ spread** (std of σ across candidates at the same decision) has coefficient of variation ~4-5%.

That means at most decisions, `√(σ²_us + σ²_opp)` is approximately constant across candidates, so the P(win) ranker reduces to `argmax(μ_us − μ_opp)` which is argmax-μ up to a shift. The formula only matters at decisions where candidates differ meaningfully in risk — and those are rare in Qwixx because the shared-dice structure means your remaining score is dominated by *how long the game continues*, which is the same across your candidates.

Qwixx's structural ceiling against a well-tuned GA is probably in the 60-62% range; squeezing further would likely require fundamentally different approaches (e.g., search at decision time).

### Diagnostic tools retained

- `examples/sigma_probe.rs` — prints per-candidate `(μ, σ)` at every decision of a single game.
- `examples/sigma_validation.rs` — batched MC rollouts from real-game decisions, compares DQN (μ, σ) to empirical (μ, σ); writes CSV.
- `examples/score_distribution.rs` — dumps the final-score distribution from a fixed snapshot.
- `scripts/sigma_validation_plots.py` — produces μ/σ scatter and σ-vs-turn plots from the CSV.

---

## Phase 12: Pairwise Differential Network ("pair" bot)

All numbers below are under the corrected two-phase rules and paired-seed
benchmarking (the pre-refactor numbers above are not comparable; the old DQN's
baseline under the new regime is **59.10%** vs GA on the seed-42 set).

### Design

Replaced the single-board value net + hand-built `OpponentContext` with a joint
two-board MLP (45 → 128 → 64 → 2) predicting the **future score differential**
`final_diff − current_diff` as `(μ, log σ²)`. Motivation: row-level opponent
visibility (lock races) and implicit covariance handling — `σ_diff` is the
calibrated quantity, no independence assumption. Training: pure self-play
(1v1/3p/4p thirds), per-opponent TD(λ=0.8) chains, board-swap sample doubling
with exactly-recomputed pair-level features, passive skips recorded, color
permutation on both blocks in the batcher. Ranking: `(current_diff + μ)/σ` vs
the leading opponent. Spec: `docs/superpowers/specs/2026-06-10-pair-network-design.md`.

### Epochs finding: 10 epochs/iteration overfits

The pair pipeline generates ~6.2M samples/iteration (~3.4× the old pipeline:
per-opponent chains × swap doubling × skip recording), so 10 epochs over the
3-iteration replay buffer was ~10× the old gradient work. At 10 epochs the
winrate plateaued at **56.5%** (iters 4–10). Switching to **3 epochs** at
iteration 11 broke the plateau: winrate climbed to a peak of **59.02%** at
iteration 19, converging to ~58.0–58.4% by iteration 31, while avg score rose
to ~65. Fewer epochs per buffer snapshot = less overfitting to recent
self-play, plus ~2.4× faster iterations.

### Results (iter-19 checkpoint, 100k games each)

| Matchup | seed 42 | seed 7 |
|---------|---------|--------|
| PAIR vs GA | **59.22%** (CI 58.86–59.58) | **59.26%** (CI 58.90–59.62) |
| PAIR vs old DQN | 49.56% vs 48.89% (1.6% ties) | 49.35% vs 49.06% (1.6% ties) |

- vs GA: statistical **parity with the old DQN** (59.10%), confirmed on an
  independent seed.
- Head-to-head: **dead even** — PAIR takes ~50.1–50.3% of decisive games, not
  significant. PAIR scores slightly higher on average (76.0 vs 75.7).

### Conclusion

Joint two-board evaluation reaches parity with compressed-context evaluation,
not superiority. This is strong evidence for the structural-ceiling
hypothesis: at ~59–60% vs GA, the binding constraint is not the evaluation
function's opponent information — both representations extract essentially all
the available signal. What the pair net does buy: ~2× cheaper inference per
decision (N feature rows instead of 2N interleaved), much simpler context
handling (no per-candidate `OpponentContext` rebuild), and a cleaner platform
for decision-time search — which is the next lever that can plausibly move the
number.

## Phase 13: Decision-Time Search (pair-search)

All numbers below are under the corrected two-phase rules and paired-seed
benchmarking, building directly on the Phase 12 pair net.

### Design

Decision-time search layered on the pair network, gated so it only runs where
it can matter. At each active decision we score the top-2 static candidates by
**truncated rollouts**: complete the current turn deterministically, simulate
one full round with every player played greedily by the value net
(lockstep-batched through `Bot::evaluate_batch_multi`), then take the leaf
value as the exact game outcome if the round ended the game, otherwise the
net's win probability (`WinProb` trait) at the horizon. 64 samples per
candidate with **common-random-number** dice derived by hashing the decision
context, so the same decision always draws the same dice across candidates and
runs — the bot is stateless-deterministic and benches are byte-reproducible.
The per-player greedy sim policy uses the Lite pipeline (`*_choices()` pure
logic shared with the production bot, no allocation-heavy ranking machinery).
Meta-rules stay single-source: the shared bot_impl `*_choices()` are consumed
by both the blanket `Bot -> Strategy` impl and the search bot, and the
simulator is pinned to `Game::play` by an equivalence test covering both
strike- and lock-terminated games.

### Calibration and the GATE_MARGIN decision

`cargo run --release --example search_calibration -- 50` with search forced
everywhere over the paired-seed game set:

- 1260 active decisions, 1019 eligible; the close gate (top-2 static gap below
  GATE_MARGIN) fires on 51.3% of eligible decisions, the endgame gate (any
  locked row, 3 strikes, or a locking candidate) on 19.9%.
- Disagreements (search picked a different move than the static net): 6.6% of
  searched decisions. **100% of those disagreements occurred at decisions where
  a gate had already fired** — gating loses no realized search value.
- Top-2 static gap percentiles: p10 0.015, p25 0.053, p50 0.147, p75 0.281,
  p90 0.500.

Since every disagreement fell inside a fired gate, **GATE_MARGIN stays at
0.15** — no constant change. Cost: ~0.17 s/game with search forced everywhere;
gated benches run ~27 games/s wall on 8 cores, ~100× the static pair bot.
Optimization is deferred; the identified levers are profiling, allocation reuse
in the rollout driver, cached opponent feature blocks, and a lighter sim
policy.

### Acceptance results (50k games each, paired dice)

| Matchup | Result | Margin |
|---------|--------|--------|
| pair-search vs pair, seed 42 | 24939 vs 24083 (978 ties) — 49.88% / 48.17% | **+856 (+1.71%)**, paired SE 0.155% |
| pair-search vs pair, seed 7 | 24924 vs 24137 (939 ties) | **+787 (+1.57%)**, replicates on independent seed |
| pair-search vs GA, seed 42 | **59.83%** (99% CI 59.32–60.34) | static pair bot 59.2% — strongest bot under the corrected rules |
| pair-search vs DQN (old), seed 42 | 50.26% vs 48.25% (1.5% ties) | — |

### Verdict

Decision-time search produces the **first measurable strength gain since the
two-phase rules fix**: +0.6% vs GA and a +1.6–1.7% head-to-head win margin over
the static pair bot, statistically solid and replicated across two independent
seeds. The size is small and consistent with the structural-ceiling hypothesis
from Phase 12. Criterion caveat: with ~2% ties no bot's raw win rate exceeds
50% head-to-head, so the meaningful statistic is the **win margin**, not a
">50%" threshold. Cost is ~100× the static bot (~27 games/s on 8 cores), with
optimization deferred until it is needed.

The decisive takeaway: search now **demonstrably beats static play**, which
justifies the next escalation — **expert iteration**, training the net on
search-improved play.

### Search-budget scan (post-perf-pass)

After the 2.6x perf pass (single-threaded matmul + mimalloc; see the perf
commit), a budget scan over the search constants, 50k-game margins vs the
static pair bot on seed 42 (seed 7 confirmations in parentheses):

| Config | Margin | Verdict |
|--------|--------|---------|
| K_SAMPLES=64, 1 round, 2 cands (baseline) | +1.71% (+1.57%) | — |
| HORIZON_ROUNDS=2 | +1.33% | worse — greedy-rollout noise compounds faster than the bootstrap improves |
| K_CANDIDATES=3 | +1.67% | neutral — there are rarely more than 2 good moves |
| **K_SAMPLES=128** | **+2.50% (+2.16%)** | **adopted** |
| K_SAMPLES=256 | +2.79% | past the knee — +0.3% for 2x cost |

Search quality is sample-noise-bound, not depth- or width-bound: gates fire at
close decisions where candidate values differ by a hair, so resolution is what
pays. With K_SAMPLES=128, **pair-search vs GA reaches 60.09% (99% CI
59.58–60.60)** — the first bot over 60% under the corrected rules, entering
the hypothesized 60–62% structural-ceiling zone. Head-to-head cost ~41 games/s
on 8 cores.

---

## Phase 14: Expert Iteration

Closed the loop on Phase 13: `pair-train --search` generates self-play training
games with the K=128 search bot as every player's policy (ε-coin fires before
search, so exploration skips it; recording, TD(λ) targets, replay, and losses
unchanged — pure trajectory-level distillation). Run configuration: 5k
games/iteration, 5 epochs (gradient-passes band reasoning), 500k-game
per-iteration static benchmark (paired SE ≈ 0.05%) as the primary metric, warm
start from the Phase 12 model at the ε floor.

### Result: small, real, and short of the bar

Over 11–12 iterations the static-V winrate climbed **58.3% → 58.7%** on the
fixed 500k bench — ≈ +0.04%/iteration, ~8σ in total, so the distillation signal
is genuinely there — but it plateaued below the pre-registered adoption bar
(+0.5% over the re-baseline). The checkpoint was not adopted:
`pair_model/model.mpk` remains the Phase 12 net and the pair-search headline
remains **60.1% vs GA**.

### Interpretation

The dilution mechanism flagged at design time is quantitatively consistent with
the observed rate: the expert corrects ~1.3 decisions/game (6.6% of searched
decisions), so ~93% of TD targets are statistically indistinguishable from what
the net already fits, and the candidate *ranking* — the only thing that affects
play — shifts even less than the values. Trajectory-level expert iteration
works, but at this expert-student gap the per-iteration gain is a few
hundredths of a percent and saturates.

Ladder amendment: the "disagreement-weighted trajectory sampling" escalation is
struck — at 1.3 disagreements/game nearly every trajectory contains one, so
trajectory-level weighting is a no-op. The remaining untried rung is
**search-value distillation**: emit extra training samples at searched
decisions regressing V toward the rollouts' mean outcome in diff units (the
sims carry actual final diffs, so no unit mismatch with μ). Estimated odds
~30–40% of +0.3–0.5%; design questions around σ-target semantics and mixing
weight. Not pursued in this campaign.

### Campaign conclusion

Every attack on the ~60% ceiling vs GA under the corrected rules has now been
mounted and measured: joint two-board representation (parity with compressed
context), decision-time search (+0.9% total, sample-noise-bound), and expert
iteration (+0.4% static-V, below adoption). The structural-ceiling hypothesis
— shared-dice luck dominating beyond ~60–62% — survived all of them. Final
leaderboard vs GA (50k+ paired games, seed 42):

| Bot | vs GA |
|-----|-------|
| **pair-search (K=128)** | **60.1%** |
| pair-search (K=64) | 59.8% |
| pair (static) | 59.2% |
| old DQN | 59.1% |

## Phase 15: Search-Divergence Analysis (when and why search overrides static)

Phase 13 established that decision-time search beats static pair play by
+2.3–2.5% head-to-head, but left open *which* decisions search rescues. If the
override pattern reduced to a crisp board feature, the static net could be
taught part of search's edge at static speed (no rollouts). This phase mines
that question and returns a verdict on encoding a meta-rule.

### Setup and data

Shadow collection: 10,000 games of static-PAIR vs GA, rotating seats, paired
dice, seed 0. At every eligible decision on the static bot's own trajectory,
search was run with `force=true` (gate ignored) and the static-vs-search
comparison logged — one JSONL event per eligible decision. Output
`divergence-10k.jsonl` is ~96 MB / 201,749 lines (191,749 decision events +
10,000 game records), collected in ~6 min wall. Every disagreement plus a 10%
sample of agreements was then **relabeled** at K=2048 rollouts per candidate
with CRN-paired stats (seed 1, 10m23s wall / 82m user, zero drift-guard panics),
attaching `hk_gap_mean`/`hk_gap_se`/`verdict` to 35,927 events. A verdict of
**flip** means cands[1] is confidently better than cands[0] (|z|>2), **keep**
the reverse, **coinflip** otherwise. `search_right` = a disagreement whose
high-K verdict confirms search's pick.

Recomputed headline rates (from the file, the original run summary was lost):

- **191,749 eligible decisions.** Close gate (top-2 static gap below
  GATE_MARGIN) fires on **53.7%**, endgame gate on **23.3%**, any gate on
  **70.1%** (134,350 decisions).
- **Disagreements: 18,644 = 9.72% of eligible / 13.82% of gate-fired.** Only 72
  of 18,644 (0.4%) fell outside a fired gate — gating still loses essentially no
  realized search value, consistent with Phase 13's 100%.
- **Pair (static) win rate vs GA: 58.95%** over the 10,000 games — in line with
  the 59.2% leaderboard figure.

These track the 500-game seed-7 validation (9.79% of eligible / 14.15% of
gate-fired, 57.6% pair wins) closely.

### Disagreement survival at high K: search is a tie-breaker, not a fixer

Among the 18,644 disagreements, the K=2048 verdict split is **flip 39.3%
(7,325) / keep 10.7% (2,001) / coinflip 50.0% (9,318)**. Search's pick is
confirmed (`search_right`) on **7,326 = 39.3%** of disagreements (the 7,325
flips plus one keep-verdict event where search had picked cands[0]), refuted
on 10.7%, and **half are statistical ties even at 16× the search budget.**

Agreement-control noise floor (verdict present, `disagree==False`, n=17,283):
**keep 96.3% / coinflip 3.2% / flip 0.6%.** Here "keep" is the expected null
(both candidates were the same static pick), so the floor of "the high-K
estimate would have changed something" is flip+coinflip ≈ 3.8% — matching the
500-game floor (0.5% + 3.5%). The disagreement coinflip rate (50%) sits far
above this floor, so coinflips at disagreements are real near-ties, not
relabel noise — but they carry no decidable signal.

The headline: **search's edge is overwhelmingly a diffuse sampling-advantage,
not the correction of identifiable static blunders.** Quantitatively, 26.4% of
disagreements (4,918) have `hk_gap_se==0`, and **every one of them is a
coinflip** — these are decisions where both rollout branches already terminate
the game deterministically, so the K=2048 estimate is exact and the two
candidates are genuinely equal-valued (an unbreakable tie, not a
high-confidence stochastic call).

### Transition matrices: no directional move-kind pattern

Among confirmed disagreements (search_right, n=7,326), move-kind transitions are
near-symmetric: **mark→mark 47% (3,453), skip→mark 27% (1,959), mark→skip 22%
(1,592), mark→strike 2% (134).** Search marks slightly more than static skips
(search marks 5,600 vs static 5,179; search skips 1,592 vs static 1,959), but
there is no clean "search is bolder/more conservative" axis. Within confirmed
mark→mark, the row-transition matrix is essentially uniform off-diagonal (every
row pair ~260–360 events) and the **jump-size shift is symmetric: mean +0.005,
median 0, std 0.80, |shift|>0 only 31% of the time.** Search is not
systematically taking shorter or longer jumps — it is re-picking among nearly
equivalent marks.

### Decision tree and importances: it's all `static_gap`

A depth-3 balanced tree predicting `search_right` over all relabeled events
scores 0.685 held-out (base rate 0.796 — barely above majority-class). The only
feature with meaningful permutation importance is **`static_gap` (0.162)**;
everything else is ≤0.012 (`cdiff` 0.012, `static_jump` 0.010, all board/move
features 0.000). `export_text` rounds the splits; the true thresholds are
`static_gap <= 0.0741` (root), then `<= 0.0000` and `<= 0.0348`. The tree says
`search_right` is most likely in a **low-but-nonzero static_gap band
(0 < gap ≲ 0.074)** — i.e. exactly where the static net is nearly indifferent
between its top two candidates. That is the diffuse-tie-breaking story restated,
not a board feature. (`our_locked` and `opp_locked` are byte-identical in this
data too — confirmed equal on all 191,749 rows — so they are one feature, not
two, and neither carries signal.)

### Candidate rule, tuned held-in (games < 5000) and reported held-out (≥ 5000)

The only rule the data suggests is "trust search when the static top-2 gap is
small." Best variant, **`0 < static_gap ≤ 0.074`**, held-out:

- **Coverage** (of confirmed disagreements matched): 86.5%
- **Precision** (rule-matched disagreements where search is confirmed): **53.8%**
- **False-fire** (rule-matched relabeled agreements that aren't "keep"): 12.4%
- **Estimated value** (Σ|hk_gap_mean| over rule-matched confirmed disagreements
  / 5,000 held-out games): **+0.0072 win-prob points/game**

Tightening to `≤ 0.035` lifts precision only to 55.3% while coverage drops to
70% and false-fire climbs to 21.4%. Even **"adopt search on every
disagreement"** has precision just 39.5% and a net realized value of **+0.0072
wpp/game** (sign-checked: Σ of signed hk_gap over held-out disagreements,
oriented to search's pick / 5,000). The upper bound — Σ|hk| over confirmed only — is
**+0.0085 wpp/game**.

### Two estimator caveats

(a) The K=2048 relabel reuses CRN samples 0..128 — the very draws that
determined the original static-vs-search pick — inside the K=2048 estimate.
Conditioning on the selection boundary biases the estimate toward the
originally-chosen side: **conservative for "flip" verdicts** (a confirmed flip
had to overcome its own initial-sample headwind, so the true flip rate is if
anything higher) and **mildly anti-conservative for "keep"** (keeps are partly
self-fulfilling). (b) `hk_gap_se==0` rows are **deterministic** outcomes — both
rollout entries are already game-over, so the gap is exact-zero, not a
high-confidence stochastic near-zero. All 4,918 such disagreements are
coinflips and should be read as true ties, not confident calls.

### Example positions (abridged, top |z|)

The highest-|z| disagreements illustrate the tie-breaking character:

- **game 8738 t5** (flip, z=+420): static skips (v=−0.205); search marks **R11**
  (v=−0.217). Two essentially co-valued options; rollouts confirm the mark by a
  hair (hk_gap +0.176).
- **game 84 t8** (flip, z=+65): static marks R11 (v=+3.43), search marks **Y11**
  (v=+2.69) — both strong marks one number apart; search prefers the lower-row
  number. A within-mark re-pick, not a skip-vs-mark blunder.
- **game 161 t8** (flip, z=+90): static marks R12 to lock-pace (v=+0.293),
  search **skips** (v=−0.066) with a row already locked and the opponent far
  ahead on R — the rare endgame case with a coherent "don't commit" story, but
  it is one position among ~7,300 confirmed and does not generalize into the
  stats.

The top examples are dominated by low-gap re-picks between adjacent or
co-valued marks, matching the `static_gap` tree split and the symmetric
transition/jump statistics — there is no recurring human-legible board motif.

### Verdict: NO-GO on encoding a meta-rule

The numbers do not support a static-speed meta-rule:

1. **Half of all disagreements are coinflips even at K=2048** (and 26.4% are
   provably exact ties), so search's measured +2.3–2.5% cannot be a handful of
   board-pattern fixes — it is a diffuse reduction of decision-sampling noise
   spread across thousands of near-ties.
2. **The only predictive feature is `static_gap`** — a statement about the net's
   own indifference, not an encodable board condition. A `static_gap` threshold
   *is* what the close gate already implements; there is nothing new to teach.
3. **The best held-out rule peaks at 53.8% precision** (barely above a coin) with
   a net value ceiling of **~+0.007–0.009 win-prob points/game**. Even adopting
   search on every disagreement nets only +0.0072 wpp/game of the +2.3–2.5%
   head-to-head edge — confirming the bulk of search's advantage lives in the
   coinflip mass that no rule can adjudicate, and is consistent with Phase 13's
   "sample-noise-bound" conclusion and Phase 14's dilution finding.

A humanly-encodable rule that hands the static bot part of search's edge does
not exist in this data. The realized override value is symmetric, gap-driven,
and concentrated in genuine ties.

**Next steps.** No Rust meta-rule. The remaining lever consistent with the
diffuse-tie picture is the unstruck Phase 14 rung — **search-value
distillation** (regress V toward rollout-mean outcomes at searched decisions),
which targets the *value calibration* that produces these near-ties rather than
trying to name them. The structural-ceiling hypothesis (shared-dice luck
dominating beyond ~60–62%) again survives: search wins by sampling, not by
knowing something nameable that the net doesn't.

## Phase 16: Safe-Lock Adjudication (when is the "always force a safe lock" meta-rule wrong?)

The `find_safe_lock` meta-rule forces a non-game-ending lock whenever one is
available — a hard, unconditional override sitting in front of the value net.
Phase 15 showed search's edge is diffuse and unnameable; this phase asks the
inverse for an *existing* named rule: on production trajectories, how often does
forcing the safe lock cost win-probability, and is the failure mode crisp enough
to gate the rule on a board feature?

### Setup

Shadow collection on **10,000 games of static-PAIR vs GA**, rotating seats,
paired dice, seed 0, **rule ON** (the bot plays exactly as in production). On
the static bot's own trajectory, every time `find_safe_lock` fired we logged the
forced lock, the full candidate list with static `v`, and the picked
alternative/runner-up indices (force detection + value-equivalence guards decide
whether the rule actually changed the move; equivalence-tie firings are kept).
Each event was then **rebuilt rule-free** and adjudicated with **K=2048
FULL-GAME CRN-paired rollouts** (no truncation, no win-prob bootstrap — terminal
outcomes only) comparing the forced lock against the best non-lock alternative
(and, when present, the runner-up lock as `alt2`). The reported quantity is
**gap = alternative − lock in win-probability units** (positive ⇒ the rule is
wrong); verdicts at |z|>2. Files (git-ignored): `lock-events.jsonl` 3.0 MB /
15,472 lines (5,472 events + 10,000 game records), collected in **0.7 s**;
`lock-events.adj.jsonl` adjudicated in **3 m 20 s** wall / 25 m user, **zero
drift-guard panics**.

### Firing statistics

`find_safe_lock` fired on **5,535 total decisions; 63 (1.1%) were skipped**
(lock_pruned 58, forced_lock_itself 3, lt2_cands 2), leaving **5,472 adjudicated
events = 0.55/game**. Context split: **ap2 3,180 / pp1 1,205 / ap1 1,087**
(ap = our active-player mark, pp = passive/white-die mark). Multi-lock states
(a runner-up lock present): **365**. `rule_free_forced` (the rule-free rebuild
still independently picks the lock): only **4** — i.e. the rule is genuinely
overriding the net almost every time.

**Structural fact: `locks_on_board == 0` for all 5,472 events.** Safe locks are
by construction the *first* lock of the game — any *second* lock would end the
game and so is never a "safe" (non-terminal) lock. This means the whole dataset
is "first lock now vs. don't," never "stack a second lock."

### Verdict split

Best-non-lock alternative (alternative − lock):

| verdict | n | share |
|---|---|---|
| lock_right | 4,683 | 85.6% |
| coinflip | 305 | 5.6% |
| **lock_wrong** | **484** | **8.8%** |

Effect-size-floored (|gap| ≥ 0.02 wp): **lock_wrong 438** (46 negligible-gap
events dropped). Per context the wrong rate is wildly uneven: **ap1 231/1,087 =
21.3%**, **pp1 169/1,205 = 14.0%**, **ap2 84/3,180 = 2.6%** — ap2 (the larger,
"mark the active-roll number" context) is where the rule is almost always right.

Gap-magnitude distributions (|gap| wp): lock_wrong (n=484) median **0.102**, p90
0.41, p99 0.63, max 0.81; lock_right (n=4,683) median **0.157**, p90 0.39, p99
0.77, max 1.0. **Both directions carry large gaps** — when the rule is right it
is often very right, and when wrong it is sometimes catastrophically wrong, so
this is not a "tiny-margin" rule like Phase 15's coinflips.

Determinism diagnostics: 96 events have `alt_gap_se == 0` (both arms terminate
deterministically — exact outcomes); 91 lock_wrong events have se < 0.005. These
low-variance wrongs are treated separately and are not z-inflation artifacts.

Runner-up lock (`alt2`, the "did the rule pick the *wrong lock*?" check, 365
multi-lock events): **lock_right 169 / coinflip 119 / lock_wrong 77** — so even
when forcing a lock is correct, the rule picks a sub-optimal lock about 1 time in
5 among multi-lock states.

### The signal: game-shortening when behind

The wrong-verdict rate is governed almost entirely by the score margin
`cdiff = our_points − opp_points` at the decision:

| cdiff bin | n | lock_right | coinflip | **lock_wrong** |
|---|---|---|---|---|
| ≤ −10 (far behind) | 913 | 71.3% | 7.4% | **21.2%** |
| −9 .. −1 (behind) | 1,498 | 77.2% | 6.9% | **16.0%** |
| 0 (tied) | 219 | 84.0% | 9.6% | **6.4%** |
| 1 .. 9 (ahead) | 1,620 | 94.0% | 3.9% | **2.2%** |
| ≥ +10 (far ahead) | 1,222 | 95.7% | 4.1% | **0.16%** |

The knee is at **cdiff = 0**: behind (cdiff<0) the wrong rate is **18.0%**
(433/2,411, mean positive gap +0.152 wp); ahead-or-tied it is **1.7%** (51/3,061,
mean positive gap +0.070 wp). The mechanism is **game-shortening**: a lock
removes a whole row from play and accelerates the end of the game; when you are
behind you *want* more turns to catch up, so forcing the lock locks in your
deficit. Of the 484 wrong events, **323 (66.7%)** have the better alternative
being **skip/defer** (don't lock, keep the row alive), and **28** are static
near-ties (|v_lock − v_alt| < 0.02) where the net itself was indifferent and the
rule broke the tie the wrong way — the exact pattern flagged in the rehearsal.

Secondary conditioners are weak by comparison: `alt_is_defer` True 10.4% vs
False 6.8%; `lock_row` 0 (red) 10.9% highest / row 1 (yellow) 7.2% lowest; stage
(turn) shows no clean cut independent of cdiff. **cdiff is the rule.**

### Candidate rule variant, held out

Candidate: **suppress the forced lock when `cdiff < threshold` and let the value
net decide.** Split games <5000 (train) / ≥5000 (test). Sweeping the threshold
on train, the natural operating point is **cdiff < 0 (suppress when behind)**.
Held-out (test, 2,757 events, base lock_wrong rate 7.94%):

| threshold | fired | true wrong | precision | lift | coverage | UB wp/game |
|---|---|---|---|---|---|---|
| cdiff < 0 | 1,244 | 199 | 16.0% | 2.0× | **90.9%** | 0.00693 |
| cdiff < −5 | 774 | 141 | 18.2% | 2.3× | 64.4% | 0.00398 |
| cdiff < −10 | 415 | 75 | 18.1% | 2.3× | 34.2% | 0.00156 |

The cdiff<0 cut **generalizes cleanly** to the held-out split (test bin rates:
≥+10 → **0.0%** wrong, 1..9 → 2.1%, behind → 14.7–18.2%), capturing **91%** of
all wrong verdicts. But precision peaks at ~18%: because lock_wrong is rare,
even the best cut still suppresses ~4–5 *correct* locks for every wrong one it
catches. The **wp/game upper bound** from acting on this rule is the sum of
positive gaps over rule-matched lock_wrong events ÷ games: **≈0.0069 wp/game**
at cdiff<0 (vs a full removable ceiling of ~0.0074 wp/game if *all* wrongs were
fixed). **This is explicitly an upper bound under perfect substitution** — it
assumes suppression always lands on the adjudicated alternative, which is not
measurable here: suppressing the rule hands the decision back to the value net,
which (per `rule_free_forced` being tiny) usually still avoids the lock but is
not guaranteed to pick the *best* alternative.

### Rendered examples (abridged)

- **game 9221, turn 7, ap1, cdiff −1, gap +0.79 wp (z=89):** behind by 1, forced
  G2 terminal lock (v=+0.42); the net's own runner-up was **skip** (v=+0.61).
  Rollouts favor skip by 0.79 wp — locking the green row away when nearly even
  throws the game.
- **game 9423, turn 9, pp1, cdiff −5, gap +0.60 wp (z=56):** forced R12 lock
  (static v=+2.35, the net *loves* it) over skip (v=+1.07); rollouts say skip is
  +0.60 wp better. A textbook static-value-vs-rollout disagreement where
  game-shortening while behind is the hidden cost the net underweights.
- **game 1768, turn 10, pp1, cdiff 0 (tied), gap +0.61 wp (z=57):** dead-even,
  forced R12 (v=+1.43) vs skip (v=+0.91); rollouts prefer skip — even at a tie
  the lock prematurely ends a winnable game.

### Estimator caveats

- **(a) Rollout-policy circularity.** The pair value net (trained rule-ON) drives
  every rollout move *and* models the GA opponent inside entries/rollouts. This
  almost certainly **flatters the rule** (the rollout policy shares the net's
  blind spots), so the true cost of the rule is plausibly larger than measured.
  It cannot be removed without an independent rollout policy.
- **(b) pp1 sequential-completion approximation.** On passive (white-die) marks
  the opponent completes its turn after seeing our mark; this is identical across
  candidates, so CRN-paired *gaps* remain valid even though the absolute states
  are an approximation.
- **(c) No CRN-selection bias.** Unlike Phase 15, the rule is **unconditional** —
  no K=128 pre-pick selected which events to adjudicate, so there is no
  winner's-curse inflation in the estimator. The estimator is clean in that
  respect.
- **(d) z>2 alone fires on negligible gaps** (46 of the 484 wrongs are <0.02 wp);
  hence the 0.02 wp effect-size floor on all headline magnitude claims.
- **(e) Deterministic-arm events (se==0)** are exact terminal outcomes, not
  stochastic estimates, and are reported separately (96 events / 91 low-var
  wrongs).

### Verdict and recommendation

`find_safe_lock` is **net-positive but miscalibrated when behind**: right 85.6%
of the time and very right in ap2, but wrong ~18% of the time once cdiff<0, where
the game-shortening cost of locking dominates. The failure mode is **crisp**
(cdiff<0, alt usually = skip) — crisper than anything Phase 15 found — and
generalizes out-of-sample. The realized headline cost is modest (the 484 wrongs
sum to **0.0080 wp/game** over all 10k games; the removable upper bound is
~0.0074 wp/game), but it is concentrated exactly where games are close, i.e. the
games most likely to be decided by it.

**Recommendation: A/B a conditional variant**, not a keep-as-is and not a
src rewrite. Per the adjudication spec §5, implement the gate **example-side**
(a `Strategy` that suppresses the forced lock when `cdiff < 0` and defers to the
value net, with **zero `src/` contamination**) and evaluate head-to-head: the
conditional bot vs the untouched rule-ON bot, **and** both vs the untouched GA,
paired dice. The held-out numbers justify the test (91% wrong-coverage at the
cdiff<0 cut, clean generalization, the close-game concentration) while the
~18% precision and the perfect-substitution caveat (b/a) mean the *measured*
upper bound (~0.007 wp/game) is small and possibly optimistic — so the A/B, not
the analysis, must decide. If the A/B is flat or negative (plausible, given the
circularity caveat flatters the current rule), **keep the rule as-is**; the
multi-lock `alt2` finding (sub-optimal lock chosen 21% of the time) is a separate,
smaller lever not addressed by this gate.

## Phase 17: Safe-Lock Suppression A/B (does gating the rule on cdiff actually pay?)

Phase 16's adjudication recommended an **A/B**, not a rewrite: implement the
`cdiff`-gated suppression of `find_safe_lock` **example-side** (zero `src/`
contamination) and measure head-to-head against the untouched rule-ON bot, plus
both vs GA. This phase runs that A/B at 1M games/arm.

### Suppression semantics

`VariantPair(suppress_below = t)` is the baseline static-PAIR bot with one
change: when `find_safe_lock` would fire and the score margin
`cdiff = our_points − opp_points` satisfies `cdiff < t`, the forced lock is
**suppressed** — the safe lock is demoted from a hard override to an ordinary
value candidate and competes in the normal value-net argmax (it may still be
picked if the net actually prefers it). Arms: `t=0` (suppress when strictly
behind, `cdiff<0`), `t=1` (suppress when behind-or-tied, `cdiff<=0`), `t=-5`
(suppress only when far behind, `cdiff<-5`). `suppress_below = None` is the
untouched production bot. Reported metric is **tie-inclusive score**
(win=1, tie=0.5, loss=0) with paired SE and z vs an exact 50% null; ties are
explicitly scored 0.5 so the null is exact, not approximate.

### Setup and gates

Equivalence gate (cheap insurance, seed 3, 1000 games): `VariantPair(None)`
replays the baseline move-for-move — **holds**. All benches 1M games, rotating
seats, paired dice, seed 0. Reference 10k smokes (seed 1) put arm 0 essentially
flat head-to-head (49.94%, z −0.94) and 59.90% vs GA. The suppression branch's
correctness is fenced by the Phase-16 mirror guards (force detection +
value-equivalence) plus this `VariantPair(None)` equivalence gate.

### Head-to-head: VariantPair(t) vs untouched rule-ON pair (1M, seed 0)

| arm | score | strict wins | ties | paired SE | z vs 50% |
|---|---|---|---|---|---|
| cdiff<0   (t=0)  | 50.06% | 49.10% | 1.93% | 0.007pp | **+9.70** |
| cdiff<=0  (t=1)  | 50.06% | 49.11% | 1.91% | 0.007pp | **+9.18** |
| cdiff<-5  (t=-5) | 50.03% | 49.06% | 1.93% | 0.005pp | **+5.08** |

All three arms are **statistically, decisively positive** (z +5.08 to +9.70 at
1M) — this is **not** a flat result in the spec's sense. The effect is real but
**tiny: +0.03..+0.06pp**, about 0.7% of the bot's 9.1pp edge over GA and ~1/10
of Phase 16's +0.7pp perfect-substitution ceiling. The seed-1 10k smoke (arm 0
at −0.06pp, z −0.94) does **not** contradict this: its SE (~0.064pp) makes it
fully consistent with a true +0.06pp at <1 SE — a 10k sample simply cannot
resolve a sub-0.1pp effect, and carries no weight against the 1M measurement.
Cross-arm ordering (t=0 ≈ t=1 > t=-5) tracks Phase 16's wrong-verdict coverage
(cdiff<0 = 90.9% vs cdiff<-5 = 64.4%): wider suppression recovers marginally
more, as expected. (The tiny SEs are a structural feature, not an anomaly:
rotation-pairs where suppression never fires play bit-identically to baseline
and contribute exactly 0.5 to the paired statistic, so all variance comes from
suppression-affected pairs. The harness does not report the realized per-game
suppression fire rate; Phase 16's firing stats imply ~0.24/game for t=0.)

### vs GA, per arm (1M, seed 0) — against the baseline yardstick

Baseline untouched pair vs GA, 1M, seed 42 (`bench ga pair`): **PAIR 59.1%
strict wins** (avg 74.6 vs 66.5 pts), 99% paired CI **58.97%–59.20%**, paired SE
0.044pp.

| arm | score (tie-incl.) | strict wins | paired SE | z vs 50% |
|---|---|---|---|---|
| cdiff<0   (t=0)  | 60.13% | 59.25% | 0.044pp | +231.64 |
| cdiff<=0  (t=1)  | 60.13% | 59.26% | 0.044pp | +231.71 |
| cdiff<-5  (t=-5) | 60.05% | 59.17% | 0.044pp | +229.81 |

On the comparable strict-win basis the arms (59.17–59.26%) sit on top of the
baseline yardstick (59.1%, CI upper 59.20%) — **no regression vs GA**; arm 0/1
are marginally at/above the CI top, arm t=-5 is squarely inside it.

### Verdict

Decision rule (spec `docs/superpowers/specs/2026-06-12-lock-ab-design.md`):
adopt requires head-to-head z>2 above 50% **and** no regression vs GA; a flat
result keeps the unconditional rule. **The pre-registered criteria for adoption
were met:** every arm clears z>2 (min +5.08) and none regresses vs GA (strict
59.17–59.26% vs the 59.1% baseline yardstick, CI 58.97–59.20% — arms 0/1 sit
at/just above the CI top, though that comparison is **unpaired** and on a
different seed/run than the head-to-head, so the apparent edge over baseline is
not itself decisive). **The recommendation is nonetheless to keep the
unconditional `find_safe_lock` rule as-is, overriding the registered
adopt-trigger on effect-size grounds:** the head-to-head win is only
+0.03..+0.06pp — ~0.7% of the bot's edge vs GA and ~1/10 of Phase 16's ceiling
— far too small to justify shipping a conditional gate and maintaining the
behavioral divergence it introduces. The cdiff<0 and cdiff<=0 arms are
statistically indistinguishable from each other, and widening to cdiff<-5 only
shrinks the edge, consistent with its lower wrong-verdict coverage (64.4% vs
90.9%, Phase 16). This is an **explicit override of a met criterion, not a
"flat" result** — the distinction matters for anyone re-reading the
pre-registration. Adoption remains the user's call with these numbers in hand.

### Interpretation — consistent with Phase 16, not a contradiction

The tiny realized effect does **not** refute Phase 16. The rule's measured ~0.0080 wp/game
cost is real, but Phase 16 already flagged that figure as an *upper bound under
perfect substitution* (≈0.0069–0.0074 wp/game removable). The realized A/B
recovers essentially none of it because the precision gap Phase 16 measured is
the binding constraint: at the cdiff<0 cut precision peaks at ~16–18%, i.e. for
every genuinely-wrong lock the rule catches, suppression also hands ~4–5
correct locks back to the value net — and the net, freed of the override, does
**not** re-pick the good locks reliably enough (the 84%-precision gap, `lock_right`
85.6%) to net any gain. Suppressing trades the rule's wrong locks for the net's
own mistakes on the locks the rule was right about, and the two *nearly*
cancel — the residual is the measured +0.03..+0.06pp. The circularity caveat
(Phase 16 §a) — the rollout policy shares the net's blind spots and flatters
the rule — called a near-null A/B the plausible outcome, and a small positive
residual an order of magnitude under the ceiling is what landed.

---

## Phase 18: Search-Value Distillation (rollout targets at gated & lock-forced decisions)

**Status: run complete, checkpoint ADOPTED.** The user executed the recipe
(40 iterations, separate machine); iteration 35 was selected and passed the
pre-registered +0.5% bar on the independent acceptance bench. Results at the
end of this section.

### Setup

Phases 13–17 established that decision-time search beats static pair play but
its edge is diffuse and that hand-named rule fixes (safe-lock suppression)
recover almost none of it. This phase takes the opposite tack: instead of
naming the fix, distill search's *value judgements* directly into the static
net's training signal. `pair-train --distill` emits extra `PairSample`s whose
targets are **full-game CRN rollout future-diffs**, computed at exactly the
decisions where search disagrees with static play:

- **Gated decisions** (close/endgame, active phase 1 and phase 2): the top-2
  static candidates are each completed deterministically and rolled out; both
  candidates are emitted, so the net learns the *relative* value of the
  decision tail, not just the chosen branch.
- **Safe-lock firings** (active p1/p2 and passive p1): the forced lock, the
  best non-lock alternative, and the runner-up safe lock (a trio) are each
  rolled out and emitted — this is the Phase 16 adjudication signal turned into
  a supervised target rather than a runtime gate.

Targets flow through the **unchanged decoupled loss** (μ-MSE / σ-NLL, μ
detached) in diff-space: `value` = mean rollout future-diff over K rollouts,
`final_diff` = an individual rollout's future-diff; each sample is swap-doubled
with negated targets exactly as `build_pair_samples` does. Generation also adds
**targeted ε-decline of forced safe locks** (`--epsilon-lock`, player-0 only,
firing at every safe-lock decision incl. passive p1) so the buffer is not
starved of the alternative-to-lock branch that the net otherwise never sees.

Generation is **static-policy** (no `--search`; the flags are independent and
the recipe omits `--search`) — the rollout *targets* carry the search signal,
while move selection during generation stays cheap.

**K=32, m=2 rationale.** K=32 full-game rollouts per candidate balances target
variance against generation cost. m=2 (samples per candidate×opponent pairing)
was chosen over the plan's default m=4 after measuring concentration: at m=4 the
distill samples are ~40% of the buffer (≈217 distill vs 325 TD per game) — too
aggressive given Phase 14's dilution/saturation history, where over-weighting a
narrow signal stalled the curve. m=2 lands distill at ~25% of the buffer
(measured below: ~106/game distill), the first-run setting; m=4 is the
escalation lever if the curve is healthy and not over-fitting the decision tail.

### Smoke (local, m=2)

`cargo run --release -- pair-train -i 2 -g 300 -e 2 -b 20000 --distill --distill-m 2 --epsilon-lock 0.05`

- Distill counts: iter 1 = 31,824 distill of 126,990 total (~106/game, ~25% of
  buffer); iter 2 = 32,094 distill of 125,344 total (~107/game). Plausible and
  on target for m=2.
- Both iterations completed; no NaN, no panic.
- Bench (20k buffer, noisy): iter 1 = 57.6%, iter 2 = 58.2% — sane (56–60%).
- Train loss: iter 1 min 76.95, iter 2 min 76.78 (valid 67–70, noisy across
  buffer growth) — still descending at iter 2.
- Generation wall time ~6s/iter for 300 games (~50 games/s) ⇒ 5000 games ≈
  ~100s/iter, consistent with the ~2 min/5k extrapolation.

**K diagnostic note.** The per-iteration output does **not** split TD-MSE vs
distill-MSE; the proper K diagnostic (loss split) was **not instrumented** (this
task stayed doc-only after the param-comment cleanup). The cheap proxy —
aggregate burn train-loss trend — is readable and still declining at iter 2.
The run-watcher should escalate `--distill-k` to 64 **only if** the curve
flatlines from iteration ~3 while Phase 14's curve at the same point was still
climbing.

**Rayon count-variance note.** Enabled-loop per-iteration sample counts vary
slightly run-to-run (pre-existing rayon reduction nondeterminism, also present
with distill off) — not a regression signal.

### Run recipe (pending user run)

```bash
rm -f pair_model/iter-*.mpk
cargo run --release -- pair-train --distill --distill-m 2 --epsilon-lock 0.05 -g 5000 -e 5 -b 500000 -c --start-iteration 20
```

(`--start-iteration 20` keeps ε at its floor and the checkpoint numbering
distinct, as in Phase 14.)

**What to watch.** The winrate-vs-GA curve against the **+0.5% bar over the
re-baseline** (re-bench the starting checkpoint, don't compare to a stale
headline). Escalation levers if the curve underperforms: `--distill-k 64`
(only on a iter-~3 flatline per the diagnostic above) and `--distill-m 4` (the
concentration escalation, only if healthy and not over-fitting the tail).
Expect the rayon count-variance noted above; it is not signal.

### Post-run acceptance (run on the selected checkpoint; results pending)

```bash
# Static headline vs GA
cargo run --release -- bench ga pair -n 1000000
# Search-on
cargo run --release -- bench ga pair-search -n 200000
# Lock-wrong rate vs Phase 16's 8.8% (18.0% behind)
./target/release/examples/divergence lock-run -n 10000 --seed 0 --out lock-retrained.jsonl \
  && ./target/release/examples/divergence lock-adjudicate --input lock-retrained.jsonl --out lock-retrained.adj.jsonl -k 2048 \
  && analysis/.venv/bin/python analysis/lock_analysis.py lock-retrained.adj.jsonl 10
# Suppression edge vs Phase 17's +0.06pp
./target/release/examples/divergence lock-ab --equivalence-check 2000 \
  && ./target/release/examples/divergence lock-ab -n 1000000 --seed 0 --suppress-below=0
```

### Results

**Run** (user-executed, 40 iterations at the recipe settings): the fixed-set
curve climbed from the 58.3% re-baseline to **~59.6% by iteration ~20** and
**59.7% at iteration 35** (the selected checkpoint) — +1.4% on the
per-iteration bench, vs Phase 14's +0.4% plateau on the same set. (The
`training_scores.csv` was lost to a resume-path bug — header only written at
`start_iteration == 0` and appends lacked `.create(true)` — fixed in
`9930f0a`; curve numbers above are from the run log.)

**Acceptance on the selected checkpoint (iteration 35):**

| test | old net (Phases 14–17) | retrained | verdict |
|---|---|---|---|
| static 1M vs GA, seed 42 | 59.1% (CI 58.97–59.20) | **59.6% (CI 59.53–59.76)** | **+0.5pp, CIs separated — bar met** |
| search-on 50k vs GA, seed 42 | 60.1% | **60.4% (CI 59.85–60.86)** | no regression (+0.3pp, ~1σ) |
| lock-adjudicate: lock_wrong | 8.8% (484/5,472) | **8.8%** (480/5,459; ap1 20.5% / pp1 14.5% / ap2 2.5%) | unchanged |
| lock-ab suppression edge (t=0, 1M) | +0.06pp (z 9.7) | **+0.04pp (z 5.0)** | unchanged |

**The checkpoint is adopted** (`pair_model/model.mpk` updated): the
pre-registered +0.5% static-V bar is met on the independent seed-42 set, with
no search-on regression. New headline numbers: **pair (static) 59.6%,
pair-search 60.4% vs GA.** The static-vs-search gap narrowed 1.0pp → 0.8pp —
consistent with distillation moving part of search's knowledge into the
static net.

**Lock secondaries: the lock-pool mechanism did NOT land.** Two corrections
to the pre-registration first: (a) `lock-adjudicate`'s lock_wrong rate
measures the *rule* against rollouts, and the rule didn't change — an
unchanged 8.8% was the expected outcome, not a failure signal; the spec
should not have listed it as "expect a drop". (b) The spec's lock-ab
direction was **backwards**: a net that had truly learned lock valuation
would make the suppression edge *grow* toward the ~0.7pp ceiling (the freed
net finally chooses well), not shrink. With the corrected reading, both
secondaries agree: the edge stayed at +0.04pp (vs +0.06pp), so when freed
from the rule the retrained net still cannot choose better at lock firings.
The +0.5pp came from the **diffuse gated pool**; the lock blind spot is
intact.

**Why the lock targets likely failed: volume.** Lock firings are
~0.55/game vs ~8 gated decisions/game; at the shared m=2 the lock trio was
~1% of the buffer, against 1.5M TD samples whose values all embed
"the lock always happens" (declined-lock continuations exist only in the ~3%
of games where ε-decline fired). The direct supervision was present but
outvoted.

**Follow-up (pre-scoped, not run):** a second distillation leg from the
adopted checkpoint with **per-pool emission weights** (e.g. `m_gated=2`,
`m_lock=16` — lock samples to ~10–15% of buffer; small `DistillCfg` change)
and `--epsilon-lock 0.10–0.15`, pre-registering the **lock-ab edge with the
corrected direction** as the primary lock metric (success = edge grows toward
~0.7pp, then the conditional-rule adoption decision converts it to win rate
via the Phase 17 harness). If 10× lock-sample weight doesn't move the
lock-ab needle, the lock pool should be declared closed (representation
limit, not data) and the campaign banked.

## Phase 19: Distillation leg 2 — lock-pool weighting

**Status: run pending user execution.**

The Phase 18 follow-up, implemented as pre-scoped: `DistillCfg` gained
per-pool emission weights (`--distill-m` for gated decisions, new
`--distill-m-lock` for lock firings, defaults 2/16), and the per-iteration
summary now reports the lock-pool sample count. Mechanism otherwise unchanged
(commit `999c666`; new test `lock_ctx_uses_m_lock_and_is_counted`).

**Hypothesis under test:** Phase 18's lock-target failure was a volume
problem — the lock trio was ~1% of the buffer at the shared m, outvoted by
1.5M TD samples whose values embed "the lock always happens." Leg 2 raises
the lock pool to **~9% of the buffer** (measured in a 300-game smoke:
43,398 distill of 136,294 total, 12,446 lock-pool) and doubles ε-decline so
declined-lock continuations appear in ~5–10% of games.

### Run recipe

```bash
rm -f pair_model/iter-*.mpk
cargo run --release -- pair-train --distill --epsilon-lock 0.10 -g 5000 -e 5 -b 500000 -c --start-iteration 20
```

(m defaults are already the leg-2 values: m_gated=2, m_lock=16. Warm start =
the adopted Phase 18 checkpoint already in `pair_model/`.)

### Pre-registration (corrected directions, learned from Phase 18)

- **Primary lock metric: the lock-ab suppression edge** (t=0, 1M, seed 0) on
  the selected checkpoint. Status quo +0.04pp; ceiling ≈ +0.7pp.
  **Success: edge ≥ +0.3pp** (the freed net demonstrably chooses better at
  firings — then the conditional-rule adoption decision via the Phase 17
  harness converts it into realized win rate). **Kill: edge < +0.15pp** ⇒
  the lock pool is representation-limited, not data-limited; declare it
  closed and bank the campaign. Between: judgment call, lean kill.
- **Guard: static 1M vs GA** (seed 42) must not regress from 59.6%
  (CI overlap acceptable). Checkpoint selection stays on the per-iteration
  static curve as before.
- `lock-adjudicate` is NOT an acceptance metric (it measures the unchanged
  rule; expected to stay ~8.8% regardless — Phase 18's correction).
- Curve expectation: the gated pool is already partially harvested, so the
  static curve may rise less than leg 1 (or plateau immediately); that alone
  is not failure — the lock-ab edge is the point of this leg.

### Post-run acceptance

```bash
cargo run --release -- bench ga pair -n 1000000
./target/release/examples/divergence lock-ab --equivalence-check 2000 \
  && ./target/release/examples/divergence lock-ab -n 1000000 --seed 0 --suppress-below=0
```

### Results

**Run** (user-executed, 40 iterations, global 21–60): fixed-set curve hovered
59.5–59.9%, **max 59.88% at iterations 57 and 39** (vs leg 1's 59.7 peak).
The user benched iterations 35/55/57 independently; **iteration 57 selected:
59.85% vs GA confirmed at 1M, and 50.14% head-to-head vs the old DQN bot.**
`training_scores.csv` logged correctly this time (the `9930f0a` fix).

**Acceptance:**

| metric | pre-registered | measured | verdict |
|---|---|---|---|
| static guard (1M vs GA) | hold 59.6% | **59.85%** | passed — improved |
| lock-ab suppression edge (t=0, 1M) | success ≥ +0.3pp; kill < +0.15pp | **+0.01pp (z +0.78)** | **KILL** |

(Equivalence gate held over 2,000 games before the edge bench.)

**Verdict: the lock pool is CLOSED — representation-limited, not
data-limited.** At ~9% of the buffer (10× leg 1) plus doubled ε-decline, the
freed net still cannot distinguish good locks from bad at firing states: the
suppression edge is statistically zero (it *fell* from +0.04pp — if anything
the heavier lock training taught the net the value *under the rule* even more
faithfully). Per the pre-registration, no further lock-targeted training is
warranted; the residual ~0.7pp lock ceiling would require a representation
change (e.g. explicit features for lock-while-behind interactions) or a
policy head — both out of scope. The Phase 17 decision stands: keep the
unconditional `find_safe_lock` rule.

**Checkpoint iteration 57 is ADOPTED** (`pair_model/model.mpk` updated, real
file): the static guard improved, so leg 2's gated-pool refinement banked
another ~+0.25pp. **Final campaign headline: pair (static) 59.85% vs GA,
pair-search 60.7%** (user bench on iter-57; from 59.1%/60.1%
pre-distillation). 60.7 is the first move past the 60.1% line that stood
through Phases 13–14 and motivated the structural-ceiling hypothesis — the
ceiling, if it exists, is higher than where we'd drawn it.

### Campaign conclusion (Phases 15–19)

The arc that began with "when does search disagree with the static net"
ends here: the diffuse calibration pool was real and distillable (+0.75pp
static across two legs — the first material gain since Phase 12, achieved
after Phase 14's expert iteration plateaued); the lock blind spot was real,
measurable, and ultimately **not fixable by data alone** under the current
representation. Every claim along the way was adjudicated by harnesses that
remain in the tree (divergence, lock-adjudicate, lock-ab) and were reused
across phases as regression guards — including catching nothing-burgers
(unchanged production behavior through two production refactors) and real
bugs (CSV resume, tie-null bias). Remaining ideas, none pre-committed:
representation work for lock/endgame interactions, a third distillation leg
on fresh rollouts (flywheel), or accepting the ~60% structural ceiling.

## Phase 20: AZ Representation (aznet) — shared-encoder one-hot value net

**Status: implemented; run pending.** Spec:
`docs/superpowers/specs/2026-06-15-aznet-representation-design.md`.

Tests AlphaZero's *representation* lever in the existing pair-train harness: a
new afterstate value net (`aznet`) encoding each board as one-hot crossing-order
per-row blocks (count 0..=12, free-pointer slot, is_locked, is_lockable on both
boards, wprob + blanks scalars) through a shared `f_row` 28→32→16 encoder, then
a 128→64 trunk with μ/σ diff-space heads (~27.5k params). Everything else is the
pair net's pipeline unchanged: TD(λ=0.8) diff targets, decoupled μ/σ loss,
board-swap doubling, `(cdiff+μ)/σ` ranking, `SearchBot`, paired-CRN bench.
2-player-only plain self-play; burn-only inference.

### Run recipe

```bash
rm -f aznet_model/iter-*.mpk
# Match the pair net's plain recipe; 200k per-iteration bench for checkpoint selection.
cargo run --release -- aznet-train -i 40 -g 20000 -e 3 -b 200000 -c
```

### Pre-registration

- **Control:** recorded plain pair numbers (~59.2% static / ~60.1% search;
  weaker isolation — recorded run used 1v1/3p/4p thirds, not 2p-only).
- **Primary:** selected checkpoint's win rate vs GA @ 1M, seed 42, static AND
  search (`bench ga aznet -n 1000000`, `bench ga aznet-search -n 1000000`).
- **PAY (≥ +0.3pp, CI-separated):** build the hand-rolled inference kernel, run
  a distillation leg, proceed to Arm B (directly-learned win head).
- **KILL (< +0.1pp):** representation is not the lever; stop before Arm B.
- **Capacity-control (only if PAY):** re-run old repr ~27k OR new repr ~14k to
  disentangle representation from the ~2× capacity.
- **Guard:** checkpoint selection on the per-iteration 200k static curve;
  90/10 valid loss watched for overfit (reduce epochs before resizing).

### Implementation note — memory (2026-06-15)

The first run (`-i 40 -g 20000 -e 3 -b 200000 -c`) OOM'd after ~4 iterations:
each `AzSample` stored the expanded `[f32; 233]` input (940 B, ~5x the pair
net's sample), and sample count grows as play improves, so the 3-iteration
replay buffer plus transient copies exceeded RAM. A reduced `-g 5000` run
completed but **plateaued at 58.3%** (200k bench) — below the ~59.2% plain
baseline, **but confounded**: a larger-input/larger net was fed 4x less data
per iteration than the recipe. Fix: `AzSample` now stores the two compact
`State`s and the batcher expands them to the 233-float input (~20x less
memory), so the pre-registered `-g 20000` recipe runs. Adjudication waits on a
clean full-data run benched at 1M seed-42 (static + search).

### Results

_(to be filled after the run)_
