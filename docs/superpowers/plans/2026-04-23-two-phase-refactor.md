# Two-Phase Game Loop Refactor

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the game loop to separate Phase 1 (all players may mark white sum) from Phase 2 (active player only may mark white+color), matching the physical Qwixx rules. This fixes a subtle behavioral bug: opponents' Phase 1 locks should deny the active player's Phase 2 options, and the active player should see updated opponent states before Phase 2.

**Architecture:** Introduce a `Bot` trait (pure state evaluator) with a blanket `impl Strategy for Bot` that handles all two-phase move selection including the RISKY/LOCKABLE edge-case analysis. Strategies that aren't pure evaluators (Random, Interactive, MCTS, etc.) implement `Strategy` directly. The game loop collects all Phase 1 decisions simultaneously (each player sees pre-phase-1 states), applies them, propagates locks, checks for game-over, then runs Phase 2 for the active player only.

**Tech Stack:** Rust, burn (DQN), rayon (parallel benchmarks/training)

**Scope:** Gameplay only. No web UI changes, no DQN retraining, no meta-rules (smart lock/strike/pruning removed initially, re-added later). Benchmarks will shift; that's expected.

---

## File structure

```
Modify: src/state.rs          — add helper methods (can_mark, apply_mark, apply_strike, generate_white_moves, generate_color_moves, would_lock_row)
Modify: src/strategy/mod.rs   — new Strategy trait (3 methods), Bot trait, remove old methods
Create: src/strategy/bot_impl.rs — blanket impl with RISKY/LOCKABLE analysis
Modify: src/game.rs           — two-phase Game::play(), updated Player methods
Modify: src/bot.rs            — DNA implements Bot instead of Strategy
Modify: src/dqn/mod.rs        — DqnStrategy implements Bot, add evaluate/evaluate_batch
Modify: src/dqn/train.rs      — RecordingDqn uses new Strategy trait
Modify: src/mcts.rs           — MonteCarlo uses new Strategy trait
Modify: src/main.rs           — strategy construction unchanged, but trait bounds may shift
Modify: src/lib.rs            — re-export Bot, updated module structure
Modify: examples/*.rs         — update any custom Strategy impls
```

---

### Task 1: State helper methods

**Files:**
- Modify: `src/state.rs`

- [ ] **Step 1: Add `can_mark` public method on State**

```rust
// In impl State, after generate_opponent_moves:
pub fn can_mark(&self, row: usize, number: u8) -> bool {
    self.rows[row].can_mark(number)
}
```

- [ ] **Step 2: Add `apply_mark` and `apply_strike`**

```rust
pub fn apply_mark(&mut self, mark: Mark) {
    self.rows[mark.row].mark(mark.number);
}

pub fn apply_strike(&mut self) {
    assert!(self.strikes < 4);
    self.strikes += 1;
}
```

- [ ] **Step 3: Add `generate_white_moves`**

Returns all marks using only the white dice sum (for Phase 1 / passive turns). Returns `Vec<Mark>` not `Vec<Move>`.

```rust
pub fn generate_white_moves(&self, white_sum: u8) -> Vec<Mark> {
    (0..4)
        .filter(|&row| self.rows[row].can_mark(white_sum))
        .map(|row| Mark { row, number: white_sum })
        .collect()
}
```

- [ ] **Step 4: Add `generate_color_moves`**

Returns all marks using one white die + matching color die (for Phase 2). Returns `Vec<Mark>`.

```rust
pub fn generate_color_moves(&self, dice: [u8; 6]) -> Vec<Mark> {
    let mut moves = Vec::new();
    for i in 0..4 {
        for &w in &[dice[0], dice[1]] {
            let num = w + dice[i + 2];
            if self.rows[i].can_mark(num) {
                moves.push(Mark { row: i, number: num });
            }
        }
    }
    moves.sort_unstable_by_key(|m| (m.row, m.number));
    moves.dedup();
    moves
}
```

Note: `Mark` needs `Ord` or we sort by `(row, number)` tuple.

- [ ] **Step 5: Add `would_lock_row` helper**

Returns true if applying the given mark would lock a row (used by RISKY analysis).

```rust
pub fn would_lock_row(&self, mark: Mark) -> bool {
    let pre_locked = self.count_locked();
    let mut s = *self;
    s.apply_mark(mark);
    s.count_locked() > pre_locked
}
```

- [ ] **Step 6: Add `is_ascending` helper for row direction**

```rust
pub fn is_ascending(&self, row: usize) -> bool {
    row < 2
}
```

(Used by the RISKY analysis to determine terminal numbers.)

- [ ] **Step 7: Add `row_terminal` helper**

```rust
pub fn row_terminal(row: usize) -> u8 {
    if row < 2 { 12 } else { 2 }
}
```

- [ ] **Step 8: Write tests for new helpers**

```rust
#[test]
fn can_mark_fresh_state() {
    let s = State::default();
    assert!(s.can_mark(0, 2));  // red, ascending, free=2
    assert!(!s.can_mark(0, 12)); // need 5+ marks to mark terminal
    assert!(s.can_mark(2, 12)); // green, descending, free=12
}

#[test]
fn apply_mark_updates_state() {
    let mut s = State::default();
    s.apply_mark(Mark { row: 0, number: 5 });
    assert_eq!(s.row_totals()[0], 1);
    assert_eq!(s.row_free_values()[0], Some(6));
}

#[test]
fn generate_white_moves_fresh() {
    let s = State::default();
    let moves = s.generate_white_moves(7);
    assert_eq!(moves.len(), 4); // all 4 rows can mark 7
}

#[test]
fn generate_color_moves() {
    let s = State::default();
    let dice = [3, 4, 2, 3, 5, 1]; // W1=3 W2=4 R=2 Y=3 G=5 B=1
    let moves = s.generate_color_moves(dice);
    // Row 0 (red,asc): 3+2=5 ✓, 4+2=6 ✓
    // Row 1 (yel,asc): 3+3=6 ✓, 4+3=7 ✓
    // Row 2 (grn,desc): 3+5=8 ✓, 4+5=9 ✓
    // Row 3 (blu,desc): 3+1=4 ✓, 4+1=5 ✓
    assert_eq!(moves.len(), 8);
}

#[test]
fn would_lock_row_detects_lock() {
    let mut s = State::default();
    // Mark red row: 2,3,4,5,6 (5 marks), then marking 12 locks
    for n in 2..=6 { s.apply_mark(Mark { row: 0, number: n }); }
    assert!(s.would_lock_row(Mark { row: 0, number: 12 }));
    assert!(!s.would_lock_row(Mark { row: 0, number: 7 }));
}
```

- [ ] **Step 9: Run tests**

Run: `cargo test --lib state::tests`

- [ ] **Step 10: Commit**

```bash
git add src/state.rs
git commit -m "feat: add State helpers for two-phase game loop (can_mark, apply_mark, generate_white/color_moves)"
```

---

### Task 2: New trait definitions

**Files:**
- Modify: `src/strategy/mod.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Define the new `Strategy` trait**

Replace the old 2-method Strategy trait with the 3-method version. Keep `is_interactive` as-is.

```rust
use crate::state::{Mark, Move, State, format_dice};

pub trait Strategy: std::fmt::Debug {
    /// Phase 1 (active player): decide whether to mark the white dice sum.
    /// Called before Phase 1 locks are applied. `opp_states` are pre-phase-1 snapshots.
    /// Return `Some(mark)` to mark, `None` to skip.
    fn active_phase1(&mut self, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark>;

    /// Phase 2 (active player only): decide whether to mark a white+color sum.
    /// Called after Phase 1 locks propagated. `opp_states` reflect Phase 1 results.
    /// `has_marked` is true if Phase 1 produced a mark (affects strike logic).
    /// Return `Some(mark)` to mark, `None` to skip (strike if `!has_marked`).
    fn active_phase2(&mut self, state: &State, opp_states: &[State], dice: [u8; 6], has_marked: bool) -> Option<Mark>;

    /// Phase 1 (passive player): decide whether to mark the white dice sum.
    /// Full dice are provided so strategies can reason about the active player's Phase 2 options.
    /// Return `Some(mark)` to mark, `None` to skip (no penalty for passives).
    fn passive_phase1(&mut self, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark>;

    fn is_interactive(&self) -> bool { false }
}
```

**Important**: Remove `your_move`, `opponents_move`, and `observe_opponents` entirely from the trait. All state observation now comes through `opp_states` parameters.

- [ ] **Step 2: Define the `Bot` trait**

Add directly below `Strategy` in the same file:

```rust
/// A pure state evaluator. Implement this for strategies that rank states by
/// a scalar value (GA instinct, DQN win-rank-score). The blanket
/// `impl Strategy for Bot` in `bot_impl.rs` handles all move selection,
/// including two-phase RISKY/LOCKABLE analysis.
pub trait Bot: std::fmt::Debug {
    /// Evaluate how good `our_state` is for us, given opponents' states.
    /// Higher is better. Must be comparable across candidates at the same decision.
    fn evaluate(&self, our_state: &State, opp_states: &[State]) -> f32;

    /// Batch evaluation for efficiency. Default calls `evaluate` in a loop.
    /// Override for batched forward passes (e.g., DQN).
    fn evaluate_batch(&self, candidates: &[State], opp_states: &[State]) -> Vec<f32> {
        candidates.iter().map(|s| self.evaluate(s, opp_states)).collect()
    }
}
```

- [ ] **Step 3: Update `src/lib.rs` re-exports**

Make sure `Bot` is publicly exported alongside `Strategy`.

- [ ] **Step 4: Commit**

```bash
git add src/strategy/mod.rs src/lib.rs
git commit -m "feat: define new Strategy trait (3-method two-phase) and Bot trait"
```

---

### Task 3: Blanket `impl Strategy for Bot` with RISKY/LOCKABLE analysis

**Files:**
- Create: `src/strategy/bot_impl.rs`
- Modify: `src/strategy/mod.rs` (add `mod bot_impl;`)

This is the most complex task. The blanket impl lets any `Bot` be used as a `Strategy` by handling move generation, candidate evaluation, and the RISKY/LOCKABLE edge-case analysis for Phase 1.

- [ ] **Step 1: Create `src/strategy/bot_impl.rs` with the blanket impl structure**

```rust
use crate::state::{Mark, State};
use super::{Bot, Strategy};

impl<T: Bot + std::fmt::Debug> Strategy for T {
    fn active_phase1(&mut self, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
        active_phase1_impl(self, state, opp_states, dice)
    }

    fn active_phase2(&mut self, state: &State, opp_states: &[State], dice: [u8; 6], has_marked: bool) -> Option<Mark> {
        active_phase2_impl(self, state, opp_states, dice, has_marked)
    }

    fn passive_phase1(&mut self, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
        passive_phase1_impl(self, state, opp_states, dice)
    }
}
```

Using free functions so the logic is testable independently.

- [ ] **Step 2: Implement `passive_phase1_impl`**

Simplest of the three. Generate white-sum mark candidates + skip (no-op). Evaluate each. Pick best.

```rust
fn passive_phase1_impl(bot: &impl Bot, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
    let white_sum = dice[0] + dice[1];
    let marks = state.generate_white_moves(white_sum);
    if marks.is_empty() {
        return None;
    }

    // Evaluate mark candidates + skip
    let skip_value = bot.evaluate(state, opp_states);
    let mut candidates: Vec<State> = marks.iter().map(|&m| {
        let mut s = *state;
        s.apply_mark(m);
        s
    }).collect();

    let values = bot.evaluate_batch(&candidates, opp_states);

    let best_mark_idx = values.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0;

    if values[best_mark_idx] > skip_value {
        Some(marks[best_mark_idx])
    } else {
        None
    }
}
```

- [ ] **Step 3: Implement `active_phase2_impl`**

Generate color-mark candidates. If `has_marked`, include skip as no-penalty option. If `!has_marked`, skip means strike.

```rust
fn active_phase2_impl(bot: &impl Bot, state: &State, opp_states: &[State], dice: [u8; 6], has_marked: bool) -> Option<Mark> {
    let marks = state.generate_color_moves(dice);

    // Evaluate the "no mark" baseline
    let no_mark_state = if has_marked {
        *state  // skip is free
    } else {
        let mut s = *state;
        s.apply_strike();
        s  // skip means strike
    };
    let no_mark_value = bot.evaluate(&no_mark_state, opp_states);

    if marks.is_empty() {
        return None;
    }

    let candidates: Vec<State> = marks.iter().map(|&m| {
        let mut s = *state;
        s.apply_mark(m);
        s
    }).collect();
    let values = bot.evaluate_batch(&candidates, opp_states);

    let best_idx = values.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0;

    if values[best_idx] > no_mark_value {
        Some(marks[best_idx])
    } else {
        None
    }
}
```

- [ ] **Step 4: Implement `active_phase1_impl` — plan generation**

This is the core logic. Generate all possible (phase1, phase2) plan pairs, evaluate each by the resulting state, then return the phase1 part of the best plan.

A "plan" is `(Option<Mark>, Option<Mark>)` = (phase1_mark, phase2_mark). We represent them as:

```rust
#[derive(Clone, Copy)]
struct Plan {
    phase1: Option<Mark>,
    phase2: Option<Mark>,
}
```

Generate all valid plans:

```rust
fn generate_plans(state: &State, dice: [u8; 6]) -> Vec<(Plan, State)> {
    let white_sum = dice[0] + dice[1];
    let white_marks = state.generate_white_moves(white_sum);
    let color_marks = state.generate_color_moves(dice);
    let mut plans = Vec::new();

    // Plan: skip both → strike
    {
        let mut s = *state;
        s.apply_strike();
        plans.push((Plan { phase1: None, phase2: None }, s));
    }

    // Plans: skip phase1, mark color in phase2
    for &cm in &color_marks {
        let mut s = *state;
        s.apply_mark(cm);
        plans.push((Plan { phase1: None, phase2: Some(cm) }, s));
    }

    // Plans: mark white in phase1, skip phase2
    for &wm in &white_marks {
        let mut s = *state;
        s.apply_mark(wm);
        plans.push((Plan { phase1: Some(wm), phase2: None }, s));
    }

    // Plans: mark white in phase1 + mark color in phase2
    for &wm in &white_marks {
        let mut s_after_white = *state;
        s_after_white.apply_mark(wm);
        for &cm in &color_marks {
            // Color mark must still be valid after white mark
            if s_after_white.can_mark(cm.row, cm.number) {
                let mut s = s_after_white;
                s.apply_mark(cm);
                plans.push((Plan { phase1: Some(wm), phase2: Some(cm) }, s));
            }
        }
        // Also check same-row color marks enabled by white mark
        let post_color = s_after_white.generate_color_moves(dice);
        for cm in post_color {
            if cm.row == wm.row && !color_marks.contains(&cm) {
                // This color mark was unlocked by the white mark on the same row
                let mut s = s_after_white;
                s.apply_mark(cm);
                plans.push((Plan { phase1: Some(wm), phase2: Some(cm) }, s));
            }
        }
    }

    plans
}
```

- [ ] **Step 5: Implement RISKY/LOCKABLE filtering**

Only applies when `white_sum == 2` or `white_sum == 12` (the terminal numbers that can trigger locks).

```rust
/// Which rows might opponents lock during Phase 1 by marking white_sum?
fn at_risk_rows(opp_states: &[State], white_sum: u8) -> Vec<usize> {
    (0..4).filter(|&row| {
        let terminal = State::row_terminal(row);
        if white_sum != terminal { return false; }
        opp_states.iter().any(|opp| {
            opp.can_mark(row, white_sum) && opp.would_lock_row(Mark { row, number: white_sum })
        })
    }).collect()
}

/// How many rows are globally locked across all players?
fn global_locked_count(state: &State, opp_states: &[State]) -> u8 {
    let our_locked = state.locked();
    (0..4).filter(|&r| {
        our_locked[r] || opp_states.iter().any(|o| o.locked()[r])
    }).count() as u8
}

/// Which rows can WE lock in phase1 with a white-sum mark?
fn our_lockable_rows(state: &State, white_sum: u8) -> Vec<usize> {
    state.generate_white_moves(white_sum).iter()
        .filter(|wm| state.would_lock_row(**wm))
        .map(|wm| wm.row)
        .collect()
}
```

Then in `active_phase1_impl`, after generating plans:

```rust
fn filter_risky_plans(
    plans: &mut Vec<(Plan, State)>,
    state: &State,
    opp_states: &[State],
    white_sum: u8,
) {
    // Only terminal white sums can cause Phase 1 locks
    if white_sum != 2 && white_sum != 12 {
        return;
    }

    let risky = at_risk_rows(opp_states, white_sum);
    if risky.is_empty() {
        return;
    }

    let global_locked = global_locked_count(state, opp_states);
    let our_lockable = our_lockable_rows(state, white_sum);
    let max_new_locks = risky.len() as u8 + our_lockable.len() as u8;

    // Can game end in Phase 1? Need 2+ total locked rows.
    let game_can_end = global_locked + max_new_locks >= 2;

    if game_can_end {
        // Remove ALL plans that rely on Phase 2 (Phase 2 might not happen)
        plans.retain(|(plan, _)| plan.phase2.is_none());
    } else {
        // Game won't end, but at-risk rows might become locked → deny Phase 2 on those rows
        plans.retain(|(plan, _)| {
            match plan.phase2 {
                None => true,
                Some(cm) => !risky.contains(&cm.row),
            }
        });
    }

    // If filtering removed everything, keep at least the strike plan
    if plans.is_empty() {
        let mut s = *state;
        s.apply_strike();
        plans.push((Plan { phase1: None, phase2: None }, s));
    }
}
```

**Important nuance from the user's design:** The `game_can_end` check is conservative — it counts the theoretical maximum new locks (all risky + all our lockable). In practice, fewer may lock. But removing phase2-dependent plans is the safe choice.

A more refined version (matching the user's case tree) would check whether our lockable rows overlap with risky rows, whether we actually need to lock to trigger game-end, etc. For the initial implementation, the conservative version above is correct and simpler. It can be refined later.

- [ ] **Step 6: Implement `active_phase1_impl` putting it all together**

```rust
fn active_phase1_impl(bot: &impl Bot, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
    let white_sum = dice[0] + dice[1];
    let mut plans = generate_plans(state, dice);

    filter_risky_plans(&mut plans, state, opp_states, white_sum);

    // Evaluate all plan end-states
    let plan_states: Vec<State> = plans.iter().map(|(_, s)| *s).collect();
    let scores = bot.evaluate_batch(&plan_states, opp_states);

    let best_idx = scores.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0;

    let best_plan = &plans[best_idx].0;

    // Return the Phase 1 part of the best plan.
    // If best plan has no Phase 1 mark (color-only or strike), defer to Phase 2.
    best_plan.phase1
}
```

- [ ] **Step 7: Register the module**

In `src/strategy/mod.rs`, add:

```rust
mod bot_impl;
```

- [ ] **Step 8: Write tests for the blanket impl**

```rust
// In bot_impl.rs:
#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{Mark, State};

    /// Trivial bot that evaluates by score (higher is better).
    #[derive(Debug)]
    struct ScoreBot;
    impl Bot for ScoreBot {
        fn evaluate(&self, state: &State, _opp: &[State]) -> f32 {
            state.count_points() as f32
        }
    }

    #[test]
    fn passive_skips_when_no_moves() {
        let s = State::default();
        let mut bot = ScoreBot;
        // white_sum=13 is impossible, so no marks available
        // Actually white_sum max is 12. Use a state where no row can mark.
        // For simplicity, use white_sum=2 on a fresh state: only desc rows can mark 12... wait.
        // Fresh state: ascending free=2 (can mark 2..=12 minus terminal rule), desc free=12.
        // white_sum=2: asc rows need can_mark(2) = true (free=2), but 2 is terminal for desc rows
        // ... test a case with locked rows instead
    }

    #[test]
    fn active_phase1_prefers_double_plan() {
        let s = State::default();
        let mut bot = ScoreBot;
        let dice = [3, 4, 2, 3, 5, 1]; // white=7
        let result = bot.active_phase1(&s, &[], dice);
        // ScoreBot prefers max points → should pick a Double plan's white part
        // white_sum=7 is markable on all rows; color marks available too
        // The best plan is likely a Double (most points) → phase1 returns Some(mark)
        assert!(result.is_some());
    }

    #[test]
    fn active_phase1_defers_to_phase2_when_color_only_is_best() {
        // Construct a state where white_sum mark is bad but a color mark is good
        // This would require a specific state setup...
    }
}
```

Tests for the RISKY analysis are covered in Task 12.

- [ ] **Step 9: Commit**

```bash
git add src/strategy/bot_impl.rs src/strategy/mod.rs
git commit -m "feat: blanket impl Bot->Strategy with RISKY/LOCKABLE phase1 analysis"
```

---

### Task 4: Game loop refactor

**Files:**
- Modify: `src/game.rs`

- [ ] **Step 1: Update `Player` methods**

Remove `your_move`, `opponents_move`, `observe_opponents`. Add phase-specific methods:

```rust
impl Player {
    // ... keep new, new_with_state, is_interactive, roll ...

    fn active_phase1(&mut self, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
        self.strategy.active_phase1(&self.state, opp_states, dice)
    }

    fn active_phase2(&mut self, opp_states: &[State], dice: [u8; 6], has_marked: bool) -> Option<Mark> {
        self.strategy.active_phase2(&self.state, opp_states, dice, has_marked)
    }

    fn passive_phase1(&mut self, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
        self.strategy.passive_phase1(&self.state, opp_states, dice)
    }

    fn apply_mark(&mut self, mark: Mark) {
        self.state.apply_mark(mark);
    }

    fn apply_strike(&mut self) {
        self.state.apply_strike();
    }
}
```

- [ ] **Step 2: Rewrite `Game::play()` with two-phase flow**

```rust
pub fn play(&mut self) {
    let n = self.players.len();
    let mut active_player = self.start_turn.take().unwrap_or(0);

    while !self.game_over() {
        let dice = self.players[active_player].roll();

        // === PHASE 1: All players decide on white sum (simultaneous) ===
        // Snapshot pre-phase1 states — every player sees THIS view.
        let pre_phase1: Vec<State> = self.players.iter().map(|p| p.state).collect();

        // Collect Phase 1 decisions (no state mutations yet)
        let mut phase1_marks: Vec<Option<Mark>> = Vec::with_capacity(n);
        for i in 0..n {
            let opp_states: Vec<State> = pre_phase1.iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, s)| *s)
                .collect();

            let mark = if i == active_player {
                self.players[i].strategy.active_phase1(&pre_phase1[i], &opp_states, dice)
            } else {
                self.players[i].strategy.passive_phase1(&pre_phase1[i], &opp_states, dice)
            };
            phase1_marks.push(mark);
        }

        // Apply all Phase 1 marks simultaneously
        let active_marked_phase1 = phase1_marks[active_player].is_some();
        for (i, mark) in phase1_marks.iter().enumerate() {
            if let Some(m) = mark {
                self.players[i].apply_mark(*m);
            }
        }

        // Propagate Phase 1 locks
        let phase1_locked = self.aggregate_locks();
        for player in self.players.iter_mut() {
            player.state.lock(phase1_locked);
        }

        // Check game-over after Phase 1 (2+ rows locked or 4 strikes)
        if self.game_over() {
            break;
        }

        // === PHASE 2: Active player only, with updated opponent states ===
        let opp_states: Vec<State> = self.players.iter()
            .enumerate()
            .filter(|(i, _)| *i != active_player)
            .map(|(_, p)| p.state)
            .collect();

        let phase2_mark = self.players[active_player]
            .strategy
            .active_phase2(
                &self.players[active_player].state,
                &opp_states,
                dice,
                active_marked_phase1,
            );

        match phase2_mark {
            Some(m) => self.players[active_player].apply_mark(m),
            None if !active_marked_phase1 => self.players[active_player].apply_strike(),
            None => {} // Skip Phase 2, no penalty (already marked in Phase 1)
        }

        // Propagate Phase 2 locks
        let phase2_locked = self.aggregate_locks();
        for player in self.players.iter_mut() {
            player.state.lock(phase2_locked);
        }

        active_player = (active_player + 1) % n;
    }
}
```

- [ ] **Step 3: Add `aggregate_locks` helper**

```rust
fn aggregate_locks(&self) -> [bool; 4] {
    let mut locked = [false; 4];
    for player in &self.players {
        for (row, is_locked) in player.state.locked().iter().enumerate() {
            locked[row] |= *is_locked;
        }
    }
    locked
}
```

- [ ] **Step 4: Update `game_over_reason`**

The current implementation checks per-player locked counts, but with Phase 1 simultaneous locks, 3+ rows could be locked at once. The check `total_locked >= 2` using the max across players should still work because `State::lock` propagates all global locks to all players. However, verify the logic is correct: after `lock(locked)`, every player has the same locked rows, so `player.state.count_locked()` is the same for all players. The current implementation that takes `max()` is fine.

No code change needed here — just verify during testing.

- [ ] **Step 5: Update verbose logging**

Update the verbose logging in `play()` to show Phase 1 and Phase 2 separately:

```rust
if self.verbose {
    // After Phase 1 marks applied:
    for (i, mark) in phase1_marks.iter().enumerate() {
        match mark {
            Some(m) => eprintln!("  Phase 1: Player {} marked {}", i + 1, Move::Single(*m)),
            None => {}
        }
    }
    // After Phase 2:
    match phase2_mark {
        Some(m) => eprintln!("  Phase 2: Player {} marked {}", active_player + 1, Move::Single(m)),
        None if !active_marked_phase1 => eprintln!("  Player {} struck", active_player + 1),
        None => eprintln!("  Phase 2: Player {} skipped", active_player + 1),
    }
}
```

- [ ] **Step 6: Commit**

```bash
git add src/game.rs
git commit -m "feat: two-phase game loop — simultaneous Phase 1, active-only Phase 2"
```

---

### Task 5: Migrate GA bot to `Bot` trait

**Files:**
- Modify: `src/bot.rs`

- [ ] **Step 1: Replace `impl Strategy for DNA` with `impl Bot for DNA`**

Remove the `Strategy` impl entirely (including `your_move`, `opponents_move`, `observe_opponents`). Remove the `score_gap` field from `DNA` — it's no longer needed since `opp_states` is passed directly.

```rust
impl Bot for DNA {
    fn evaluate(&self, our_state: &State, opp_states: &[State]) -> f32 {
        self.instinct(our_state) as f32
    }
}
```

GA's `instinct` currently doesn't use opponent information. It could incorporate score gap, but for the initial migration this is fine — the blanket impl handles everything else.

- [ ] **Step 2: Remove `score_gap` field from DNA**

```rust
pub struct DNA {
    weights: Vec<f64>,
    genes: Arc<Vec<GeneFn>>,
}
```

Update `new_random`, `crossover`, `mutate`, `load_weights`, `save_weights` to remove `score_gap: 0` initialization.

- [ ] **Step 3: Import `Bot` instead of `Strategy`**

```rust
use crate::strategy::Bot;
```

Remove the `use crate::strategy::Strategy;` import.

- [ ] **Step 4: Verify compilation**

Run: `cargo check --lib`

- [ ] **Step 5: Commit**

```bash
git add src/bot.rs
git commit -m "refactor: migrate GA DNA to Bot trait"
```

---

### Task 6: Migrate DQN to `Bot` trait

**Files:**
- Modify: `src/dqn/mod.rs`

- [ ] **Step 1: Implement `Bot` for `DqnStrategy`**

```rust
impl Bot for DqnStrategy {
    fn evaluate(&self, our_state: &State, opp_states: &[State]) -> f32 {
        if opp_states.is_empty() {
            let ctx = OpponentContext::default();
            let features = state_features(our_state, &ctx);
            let (mean, _) = self.model.evaluate_state(&features, &self.device);
            return mean;
        }

        // Find leader among opponents
        let leader_idx = (0..opp_states.len())
            .max_by_key(|&i| opp_states[i].count_points())
            .unwrap();
        let leader = &opp_states[leader_idx];
        let non_leaders: Vec<State> = opp_states.iter()
            .enumerate()
            .filter(|(i, _)| *i != leader_idx)
            .map(|(_, s)| *s)
            .collect();

        // Build our context (post-move perspective)
        let our_ctx = build_opponent_context_for(
            our_state.count_points(), leader, &non_leaders,
        );
        let our_features = state_features(our_state, &our_ctx);
        let (us_mean, us_log_var) = self.model.evaluate_state(&our_features, &self.device);

        // Build opp context (leader's perspective of us)
        let opp_ctx = build_opponent_context_for(
            leader.count_points(), our_state, &non_leaders,
        );
        let opp_features = state_features(leader, &opp_ctx);
        let (opp_mean, opp_log_var) = self.model.evaluate_state(&opp_features, &self.device);

        win_rank_score(us_mean, us_log_var, opp_mean, opp_log_var)
    }

    fn evaluate_batch(&self, candidates: &[State], opp_states: &[State]) -> Vec<f32> {
        if opp_states.is_empty() {
            let ctx = OpponentContext::default();
            let feats: Vec<[f32; NUM_FEATURES]> = candidates.iter()
                .map(|s| state_features(s, &ctx))
                .collect();
            return batch_forward_features(&self.model, &self.device, &feats)
                .into_iter()
                .map(|(mean, _)| mean)
                .collect();
        }

        let leader_idx = (0..opp_states.len())
            .max_by_key(|&i| opp_states[i].count_points())
            .unwrap();
        let leader = &opp_states[leader_idx];
        let non_leaders: Vec<State> = opp_states.iter()
            .enumerate()
            .filter(|(i, _)| *i != leader_idx)
            .map(|(_, s)| *s)
            .collect();

        // Build 2N features: [our_0, opp_0, our_1, opp_1, ...]
        let mut features_list = Vec::with_capacity(candidates.len() * 2);
        for cand in candidates {
            let our_ctx = build_opponent_context_for(
                cand.count_points(), leader, &non_leaders,
            );
            features_list.push(state_features(cand, &our_ctx));
            let opp_ctx = build_opponent_context_for(
                leader.count_points(), cand, &non_leaders,
            );
            features_list.push(state_features(leader, &opp_ctx));
        }

        let values = batch_forward_features(&self.model, &self.device, &features_list);
        (0..candidates.len()).map(|i| {
            let (mean, log_var) = values[2 * i];
            let (opp_mean, opp_log_var) = values[2 * i + 1];
            win_rank_score(mean, log_var, opp_mean, opp_log_var)
        }).collect()
    }
}
```

- [ ] **Step 2: Remove `impl Strategy for DqnStrategy`**

Delete the entire `impl Strategy for DqnStrategy` block (lines ~475-590). The blanket impl from `Bot` provides `Strategy` automatically.

- [ ] **Step 3: Remove stale fields from `DqnStrategy`**

Remove `context`, `leader_state`, `non_leader_states`, `cache` fields — these were used by `observe_opponents` and the old `your_move`/`opponents_move`. Now all context is computed per-call in `evaluate`/`evaluate_batch`.

```rust
pub struct DqnStrategy {
    model: QwixxModel<MyBackend>,
    device: burn::backend::ndarray::NdArrayDevice,
}
```

Update `from_model` and `new` constructors accordingly.

- [ ] **Step 4: Keep `rank_candidates_with_opp_context` and `evaluate_with_context`**

These are still used by training code and examples. Keep them as standalone functions/methods, but they're no longer part of the Strategy impl path.

- [ ] **Step 5: Verify compilation**

Run: `cargo check --lib`

- [ ] **Step 6: Commit**

```bash
git add src/dqn/mod.rs
git commit -m "refactor: migrate DQN to Bot trait with evaluate_batch"
```

---

### Task 7: Migrate simple strategies (Random, Conservative, Opportunist)

**Files:**
- Modify: `src/strategy/mod.rs`

These strategies implement `Strategy` directly (not `Bot`), since they have custom move-filtering logic (blank caps) that doesn't map to a pure evaluator.

- [ ] **Step 1: Migrate Random**

```rust
impl Strategy for Random {
    fn active_phase1(&mut self, state: &State, _opp: &[State], dice: [u8; 6]) -> Option<Mark> {
        let marks = state.generate_white_moves(dice[0] + dice[1]);
        if marks.is_empty() { return None; }
        let mut rng = rand::thread_rng();
        // 50% chance to mark, 50% to defer to phase2
        if rng.gen_bool(0.5) {
            Some(marks[rng.gen_range(0..marks.len())])
        } else {
            None
        }
    }

    fn active_phase2(&mut self, state: &State, _opp: &[State], dice: [u8; 6], has_marked: bool) -> Option<Mark> {
        let marks = state.generate_color_moves(dice);
        if marks.is_empty() { return None; }
        let mut rng = rand::thread_rng();
        if has_marked {
            // 50% chance to also mark color
            if rng.gen_bool(0.5) {
                Some(marks[rng.gen_range(0..marks.len())])
            } else {
                None
            }
        } else {
            // Must mark or strike — try to mark
            Some(marks[rng.gen_range(0..marks.len())])
        }
    }

    fn passive_phase1(&mut self, state: &State, _opp: &[State], dice: [u8; 6]) -> Option<Mark> {
        let marks = state.generate_white_moves(dice[0] + dice[1]);
        if marks.is_empty() { return None; }
        let mut rng = rand::thread_rng();
        if rng.gen_bool(0.5) {
            Some(marks[rng.gen_range(0..marks.len())])
        } else {
            None
        }
    }
}
```

- [ ] **Step 2: Migrate Conservative**

Remove `score_gap` field. Evaluate by blanks + score directly per phase.

```rust
impl Strategy for Conservative {
    fn active_phase1(&mut self, state: &State, _opp: &[State], dice: [u8; 6]) -> Option<Mark> {
        let white_sum = dice[0] + dice[1];
        let marks = state.generate_white_moves(white_sum);
        Self::best_mark(state, &marks, self.max_new_blanks)
    }

    fn active_phase2(&mut self, state: &State, _opp: &[State], dice: [u8; 6], _has_marked: bool) -> Option<Mark> {
        let marks = state.generate_color_moves(dice);
        Self::best_mark(state, &marks, self.max_new_blanks)
    }

    fn passive_phase1(&mut self, state: &State, _opp: &[State], dice: [u8; 6]) -> Option<Mark> {
        let white_sum = dice[0] + dice[1];
        let marks = state.generate_white_moves(white_sum);
        Self::best_mark(state, &marks, 0)
    }
}

impl Conservative {
    fn best_mark(state: &State, marks: &[Mark], max_new_blanks: u8) -> Option<Mark> {
        let current_blanks = state.blanks();
        marks.iter()
            .filter_map(|&mark| {
                let mut s = *state;
                s.apply_mark(mark);
                let new_blanks = s.blanks().saturating_sub(current_blanks);
                if new_blanks <= max_new_blanks {
                    Some((mark, new_blanks, s.count_points()))
                } else {
                    None
                }
            })
            .min_by_key(|(_, blanks, score)| (*blanks, -*score))
            .map(|(mark, _, _)| mark)
    }
}
```

- [ ] **Step 3: Migrate Opportunist**

Similar to Conservative but evaluates by probability.

```rust
impl Strategy for Opportunist {
    fn active_phase1(&mut self, state: &State, _opp: &[State], dice: [u8; 6]) -> Option<Mark> {
        let marks = state.generate_white_moves(dice[0] + dice[1]);
        Self::best_mark(state, &marks, 2)
    }

    fn active_phase2(&mut self, state: &State, _opp: &[State], dice: [u8; 6], _has_marked: bool) -> Option<Mark> {
        let marks = state.generate_color_moves(dice);
        Self::best_mark(state, &marks, 2)
    }

    fn passive_phase1(&mut self, state: &State, _opp: &[State], dice: [u8; 6]) -> Option<Mark> {
        let marks = state.generate_white_moves(dice[0] + dice[1]);
        Self::best_mark(state, &marks, 1)
    }
}

impl Opportunist {
    fn best_mark(state: &State, marks: &[Mark], max_new_blanks: u8) -> Option<Mark> {
        let current_blanks = state.blanks();
        marks.iter()
            .copied()
            .filter_map(|mark| {
                let mut s = *state;
                s.apply_mark(mark);
                let new_blanks = s.blanks().saturating_sub(current_blanks);
                if new_blanks <= max_new_blanks {
                    Some((mark, s.probability()))
                } else {
                    None
                }
            })
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(mark, _)| mark)
    }
}
```

- [ ] **Step 4: Remove `observe_opponents` usages and `score_gap` fields**

Remove `score_gap` from Conservative and Opportunist structs.

- [ ] **Step 5: Verify compilation**

Run: `cargo check --lib`

- [ ] **Step 6: Commit**

```bash
git add src/strategy/mod.rs
git commit -m "refactor: migrate Random, Conservative, Opportunist to new Strategy trait"
```

---

### Task 8: Migrate MCTS

**Files:**
- Modify: `src/mcts.rs`

- [ ] **Step 1: Implement new Strategy trait for MonteCarlo**

MCTS runs rollouts for each candidate. For Phase 1, evaluate white-sum marks vs skip via rollouts. For Phase 2, evaluate color marks vs skip/strike.

The rollout games themselves use the inner rollout strategy (GA), which now uses the new two-phase game loop automatically since `Game::play()` was refactored.

```rust
impl Strategy for MonteCarlo {
    fn active_phase1(&mut self, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
        let white_sum = dice[0] + dice[1];
        let marks = state.generate_white_moves(white_sum);
        if marks.is_empty() {
            return None;
        }

        // Evaluate skip (pass to Phase 2)
        let skip_value = self.evaluate_state(state, opp_states);
        let best_mark = marks.iter()
            .map(|&m| {
                let mut s = *state;
                s.apply_mark(m);
                (m, self.evaluate_state(&s, opp_states))
            })
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        if best_mark.1 > skip_value { Some(best_mark.0) } else { None }
    }

    fn active_phase2(&mut self, state: &State, opp_states: &[State], dice: [u8; 6], has_marked: bool) -> Option<Mark> {
        let marks = state.generate_color_moves(dice);
        let no_mark_state = if has_marked {
            *state
        } else {
            let mut s = *state;
            s.apply_strike();
            s
        };
        let skip_value = self.evaluate_state(&no_mark_state, opp_states);

        if marks.is_empty() {
            return None;
        }

        let best_mark = marks.iter()
            .map(|&m| {
                let mut s = *state;
                s.apply_mark(m);
                (m, self.evaluate_state(&s, opp_states))
            })
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        if best_mark.1 > skip_value { Some(best_mark.0) } else { None }
    }

    fn passive_phase1(&mut self, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
        let white_sum = dice[0] + dice[1];
        let marks = state.generate_white_moves(white_sum);
        if marks.is_empty() { return None; }

        let skip_value = self.evaluate_state(state, opp_states);
        let best_mark = marks.iter()
            .map(|&m| {
                let mut s = *state;
                s.apply_mark(m);
                (m, self.evaluate_state(&s, opp_states))
            })
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        if best_mark.1 > skip_value { Some(best_mark.0) } else { None }
    }
}
```

`evaluate_state` is the existing `evaluate_move_public` adapted to evaluate a state (not a move) via rollout. This may require a small refactor of the existing rollout logic.

- [ ] **Step 2: Remove `score_gap` field and old Strategy methods**

- [ ] **Step 3: Commit**

```bash
git add src/mcts.rs
git commit -m "refactor: migrate MCTS to new Strategy trait"
```

---

### Task 9: Migrate Interactive

**Files:**
- Modify: `src/strategy/mod.rs`

- [ ] **Step 1: Implement two-phase prompting**

```rust
impl Strategy for Interactive {
    fn active_phase1(&mut self, state: &State, _opp: &[State], dice: [u8; 6]) -> Option<Mark> {
        let white_sum = dice[0] + dice[1];
        println!("\n  {BOLD}═══ YOUR TURN — Phase 1 (white sum: {white_sum}) ═══{RESET}\n");
        println!("{state}");
        println!("{}", format_dice(dice));

        let marks = state.generate_white_moves(white_sum);
        if marks.is_empty() {
            println!("  {DIM}No white-sum marks available.{RESET}");
            return None;
        }

        println!("\n  {DIM}Mark white sum ({white_sum}):{RESET}");
        for (i, mark) in marks.iter().enumerate() {
            println!("  {DIM}{}){RESET} {}", i + 1, Move::Single(*mark));
        }
        println!("  {DIM}Enter number or Enter to skip:{RESET}");

        loop {
            let input = Self::read_line();
            let input = input.trim();
            if input.is_empty() { return None; }
            if let Ok(n) = input.parse::<usize>() {
                if n >= 1 && n <= marks.len() {
                    return Some(marks[n - 1]);
                }
            }
            println!("  \x1b[91mInvalid.{RESET}");
        }
    }

    fn active_phase2(&mut self, state: &State, _opp: &[State], dice: [u8; 6], has_marked: bool) -> Option<Mark> {
        println!("\n  {BOLD}═══ Phase 2 (color marks) ═══{RESET}\n");
        println!("{state}");

        let marks = state.generate_color_moves(dice);
        if marks.is_empty() {
            if !has_marked {
                println!("  {DIM}No marks available — strike.{RESET}");
            } else {
                println!("  {DIM}No color marks available.{RESET}");
            }
            return None;
        }

        for (i, mark) in marks.iter().enumerate() {
            println!("  {DIM}{}){RESET} {}", i + 1, Move::Single(*mark));
        }
        if has_marked {
            println!("  {DIM}Enter number or Enter to skip:{RESET}");
        } else {
            println!("  {DIM}Enter number (or Enter for strike):{RESET}");
        }

        loop {
            let input = Self::read_line();
            let input = input.trim();
            if input.is_empty() { return None; }
            if let Ok(n) = input.parse::<usize>() {
                if n >= 1 && n <= marks.len() {
                    return Some(marks[n - 1]);
                }
            }
            println!("  \x1b[91mInvalid.{RESET}");
        }
    }

    fn passive_phase1(&mut self, state: &State, _opp: &[State], dice: [u8; 6]) -> Option<Mark> {
        let white_sum = dice[0] + dice[1];
        println!("\n  {BOLD}═══ OPPONENT'S TURN — mark white sum? ({white_sum}) ═══{RESET}\n");
        println!("{state}");

        let marks = state.generate_white_moves(white_sum);
        if marks.is_empty() {
            println!("  {DIM}No marks available.{RESET}");
            return None;
        }

        for (i, mark) in marks.iter().enumerate() {
            println!("  {DIM}{}){RESET} {}", i + 1, Move::Single(*mark));
        }
        println!("  {DIM}Enter number or Enter to skip:{RESET}");

        loop {
            let input = Self::read_line();
            let input = input.trim();
            if input.is_empty() { return None; }
            if let Ok(n) = input.parse::<usize>() {
                if n >= 1 && n <= marks.len() {
                    return Some(marks[n - 1]);
                }
            }
            println!("  \x1b[91mInvalid.{RESET}");
        }
    }

    fn is_interactive(&self) -> bool { true }
}
```

- [ ] **Step 2: Commit**

```bash
git add src/strategy/mod.rs
git commit -m "refactor: migrate Interactive to two-phase Strategy"
```

---

### Task 10: Migrate training code (RecordingDqn)

**Files:**
- Modify: `src/dqn/train.rs`

- [ ] **Step 1: Update `RecordingDqn` struct**

Remove `context`, `leader_state`, `non_leader_states` fields. Add model and device only.

```rust
struct RecordingDqn {
    model: QwixxModel<MyBackend>,
    device: burn::backend::ndarray::NdArrayDevice,
    epsilon: f32,
    rng: SmallRng,
    recorded: Rc<RefCell<Vec<[f32; NUM_FEATURES]>>>,
}
```

- [ ] **Step 2: Implement new `Strategy` for `RecordingDqn`**

RecordingDqn should record features AND handle epsilon-greedy exploration. It implements Strategy directly (not Bot) because it needs to intercept decisions for recording.

For each phase, it:
1. Generates candidate marks
2. With probability epsilon, picks randomly
3. Otherwise, evaluates via the DQN model (using `rank_candidates_with_opp_context` or `evaluate_batch`-style logic)
4. Records the post-decision features

```rust
impl Strategy for RecordingDqn {
    fn active_phase1(&mut self, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
        let white_sum = dice[0] + dice[1];
        let marks = state.generate_white_moves(white_sum);
        // For simplicity in training: evaluate the full move set like the blanket impl
        // and record post-decision state features.
        // ... (detailed impl mirrors blanket impl but with epsilon-greedy + recording)
        todo!("implement with epsilon-greedy and feature recording")
    }

    fn active_phase2(&mut self, state: &State, opp_states: &[State], dice: [u8; 6], has_marked: bool) -> Option<Mark> {
        todo!("implement with epsilon-greedy and feature recording")
    }

    fn passive_phase1(&mut self, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
        // On passive turns, use DQN evaluation (no exploration, no recording)
        // Similar to the Bot blanket impl's passive_phase1
        todo!("implement passive — delegate to model evaluation")
    }
}
```

**Important implementation note**: The recording logic should record features for the chosen post-move state, same as today. The TD(λ) target computation in `play_training_game` works the same way — it just processes a sequence of recorded features.

- [ ] **Step 3: Update `play_training_game`**

This function creates a `RecordingDqn` and opponent strategies, builds a `Game`, and calls `game.play()`. Since `Game::play()` now uses the two-phase protocol, the recording happens through the `Strategy` methods. The TD target computation at the end is unchanged — it's still a backwards pass through the feature sequence.

Remove the `make_context` helper if it's no longer used.

- [ ] **Step 4: Update `TrainingSample` and batcher**

The `TrainingSample` structure is unchanged: `{features, value, final_score}`. The batcher is unchanged. Only the Strategy impl that produces samples changes.

- [ ] **Step 5: Verify compilation**

Run: `cargo check --features dqn`

- [ ] **Step 6: Commit**

```bash
git add src/dqn/train.rs
git commit -m "refactor: migrate RecordingDqn to two-phase Strategy"
```

---

### Task 11: Update main.rs, lib.rs, and examples

**Files:**
- Modify: `src/main.rs`
- Modify: `src/lib.rs`
- Modify: `examples/sigma_validation.rs`
- Modify: `examples/sigma_probe.rs`
- Modify: `examples/score_distribution.rs`

- [ ] **Step 1: Update `src/lib.rs`**

Add `Bot` to public exports:

```rust
pub use strategy::{Bot, Strategy};
```

Ensure `strategy::bot_impl` module is included.

- [ ] **Step 2: Update `src/main.rs`**

`make_strategy` returns `Box<dyn Strategy>`. Since `Bot` types get `Strategy` via blanket impl, this should work without changes. But verify:
- `DNA` implements `Bot` → gets `Strategy` → can be `Box<dyn Strategy>` ✓
- `DqnStrategy` implements `Bot` → gets `Strategy` → can be `Box<dyn Strategy>` ✓
- Other strategies implement `Strategy` directly ✓

There may be object safety issues. `Bot` has `evaluate_batch` with a default impl using `&self` — this should be object-safe. But `impl<T: Bot + Debug> Strategy for T` requires `T: Bot + Debug`, and we need `Box<dyn Strategy>`. Since the blanket impl converts Bot to Strategy, `Box::new(dna) as Box<dyn Strategy>` should work.

Check for any `observe_opponents` calls in main.rs that need removal.

- [ ] **Step 3: Update examples**

`sigma_validation.rs` has `ValidateDqn` which implements `Strategy`. Update it to implement the new 3-method trait. The validation logic (comparing DQN σ to MC rollouts) runs during active_phase1/phase2.

`sigma_probe.rs` and `score_distribution.rs` may have custom `Strategy` impls — update them similarly.

- [ ] **Step 4: Verify full compilation**

Run: `cargo check --lib --examples`

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs src/main.rs examples/
git commit -m "refactor: update main, lib, and examples for two-phase Strategy"
```

---

### Task 12: Web crate compatibility

**Files:**
- Modify: `web/src/lib.rs`

**Note:** The user said "not touching web." However, since the `Strategy` trait changed, the web crate won't compile. The minimal fix: update `WebGame` to call the new phase methods instead of `your_move`/`opponents_move`. No UI changes.

- [ ] **Step 1: Update bot turn logic in WebGame**

Where the web crate currently calls `strategy.your_move(state, dice)`, split into:
1. `strategy.active_phase1(state, opp_states, dice)` → apply mark if Some
2. Propagate locks
3. `strategy.active_phase2(state, opp_states, dice, has_marked)` → apply mark if Some, else strike if !has_marked

Where it calls `strategy.opponents_move(state, white_sum, locked)`, replace with:
1. `strategy.passive_phase1(state, opp_states, dice)` → apply mark if Some

The web crate currently has the full dice in scope for bot turns, so passing it to phase methods is straightforward.

- [ ] **Step 2: Remove `observe_opponents` calls**

- [ ] **Step 3: Verify WASM compilation**

Run: `cd web && cargo check --target wasm32-unknown-unknown`

- [ ] **Step 4: Commit**

```bash
git add web/src/lib.rs
git commit -m "refactor: minimal web crate update for two-phase Strategy compatibility"
```

---

### Task 13: Integration tests and edge-case tests

**Files:**
- Modify: `src/state.rs` (add tests)
- Create or modify: test files

- [ ] **Step 1: Test that Phase 1 locks deny Phase 2**

The key behavioral difference this refactor introduces:

```rust
#[test]
fn phase1_lock_denies_phase2() {
    // Setup: opponent has 5 marks in red, white_sum=12 (terminal for red)
    // Active player wanted to mark red+color in phase2
    // After phase1: opponent locks red → active can't mark red in phase2
    // This is the whole reason for the refactor.

    // Build opponent state with 5 marks in row 0 (red ascending)
    let mut opp = State::default();
    for n in 2..=6 { opp.apply_mark(Mark { row: 0, number: n }); }
    // opp can mark 12 → locks red

    let active = State::default();
    let dice = [6, 6, 6, 6, 6, 6]; // white_sum = 12

    // In the old code, active would move first and could mark red.
    // In the new code, opp marks red 12 in Phase 1 (locks it),
    // then active's Phase 2 can't use red.

    // ... test via Game::play with specific strategies and dice ...
}
```

- [ ] **Step 2: Test simultaneous Phase 1 (pre-phase1 state snapshot)**

Verify that all players see the same pre-phase1 states, not each other's Phase 1 results:

```rust
#[test]
fn phase1_sees_pre_phase1_state() {
    // Two passive players, both can mark white_sum.
    // After phase1, both should have marked independently.
    // Neither should see the other's phase1 mark.
}
```

- [ ] **Step 3: Test game-over during Phase 1**

```rust
#[test]
fn game_ends_in_phase1_three_locks() {
    // 1 row already locked. Two players each lock a row in Phase 1.
    // Total: 3 locked rows. Game ends. Phase 2 doesn't happen.
}
```

- [ ] **Step 4: Test strike when both phases skip**

```rust
#[test]
fn strike_when_both_phases_skip() {
    // Active player returns None from phase1 and None from phase2.
    // Should get a strike.
}
```

- [ ] **Step 5: Test no strike for passive skip**

```rust
#[test]
fn passive_skip_no_strike() {
    // Passive player returns None from phase1.
    // Should NOT get a strike.
}
```

- [ ] **Step 6: Run full test suite**

```bash
cargo test --lib
cargo test --examples
```

- [ ] **Step 7: Run benchmarks to verify correctness**

```bash
cargo run --release -- bench dqn ga -n 100000
cargo run --release -- bench ga opportunist -n 100000
```

Winrates will shift — document the new baselines.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "test: add two-phase game loop edge case tests"
```

---

## Execution notes

- **Compilation order**: Tasks 1-2 are additive (no breakage). Task 3 (blanket impl) requires Task 2. Task 4 (game loop) requires Task 2. Tasks 5-10 (migrations) require Tasks 2-4. Task 11 requires all migrations. Task 12 (web) can be done last.
- **The code won't compile between Tasks 4 and the completion of all migrations** (Tasks 5-10), because `Game::play()` calls new `Strategy` methods that old impls don't have yet. Migrate all strategies before attempting `cargo check`.
- **Meta-rules (smart lock/strike/pruning) are intentionally omitted.** They'll be re-added in a follow-up once the two-phase scaffolding is correct. This means benchmarks will regress compared to the current model. That's expected.
- **Web crate**: Task 12 is the minimal change to keep it compiling. No UI redesign.
- **DQN retraining**: Not in scope. The model still works (Bot::evaluate uses the same neural net), but winrate may shift because (a) the game plays differently now and (b) meta-rules are removed.
