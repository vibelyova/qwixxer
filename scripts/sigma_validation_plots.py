#!/usr/bin/env python3
"""Analyze /tmp/sigma_validation.csv from examples/sigma_validation.rs.

Produces three plots:
  1. Scatter: DQN μ vs empirical μ
  2. Scatter: DQN σ vs empirical σ
  3. Line: σ vs turn number (one series per game, DQN solid vs empirical dashed)

And prints correlation / spread stats to stdout.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import statistics
import sys

CSV_PATH = '/tmp/sigma_validation.csv'
OUT_DIR = '/tmp'


def parse_row(line):
    """The `move` column contains commas — split around the fixed-position
    numeric columns instead of using csv.DictReader.
    """
    parts = line.rstrip().split(',')
    return {
        'game': int(parts[0]),
        'turn': int(parts[1]),
        'decision': int(parts[2]),
        'cand_idx': int(parts[3]),
        'move': ','.join(parts[4:-5]),
        'dqn_mean': float(parts[-5]),
        'dqn_sigma': float(parts[-4]),
        'emp_mean': float(parts[-3]),
        'emp_sigma': float(parts[-2]),
        'n_rollouts': int(parts[-1]),
    }


def pearson(x, y):
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    den = (sum((x[i] - mx) ** 2 for i in range(n)) * sum((y[i] - my) ** 2 for i in range(n))) ** 0.5
    return num / den if den > 0 else 0.0


def main():
    rows = []
    with open(CSV_PATH) as f:
        next(f)  # skip header
        for line in f:
            rows.append(parse_row(line))

    n_decisions = len({(r['game'], r['turn']) for r in rows})
    print(f"candidates: {len(rows)}, decisions: {n_decisions}")

    # --- Stats ---
    dqn_mean = [r['dqn_mean'] for r in rows]
    emp_mean = [r['emp_mean'] for r in rows]
    dqn_sigma = [r['dqn_sigma'] for r in rows]
    emp_sigma = [r['emp_sigma'] for r in rows]

    print(f"\ndqn_sigma  median {statistics.median(dqn_sigma):.2f}  range [{min(dqn_sigma):.2f}, {max(dqn_sigma):.2f}]")
    print(f"emp_sigma  median {statistics.median(emp_sigma):.2f}  range [{min(emp_sigma):.2f}, {max(emp_sigma):.2f}]")
    print(f"\nPearson correlations:")
    print(f"  μ:  {pearson(dqn_mean, emp_mean):+.4f}")
    print(f"  σ:  {pearson(dqn_sigma, emp_sigma):+.4f}")

    # Within-decision σ coefficient of variation
    per_decision = defaultdict(list)
    for r in rows:
        per_decision[(r['game'], r['turn'])].append(r)

    dqn_cv = []
    emp_cv = []
    for cands in per_decision.values():
        if len(cands) < 2:
            continue
        dqn_ss = [c['dqn_sigma'] for c in cands]
        emp_ss = [c['emp_sigma'] for c in cands]
        if statistics.mean(dqn_ss) > 0:
            dqn_cv.append(statistics.stdev(dqn_ss) / statistics.mean(dqn_ss))
        if statistics.mean(emp_ss) > 0:
            emp_cv.append(statistics.stdev(emp_ss) / statistics.mean(emp_ss))

    print(f"\nWithin-decision σ coef of variation (std / mean across candidates):")
    print(f"  DQN:       median {statistics.median(dqn_cv):.2%}  max {max(dqn_cv):.2%}")
    print(f"  Empirical: median {statistics.median(emp_cv):.2%}  max {max(emp_cv):.2%}")

    # --- Scatter plots: μ and σ, DQN vs empirical ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    ax.scatter(emp_mean, dqn_mean, alpha=0.5, s=20)
    lo = min(min(emp_mean), min(dqn_mean))
    hi = max(max(emp_mean), max(dqn_mean))
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5, label='y = x')
    ax.set_xlabel('Empirical μ (MC rollouts)')
    ax.set_ylabel('DQN predicted μ')
    ax.set_title(f'Mean: DQN vs MC (corr = {pearson(dqn_mean, emp_mean):+.3f})')
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.scatter(emp_sigma, dqn_sigma, alpha=0.5, s=20, color='red')
    hi = max(max(emp_sigma), max(dqn_sigma))
    ax.plot([0, hi], [0, hi], 'k--', alpha=0.5, label='y = x')
    ax.set_xlabel('Empirical σ (MC rollouts)')
    ax.set_ylabel('DQN predicted σ')
    ax.set_title(f'σ: DQN vs MC (corr = {pearson(dqn_sigma, emp_sigma):+.3f})')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/sigma_validation.png', dpi=120)
    print(f'\nwrote {OUT_DIR}/sigma_validation.png')

    # --- σ vs turn number (one series per game) ---
    per_turn_stats = []
    for (game, turn), cands in per_decision.items():
        dqn_s = statistics.mean(c['dqn_sigma'] for c in cands)
        emp_s = statistics.mean(c['emp_sigma'] for c in cands)
        per_turn_stats.append((turn, dqn_s, emp_s, game))
    per_turn_stats.sort()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.get_cmap('tab10')
    games = sorted({p[3] for p in per_turn_stats})
    for gi, game in enumerate(games):
        gps = [p for p in per_turn_stats if p[3] == game]
        turns = [p[0] for p in gps]
        dqns = [p[1] for p in gps]
        emps = [p[2] for p in gps]
        c = colors(gi)
        ax.plot(turns, dqns, 'o-', color=c, alpha=0.8, label=f'game {game} DQN σ')
        ax.plot(turns, emps, 's--', color=c, alpha=0.5, label=f'game {game} empirical σ')

    ax.set_xlabel('turn number (DQN active turn)')
    ax.set_ylabel('σ (averaged across candidates at that decision)')
    ax.set_title('σ vs turn number: DQN (solid) vs empirical MC (dashed)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=len(games))
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/sigma_vs_turn.png', dpi=120)
    print(f'wrote {OUT_DIR}/sigma_vs_turn.png')


if __name__ == '__main__':
    main()
