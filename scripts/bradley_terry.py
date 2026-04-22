#!/usr/bin/env python3
"""
Bradley-Terry Model Solver
---------------------------
Input pairwise win rates between players and compute their
Bradley-Terry skill scores, normalized so the weakest player = 1.

Usage:
    python bradley_terry.py
"""

import sys
from itertools import combinations

try:
    import numpy as np
    from scipy.optimize import minimize
except ImportError:
    print("Required packages missing. Install with:")
    print("  pip install numpy scipy")
    sys.exit(1)


def parse_win_rate(s: str) -> float:
    s = s.strip()
    if "%" in s:
        return float(s.replace("%", "")) / 100
    return float(s)


def fit_bradley_terry(players: list[str], matchups: list[tuple]) -> dict[str, float]:
    """
    Fit Bradley-Terry model via maximum likelihood.
    matchups: list of (player_i, player_j, p_i_wins)
        where p_i_wins is the probability that player_i beats player_j.
    Returns dict of {player: score}, normalized so min score = 1.
    """
    n = len(players)
    idx = {p: i for i, p in enumerate(players)}

    # Log-likelihood: sum over matchups of [p*log(sig) + (1-p)*log(1-sig)]
    # where sig = exp(a_i) / (exp(a_i) + exp(a_j))
    # We optimize log-scores (unconstrained), fix last player = 0 for identifiability
    def neg_log_likelihood(log_scores):
        ll = 0.0
        for pi, pj, p_win in matchups:
            ai = log_scores[idx[pi]]
            aj = log_scores[idx[pj]]
            # log P(i beats j) = ai - log(exp(ai) + exp(aj))
            log_p = ai - np.logaddexp(ai, aj)
            log_q = aj - np.logaddexp(ai, aj)
            ll += p_win * log_p + (1 - p_win) * log_q
        return -ll

    x0 = np.zeros(n)
    # Fix one player as anchor to remove scale ambiguity (soft constraint)
    result = minimize(neg_log_likelihood, x0, method="L-BFGS-B",
                      options={"maxiter": 10000, "ftol": 1e-12})

    log_scores = result.x
    scores = np.exp(log_scores)

    # Normalize: weakest player = 1
    scores /= scores.min()

    return {p: float(scores[idx[p]]) for p in players}


def get_players() -> list[str]:
    print("\n=== Bradley-Terry Model Solver ===\n")
    raw = input("Enter player names (comma-separated): ")
    players = [p.strip() for p in raw.split(",") if p.strip()]
    if len(players) < 2:
        print("Need at least 2 players.")
        sys.exit(1)
    return players


def get_matchups(players: list[str]) -> list[tuple]:
    n = len(players)
    pairs = list(combinations(players, 2))
    print(f"\nEnter win rates for {len(pairs)} matchup(s).")
    print("Format: probability/percentage that the FIRST player beats the second.")
    print("(e.g. '0.6' or '60%' or '60'  — press Enter to skip a matchup)\n")

    matchups = []
    for a, b in pairs:
        while True:
            raw = input(f"  P({a} beats {b}): ").strip()
            if raw == "":
                print(f"    Skipped.")
                break
            try:
                p = parse_win_rate(raw)
                if not (0 < p < 1):
                    print("    Must be strictly between 0 and 1.")
                    continue
                matchups.append((a, b, p))
                break
            except ValueError:
                print("    Invalid input. Try again.")

    if len(matchups) < len(players) - 1:
        print("\nWarning: not enough matchups to fully constrain the model.")
        print("Some players may not be comparable. Results may be unreliable.\n")

    return matchups


def print_results(scores: dict[str, float]):
    sorted_players = sorted(scores.items(), key=lambda x: -x[1])
    max_name = max(len(p) for p in scores)

    print("\n=== Bradley-Terry Scores (baseline = 1.0) ===\n")
    print(f"  {'Rank':<6} {'Player':<{max_name + 2}} {'Score':>8}   {'Win% vs baseline':>16}")
    print("  " + "-" * (max_name + 38))

    baseline_score = min(scores.values())
    for rank, (player, score) in enumerate(sorted_players, 1):
        win_vs_baseline = score / (score + baseline_score) * 100
        print(f"  {rank:<6} {player:<{max_name + 2}} {score:>8.4f}   {win_vs_baseline:>15.1f}%")

    print()
    print("=== Predicted Win Rates (%) ===\n")
    players = [p for p, _ in sorted_players]
    header = " " * (max_name + 2) + "  " + "  ".join(f"{p:>8}" for p in players)
    print("  " + header)
    for pi in players:
        row = f"  {pi:<{max_name + 2}}"
        for pj in players:
            if pi == pj:
                row += "        --"
            else:
                si, sj = scores[pi], scores[pj]
                pct = si / (si + sj) * 100
                row += f"  {pct:>7.1f}%"
        print(row)
    print()


def main():
    players = get_players()
    matchups = get_matchups(players)

    if not matchups:
        print("No matchups entered. Exiting.")
        sys.exit(1)

    print("\nFitting Bradley-Terry model...")
    scores = fit_bradley_terry(players, matchups)
    print_results(scores)


if __name__ == "__main__":
    main()
