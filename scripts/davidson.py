#!/usr/bin/env python3
"""
Davidson Model Solver
----------------------
Extension of Bradley-Terry that handles ties natively.
Input pairwise win/tie rates and compute skill scores + tie parameter ν.

For each matchup (i vs j) with scores λᵢ, λⱼ and global tie parameter ν:
    P(i wins) = λᵢ / (λᵢ + λⱼ + ν√(λᵢλⱼ))
    P(tie)    = ν√(λᵢλⱼ) / (λᵢ + λⱼ + ν√(λᵢλⱼ))
    P(j wins) = λⱼ / (λᵢ + λⱼ + ν√(λᵢλⱼ))

Scores are normalized so the weakest player = 1.

Usage:
    python davidson.py
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


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_rate(s: str) -> float:
    s = s.strip()
    if "%" in s:
        v = float(s.replace("%", "")) / 100
    else:
        v = float(s)
        if v > 1:          # treat bare numbers like 60 as percentages
            v /= 100
    return v


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def davidson_probs(li: float, lj: float, nu: float) -> tuple[float, float, float]:
    """Return (P(i wins), P(tie), P(j wins)) under the Davidson model."""
    denom = li + lj + nu * np.sqrt(li * lj)
    return li / denom, nu * np.sqrt(li * lj) / denom, lj / denom


def fit_davidson(
    players: list[str],
    matchups: list[tuple],   # (pi, pj, w_win, w_tie, w_loss)  — fractional weights summing to 1
) -> tuple[dict[str, float], float]:
    """
    Fit Davidson model via maximum likelihood.

    matchups entries: (player_i, player_j, p_win, p_tie, p_loss)
        p_win  = observed fraction of games where i won
        p_tie  = observed fraction of ties
        p_loss = observed fraction of games where j won
        (p_win + p_tie + p_loss == 1)

    Returns:
        scores : dict {player: score}, normalized so min = 1
        nu     : fitted tie parameter (ν ≥ 0)
    """
    n = len(players)
    idx = {p: i for i, p in enumerate(players)}
    has_ties = any(t > 0 for _, _, _, t, _ in matchups)

    def neg_log_likelihood(params):
        log_lambdas = params[:n]
        log_nu = params[n]

        lambdas = np.exp(log_lambdas)
        nu = np.exp(log_nu)

        ll = 0.0
        for pi, pj, p_win, p_tie, p_loss in matchups:
            li = lambdas[idx[pi]]
            lj = lambdas[idx[pj]]
            denom = li + lj + nu * np.sqrt(li * lj)

            log_denom = np.log(denom)
            log_li = np.log(li)
            log_lj = np.log(lj)
            log_nu_term = np.log(nu) + 0.5 * (log_li + log_lj)

            if p_win > 0:
                ll += p_win * (log_li - log_denom)
            if p_tie > 0:
                ll += p_tie * (log_nu_term - log_denom)
            if p_loss > 0:
                ll += p_loss * (log_lj - log_denom)

        return -ll

    # Initial params: all log-scores = 0, log_nu = 0 (nu=1)
    x0 = np.zeros(n + 1)

    result = minimize(
        neg_log_likelihood, x0,
        method="L-BFGS-B",
        options={"maxiter": 20000, "ftol": 1e-14, "gtol": 1e-10},
    )

    log_lambdas = result.x[:n]
    log_nu = result.x[n]

    lambdas = np.exp(log_lambdas)
    nu = float(np.exp(log_nu))

    # Normalize: weakest player = 1
    lambdas /= lambdas.min()

    scores = {p: float(lambdas[idx[p]]) for p in players}

    if not has_ties:
        # nu is unidentified without tie data — report as N/A
        nu = None

    return scores, nu


# ---------------------------------------------------------------------------
# Input collection
# ---------------------------------------------------------------------------

def get_players() -> list[str]:
    print("\n=== Davidson Model Solver (Bradley-Terry + Ties) ===\n")
    raw = input("Enter player names (comma-separated): ")
    players = [p.strip() for p in raw.split(",") if p.strip()]
    if len(players) < 2:
        print("Need at least 2 players.")
        sys.exit(1)
    return players


def get_matchups(players: list[str]) -> list[tuple]:
    pairs = list(combinations(players, 2))
    print(f"\nEnter results for {len(pairs)} matchup(s).")
    print("For each pair provide win%, tie%, and loss% (must sum to 100).")
    print("You can omit tie% — it defaults to 0 (pure Bradley-Terry).")
    print("Press Enter to skip a matchup entirely.\n")

    matchups = []
    for a, b in pairs:
        print(f"  {a}  vs  {b}")
        while True:
            raw_win = input(f"    P({a} wins): ").strip()
            if raw_win == "":
                print("    Skipped.\n")
                break
            try:
                p_win = parse_rate(raw_win)
            except ValueError:
                print("    Invalid. Try again.")
                continue

            raw_tie = input(f"    P(tie)     [default 0]: ").strip()
            p_tie = 0.0
            if raw_tie:
                try:
                    p_tie = parse_rate(raw_tie)
                except ValueError:
                    print("    Invalid tie rate, defaulting to 0.")

            p_loss = 1.0 - p_win - p_tie
            if p_loss < -1e-9:
                print(f"    Win% + Tie% > 100% ({(p_win+p_tie)*100:.1f}%). Try again.")
                continue
            p_loss = max(p_loss, 0.0)

            print(f"    → {a} wins {p_win*100:.1f}%  |  tie {p_tie*100:.1f}%  |  {b} wins {p_loss*100:.1f}%")
            matchups.append((a, b, p_win, p_tie, p_loss))
            print()
            break

    if len(matchups) < len(players) - 1:
        print("Warning: not enough matchups to fully constrain the model.")
        print("Results may be unreliable.\n")

    return matchups


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_results(players: list[str], scores: dict[str, float], nu: float | None):
    sorted_players = sorted(scores.items(), key=lambda x: -x[1])
    max_name = max(len(p) for p in scores)
    W = max_name + 2

    print("\n=== Davidson Scores (baseline = 1.0) ===\n")
    if nu is not None:
        print(f"  Tie parameter ν = {nu:.4f}  "
              f"({'low tie tendency' if nu < 1 else 'high tie tendency'})\n")
    else:
        print("  Tie parameter ν = N/A (no ties observed — reduces to Bradley-Terry)\n")

    baseline = min(scores.values())
    print(f"  {'Rank':<6} {'Player':<{W}} {'Score':>8}   {'Win% vs baseline':>16}")
    print("  " + "-" * (W + 38))
    for rank, (player, score) in enumerate(sorted_players, 1):
        p_win, _, _ = davidson_probs(score, baseline, nu if nu is not None else 0.0)
        print(f"  {rank:<6} {player:<{W}} {score:>8.4f}   {p_win*100:>15.1f}%")

    print()
    _print_matrix("Predicted Win %", sorted_players, scores, nu, "win")
    if nu is not None and nu > 0.01:
        _print_matrix("Predicted Tie %", sorted_players, scores, nu, "tie")


def _print_matrix(title: str, sorted_players, scores, nu, which: str):
    max_name = max(len(p) for p, _ in sorted_players)
    W = max_name + 2
    players = [p for p, _ in sorted_players]
    col_w = max(max(len(p) for p in players), 7)

    print(f"=== {title} ===\n")
    header = " " * (W + 2) + "  ".join(f"{p:>{col_w}}" for p in players)
    print("  " + header)
    for pi in players:
        row = f"  {pi:<{W}}"
        for pj in players:
            if pi == pj:
                row += " " * (col_w - 1) + "--  "
            else:
                si, sj = scores[pi], scores[pj]
                _nu = nu if nu is not None else 0.0
                p_win, p_tie, p_loss = davidson_probs(si, sj, _nu)
                val = p_win if which == "win" else p_tie
                row += f"  {val*100:>{col_w-1}.1f}%"
        print(row)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    players = get_players()
    matchups = get_matchups(players)

    if not matchups:
        print("No matchups entered. Exiting.")
        sys.exit(1)

    print("Fitting Davidson model...")
    scores, nu = fit_davidson(players, matchups)
    print_results(players, scores, nu)


if __name__ == "__main__":
    main()
