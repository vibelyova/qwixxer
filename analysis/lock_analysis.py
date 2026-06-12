"""Adjudicated safe-lock events: verdict splits, conditioning, cost, examples.

Usage: lock_analysis.py <adjudicated.jsonl> [top_n]

UNITS: alt_gap / alt2_gap are WIN-PROBABILITY differences (full-game rollout
outcomes in [0,1], oriented alternative - lock). Positive = the lock rule is
wrong (the alternative wins more often). They are NOT points.
"""
import json
import math
import sys

import pandas as pd

from examples import fmt_mark, render_state

pd.set_option("display.width", 200)

# Effect-size floor: ignore lock_wrong verdicts whose win-prob gap is below
# this magnitude when reporting the "floored" count (statistically significant
# but practically negligible).
GAP_FLOOR = 0.02


def z_or_inf(gap, se, z):
    """alt_z is null in JSON when se==0 (inf not representable); reconstruct."""
    if z is not None:
        return z
    if se == 0:
        return math.copysign(math.inf, gap) if gap else 0.0
    return gap / se


def load_lock(path):
    rows, raw, games = [], [], {}
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            try:
                e = json.loads(line)
            except json.JSONDecodeError as err:
                sys.exit(f"{path}:{lineno}: bad JSON: {err}")
            if e["t"] == "g":
                games[e["game"]] = e
                continue
            if e["t"] != "l":
                continue
            try:
                our, opp = e["our"], e["opps"][0]
                z = z_or_inf(e["alt_gap_mean"], e["alt_gap_se"], e.get("alt_z"))
                r = {
                    "game": e["game"],
                    "turn": e["turn"],
                    "ctx": e["ctx"],
                    "our_points": e["our_points"],
                    "opp_points": e["opp_points"],
                    "cdiff": e["our_points"] - e["opp_points"],
                    "our_marks": sum(t for t, _ in our["rows"]),
                    "opp_marks": sum(t for t, _ in opp["rows"]),
                    "locks_on_board": sum(f is None for _, f in our["rows"])
                    + sum(f is None for _, f in opp["rows"]),
                    "our_strikes": our["strikes"],
                    "n_cands": len(e["cands"]),
                    "n_safe_locks": e["n_safe_locks"],
                    "rule_free_forced": e["rule_free_forced"],
                    "lock_row": e["lock_mark"][0],
                    "v_lock": e["cands"][e["lock_idx"]]["v"],
                    "v_alt": e["cands"][e["alt_idx"]]["v"],
                    "alt_is_defer": e["cands"][e["alt_idx"]]["mark"] is None,
                    "alt_gap": e["alt_gap_mean"],
                    "alt_se": e["alt_gap_se"],
                    "alt_z": z,
                    "alt_verdict": e["alt_verdict"],
                    "alt2_gap": e.get("alt2_gap_mean"),
                    "alt2_verdict": e.get("alt2_verdict"),
                }
            except KeyError as err:
                sys.exit(f"{path}:{lineno}: missing field {err}")
            rows.append(r)
            raw.append(e)
    return pd.DataFrame(rows), raw, games


def describe_gaps(label, sub):
    """Print the gap-magnitude (|alt_gap| win-prob) distribution for a subset."""
    if sub.empty:
        print(f"  {label}: (none)")
        return
    mag = sub.alt_gap.abs()
    print(f"  {label} (n={len(sub)}) |alt_gap| win-prob:")
    print(mag.describe().to_string().replace("\n", "  "))
    q = mag.quantile([0.5, 0.75, 0.9, 0.95, 0.99]).round(4)
    print("  quantiles:", q.to_dict())


def main(path, top_n=10):
    df, raw, games = load_lock(path)
    n_games = len(games)
    print(f"{len(df)} adjudicated lock firings from {n_games} games ({len(df) / n_games:.2f}/game)")
    print("ctx split:", df.ctx.value_counts().to_dict())

    print("\n== Verdict split (lock vs best non-lock; positive gap = rule wrong) ==")
    print(df.alt_verdict.value_counts().to_string())
    print("\n== ...by context ==")
    print(pd.crosstab(df.ctx, df.alt_verdict, margins=True))

    # --- Extension 2: effect-size floor on lock_wrong ---
    wrong = df[df.alt_verdict == "lock_wrong"]
    right = df[df.alt_verdict == "lock_right"]
    wrong_floored = wrong[wrong.alt_gap.abs() >= GAP_FLOOR]
    print("\n== lock_wrong, two ways ==")
    print(f"  raw z-verdict:            {len(wrong)}")
    print(f"  with |alt_gap| >= {GAP_FLOOR} wp: {len(wrong_floored)} "
          f"(dropped {len(wrong) - len(wrong_floored)} negligible-gap)")

    print("\n== Gap-magnitude distribution (|alt_gap| win-prob), by verdict ==")
    describe_gaps("lock_wrong", wrong)
    describe_gaps("lock_right", right)

    # --- Extension 3: deterministic-arm / low-variance diagnostics ---
    det = df[df.alt_se == 0]
    wrong_lowvar = wrong[wrong.alt_se < 0.005]
    print("\n== Determinism diagnostics ==")
    print(f"  events with alt_gap_se == 0 (both arms deterministic): {len(det)}")
    print(f"  lock_wrong with se < 0.005 (low-variance, treat separately): {len(wrong_lowvar)}")
    if len(wrong_lowvar):
        print("  low-variance lock_wrong events (game/turn/ctx/gap/se):")
        for _, r in wrong_lowvar.iterrows():
            print(f"    game {r.game} turn {r.turn} {r.ctx} "
                  f"gap {r.alt_gap:+.4f} se {r.alt_se:.5f}")

    print("\n== lock_wrong rate by score situation (cdiff bins) ==")
    df["cdiff_bin"] = pd.cut(df.cdiff, [-200, -10, -1, 0, 9, 200], labels=["<=-10", "-9..-1", "0", "1..9", ">=10"])
    print(df.groupby("cdiff_bin", observed=True).alt_verdict.value_counts(normalize=True).unstack(fill_value=0).to_string())
    print(df.groupby("cdiff_bin", observed=True).size().to_string())

    # --- Extension 4: game-shortening hypothesis cut (behind vs not-behind) ---
    print("\n== Game-shortening cut: cdiff<0 (behind) vs cdiff>=0 ==")
    for label, mask in [("cdiff<0  (behind)", df.cdiff < 0), ("cdiff>=0 (ahead/tied)", df.cdiff >= 0)]:
        sub = df[mask]
        sw = sub[sub.alt_verdict == "lock_wrong"]
        rate = len(sw) / len(sub) if len(sub) else float("nan")
        mean_pos = sub[sub.alt_gap > 0].alt_gap.mean()
        mean_pos = 0.0 if pd.isna(mean_pos) else mean_pos
        print(f"  {label}: n={len(sub)}  lock_wrong={len(sw)} ({rate:.1%})  "
              f"mean positive gap={mean_pos:+.4f} wp")

    print("\n== lock_wrong rate by stage / board ==")
    for col in ["locks_on_board", "n_safe_locks", "rule_free_forced", "alt_is_defer", "lock_row"]:
        sub = df.groupby(col).alt_verdict.value_counts(normalize=True).unstack(fill_value=0)
        sub["n"] = df.groupby(col).size()
        print(sub.to_string(), "\n")

    print(f"== Cost: {len(wrong)} lock_wrong events, "
          f"sum gap {wrong.alt_gap.sum():.3f} win-prob over {n_games} games "
          f"= {wrong.alt_gap.sum() / n_games:.5f} wp/game ==")

    if df.alt2_verdict.notna().any():
        print("\n== Runner-up lock (wrong-lock-chosen check; win-prob, alt2 - lock) ==")
        a2 = df[df.alt2_verdict.notna()]
        print(f"  {len(a2)} multi-lock events with a runner-up lock")
        print(a2.alt2_verdict.value_counts(dropna=True).to_string())

    print(f"\n== Top {top_n} lock_wrong positions by z ==")
    wrong_raw = [e for e in raw if e["alt_verdict"] == "lock_wrong"]
    wrong_raw.sort(key=lambda e: -z_or_inf(e["alt_gap_mean"], e["alt_gap_se"], e.get("alt_z")))
    for e in wrong_raw[:top_n]:
        print("=" * 78)
        print(
            f"game {e['game']} turn {e['turn']} ctx {e['ctx']} dice {e['dice']} "
            f"| gap {e['alt_gap_mean']:+.4f} wp z={z_or_inf(e['alt_gap_mean'], e['alt_gap_se'], e.get('alt_z')):+.1f}"
        )
        render_state("OUR", e["our"])
        render_state("OPP", e["opps"][0])
        phase = 1 if e["ctx"] in ("ap1", "pp1") else 2
        lk = e["cands"][e["lock_idx"]]
        alt = e["cands"][e["alt_idx"]]
        print(f"  forced lock: {fmt_mark(lk['mark'], phase, e['has_marked'])} (v={lk['v']:+.3f})")
        print(f"  better alt:  {fmt_mark(alt['mark'], phase, e['has_marked'])} (v={alt['v']:+.3f})")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10)
