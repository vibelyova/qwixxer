"""Load divergence JSONL into a tidy DataFrame with engineered features.

Usage: load.py <events.jsonl>   (prints a summary; import load() elsewhere)
"""
import json
import sys

import pandas as pd

ROW_ASC = (True, True, False, False)
TERMINAL = (12, 12, 2, 2)


def tri(t):
    return t * (t + 1) // 2


def points(state):
    return sum(tri(t) for t, _ in state["rows"]) - 5 * state["strikes"]


def move_feats(prefix, mark, state, phase, has_marked):
    """Features of one candidate move from the pre-move state.

    Numeric move features (jump, to_terminal, row_total) are NaN for non-mark
    moves; kind carries that information.
    """
    if mark is None:
        kind = "skip" if phase == 1 or has_marked else "strike"
        return {
            f"{prefix}_kind": kind,
            f"{prefix}_row": -1,
            f"{prefix}_jump": float("nan"),
            f"{prefix}_points": -5 if kind == "strike" else 0,
            f"{prefix}_locks": False,
            f"{prefix}_to_terminal": float("nan"),
            f"{prefix}_row_total": float("nan"),
        }
    row, number = mark
    total, free = state["rows"][row]
    asc = ROW_ASC[row]
    jump = (number - free) if asc else (free - number)  # numbers skipped over
    locks = number == TERMINAL[row]
    return {
        f"{prefix}_kind": "mark",
        f"{prefix}_row": row,
        f"{prefix}_jump": jump,
        f"{prefix}_points": (total + 2) if locks else (total + 1),
        f"{prefix}_locks": locks,
        f"{prefix}_to_terminal": (TERMINAL[row] - number) if asc else (number - TERMINAL[row]),
        f"{prefix}_row_total": total,
    }


def load(path):
    decisions, games = [], {}
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                sys.exit(f"{path}:{lineno}: bad JSON: {e}")
            if obj["t"] == "g":
                games[obj["game"]] = obj
            else:
                decisions.append((lineno, obj))
    rows = []
    for lineno, ev in decisions:
        try:
            our, opp = ev["our"], ev["opps"][0]
            static_mark = ev["cands"][ev["static_pick"]]["mark"]
            r = {
                "game": ev["game"],
                "turn": ev["turn"],
                "phase": ev["phase"],
                "has_marked": ev["has_marked"],
                "gate_close": ev["gate_close"],
                "gate_endgame": ev["gate_endgame"],
                "static_gap": ev["static_gap"],
                "n_cands": len(ev["cands"]),
                "disagree": ev["disagree"],
                "our_points": ev["our_points"],
                "opp_points": ev["opp_points"],
                "cdiff": ev["our_points"] - ev["opp_points"],
                "our_strikes": our["strikes"],
                "opp_strikes": opp["strikes"],
                "our_marks": sum(t for t, _ in our["rows"]),
                "opp_marks": sum(t for t, _ in opp["rows"]),
                "our_locked": sum(f is None for _, f in our["rows"]),
                "opp_locked": sum(f is None for _, f in opp["rows"]),
                "v_static": ev["cands"][ev["static_pick"]]["v"],
                "v_search": ev["cands"][ev["search_pick"]]["v"],
                "static_pick": ev["static_pick"],
                "search_pick": ev["search_pick"],
                "hk_gap_mean": ev.get("hk_gap_mean"),
                "hk_gap_se": ev.get("hk_gap_se"),
                "verdict": ev.get("verdict"),
            }
            r.update(move_feats("static", static_mark, our, ev["phase"], ev["has_marked"]))
            r.update(move_feats("search", ev["search_mark"], our, ev["phase"], ev["has_marked"]))
            g = games.get(ev["game"])
            r["game_won"] = g["pair_won"] if g else None
            rows.append(r)
        except KeyError as e:
            sys.exit(f"{path}:{lineno}: missing field {e}")
    df = pd.DataFrame(rows)
    # search_right: relabeled disagreement where high-K confirms search's side.
    # verdict 'flip' favors cands[1], 'keep' favors cands[0].
    df["search_right"] = df["disagree"] & (
        ((df["verdict"] == "flip") & (df["search_pick"] == 1))
        | ((df["verdict"] == "keep") & (df["search_pick"] == 0))
    )
    return df


if __name__ == "__main__":
    df = load(sys.argv[1])
    print(f"{len(df)} events from {df.game.nunique()} games")
    print(f"disagree: {df.disagree.mean():.2%}")
    print("verdicts among disagreements:")
    print(df[df.disagree].verdict.value_counts(dropna=False))
    print(f"search_right: {df.search_right.sum()}")
