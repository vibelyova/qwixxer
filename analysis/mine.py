"""Cross-tabs and decision-tree mining over relabeled divergence events.

Usage: mine.py <relabeled.jsonl>
"""
import sys

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

from load import load

pd.set_option("display.width", 200)


def main(path):
    df = load(path)
    print(
        f"{len(df)} events, {df.disagree.sum()} disagreements ({df.disagree.mean():.2%}), "
        f"{df.search_right.sum()} confirmed search_right"
    )

    print("\n== Disagreement rate by single features ==")
    for col in [
        "phase", "gate_close", "gate_endgame", "our_strikes", "opp_strikes",
        "our_locked", "opp_locked", "static_kind",
    ]:
        print(df.groupby(col)["disagree"].agg(["mean", "count"]).to_string(), "\n")

    print("== Move-kind transition matrix: all disagreements ==")
    d = df[df.disagree]
    print(pd.crosstab(d.static_kind, d.search_kind, margins=True))
    print("\n== ...confirmed (search_right) only ==")
    c = df[df.search_right]
    print(pd.crosstab(c.static_kind, c.search_kind, margins=True))
    mm = c[(c.static_kind == "mark") & (c.search_kind == "mark")]
    print("\n== Row transition among confirmed mark->mark ==")
    print(pd.crosstab(mm.static_row, mm.search_row, margins=True))
    print("\n== Jump-size shift among confirmed mark->mark ==")
    print((mm.search_jump - mm.static_jump).describe())

    # Tree over every event that has a verdict (relabeled agreements act as
    # controls; refuted/coinflip disagreements as hard negatives).
    lab = df[df.verdict.notna()].copy()
    feats = [
        "phase", "turn", "static_gap", "cdiff", "our_strikes", "opp_strikes",
        "our_marks", "opp_marks", "our_locked", "opp_locked", "n_cands",
        "static_jump", "static_points", "static_row_total", "static_to_terminal",
        "search_jump", "search_points", "search_row_total", "search_to_terminal",
    ]
    X = pd.get_dummies(lab[feats + ["static_kind", "search_kind"]], columns=["static_kind", "search_kind"])
    y = lab["search_right"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    tree = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=0)
    tree.fit(Xtr, ytr)
    print(f"\n== Decision tree (held-out acc {tree.score(Xte, yte):.3f}, base rate {1 - yte.mean():.3f}) ==")
    print(export_text(tree, feature_names=list(X.columns)))
    imp = permutation_importance(tree, Xte, yte, n_repeats=10, random_state=0)
    print("== Permutation importance (top 10) ==")
    for v, name in sorted(zip(imp.importances_mean, X.columns), reverse=True)[:10]:
        print(f"  {name:<30} {v:.4f}")


if __name__ == "__main__":
    main(sys.argv[1])
