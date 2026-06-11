"""Print the most confident disagreements as ASCII boards for eyeballing.

Usage: examples.py <relabeled.jsonl> [top_n]
"""
import json
import math
import sys

COLORS = ["R", "Y", "G", "B"]


def render_row(i, total, free):
    asc = i < 2
    nums = range(2, 13) if asc else range(12, 1, -1)
    if free is None:
        cells = " ".join("##" for _ in nums)
        return f"  {COLORS[i]} [{cells}]  marks={total} LOCKED"
    open_ = (lambda n: n >= free) if asc else (lambda n: n <= free)
    cells = " ".join(f"{n:>2}" if open_(n) else " ." for n in nums)
    return f"  {COLORS[i]} [{cells}]  marks={total} free={free}"


def render_state(label, s):
    print(f"{label}: strikes={s['strikes']}")
    for i, (t, f) in enumerate(s["rows"]):
        print(render_row(i, t, f))


def fmt_mark(m, phase, has_marked):
    if m is None:
        return "skip" if phase == 1 or has_marked else "STRIKE"
    return f"{COLORS[m[0]]}{m[1]}"


def z_of(e):
    if e["hk_gap_se"] > 0:
        return e["hk_gap_mean"] / e["hk_gap_se"]
    return math.copysign(math.inf, e["hk_gap_mean"]) if e["hk_gap_mean"] else 0.0


def main(path, top_n=15):
    evs = []
    with open(path) as fh:
        for line in fh:
            e = json.loads(line)
            if e["t"] == "d" and e.get("verdict") and e["disagree"]:
                evs.append(e)
    evs.sort(key=lambda e: -abs(z_of(e)))
    for e in evs[:top_n]:
        print("=" * 78)
        print(
            f"game {e['game']} turn {e['turn']} phase {e['phase']} dice {e['dice']} "
            f"| verdict {e['verdict']} z={z_of(e):+.1f} hk_gap={e['hk_gap_mean']:+.4f}"
        )
        render_state("OUR", e["our"])
        render_state("OPP", e["opps"][0])
        st = e["cands"][e["static_pick"]]
        print(f"  static: {fmt_mark(st['mark'], e['phase'], e['has_marked'])} (v={st['v']:+.3f})")
        print(f"  search: {fmt_mark(e['search_mark'], e['phase'], e['has_marked'])}")
        print(f"  cands: {[(fmt_mark(c['mark'], e['phase'], e['has_marked']), round(c['v'], 3)) for c in e['cands'][:5]]}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 15)
