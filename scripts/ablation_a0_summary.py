"""Summarize an ablation_a0*.json results file: win rate + Wilson 95% CI per arm."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS_PATH = REPO / "reports" / "ablation_a0.json"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glicko1 import wilson_ci  # noqa: E402


def aggregate(entries):
    """Sum multiple runs of the same (arm, opponent) into one row."""
    grouped = {}
    for e in entries:
        key = (e["arm"], e["opponent"])
        g = grouped.setdefault(key, {"arm": e["arm"], "opponent": e["opponent"],
                                      "n_games": 0, "a0_wins": 0, "opp_wins": 0,
                                      "draws": 0, "elapsed_sec": 0.0})
        g["n_games"] += e["n_games"]
        g["a0_wins"] += e["a0_wins"]
        g["opp_wins"] += e["opp_wins"]
        g["draws"] += e["draws"]
        g["elapsed_sec"] += e["elapsed_sec"]
    return list(grouped.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-path", default=str(RESULTS_PATH))
    args = ap.parse_args()
    raw = json.loads(Path(args.results_path).read_text())
    entries = aggregate(raw)
    print(f"{'arm':<16} {'opp':<28} {'games':>6} {'wins':>5} {'losses':>7} {'draws':>6} "
          f"{'win%':>7} {'95% CI':>16} {'s/game':>8}")
    for e in entries:
        n_decided = e["n_games"] - e["draws"]
        p, lo, hi = wilson_ci(e["a0_wins"], e["n_games"])
        print(f"{e['arm']:<16} {e['opponent']:<28} {e['n_games']:>6} {e['a0_wins']:>5} "
              f"{e['opp_wins']:>7} {e['draws']:>6} {p*100:>6.1f}% "
              f"[{lo*100:>5.1f},{hi*100:>5.1f}]  {e['elapsed_sec']/e['n_games']:>7.1f}s")

    print("\nPairwise overlap check (non-overlapping CIs = distinguishable at this N):")
    for i, e1 in enumerate(entries):
        for e2 in entries[i + 1:]:
            _, lo1, hi1 = wilson_ci(e1["a0_wins"], e1["n_games"])
            _, lo2, hi2 = wilson_ci(e2["a0_wins"], e2["n_games"])
            overlap = not (hi1 < lo2 or hi2 < lo1)
            verdict = "tied at this sample size" if overlap else "distinguishable"
            print(f"  {e1['arm']:<16} vs {e2['arm']:<16}: {verdict}")


if __name__ == "__main__":
    main()
