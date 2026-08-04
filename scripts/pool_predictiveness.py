"""Does the local benchmark pool predict the real ladder? Measure it, don't assert it.

Correlates local Glicko ratings (from a benchmark run's result JSON) against
real Kaggle-ladder readings for every agent that has both. This is the running
answer to the leaderboard-check skill's question -- two confirmed local-vs-ladder
inversions (2026-08-02, 2026-08-03) are why this number exists.

Ladder anchor table caveats (all matter, stated so the rho isn't over-read):
- Readings span DIFFERENT dates/eras of a live ladder (field grew ~4.5k -> 6.2k
  teams July->August; Elo scale drifts). Same-build spread is ~ +/-100.
- wmh_* values are self-reported in github.com/wmh/ptcg-abc docs (July readings).
- tb_* values are TomBombadyl's decoded-submission catalog mu.
- Ours are read from reports/submission_ledger.jsonl history.
A high rho here is necessary, not sufficient, for trusting local ordering.

Usage:
  uv run python scripts/pool_predictiveness.py [--result reports/wmh_pool_calibration.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# agent -> (ladder score, provenance)
LADDER_ANCHORS: dict[str, tuple[float, str]] = {
    "tb_archaludon": (1196.1, "TomBombadyl catalog mu (his leader)"),
    "romanrozen_strong_start": (950.0, "notebook title 'LB 960'/~950 claim"),
    "makthanithin_1084_baseline": (1084.5, "notebook's scored-target claim"),
    "wmh_megastarmie": (871.5, "wmh docs: megastarmie v3, 2026-06-24"),
    "wmh_alakazam": (860.3, "wmh docs: Alakazam v3 new-high, 2026-07-08"),
    "wmh_bellibolt": (836.0, "wmh README ladder Elo"),
    "wmh_garchomp": (713.8, "wmh docs: garchomp first run, 2026-07-08"),
    "wmh_typhlosion": (532.0, "wmh README ladder Elo"),
    "tb_dragapult": (880.9, "TomBombadyl catalog mu"),
    "tb_alakazam": (659.0, "TomBombadyl catalog mu"),
    "tb_starmie": (277.5, "TomBombadyl catalog mu"),
    "tb_search": (660.5, "TomBombadyl catalog mu"),
    "tb_heuristic": (633.0, "TomBombadyl catalog mu"),
    "tb_rulecore": (535.6, "TomBombadyl catalog mu"),
    "improved_prob_main": (701.6, "our sub 55169814, 2026-08-01"),
    "agent_core_improved": (804.0, "our sub 55162376 best-ever; recent reads 643-693"),
    "il_agent": (398.7, "our subs 55190924/55149903 mean"),
    "s2_e1_s43": (395.0, "our sub 55196434"),
    "ppo_u120832": (275.1, "our sub 55215267 (drifting read)"),
}


def spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):  # mean rank for ties
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            mean = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = mean
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy) if vx and vy else float("nan")


def permutation_p(xs: list[float], ys: list[float], observed: float, n_perm: int = 20000) -> float:
    import random

    rng = random.Random(42)
    ys2 = list(ys)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(ys2)
        if abs(spearman(xs, ys2)) >= abs(observed):
            hits += 1
    return (hits + 1) / (n_perm + 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--result", type=Path,
                    default=REPO / "reports" / "wmh_pool_calibration.json",
                    help="benchmark result JSON (needs a 'glicko' block)")
    args = ap.parse_args()

    res = json.loads(args.result.read_text())
    glicko = res["glicko"]

    rows = []
    for name, g in glicko.items():
        if name in LADDER_ANCHORS:
            ladder, src = LADDER_ANCHORS[name]
            rows.append((name, g["rating"], g["rd"], ladder, src))
    rows.sort(key=lambda r: -r[3])

    print(f"{len(rows)} agents with BOTH a local Glicko (this run) and a ladder reading:\n")
    print(f"{'agent':32s} {'local glicko':>14s} {'RD':>6s} {'ladder':>8s}  provenance")
    for name, rating, rd, ladder, src in rows:
        print(f"{name:32s} {rating:>10.1f} {'+/-':>3s}{rd:>4.0f} {ladder:>8.1f}  {src}")

    if len(rows) < 4:
        print("\nToo few overlapping agents for a correlation; wire more ladder-scored anchors.")
        return

    local = [r[1] for r in rows]
    ladder = [r[3] for r in rows]
    rho = spearman(local, ladder)
    p = permutation_p(local, ladder, rho)
    print(f"\nSpearman rho (local Glicko vs ladder) = {rho:+.3f}  "
          f"(n={len(rows)}, permutation p={p:.4f})")
    print("Read with the header caveats: mixed-era ladder readings, ~+/-100 same-build drift.")

    # The two most decision-relevant slices: does the pool order OUR agents right?
    ours = [r for r in rows if r[0] in ("improved_prob_main", "agent_core_improved",
                                        "il_agent", "s2_e1_s43", "ppo_u120832")]
    if len(ours) >= 3:
        rho_ours = spearman([r[1] for r in ours], [r[3] for r in ours])
        print(f"Ours-only rho = {rho_ours:+.3f} (n={len(ours)}) -- the slice the "
              f"2026-08-02/03 inversions live in.")


if __name__ == "__main__":
    main()
