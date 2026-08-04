"""Merge per-arm benchmark JSONs into one local leaderboard.

Each sweep process benchmarks ONE arm against the shared opponent pool, so its
own Glicko block starts from a fresh 1500 prior and is NOT comparable across
processes. This script pools every game from every run and recomputes Glicko
once, which puts all arms on a common scale: they never play each other, but
they all play the same 8 opponents, and Glicko propagates strength through
those shared results.

Reports, per arm:
  * win% +/- sigma vs each opponent (sigma = sqrt(p(1-p)/n), the same
    estimator the earlier deck study used, so numbers stay comparable)
  * aggregate win% vs the FIELD (self-play and arm-vs-arm excluded)
  * Wilson 95% CI on the aggregate
  * pooled Glicko rating with RD

Also reports pairwise separation between arms: two arms are only called
different when their aggregate 95% CIs do not overlap.

Usage:
  uv run python scripts/merge_il_sweep.py --dirs reports/il_sweep/stageA \
      --out reports/il_sweep/stageA_merged.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO / "scripts"))
import glicko1  # noqa: E402


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, c - m), 100 * min(1.0, c + m))


def sigma(wins: int, n: int) -> float:
    if n == 0:
        return 0.0
    p = wins / n
    return 100 * math.sqrt(p * (1 - p) / n)


def is_arm(name: str) -> bool:
    return "@" in name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    files: list[Path] = []
    for d in args.dirs:
        files.extend(sorted(Path(d).glob("*.json")))
    if not files:
        print("no result JSONs found")
        return 1

    # pooled[a][b] = [wins_by_a, games]
    pooled: dict[str, dict[str, list[int]]] = {}
    sources = []
    for f in files:
        data = json.loads(f.read_text())
        sources.append(f.name)
        for a, row in data["wins"].items():
            for b, w in row.items():
                if a == b:
                    continue  # self-play: excluded from rating periods
                g = data["games"][a][b]
                cell = pooled.setdefault(a, {}).setdefault(b, [0, 0])
                cell[0] += w
                cell[1] += g

    # --- pooled Glicko over the union of all games -------------------------
    # One rating period over every game from every run, via the repo's own
    # Glicko-1 implementation. Ratings start from a flat 1500 prior (NOT the
    # standing reports/glicko_ratings.json) -- these arms are synthetic
    # identities that have never played on the ladder, so inheriting history
    # would import strength they never earned.
    names = sorted(pooled)
    glicko_games: list[tuple[str, str, float]] = []
    seen: set[tuple[str, str]] = set()
    for a in names:
        for b, (w, g) in pooled[a].items():
            if g == 0 or (b, a) in seen:
                continue
            seen.add((a, b))
            w_b = pooled.get(b, {}).get(a, [0, 0])[0]
            draws = g - w - w_b
            glicko_games.extend([(a, b, 1.0)] * w)
            glicko_games.extend([(a, b, 0.0)] * w_b)
            if draws > 0:
                glicko_games.extend([(a, b, 0.5)] * draws)
    new_ratings = glicko1.run_rating_period({}, glicko_games)

    arms = [n for n in names if is_arm(n)]
    field = [n for n in names if not is_arm(n)]

    out: dict = {
        "sources": sources,
        "pool": field,
        "arms": arms,
        "per_opponent": {},
        "aggregate": {},
        "glicko": {
            n: {"rating": r.rating, "rd": r.rd, "gxe": glicko1.gxe(r)}
            for n, r in new_ratings.items()
        },
    }

    for a in arms:
        row = {}
        tw = tg = 0
        for b in field:
            w, g = pooled[a].get(b, [0, 0])
            if g == 0:
                continue
            row[b] = {
                "wins": w, "games": g,
                "win_pct": 100 * w / g,
                "sigma": sigma(w, g),
            }
            tw += w
            tg += g
        out["per_opponent"][a] = row
        lo, hi = wilson(tw, tg)
        out["aggregate"][a] = {
            "wins": tw, "games": tg,
            "win_pct": 100 * tw / tg if tg else 0.0,
            "sigma": sigma(tw, tg),
            "wilson95": [lo, hi],
            "glicko": new_ratings[a].rating,
            "glicko_rd": new_ratings[a].rd,
        }

    ranked = sorted(arms, key=lambda a: -out["aggregate"][a]["win_pct"])

    # --- separation: only call two arms different on non-overlapping CIs ----
    seps = []
    for i, a in enumerate(ranked):
        for b in ranked[i + 1:]:
            A, Bg = out["aggregate"][a], out["aggregate"][b]
            overlap = A["wilson95"][0] <= Bg["wilson95"][1] and Bg["wilson95"][0] <= A["wilson95"][1]
            seps.append({
                "a": a, "b": b,
                "gap_pp": A["win_pct"] - Bg["win_pct"],
                "separated": not overlap,
            })
    out["separation"] = seps

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))

    # --- console report ----------------------------------------------------
    print(f"merged {len(sources)} run(s); pool = {', '.join(field)}\n")
    print(f"{'arm':38s} {'field win%':>18s} {'wilson95':>16s} {'glicko':>16s}  games")
    print("-" * 100)
    for a in ranked:
        r = out["aggregate"][a]
        print(f"{a:38s} {r['win_pct']:8.1f} +/-{r['sigma']:4.1f}%  "
              f"[{r['wilson95'][0]:5.1f},{r['wilson95'][1]:5.1f}]  "
              f"{r['glicko']:7.1f} (RD {r['glicko_rd']:4.0f})  {r['games']}")

    print("\nper-opponent win% (+/- sigma):")
    hdr = f"{'arm':38s}" + "".join(f"{b[:15]:>17s}" for b in field)
    print(hdr)
    for a in ranked:
        cells = ""
        for b in field:
            c = out["per_opponent"][a].get(b)
            cells += f"{c['win_pct']:10.1f}+-{c['sigma']:4.1f}" if c else f"{'-':>17s}"
        print(f"{a:38s}{cells}")

    print("\nseparation (95% CI non-overlap):")
    sep_yes = [s for s in seps if s["separated"]]
    if not sep_yes:
        print("  NONE — no two arms are separated at this sample size.")
    for s in sep_yes:
        print(f"  {s['a']} > {s['b']}  (+{s['gap_pp']:.1f}pp)")
    n_pairs = len(seps)
    print(f"  ({len(sep_yes)}/{n_pairs} arm pairs separated)")
    print(f"\nsaved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
