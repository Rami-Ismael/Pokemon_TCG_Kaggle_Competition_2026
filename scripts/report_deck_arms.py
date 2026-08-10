"""Per-deck and per-matchup win rates for the deck-variation arms.

WHY NOT THE POOLED WIN RATE. A single pooled number is exactly the statistic
that hides the failure this experiment exists to detect: a policy that dominates
one deck and loses on the rest can post a perfectly healthy pooled rate. So this
prints the CELLS — per learner deck, and per (learner deck ~ opponent deck)
matchup — worst first, each with its own n and binomial standard error.

READ n BEFORE READING A WIN RATE. With K decks the matchup grid is K^2 cells, so
episodes per cell fall as K^2 while episodes only grow linearly with compute. At
K=29 a 1.2M-step run puts single digits in most matchup cells, where a 0.000 is
noise, not a weakness. Cells below --min-n are printed but marked UNREADABLE and
excluded from the "worst" verdict.

These numbers come from TRAINING rollouts, so they average over a policy that
was changing throughout the run and over a mirror opponent that was changing
with it. They are a monitor, not an arm comparison — for that, evaluate the
final checkpoints on a fixed deck grid against a fixed opponent.

    uv run python scripts/report_deck_arms.py models/selfplay_deckpool29_seed42
    uv run python scripts/report_deck_arms.py models/selfplay_*_seed42 --min-n 30
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokemon_tcg import config  # noqa: E402


def se(p: float, n: int) -> float:
    return math.sqrt(max(p * (1 - p), 0.0) / n) if n > 0 else float("nan")


def load_arm(run_dir: Path) -> dict | None:
    """Prefer the end-of-run summary; fall back to the last jsonl row."""
    summ = run_dir / "deck_summary.json"
    if summ.exists():
        return json.loads(summ.read_text())
    jl = run_dir / "deck_metrics.jsonl"
    if jl.exists():
        rows = [json.loads(ln) for ln in jl.read_text().splitlines() if ln.strip()]
        if rows:
            last = rows[-1]
            return {"run_tag": last.get("run_tag", run_dir.name),
                    "global_step": last.get("global_step"),
                    "per_deck": last["per_deck"],
                    "per_matchup": last["per_matchup"],
                    "partial": True}
    meta = run_dir / "train_metrics.jsonl"
    if meta.exists():
        # Single-deck control arm: no deck tables by construction. Report the
        # pooled opponent-bucket rates so the arms are still comparable.
        rows = [json.loads(ln) for ln in meta.read_text().splitlines() if ln.strip()]
        opp = rows[-1].get("opp_wr", {}) if rows else {}
        return {"run_tag": rows[-1].get("run_tag", run_dir.name) if rows else run_dir.name,
                "global_step": rows[-1].get("global_step") if rows else None,
                "per_deck": {}, "per_matchup": {}, "opp_wr": opp,
                "single_deck": True}
    return None


def show(title: str, table: dict, min_n: int, limit: int) -> tuple | None:
    if not table:
        print(f"  ({title}: no cells — single-deck arm)")
        return None
    readable = {k: v for k, v in table.items() if v[1] >= min_n}
    total_n = sum(v[1] for v in table.values())
    print(f"  {title}: {len(table)} cells, {total_n} episodes, "
          f"{len(readable)} at n>={min_n}")
    for name, (wr, n) in list(table.items())[:limit]:
        mark = "" if n >= min_n else "   <- UNREADABLE (n too small)"
        print(f"    {wr:6.3f} +-{se(wr, n):5.3f}  n={n:<6} {name}{mark}")
    if len(table) > limit:
        print(f"    ... {len(table) - limit} more cells (tables are sorted "
              f"worst-first, so the tail is the strong end)")
    if not readable:
        print(f"    NO CELL reaches n>={min_n}: this cut has no readable "
              f"worst case. Either train longer or shrink K.")
        return None
    worst = min(readable.items(), key=lambda kv: kv[1][0])
    return worst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path,
                    help="one or more <out> dirs from train_ppo_puffer.py")
    ap.add_argument("--min-n", type=int, default=30,
                    help="cells below this many episodes are marked UNREADABLE "
                         "and excluded from the worst-case verdict. 30 gives "
                         "SE<=0.09 at p=0.5 -- already loose")
    ap.add_argument("--limit", type=int, default=15,
                    help="cells printed per table (worst first)")
    ap.add_argument("--out", type=Path,
                    default=config.REPORTS_DIR / "deck_arms_report.json")
    args = ap.parse_args()

    report = {}
    for d in args.run_dirs:
        arm = load_arm(d)
        print(f"\n=== {d}")
        if arm is None:
            print("  no metrics found (run never produced train_metrics.jsonl)")
            continue
        head = [f"step={arm.get('global_step')}"]
        if arm.get("single_deck"):
            head.append("SINGLE-DECK CONTROL")
        else:
            head.append(f"K={arm.get('deck_pool_k')}")
            head.append("mirror-deck" if arm.get("mirror_deck") else "independent-decks")
        if arm.get("partial"):
            head.append("PARTIAL (run did not finish)")
        print("  " + "  ".join(str(h) for h in head))
        if "pooled_win_rate" in arm:
            print(f"  pooled win rate {arm['pooled_win_rate']:.3f} over "
                  f"{arm['episodes']} episodes  <- do not report this alone")
        if arm.get("opp_wr"):
            print("  by opponent bucket:")
            for k, v in arm["opp_wr"].items():
                print(f"    {v[0]:6.3f} +-{se(v[0], v[1]):5.3f}  n={v[1]:<6} {k}")

        worst_deck = show("per learner deck", arm.get("per_deck", {}),
                          args.min_n, args.limit)
        worst_mu = show("per matchup (learner ~ opponent)",
                        arm.get("per_matchup", {}), args.min_n, args.limit)
        if worst_deck:
            print(f"  WORST DECK:    {worst_deck[0]}  wr={worst_deck[1][0]:.3f} "
                  f"n={worst_deck[1][1]}")
        if worst_mu:
            print(f"  WORST MATCHUP: {worst_mu[0]}  wr={worst_mu[1][0]:.3f} "
                  f"n={worst_mu[1][1]}")
        report[str(d)] = {**{k: v for k, v in arm.items()
                             if k not in ("per_deck", "per_matchup")},
                          "worst_deck": worst_deck, "worst_matchup": worst_mu,
                          "per_deck": arm.get("per_deck", {}),
                          "per_matchup": arm.get("per_matchup", {})}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nreport: {args.out}")
    print("REMINDER: these are training-rollout rates against a MOVING mirror. "
          "They monitor the run; they do not compare the arms. For that, "
          "evaluate the final checkpoints on a fixed deck grid vs fixed "
          "opponents, then the anchored local pool, then the ladder.")


if __name__ == "__main__":
    main()
