"""EDA: does our BC agent's win rate DECAY with game length?

The behaviour-cloning failure mode is compounding error: the policy matches the
expert on states the expert visits, but one off-distribution move leads to a
state no expert ever reached, where it is less accurate, which leads further
off-distribution. If that is what is costing us ~730 ladder points, the
signature is mechanical and testable:

    win rate should fall as games get longer.

If instead win rate is flat in game length, compounding error is NOT the
dominant term and the loss is something else (a systematic strategic gap that
shows up immediately, or an opponent-model failure).

This is a *within-agent* test, so it is immune to the local-pool bias that has
made every cross-agent comparison untrustworthy: we are not asking "is A better
than B", only "does this agent do worse the longer the game runs".

Usage:
  uv run python scripts/eda_error_source.py --agent il_alldays_3ep@marnies_grimmsnarl_ex \
      --opponents tb_archaludon,wmh_alakazam,tb_alakazam --games 30
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import benchmark_agents as B  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True)
    ap.add_argument("--opponents", required=True)
    ap.add_argument("--games", type=int, default=30, help="games per opponent")
    ap.add_argument("--out", type=Path, default=REPO / "reports" / "eda_error_source.json")
    args = ap.parse_args()

    from kaggle_environments import make

    me = B.load_agent(args.agent)
    opps = {o: B.load_agent(o) for o in args.opponents.split(",") if o}

    rows = []
    for oname, ofn in opps.items():
        for i in range(args.games):
            seat0_is_me = (i % 2 == 0)          # alternate seats: cancels first-player edge
            a, b = (me, ofn) if seat0_is_me else (ofn, me)
            env = make("cabt")
            trace = env.run([a, b])
            final = trace[-1]
            r = [final[k].get("reward") for k in range(2)]
            my_idx = 0 if seat0_is_me else 1
            if r[0] is None or r[1] is None:
                outcome = "crash"
            elif r[0] == r[1]:
                outcome = "draw"
            else:
                outcome = "win" if r[my_idx] > r[1 - my_idx] else "loss"
            rows.append({"opponent": oname, "turns": len(trace),
                         "outcome": outcome, "my_seat": my_idx})
            print(f"  {oname:24s} game {i+1:3d}/{args.games}  turns={len(trace):4d}  {outcome}",
                  flush=True)

    played = [r for r in rows if r["outcome"] in ("win", "loss")]
    if not played:
        print("no decisive games")
        return 1
    lens = sorted(r["turns"] for r in played)
    med = lens[len(lens) // 2]
    short = [r for r in played if r["turns"] <= med]
    long_ = [r for r in played if r["turns"] > med]

    def wr(g):
        return 100 * sum(1 for r in g if r["outcome"] == "win") / len(g) if g else float("nan")

    # quartiles, to see a trend rather than a single split
    qs = [lens[int(p * (len(lens) - 1))] for p in (0.25, 0.5, 0.75)]
    buckets = [("Q1 shortest", [r for r in played if r["turns"] <= qs[0]]),
               ("Q2", [r for r in played if qs[0] < r["turns"] <= qs[1]]),
               ("Q3", [r for r in played if qs[1] < r["turns"] <= qs[2]]),
               ("Q4 longest", [r for r in played if r["turns"] > qs[2]])]

    print(f"\n=== {args.agent} — win rate vs game length ===")
    print(f"decisive games: {len(played)}   turns: median {med}, "
          f"mean {st.mean(lens):.1f}, range {min(lens)}–{max(lens)}")
    print(f"\n{'bucket':14s} {'turns':>12s} {'games':>7s} {'win%':>8s}")
    for name, g in buckets:
        if not g:
            continue
        tl = sorted(r["turns"] for r in g)
        print(f"{name:14s} {str(tl[0])+'–'+str(tl[-1]):>12s} {len(g):7d} {wr(g):7.1f}%")
    print(f"\nshort half {wr(short):.1f}%  ({len(short)} games)   "
          f"long half {wr(long_):.1f}%  ({len(long_)} games)   "
          f"delta {wr(long_) - wr(short):+.1f}pp")
    print("\nA large NEGATIVE delta is the compounding-error signature.\n"
          "A flat delta means the loss is not primarily compounding error.")

    args.out.write_text(json.dumps(
        {"agent": args.agent, "rows": rows,
         "short_win_pct": wr(short), "long_win_pct": wr(long_),
         "median_turns": med}, indent=2))
    print(f"saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
