"""Rung 3 (sanity): play full matches with the IL agent and inspect the
transcript for the specific failure mode we're guarding against -- a policy
that never crashes and never does anything meaningful either (never
declines an optional selection, never retreats, never attacks).

Usage:
    uv run python scripts/eval_rung3_sanity.py --games 5
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for _cand in (REPO / "data" / "external" / "cg-lib", REPO / "agents" / "mega_lucario"):
    if (_cand / "cg" / "api.py").exists():
        sys.path.insert(0, str(_cand))
        break
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

# Reuse the already-proven agent-loading path (main.py-wired deck if it
# exists, else bare-module + deck.csv injection) instead of reimplementing
# it -- a first version of this script hand-rolled the opponent's loading
# and every game ended after 2 steps (a silent deck-submission failure),
# while the exact same opponent worked fine through benchmark_agents.py
# seconds earlier. Don't re-invent what's already validated.
from benchmark_agents import load_agent  # noqa: E402
from cg.api import OptionType  # noqa: E402


def make_logging_agent(name: str, stats: Counter, fn):
    def wrapped(obs_dict: dict) -> list[int]:
        sel = obs_dict.get("select")
        result = fn(obs_dict)
        if sel is None:
            return result
        stats[f"{name}/decisions"] += 1
        if not result and (sel.get("minCount") or 0) == 0:
            stats[f"{name}/declined"] += 1
        if sel.get("option") and result:
            chosen = sel["option"][result[0]] if result[0] < len(sel["option"]) else None
            if chosen:
                otype = chosen.get("type")
                if otype == int(OptionType.RETREAT):
                    stats[f"{name}/retreated"] += 1
                elif otype == int(OptionType.ATTACK):
                    stats[f"{name}/attacked"] += 1
                elif otype == int(OptionType.END):
                    stats[f"{name}/ended_turn"] += 1
        return result

    return wrapped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=5)
    args = ap.parse_args()

    from kaggle_environments import make

    il_fn = load_agent("il_agent")
    opp_fn = load_agent("rule_baseline")

    stats: Counter = Counter()
    outcomes = []
    for i in range(args.games):
        env = make("cabt")
        a = make_logging_agent("il_agent_p0", stats, il_fn)
        b = make_logging_agent("rule_baseline_p1", stats, opp_fn)

        trace = env.run([a, b])
        final = trace[-1]
        outcomes.append((final[0]["reward"], final[1]["reward"], final[0]["status"], len(trace)))
        print(f"game {i}: reward(p0,p1)=({final[0]['reward']},{final[1]['reward']}) "
              f"status={final[0]['status']} steps={len(trace)}")

    print("\n--- aggregate behavioral stats across all games (il_agent as p0) ---")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")

    n_decisions = stats.get("il_agent_p0/decisions", 0)
    print("\n--- sanity checks ---")
    print(f"  ever declined an optional selection? {'YES' if stats.get('il_agent_p0/declined', 0) > 0 else 'NO'} "
          f"({stats.get('il_agent_p0/declined', 0)} times)")
    print(f"  ever retreated? {'YES' if stats.get('il_agent_p0/retreated', 0) > 0 else 'NO'} "
          f"({stats.get('il_agent_p0/retreated', 0)} times)")
    print(f"  ever attacked? {'YES' if stats.get('il_agent_p0/attacked', 0) > 0 else 'NO'} "
          f"({stats.get('il_agent_p0/attacked', 0)} times)")
    print(f"  total decisions made: {n_decisions}")

    wins = sum(1 for r0, r1, s, _ in outcomes if r0 is not None and r1 is not None and r0 > r1)
    losses = sum(1 for r0, r1, s, _ in outcomes if r0 is not None and r1 is not None and r0 < r1)
    print(f"\n  record vs rule_baseline: {wins}W-{losses}L / {len(outcomes)} games (not statistically meaningful at this N -- see Q24)")


if __name__ == "__main__":
    main()
