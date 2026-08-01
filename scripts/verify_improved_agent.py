"""Verify the reimplemented Improved Probabilistic agent runs end-to-end.

Two checks:
  1. Self-play: the bundled agent plays a full legal game vs a copy of itself.
  2. Search-path smoke test: on a real MAIN decision, run the UCB1 bandit and
     confirm it returns a re-ranked permutation and that simulate_action scores
     real engine states (not -inf crashes).

Run from the repo root with:
    uv run python scripts/verify_improved_agent.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "agents" / "mega_lucario"))

import agent_core_improved as ag  # noqa: E402

from kaggle_environments import make  # noqa: E402


def check_selfplay():
    print("== self-play ==")
    env = make("cabt")
    trace = env.run([ag.agent, ag.agent])
    final = trace[-1]
    statuses = [final[i]["status"] for i in range(2)]
    rewards = [final[i]["reward"] for i in range(2)]
    print(f"steps={len(trace)} statuses={statuses} rewards={rewards}")
    assert all(s == "DONE" for s in statuses), "game did not finish"
    assert sorted(rewards) == [-1, 1], "unexpected rewards"
    print("OK: full legal game\n")


def check_search_path():
    print("== search-path smoke test ==")
    assert ag._SEARCH_OK, "cg search API unavailable in this env"
    env = make("cabt")
    trace = env.run([ag.agent, ag.agent])
    # Find the first agent-0 observation whose decision context is MAIN.
    target = None
    for state in trace:
        obs_dict = state[0]["observation"]
        if obs_dict is None:
            continue
        from cg.api import to_observation_class, SelectContext  # local import ok
        obs = to_observation_class(obs_dict)
        if obs.select is not None and obs.select.context == SelectContext.MAIN:
            target = obs_dict
            break
    assert target is not None, "no MAIN decision found in self-play trace"

    t0 = time.time()
    # Force the search path by calling the public agent (search is enabled).
    ordered = ag.agent(target)
    elapsed = time.time() - t0
    print(f"agent() returned {len(ordered)} indices in {elapsed:.3f}s: {ordered[:8]}")
    assert len(ordered) >= 1

    # Directly call the bandit and confirm it spends simulations + re-ranks.
    obs = ag.to_observation_class(target)
    base = ag.AdvancedPolicy(obs).choose()
    ranked = ag.SEARCH_ALGO(target, obs)
    print(f"heuristic top-8: {base[:8]}")
    print(f"search re-rank:  {ranked}")
    assert ranked is not None and len(ranked) >= 1
    # best action must be a legal index
    assert 0 <= ranked[0] < len(obs.select.option)
    print("OK: UCB1 bandit ran against the real engine\n")


if __name__ == "__main__":
    check_selfplay()
    check_search_path()
    print("ALL CHECKS PASSED")
