"""Pure random-legal-move baseline.

Uniformly samples a legal response respecting minCount/maxCount (including
sometimes legitimately declining when minCount == 0, rather than always
committing to the maximum -- see 1.3/Q10). No card, board, or strategy
awareness at all.

Purpose: a floor test, not a competitor. If a trained policy can't beat
this decisively, its offline accuracy is not credible evidence that it has
learned anything useful, no matter how it looks against stronger opponents.
Same frozen deck as il_agent so this isolates policy quality from deck
quality.
"""

from __future__ import annotations

import random

# `my_deck` intentionally not pre-declared -- see agents/il_agent/agent_core.py's
# comment on why (benchmark harness / main.py injects it from deck.csv).


def agent(obs_dict: dict) -> list[int]:
    select = obs_dict.get("select")
    if select is None:
        return my_deck

    options = select.get("option", [])
    n = len(options)
    if n == 0:
        return []

    min_count = max(select.get("minCount") or 0, 0)
    max_count = select.get("maxCount")
    max_count = n if max_count is None else min(max_count, n)

    k = random.randint(0 if min_count == 0 else min_count, max_count)
    return random.sample(range(n), k) if k > 0 else []
