"""Per-deck official rule pilots for field training and eval.

Only uses organizer-provided samples (see agent/official_registry.py).
No RuleCoreScorer / invented pilots unless explicitly requested via --non-official-brain.
"""

from __future__ import annotations

from collections.abc import Callable

from agent.official_registry import make_official_opponent, official_archetype_for_opponent


def native_brain_label(deck_ids: list[int], opp_name: str = "") -> str:
    arch = official_archetype_for_opponent(opp_name, deck_ids)
    return arch if arch else "none"


def make_opponent_brain(
    kind: str,
    deck_path: str,
    deck_ids: list[int],
    *,
    opp_name: str = "",
    seed: int = 0,
    random_agent: Callable[[dict], list[int]] | None = None,
    non_official_brain: str = "random",
) -> tuple[Callable[[dict], list[int]], str]:
    """Resolve training/eval opponent move function from CLI brain kind."""
    if kind == "native":
        return make_official_opponent(
            deck_path, deck_ids, opp_name=opp_name, seed=seed,
        )
    if kind == "random":
        if random_agent is None:
            raise ValueError("random_agent callback required for kind=random")
        return random_agent, "random"
    if kind == "non_official":
        arch = official_archetype_for_opponent(opp_name, deck_ids)
        if arch is not None:
            return make_official_opponent(
                deck_path, deck_ids, opp_name=opp_name, seed=seed,
            )
        if non_official_brain == "random":
            if random_agent is None:
                raise ValueError("random_agent required")
            return random_agent, f"random:{opp_name}"
        raise ValueError(f"unknown non_official_brain: {non_official_brain}")
    raise ValueError(
        f"unknown opponent brain kind: {kind!r}. "
        "Use native (official samples), random, or non_official (official where available)."
    )
