from __future__ import annotations
"""A narrow legality-preserving opening continuity safeguard."""

from collections import Counter
import human_controller as hc

IMPIDIMP = 646
MORGREM = 647
GRIMMSNARL_EX = 648
POKE_PAD = 1152


def reset():
    return None


def _context(obs):
    value = (obs.get("select") or {}).get("context")
    return int(value if value is not None else -1)


def _effect(obs):
    select = obs.get("select") or {}
    return hc._cid(select.get("effect")) or hc._cid(select.get("contextCard"))


def choose(obs, baseline):
    """Fetch Impidimp when a held Morgrem otherwise has no evolution base."""
    if not hc._legal(obs, baseline):
        return baseline
    if _context(obs) != 7 or _effect(obs) != POKE_PAD:
        return baseline
    if not isinstance(baseline, list) or len(baseline) != 1:
        return baseline

    _, me, _ = hc._players(obs)
    if hc._bench_free(me) <= 0:
        return baseline
    in_play = hc._inplay(me)
    if any(hc._cid(card) in (IMPIDIMP, MORGREM, GRIMMSNARL_EX) for card in in_play):
        return baseline

    hand = Counter(hc._hand_ids(me))
    if hand[MORGREM] <= 0 or hand[IMPIDIMP] > 0:
        return baseline

    options = (obs.get("select") or {}).get("option") or []
    for index, option in enumerate(options):
        if hc._cid(hc._card_at(obs, option)) != IMPIDIMP:
            continue
        action = [index]
        if action != baseline and hc._legal(obs, action):
            return action
    return baseline
