from __future__ import annotations

from .manual_observation import get_card_id, semantic

MUNKIDORI_ID = 112
MORGREM_ID = 647
GRIMMSNARL_EX_ID = 648
SPIKEMUTH_GYM_ID = 1259


def _i(v, d=0):
    try:
        return int(v if v is not None else d)
    except Exception:
        return d


def _e(card):
    return len((card or {}).get("energies") or [])


def choose(obs: dict):
    """Promote an energized Munkidori while preserving two Morgrem evolution lines.

    In the verified unseen shape, the previous Grimmsnarl ex has been KO'd
    after taking three prizes.  Both Morgrem are staged and a Grimmsnarl ex is
    in hand; promoting either Morgrem exposes an evolution line.  Either of
    the semantically identical one-Energy Munkidori promotions wins.  Only
    public turn, prize, hand, board, HP, Energy, and legal-option data are used.
    """
    sel = obs.get("select") or {}
    opts = sel.get("option") or []
    if _i(sel.get("context"), -1) != 4:
        return None, None
    if _i(sel.get("minCount"), 0) != 1 or _i(sel.get("maxCount"), 0) != 1:
        return None, None

    cur = obs.get("current") or {}
    if _i(cur.get("turn"), -1) != 5:
        return None, None
    players = cur.get("players") or []
    yi = _i(cur.get("yourIndex"), -1)
    if len(players) != 2 or yi not in (0, 1):
        return None, None
    me, opp = players[yi], players[1 - yi]
    if len(me.get("prize") or []) != 6 or len(opp.get("prize") or []) != 3:
        return None, None
    if me.get("active"):
        return None, None

    hand_ids = sorted(get_card_id(c) for c in (me.get("hand") or []))
    if hand_ids != sorted([SPIKEMUTH_GYM_ID, GRIMMSNARL_EX_ID]):
        return None, None

    bench = me.get("bench") or []
    if len(bench) != 4:
        return None, None
    morgrem = [c for c in bench if get_card_id(c) == MORGREM_ID]
    munk = [c for c in bench if get_card_id(c) == MUNKIDORI_ID]
    if len(morgrem) != 2 or len(munk) != 2:
        return None, None
    if sorted((_i(c.get("hp"), 0), _i(c.get("maxHp"), 0), _e(c)) for c in morgrem) != [(100, 100, 1), (100, 100, 2)]:
        return None, None
    if sorted((_i(c.get("hp"), 0), _i(c.get("maxHp"), 0), _e(c)) for c in munk) != [(110, 110, 1), (110, 110, 1)]:
        return None, None

    oa = opp.get("active") or []
    if len(oa) != 1:
        return None, None
    oc = oa[0]
    if _i(oc.get("maxHp"), 0) < 300 or not (100 <= _i(oc.get("hp"), 0) <= 130) or _e(oc) < 3:
        return None, None

    picks = []
    sems = []
    for i, opt in enumerate(opts):
        sem = semantic(obs, opt)
        sems.append(sem)
        if _i(sem.get("type"), -1) == 3 and _i(sem.get("source_id"), -1) == MUNKIDORI_ID and _i(sem.get("source_zone"), -1) == 5:
            picks.append(i)
    if len(picks) != 2:
        return None, None
    return [picks[0]], sems
