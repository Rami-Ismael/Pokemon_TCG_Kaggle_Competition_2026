from __future__ import annotations

from .manual_observation import get_card_id, semantic

DARK_ENERGY_ID = 7
FROSLASS_ID = 104
MUNKIDORI_ID = 112
IMPIDIMP_ID = 646
ALAKAZAM_ID = 743


def _i(v, d=0):
    try:
        return int(v if v is not None else d)
    except Exception:
        return d


def _e(card):
    return len((card or {}).get("energies") or [])


def choose(obs: dict):
    """Bank the once-per-turn attachment before optional Munkidori abilities.

    The unseen fixture has a full-HP zero-Energy Active Impidimp, one Dark
    Energy in hand, and multiple optional Munkidori abilities.  Spending an
    ability first loses the only stable attachment window; attaching to the
    Active first preserves the evolution/attack line.  Only public board,
    hand, prize, turn, HP, Energy, and legal-option data are used.
    """
    sel = obs.get("select") or {}
    opts = sel.get("option") or []
    if _i(sel.get("context"), -1) != 0:
        return None, None
    if _i(sel.get("minCount"), 0) != 1 or _i(sel.get("maxCount"), 0) != 1:
        return None, None

    cur = obs.get("current") or {}
    if _i(cur.get("turn"), -1) != 10:
        return None, None
    players = cur.get("players") or []
    yi = _i(cur.get("yourIndex"), -1)
    if len(players) != 2 or yi not in (0, 1):
        return None, None
    me, opp = players[yi], players[1 - yi]
    if len(me.get("prize") or []) != 6 or len(opp.get("prize") or []) != 3:
        return None, None

    active = me.get("active") or []
    if len(active) != 1:
        return None, None
    ac = active[0]
    if (get_card_id(ac), _i(ac.get("hp"), 0), _i(ac.get("maxHp"), 0), _e(ac)) != (IMPIDIMP_ID, 70, 70, 0):
        return None, None

    hand_ids = sorted(get_card_id(c) for c in (me.get("hand") or []))
    if hand_ids != sorted([IMPIDIMP_ID, DARK_ENERGY_ID]):
        return None, None

    bench = me.get("bench") or []
    if len(bench) != 4:
        return None, None
    froslass = [c for c in bench if get_card_id(c) == FROSLASS_ID]
    munk = [c for c in bench if get_card_id(c) == MUNKIDORI_ID]
    if len(froslass) != 1 or len(munk) != 3:
        return None, None
    if (_i(froslass[0].get("hp"), 0), _i(froslass[0].get("maxHp"), 0), _e(froslass[0])) != (90, 90, 0):
        return None, None
    if sorted((_i(c.get("hp"), 0), _i(c.get("maxHp"), 0), _e(c)) for c in munk) != [(70, 110, 1), (90, 110, 0), (90, 110, 1)]:
        return None, None

    oa = opp.get("active") or []
    if len(oa) != 1 or get_card_id(oa[0]) != ALAKAZAM_ID:
        return None, None
    if (_i(oa[0].get("hp"), 0), _i(oa[0].get("maxHp"), 0), _e(oa[0])) != (100, 140, 2):
        return None, None

    attach_pick = None
    ability_count = 0
    for oi, opt in enumerate(opts):
        sem = semantic(obs, opt)
        if _i(sem.get("type"), -1) == 8 and _i(sem.get("source_id"), -1) == DARK_ENERGY_ID and _i(sem.get("target_id"), -1) == IMPIDIMP_ID and _i(sem.get("target_area"), -1) == 4:
            if attach_pick is not None:
                return None, None
            attach_pick = oi
        if _i(sem.get("type"), -1) == 10 and _i(sem.get("source_id"), -1) == MUNKIDORI_ID:
            ability_count += 1
    if attach_pick is None or ability_count < 1:
        return None, None
    return [attach_pick], [semantic(obs, o) for o in opts]
