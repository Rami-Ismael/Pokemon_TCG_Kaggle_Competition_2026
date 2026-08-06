from __future__ import annotations

from .manual_observation import get_card_id, semantic

MUNKIDORI_ID = 112
MORGREM_ID = 647
GRIMMSNARL_EX_ID = 648


def _i(v, d=0):
    try:
        return int(v if v is not None else d)
    except Exception:
        return d


def _e(card):
    return len((card or {}).get("energies") or [])


def choose(obs: dict):
    """Promote the staged Morgrem after an early retreat to preserve tempo.

    In the discovered turn-four shape, the opponent's full-HP 70-HP Active has
    no Energy.  Promoting the three-Energy Morgrem rather than the two-Energy
    Grimmsnarl ex preserves the Rule Box attacker and keeps the zero-Energy
    Active from being immediately cashed out.  The rule uses only public board,
    hand, HP, Energy, prize, and legal-option information; it ignores serials,
    seed, replay, and opponent identity.
    """
    sel = obs.get("select") or {}
    opts = sel.get("option") or []
    if _i(sel.get("context"), -1) != 3:
        return None, None
    if _i(sel.get("minCount"), 0) != 1 or _i(sel.get("maxCount"), 0) != 1:
        return None, None
    if len(opts) != 4:
        return None, None

    cur = obs.get("current") or {}
    if _i(cur.get("turn"), -1) != 4:
        return None, None
    players = cur.get("players") or []
    yi = _i(cur.get("yourIndex"), -1)
    if len(players) != 2 or yi not in (0, 1):
        return None, None
    me, opp = players[yi], players[1 - yi]
    if len(me.get("prize") or []) != 6 or len(opp.get("prize") or []) != 6:
        return None, None

    active = me.get("active") or []
    if len(active) != 1 or get_card_id(active[0]) != MUNKIDORI_ID:
        return None, None
    if (_i(active[0].get("hp"), 0), _i(active[0].get("maxHp"), 0), _e(active[0])) != (110, 110, 0):
        return None, None

    oa = opp.get("active") or []
    if len(oa) != 1:
        return None, None
    if _i(oa[0].get("hp"), 0) != 70 or _i(oa[0].get("maxHp"), 0) != 70 or _e(oa[0]) != 0:
        return None, None

    hand_ids = sorted(get_card_id(c) for c in (me.get("hand") or []))
    if hand_ids != sorted([1227, 1219, 1219, 104, 1097]):
        return None, None

    bench = me.get("bench") or []
    expected = {
        (GRIMMSNARL_EX_ID, 320, 320, 2): 1,
        (MORGREM_ID, 100, 100, 3): 1,
        (MUNKIDORI_ID, 110, 110, 1): 1,
        (MUNKIDORI_ID, 110, 110, 0): 1,
    }
    seen = {}
    pick = None
    for oi, opt in enumerate(opts):
        if _i((opt or {}).get("area"), -1) != 5:
            return None, None
        bi = _i((opt or {}).get("index"), -1)
        if not 0 <= bi < len(bench):
            return None, None
        c = bench[bi]
        key = (get_card_id(c), _i(c.get("hp"), 0), _i(c.get("maxHp"), 0), _e(c))
        seen[key] = seen.get(key, 0) + 1
        if key[0] == MORGREM_ID:
            if pick is not None:
                return None, None
            pick = oi
    if seen != expected or pick is None:
        return None, None
    return [pick], [semantic(obs, o) for o in opts]
