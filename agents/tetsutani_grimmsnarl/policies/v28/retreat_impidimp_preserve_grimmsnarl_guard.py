from __future__ import annotations

from collections import Counter

from .manual_observation import get_card_id, semantic

MUNKIDORI_ID = 112
IMPIDIMP_ID = 646
GRIMMSNARL_EX_ID = 648
SNORUNT_ID = 860
MEGA_KANGASKHAN_EX_ID = 756
TEAL_MASK_OGERPON_EX_ID = 96


def _i(v, d=0):
    try:
        return int(v if v is not None else d)
    except Exception:
        return d


def _energy_count(card):
    return len((card or {}).get("energies") or [])


def _shape(card):
    return (
        get_card_id(card),
        _i((card or {}).get("hp"), 0),
        _i((card or {}).get("maxHp"), 0),
        _energy_count(card),
    )


def choose(obs: dict):
    """After an early retreat, preserve a zero-Energy Grimmsnarl ex on Bench.

    The discovered public board contains a full-HP zero-Energy Munkidori that
    has just paid its retreat cost, a full-HP zero-Energy Mega Kangaskhan ex
    opposing it, and five legal promotion targets.  Promoting the two-Energy
    Rule Box Grimmsnarl ex exposes the developed attacker before it can finish
    the prize race.  The unique three-Energy Impidimp is instead used as the
    temporary Active, leaving the Grimmsnarl ex protected on Bench.

    The rule intentionally ignores physical serials, seed, replay, seat, and
    opponent identity.  It requires a unique Impidimp target and
    rejects the two same-state Snorunt choices whose counterfactual outcomes
    differed by physical-copy path.
    """
    sel = obs.get("select") or {}
    opts = sel.get("option") or []
    if _i(sel.get("context"), -1) != 3:
        return None, None
    if _i(sel.get("minCount"), 0) != 1 or _i(sel.get("maxCount"), 0) != 1:
        return None, None
    if len(opts) != 5:
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

    own_active = me.get("active") or []
    if len(own_active) != 1 or _shape(own_active[0]) != (MUNKIDORI_ID, 110, 110, 0):
        return None, None

    opp_active = opp.get("active") or []
    if len(opp_active) != 1 or _shape(opp_active[0]) != (MEGA_KANGASKHAN_EX_ID, 300, 300, 0):
        return None, None
    opp_bench = Counter(_shape(c) for c in (opp.get("bench") or []) if isinstance(c, dict))
    if opp_bench != Counter({
        (TEAL_MASK_OGERPON_EX_ID, 210, 210, 1): 1,
        (MEGA_KANGASKHAN_EX_ID, 300, 300, 2): 1,
    }):
        return None, None

    hand_ids = Counter(get_card_id(c) for c in (me.get("hand") or []) if isinstance(c, dict))
    if hand_ids != Counter({1259: 1, 7: 1, 1097: 1, GRIMMSNARL_EX_ID: 1, 1080: 1}):
        return None, None

    bench = me.get("bench") or []
    expected = Counter({
        (MUNKIDORI_ID, 110, 110, 1): 1,
        (SNORUNT_ID, 70, 70, 0): 2,
        (GRIMMSNARL_EX_ID, 320, 320, 2): 1,
        (IMPIDIMP_ID, 70, 70, 3): 1,
    })
    seen = Counter()
    pick = None
    for oi, opt in enumerate(opts):
        if _i((opt or {}).get("area"), -1) != 5:
            return None, None
        bi = _i((opt or {}).get("index"), -1)
        if not 0 <= bi < len(bench):
            return None, None
        shape = _shape(bench[bi])
        seen[shape] += 1
        if shape[0] == IMPIDIMP_ID:
            if pick is not None:
                return None, None
            pick = oi
    if seen != expected or pick is None:
        return None, None
    return [pick], [semantic(obs, option) for option in opts]
