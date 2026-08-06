from __future__ import annotations

from .manual_observation import get_card_id, semantic

MUNKIDORI_ID = 112
FROSLASS_ID = 104
SNORUNT_ID = 860
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
    """Use a slightly damaged Munkidori as the early temporary Active.

    After the first prize is lost, preserve the visible Morgrem -> Grimmsnarl
    ex line and keep the developed Froslass on the Bench.  The rule only
    handles the discovered four-candidate public board shape and ignores card
    serials and opponent identity.
    """
    sel = obs.get("select") or {}
    opts = sel.get("option") or []
    if _i(sel.get("context"), -1) != 4:
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
    if me.get("active"):
        return None, None
    if len(me.get("prize") or []) != 6 or len(opp.get("prize") or []) != 5:
        return None, None
    if not any(get_card_id(c) == GRIMMSNARL_EX_ID for c in (me.get("hand") or [])):
        return None, None

    oa = opp.get("active") or []
    if len(oa) != 1:
        return None, None
    ohp, omh = _i(oa[0].get("hp"), 0), _i(oa[0].get("maxHp"), 0)
    if omh < 300 or ohp < 280:
        return None, None

    bench = me.get("bench") or []
    seen = {MUNKIDORI_ID: 0, FROSLASS_ID: 0, SNORUNT_ID: 0, MORGREM_ID: 0}
    pick = None
    for oi, opt in enumerate(opts):
        if _i((opt or {}).get("area"), -1) != 5:
            return None, None
        bi = _i((opt or {}).get("index"), -1)
        if not 0 <= bi < len(bench):
            return None, None
        c = bench[bi]
        cid = get_card_id(c)
        if cid not in seen or _e(c) != 0:
            return None, None
        seen[cid] += 1
        hp, mh = _i(c.get("hp"), 0), _i(c.get("maxHp"), 0)
        if cid == MUNKIDORI_ID:
            # Slightly damaged but still at least as durable as the full
            # Morgrem/Froslass candidates.
            if mh != 110 or not (90 <= hp < mh) or pick is not None:
                return None, None
            pick = oi
        else:
            if hp <= 0 or hp != mh:
                return None, None
    if pick is None or any(v != 1 for v in seen.values()):
        return None, None
    return [pick], [semantic(obs, o) for o in opts]
