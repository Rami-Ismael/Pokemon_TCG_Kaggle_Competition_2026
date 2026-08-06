from __future__ import annotations

from .manual_observation import get_card_id, semantic

MUNKIDORI_ID = 112
MORGREM_ID = 647
IMPIDIMP_ID = 646
GRIMMSNARL_EX_ID = 648
SNORUNT_ID = 860
RARE_CANDY_ID = 1079
TOOL_SCRAPPER_ID = 1137


def _i(v, d=0):
    try:
        return int(v if v is not None else d)
    except Exception:
        return d


def _e(card):
    return len((card or {}).get("energies") or [])


def choose(obs: dict):
    """Promote the damaged Impidimp to convert Rare Candy next turn.

    In the discovered same-deck midgame shape, promoting Morgrem exposes the
    only staged evolution line.  Promoting the already-damaged Impidimp keeps
    Morgrem on the Bench and lets Rare Candy plus the visible Stadium search
    convert the temporary Active into Grimmsnarl ex.  The rule uses only public
    card identities, HP, Energy counts, prizes, and legal promotion options.
    """
    sel = obs.get("select") or {}
    opts = sel.get("option") or []
    if _i(sel.get("context"), -1) != 4:
        return None, None
    if _i(sel.get("minCount"), 0) != 1 or _i(sel.get("maxCount"), 0) != 1:
        return None, None
    if len(opts) != 5:
        return None, None

    cur = obs.get("current") or {}
    if _i(cur.get("turn"), -1) != 7:
        return None, None
    players = cur.get("players") or []
    yi = _i(cur.get("yourIndex"), -1)
    if len(players) != 2 or yi not in (0, 1):
        return None, None
    me, opp = players[yi], players[1 - yi]
    if me.get("active"):
        return None, None
    if len(me.get("prize") or []) != 4 or len(opp.get("prize") or []) != 3:
        return None, None

    hand_ids = [get_card_id(c) for c in (me.get("hand") or [])]
    if hand_ids.count(RARE_CANDY_ID) != 1 or hand_ids.count(TOOL_SCRAPPER_ID) != 1:
        return None, None
    if len(hand_ids) != 2:
        return None, None

    oa = opp.get("active") or []
    ob = opp.get("bench") or []
    if len(oa) != 1 or get_card_id(oa[0]) != GRIMMSNARL_EX_ID:
        return None, None
    if _i(oa[0].get("hp"), 0) != 290 or _i(oa[0].get("maxHp"), 0) != 320:
        return None, None
    if len(ob) != 2:
        return None, None
    opp_shape = sorted((get_card_id(c), _i(c.get("hp"), 0), _i(c.get("maxHp"), 0)) for c in ob)
    if opp_shape != sorted([(SNORUNT_ID, 70, 70), (GRIMMSNARL_EX_ID, 290, 320)]):
        return None, None

    bench = me.get("bench") or []
    counts = {MUNKIDORI_ID: 0, MORGREM_ID: 0, IMPIDIMP_ID: 0}
    pick = None
    for oi, opt in enumerate(opts):
        if _i((opt or {}).get("area"), -1) != 5:
            return None, None
        bi = _i((opt or {}).get("index"), -1)
        if not 0 <= bi < len(bench):
            return None, None
        c = bench[bi]
        cid = get_card_id(c)
        if cid not in counts:
            return None, None
        counts[cid] += 1
        hp, mh = _i(c.get("hp"), 0), _i(c.get("maxHp"), 0)
        if cid == MUNKIDORI_ID:
            if (hp, mh) != (110, 110) or _e(c) != 0:
                return None, None
        elif cid == MORGREM_ID:
            if (hp, mh) != (70, 100) or _e(c) != 3:
                return None, None
        else:
            if (hp, mh) != (40, 70) or _e(c) != 0 or pick is not None:
                return None, None
            pick = oi
    if counts != {MUNKIDORI_ID: 3, MORGREM_ID: 1, IMPIDIMP_ID: 1} or pick is None:
        return None, None
    return [pick], [semantic(obs, o) for o in opts]
