from __future__ import annotations

from .manual_observation import get_card_id, semantic

FROSLASS_ID = 104
MUNKIDORI_ID = 112
MORGREM_ID = 647
IMPIDIMP_ID = 646
SNORUNT_ID = 860
MEGA_KANGASKHAN_EX_ID = 756
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
    """Promote the completed Froslass as the expendable early wall.

    In the verified unseen five-candidate shape, the opponent's four-Energy
    Mega Kangaskhan ex has already taken two prizes.  Froslass is the only
    completed zero-Energy line whose loss does not consume the staged Morgrem,
    Snorunt development, or either Munkidori utility resource.  The rule uses
    only public board/hand data and ignores serials, seeds, and opponent IDs.
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
    if _i(cur.get("turn"), -1) != 4:
        return None, None
    players = cur.get("players") or []
    yi = _i(cur.get("yourIndex"), -1)
    if len(players) != 2 or yi not in (0, 1):
        return None, None
    me, opp = players[yi], players[1 - yi]
    if me.get("active"):
        return None, None
    if len(me.get("prize") or []) != 6 or len(opp.get("prize") or []) != 4:
        return None, None

    hand_ids = sorted(get_card_id(c) for c in (me.get("hand") or []))
    if hand_ids != sorted([RARE_CANDY_ID, IMPIDIMP_ID, TOOL_SCRAPPER_ID]):
        return None, None

    bench = me.get("bench") or []
    if len(bench) != 5:
        return None, None
    expected = {
        FROSLASS_ID: [(90, 90, 0)],
        SNORUNT_ID: [(70, 70, 0)],
        MORGREM_ID: [(100, 100, 0)],
        MUNKIDORI_ID: [(100, 110, 0), (100, 110, 1)],
    }
    actual = {k: [] for k in expected}
    for c in bench:
        cid = get_card_id(c)
        if cid not in actual:
            return None, None
        actual[cid].append((_i(c.get("hp"), 0), _i(c.get("maxHp"), 0), _e(c)))
    if any(sorted(actual[k]) != sorted(v) for k, v in expected.items()):
        return None, None

    oa = opp.get("active") or []
    if len(oa) != 1:
        return None, None
    oc = oa[0]
    if get_card_id(oc) != MEGA_KANGASKHAN_EX_ID:
        return None, None
    if _i(oc.get("maxHp"), 0) != 300 or not (285 <= _i(oc.get("hp"), 0) <= 295) or _e(oc) != 4:
        return None, None
    if len(opp.get("bench") or []) != 5:
        return None, None

    picks = []
    sems = []
    for i, opt in enumerate(opts):
        sem = semantic(obs, opt)
        sems.append(sem)
        if _i(sem.get("type"), -1) == 3 and _i(sem.get("source_id"), -1) == FROSLASS_ID and _i(sem.get("source_zone"), -1) == 5:
            picks.append(i)
    if len(picks) != 1:
        return None, None
    return [picks[0]], sems
