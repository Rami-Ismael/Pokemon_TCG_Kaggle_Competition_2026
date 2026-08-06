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


def _e(card: dict) -> int:
    return len((card or {}).get("energies") or [])


def choose(obs: dict):
    """Preserve a prepared Morgrem evolution line after a KO.

    In an unseen late prize-race promotion, a healthy two-Energy Morgrem plus a
    Grimmsnarl ex in hand is a completed next-turn attacker.  Promote the best
    available energized Munkidori instead of exposing Morgrem as the temporary
    Active.  The rule uses only public board/hand state and never card serials.
    """
    sel = obs.get("select") or {}
    opts = sel.get("option") or []
    if _i(sel.get("context"), -1) != 4:
        return None, None
    if _i(sel.get("minCount"), 0) != 1 or _i(sel.get("maxCount"), 0) != 1:
        return None, None
    if not (4 <= len(opts) <= 6):
        return None, None

    cur = obs.get("current") or {}
    players = cur.get("players") or []
    your = _i(cur.get("yourIndex"), -1)
    if len(players) != 2 or your not in (0, 1):
        return None, None
    me, opp = players[your], players[1 - your]
    if me.get("active"):
        return None, None
    # Late deficit only: avoid changing ordinary early-game promotions.
    if len(me.get("prize") or []) != 4 or len(opp.get("prize") or []) != 2:
        return None, None
    if not any(get_card_id(c) == GRIMMSNARL_EX_ID for c in (me.get("hand") or [])):
        return None, None
    opp_active = opp.get("active") or []
    # Do not preserve Morgrem when its 60-damage attack can take the current
    # Active immediately; in that case promoting the prepared attacker is the
    # direct tempo play.
    opp_hp = _i(opp_active[0].get("hp"), 0) if len(opp_active) == 1 else 0
    if len(opp_active) != 1 or _e(opp_active[0]) < 2 or not (61 <= opp_hp <= 80):
        return None, None

    bench = me.get("bench") or []
    morgrem_options = []
    total_morgrem = 0
    munk_options = []
    for option_index, option in enumerate(opts):
        if _i((option or {}).get("area"), -1) != 5:
            return None, None
        bench_index = _i((option or {}).get("index"), -1)
        if not (0 <= bench_index < len(bench)):
            return None, None
        card = bench[bench_index]
        cid = get_card_id(card)
        hp = _i(card.get("hp"), 0)
        max_hp = _i(card.get("maxHp"), 0)
        if hp <= 0:
            return None, None
        if cid == MORGREM_ID:
            total_morgrem += 1
            if _e(card) >= 2 and hp == max_hp:
                morgrem_options.append(option_index)
        elif cid == MUNKIDORI_ID and _e(card) >= 1 and hp >= 90:
            munk_options.append((option_index, _e(card), hp))

    if total_morgrem != 1 or len(morgrem_options) != 1 or not munk_options:
        return None, None
    # Prefer the most prepared Munkidori, then the healthier one.  Option index
    # is only a deterministic final tie-breaker; physical serial is ignored.
    pick = max(munk_options, key=lambda x: (x[1], x[2], -x[0]))[0]
    return [pick], [semantic(obs, option) for option in opts]
