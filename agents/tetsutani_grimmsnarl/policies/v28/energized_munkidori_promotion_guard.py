from __future__ import annotations

from .manual_observation import get_card_id, semantic

MUNKIDORI_ID = 112
IMPIDIMP_ID = 646


def _int(value, default=0):
    try:
        return int(value if value is not None else default)
    except Exception:
        return default


def _energy_count(card: dict) -> int:
    return len((card or {}).get("energies") or [])


def choose(obs: dict):
    """Promote the healthy energized utility Basic over a zero-energy evolution Basic.

    This is a deliberately narrow public-state promotion rule for an unseen
    early prize-race state.  After our Active is knocked out, all legal
    promotion candidates must consist of exactly one full-HP, energized
    Munkidori and otherwise full-HP, zero-energy Marnie's Impidimp.  Against an
    already energized opposing Active, promote Munkidori to preserve the
    Impidimp evolution line and place the sturdier body in front.
    """
    sel = obs.get("select") or {}
    opts = sel.get("option") or []
    if _int(sel.get("context"), -1) != 4:
        return None, None
    if _int(sel.get("minCount"), 0) != 1 or _int(sel.get("maxCount"), 0) != 1:
        return None, None
    if len(opts) != 3:
        return None, None

    cur = obs.get("current") or {}
    players = cur.get("players") or []
    your = _int(cur.get("yourIndex"), -1)
    if len(players) != 2 or your not in (0, 1):
        return None, None
    me, opp = players[your], players[1 - your]
    if me.get("active"):
        return None, None
    # The discovered branch was an early one-prize deficit.  Keep the rule out
    # of late-game sacrifice/promotion decisions.
    if len(me.get("prize") or []) < 5 or len(opp.get("prize") or []) != 5:
        return None, None
    opp_active = opp.get("active") or []
    if len(opp_active) != 1 or _energy_count(opp_active[0]) < 1 or _int(opp_active[0].get("hp"), 0) < 100:
        return None, None

    bench = me.get("bench") or []
    munk_option = None
    for option_index, option in enumerate(opts):
        if _int((option or {}).get("area"), -1) != 5:
            return None, None
        bench_index = _int((option or {}).get("index"), -1)
        if not (0 <= bench_index < len(bench)):
            return None, None
        card = bench[bench_index]
        cid = get_card_id(card)
        hp = _int(card.get("hp"), 0)
        max_hp = _int(card.get("maxHp"), 0)
        if hp <= 0 or hp != max_hp:
            return None, None
        if cid == MUNKIDORI_ID:
            if _energy_count(card) != 1 or munk_option is not None:
                return None, None
            munk_option = option_index
        elif cid == IMPIDIMP_ID:
            if _energy_count(card) != 0:
                return None, None
        else:
            return None, None
    if munk_option is None:
        return None, None
    return [munk_option], [semantic(obs, option) for option in opts]
