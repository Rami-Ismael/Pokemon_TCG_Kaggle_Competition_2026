from __future__ import annotations

from .manual_observation import card_meta, get_card_id, resolve_source, semantic

BOSS_ORDERS_ID = 1182
MUNKIDORI_ID = 112
MORGREM_ID = 647
DARK_ENERGY_ID = 7


def _hp(card: dict, default: int = 999) -> int:
    try:
        return int((card or {}).get("hp", default) or default)
    except Exception:
        return default


def _max_hp(card: dict, default: int = 999) -> int:
    try:
        return int((card or {}).get("maxHp", default) or default)
    except Exception:
        return default


def _dark_count(card: dict) -> int:
    cards = (card or {}).get("energyCards") or []
    if cards:
        return sum(1 for x in cards if get_card_id(x) == DARK_ENERGY_ID)
    return sum(1 for x in ((card or {}).get("energies") or []) if int(x or 0) == DARK_ENERGY_ID)


def _energy_count(card: dict) -> int:
    cards = (card or {}).get("energyCards") or []
    if cards:
        return len(cards)
    return len((card or {}).get("energies") or [])


def choose(obs: dict):
    """Gust a damaged, stranded Basic instead of an undamaged stall target.

    The guard is intentionally narrow and public-state only. It applies to an
    early Boss's Orders target selection when our Active Munkidori has only one
    Dark Energy (so it is not yet an attacker), a one-Energy Morgrem is being
    prepared on the Bench, and the opponent exposes a damaged <=40-HP,
    zero-Energy Basic. Pulling that target Active denies immediate evolution
    and creates a much cheaper future prize than selecting an undamaged target.
    Known exact validated states are excluded by the caller.
    """
    sel = obs.get("select") or {}
    opts = sel.get("option") or []
    if int(sel.get("context", -1) if sel.get("context") is not None else -1) != 3:
        return None, None
    if get_card_id(sel.get("effect")) != BOSS_ORDERS_ID:
        return None, None
    if int(sel.get("minCount", 0) or 0) != 1 or int(sel.get("maxCount", 0) or 0) != 1:
        return None, None

    cur = obs.get("current") or {}
    turn = int(cur.get("turn", 0) or 0)
    if not (2 <= turn <= 6):
        return None, None
    players = cur.get("players") or []
    try:
        your = int(cur.get("yourIndex", 0) or 0)
    except Exception:
        return None, None
    if len(players) != 2 or your not in (0, 1):
        return None, None
    me, opp = players[your], players[1 - your]
    if len(me.get("prize") or []) != 6 or len(opp.get("prize") or []) != 6:
        return None, None

    active = me.get("active") or []
    if len(active) != 1 or get_card_id(active[0]) != MUNKIDORI_ID or _dark_count(active[0]) != 1:
        return None, None
    if not any(get_card_id(c) == MORGREM_ID and _dark_count(c) == 1 for c in (me.get("bench") or [])):
        return None, None

    opp_active = opp.get("active") or []
    if len(opp_active) != 1 or _energy_count(opp_active[0]) != 0 or _hp(opp_active[0]) < 60:
        return None, None

    sems = [semantic(obs, option) for option in opts]
    candidates = []
    for i, option in enumerate(opts):
        target, area, _ = resolve_source(obs, option)
        if area != 5 or not isinstance(target, dict):
            continue
        hp, mhp = _hp(target), _max_hp(target)
        if not (0 < hp <= 40 and mhp <= 70 and hp < mhp):
            continue
        if _energy_count(target) != 0:
            continue
        stage, _, _, _, _ = card_meta(get_card_id(target))
        if int(stage) != 2:  # manual_observation: Basic Pokemon stage code
            continue
        candidates.append(((hp, mhp, get_card_id(target), i), i))
    if not candidates:
        return None, sems
    candidates.sort()
    return [candidates[0][1]], sems
