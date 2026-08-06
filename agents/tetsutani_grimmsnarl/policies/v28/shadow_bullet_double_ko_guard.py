from __future__ import annotations

from .manual_observation import get_card_id, semantic

MORGREM_ID = 647
GRIMMSNARL_EX_ID = 648
SHADOW_BULLET_DAMAGE = 180
SHADOW_BULLET_BENCH_DAMAGE = 30
DARK_ENERGY_ID = 7


def _hp(card: dict, default: int = 999) -> int:
    try:
        return int((card or {}).get("hp", default) or default)
    except Exception:
        return default


def _dark_energy_count(card: dict) -> int:
    cards = (card or {}).get("energyCards") or []
    if cards:
        return sum(1 for x in cards if get_card_id(x) == DARK_ENERGY_ID)
    return sum(1 for x in ((card or {}).get("energies") or []) if int(x or 0) == DARK_ENERGY_ID)


def choose(obs: dict):
    """Evolve the Active Morgrem before side actions when Shadow Bullet double-KOs.

    This rule uses only public board state. It fires when the Active Morgrem is
    already able to pay Shadow Bullet's two-Dark cost, the opposing Active is
    within 180 damage, and at least one opposing Benched Pokemon is within the
    attack's 30 bench damage. The action chosen is specifically the evolution
    of the Active Morgrem, not a Benched copy.
    """
    sel = obs.get("select") or {}
    opts = sel.get("option") or []
    if int(sel.get("context", -1) if sel.get("context") is not None else -1) != 0:
        return None, None
    if int(sel.get("minCount", 0) or 0) != 1 or int(sel.get("maxCount", 0) or 0) != 1:
        return None, None

    cur = obs.get("current") or {}
    players = cur.get("players") or []
    try:
        your = int(cur.get("yourIndex", 0) or 0)
    except Exception:
        return None, None
    if len(players) != 2 or your not in (0, 1):
        return None, None
    me, opp = players[your], players[1 - your]
    active = me.get("active") or []
    opp_active = opp.get("active") or []
    if len(active) != 1 or get_card_id(active[0]) != MORGREM_ID:
        return None, None
    if _dark_energy_count(active[0]) < 2:
        return None, None
    if len(opp_active) != 1 or not (0 < _hp(opp_active[0]) <= SHADOW_BULLET_DAMAGE):
        return None, None
    if not any(0 < _hp(card) <= SHADOW_BULLET_BENCH_DAMAGE for card in (opp.get("bench") or [])):
        return None, None

    sems = [semantic(obs, option) for option in opts]
    # Require that an attack is legal in the current state; after evolution,
    # Shadow Bullet will also be legal because the Active already has 2 Dark.
    if not any(int(s.get("type", -1)) == 13 for s in sems):
        return None, sems

    candidates = []
    for i, s in enumerate(sems):
        if int(s.get("type", -1)) != 9:
            continue
        if get_card_id({"id": s.get("source_id")}) != GRIMMSNARL_EX_ID:
            continue
        if get_card_id({"id": s.get("target_id")}) != MORGREM_ID:
            continue
        if int(s.get("target_area", 0) or 0) != 4:
            continue
        candidates.append(i)
    if not candidates:
        return None, sems
    return [min(candidates)], sems
