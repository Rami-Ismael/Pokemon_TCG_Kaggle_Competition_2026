from __future__ import annotations

from .manual_observation import get_card_id, semantic

DARK_ENERGY_ID = 7
UNFAIR_STAMP_ID = 1080
IMPIDIMP_ID = 646
MORGREM_ID = 647
GRIMMSNARL_EX_ID = 648
ATTACKER_IDS = {IMPIDIMP_ID, MORGREM_ID, GRIMMSNARL_EX_ID}


def _dark_energy_count(card: dict) -> int:
    cards = (card or {}).get("energyCards") or []
    if cards:
        return sum(1 for x in cards if get_card_id(x) == DARK_ENERGY_ID)
    return sum(1 for x in ((card or {}).get("energies") or []) if int(x or 0) == DARK_ENERGY_ID)


def _hp(card: dict) -> int:
    try:
        return int((card or {}).get("hp", 0) or 0)
    except Exception:
        return 0


def _max_hp(card: dict) -> int:
    try:
        return int((card or {}).get("maxHp", 0) or 0)
    except Exception:
        return 0


def choose(obs: dict):
    """Bank a manual attachment before Unfair Stamp shuffles the hand.

    The guard is deliberately late-game and backup-specific. It fires only
    when the Active evolution line is already attack-ready, an attack is
    currently legal, Unfair Stamp is playable, and a healthy Benched
    Impidimp/Morgrem/Grimmsnarl ex with exactly one Dark Energy can be
    completed to the deck's two-Energy attack threshold. Known recorded-policy
    states are excluded by the caller.
    """
    sel = obs.get("select") or {}
    opts = sel.get("option") or []
    if int(sel.get("context", -1) if sel.get("context") is not None else -1) != 0:
        return None, None
    if int(sel.get("minCount", 0) or 0) != 1 or int(sel.get("maxCount", 0) or 0) != 1:
        return None, None

    cur = obs.get("current") or {}
    if bool(cur.get("energyAttached")):
        return None, None
    players = cur.get("players") or []
    try:
        your = int(cur.get("yourIndex", 0) or 0)
    except Exception:
        return None, None
    if len(players) != 2 or your not in (0, 1):
        return None, None
    me, opp = players[your], players[1 - your]
    if len(me.get("prize") or []) > 3 or len(opp.get("prize") or []) > 3:
        return None, None

    active = me.get("active") or []
    bench = me.get("bench") or []
    hand = me.get("hand") or []
    if len(active) != 1 or get_card_id(active[0]) not in ATTACKER_IDS:
        return None, None
    if _dark_energy_count(active[0]) < 2:
        return None, None

    sems = [semantic(obs, option) for option in opts]
    if not any(int(s.get("type", -1)) == 13 for s in sems):
        return None, sems
    # Structural setup takes precedence over this sequencing repair.
    if any(int(s.get("type", -1)) == 9 for s in sems):
        return None, sems

    stamp_present = any(
        int(s.get("type", -1)) == 7 and int(s.get("source_id", 0) or 0) == UNFAIR_STAMP_ID
        for s in sems
    )
    if not stamp_present:
        return None, sems

    candidates = []
    for i, option in enumerate(opts):
        s = sems[i]
        if int(s.get("type", -1)) != 8 or int(s.get("source_id", 0) or 0) != DARK_ENERGY_ID:
            continue
        if int(s.get("target_area", 0) or 0) != 5:
            continue
        bidx = int(s.get("inplay_index", s.get("inPlayIndex", -1)) if s.get("inplay_index", s.get("inPlayIndex", -1)) is not None else -1)
        if not (0 <= bidx < len(bench)):
            continue
        card = bench[bidx]
        cid = get_card_id(card)
        if cid not in ATTACKER_IDS or _dark_energy_count(card) != 1:
            continue
        hp, mhp = _hp(card), _max_hp(card)
        if mhp <= 0 or hp != mhp:
            continue
        role = {GRIMMSNARL_EX_ID: 0, MORGREM_ID: 1, IMPIDIMP_ID: 2}.get(cid, 3)
        candidates.append(((role, -hp, bidx, i), i))
    if not candidates:
        return None, sems
    candidates.sort()
    return [candidates[0][1]], sems
