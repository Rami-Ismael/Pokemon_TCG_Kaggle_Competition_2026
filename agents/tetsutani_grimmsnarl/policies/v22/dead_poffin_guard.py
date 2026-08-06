from __future__ import annotations

from .manual_observation import get_card_id, semantic

BUDDY_BUDDY_POFFIN_ID = 1086
DARK_ENERGY_ID = 7
IMPIDIMP_ID = 646
SNORUNT_ID = 860
MORGREM_ID = 647
GRIMMSNARL_EX_ID = 648
MUNKIDORI_ID = 112
DECK_TARGET_COUNTS = {IMPIDIMP_ID: 4, SNORUNT_ID: 2}


def _energy_count(card: dict) -> int:
    ecards = (card or {}).get("energyCards") or []
    if ecards:
        return sum(1 for x in ecards if get_card_id(x) == DARK_ENERGY_ID)
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


def _walk_card(card: dict):
    if not isinstance(card, dict):
        return
    yield card
    for prev in card.get("preEvolution") or []:
        yield from _walk_card(prev)


def _known_outside_deck(me: dict) -> dict[int, int]:
    counts = {cid: 0 for cid in DECK_TARGET_COUNTS}
    for zone in ("active", "bench", "hand", "discard"):
        for top in me.get(zone) or []:
            for card in _walk_card(top):
                cid = get_card_id(card)
                if cid in counts:
                    counts[cid] += 1
    return counts


def choose(obs: dict):
    """Skip a provably dead Buddy-Buddy Poffin and bank the attachment.

    This is a public-state conservation rule. It fires only when every Poffin-
    searchable Basic in this fixed deck is already visible outside the deck,
    the game is in a late prize race, a legal attack exists, and the once-per-
    turn manual Dark attachment is still available. It never reasons from an
    opponent id, seed, replay id, or serial number.
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
    me = players[your]
    opp = players[1 - your]
    # Restrict to an actual late prize race; this avoids changing setup turns.
    if len(me.get("prize") or []) > 3 or len(opp.get("prize") or []) > 3:
        return None, None

    known = _known_outside_deck(me)
    if any(known.get(cid, 0) < total for cid, total in DECK_TARGET_COUNTS.items()):
        return None, None

    hand = me.get("hand") or []
    active = me.get("active") or []
    bench = me.get("bench") or []
    poffin_present = False
    attack_present = False
    attachments = []
    # Preserve structural setup actions. The discovered error occurs only
    # after all available evolution has been completed; attaching before an
    # available evolution changed the intended strategic line and regressed.
    if any(int(o.get("type", -1) if o.get("type") is not None else -1) == 9 for o in opts):
        return None, [semantic(obs, option) for option in opts]
    for oi, option in enumerate(opts):
        typ = int(option.get("type", -1) if option.get("type") is not None else -1)
        if typ == 13:
            attack_present = True
            continue
        if typ == 7:
            idx = int(option.get("index", -1) if option.get("index") is not None else -1)
            if 0 <= idx < len(hand) and get_card_id(hand[idx]) == BUDDY_BUDDY_POFFIN_ID:
                poffin_present = True
            continue
        if typ != 8:
            continue
        hidx = int(option.get("index", -1) if option.get("index") is not None else -1)
        if not (0 <= hidx < len(hand)) or get_card_id(hand[hidx]) != DARK_ENERGY_ID:
            continue
        area = int(option.get("inPlayArea", option.get("area", 0)) or 0)
        tidx = int(option.get("inPlayIndex", -1) if option.get("inPlayIndex") is not None else -1)
        if area == 4:
            card = active[0] if tidx == 0 and active else None
        elif area == 5:
            card = bench[tidx] if 0 <= tidx < len(bench) else None
        else:
            card = None
        if not card:
            continue
        cid = get_card_id(card)
        ec = _energy_count(card)
        hp, mhp = _hp(card), _max_hp(card)
        healthy = int(mhp > 0 and hp == mhp)
        # Prefer a healthy, underprepared evolved backup. Then another useful
        # benched attacker; active overfill is the last resort. Deterministic
        # tie-breaking uses public board position only.
        role = {
            GRIMMSNARL_EX_ID: 0,
            MORGREM_ID: 1,
            MUNKIDORI_ID: 2,
            IMPIDIMP_ID: 3,
            SNORUNT_ID: 4,
        }.get(cid, 5)
        is_bench = int(area == 5)
        score = (
            0 if is_bench and healthy and cid in (GRIMMSNARL_EX_ID, MORGREM_ID) and ec < 2 else 1,
            0 if is_bench else 1,
            role,
            ec,
            -hp,
            tidx,
            oi,
        )
        attachments.append((score, oi))

    if not poffin_present or not attack_present or not attachments:
        return None, [semantic(obs, option) for option in opts]
    attachments.sort()
    sems = [semantic(obs, option) for option in opts]
    return [attachments[0][1]], sems
