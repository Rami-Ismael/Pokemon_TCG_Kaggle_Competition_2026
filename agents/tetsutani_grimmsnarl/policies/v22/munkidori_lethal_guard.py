from __future__ import annotations
from .manual_observation import get_card_id, semantic

MUNKIDORI_ID = 112
GRIMMSNARL_EX_ID = 648

def _hp(card):
    try: return int((card or {}).get('hp', 999) or 999)
    except Exception: return 999

def choose(obs):
    """Secure a bench KO with Munkidori while preserving a lethal active target.

    The rule is public-board-state only and intentionally excludes seed,
    opponent identity, replay identity, and physical-card serials.
    """
    sel = obs.get('select') or {}; opts = sel.get('option') or []
    if int(sel.get('context', -1) if sel.get('context') is not None else -1) != 13: return None, None
    if int(sel.get('minCount', 0) or 0) != 1 or int(sel.get('maxCount', 0) or 0) != 1: return None, None
    if get_card_id(sel.get('effect')) != MUNKIDORI_ID: return None, None
    cur = obs.get('current') or {}; players = cur.get('players') or []
    try: your = int(cur.get('yourIndex', 0) or 0)
    except Exception: return None, None
    if len(players) != 2 or your not in (0, 1): return None, None
    me, opp = players[your], players[1-your]
    my_active = me.get('active') or []; opp_active = opp.get('active') or []
    if len(my_active) != 1 or get_card_id(my_active[0]) != GRIMMSNARL_EX_ID: return None, None
    if len(opp_active) != 1 or _hp(opp_active[0]) > 30: return None, None
    if len(opp.get('prize') or []) != 1 or len(me.get('prize') or []) > 4: return None, None
    sems = [semantic(obs, option) for option in opts]
    bench = opp.get('bench') or []; opp_index = 1-your; candidates = []
    for i, option in enumerate(opts):
        if int(option.get('type', -1) if option.get('type') is not None else -1) != 3: continue
        if int(option.get('playerIndex', -1) if option.get('playerIndex') is not None else -1) != opp_index: continue
        if int(option.get('area', 0) or 0) != 5: continue
        j = int(option.get('index', -1) if option.get('index') is not None else -1)
        if not (0 <= j < len(bench)): continue
        card = bench[j]; hp = _hp(card)
        if hp <= 30:
            try: max_hp = int((card or {}).get('maxHp', 0) or 0)
            except Exception: max_hp = 0
            candidates.append((hp, -max_hp, i))
    if not candidates: return None, sems
    candidates.sort()
    return [candidates[0][2]], sems
