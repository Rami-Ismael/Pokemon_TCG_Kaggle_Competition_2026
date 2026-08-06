"""Grimmsnarl ex / Froslass / Munkidori rule-based agent.

Actions are selected from the current board, the legal option menu, and
within-game tactical state.
"""
from __future__ import annotations

import os
import sys
from collections import Counter

try:
    _ROOT_FILE = __file__
except NameError:
    _ROOT_FILE = None

for _p in ([os.path.dirname(os.path.abspath(_ROOT_FILE))] if _ROOT_FILE else []) + [
    "/kaggle_simulations/agent"
]:
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from cg.api import (
    AreaType,
    CardType,
    OptionType,
    SelectContext,
    all_card_data,
    to_observation_class,
)

# -----------------------------------------------------------------------------
# 1) Card IDs and the exact 60-card deck
# -----------------------------------------------------------------------------
DARK = 7
FROSLASS = 104
MUNKIDORI = 112
IMPIDIMP = 646
MORGREM = 647
GRIMMSNARL = 648
SNORUNT = 860
RARE_CANDY = 1079
UNFAIR_STAMP = 1080
POFFIN = 1086
NIGHT_STRETCHER = 1097
POKEGEAR = 1122
TOOL_SCRAPPER = 1137
POKE_PAD = 1152
BOSS = 1182
PETREL = 1219
LILLIE = 1227
DAWN = 1231
SPIKEMUTH = 1259

FILCH = 934
CORKSCREW_IMPIDIMP = 935
CORKSCREW_MORGREM = 936
SHADOW_BULLET = 937

DECK = [
    7,7,7,7,7,7,7,7,7,7,
    104,104,
    112,112,112,112,
    646,646,646,646,
    647,647,647,
    648,648,648,
    860,860,
    1079,1079,1079,
    1080,
    1086,1086,1086,1086,
    1097,1097,1097,
    1122,
    1137,
    1152,1152,1152,1152,
    1182,1182,
    1219,1219,1219,1219,
    1227,1227,1227,1227,
    1231,
    1259,1259,1259,1259,
]

# -----------------------------------------------------------------------------
# 2) Cached card metadata and small policy state
# -----------------------------------------------------------------------------
CARD_DB = {c.cardId: c for c in all_card_data()}
MARNIE_LINE = {IMPIDIMP, MORGREM, GRIMMSNARL}
BASIC_SETUP = {IMPIDIMP, MUNKIDORI, SNORUNT}
TRAINERS = {
    RARE_CANDY, UNFAIR_STAMP, POFFIN, NIGHT_STRETCHER, POKEGEAR,
    TOOL_SCRAPPER, POKE_PAD, BOSS, PETREL, LILLIE, DAWN, SPIKEMUTH,
}

_LAST_TURN = None
_LAST_EPISODE_STEP = None
_MOVE_COUNT = 3


# -----------------------------------------------------------------------------
# 3) Observation helpers: turn nested engine data into simple questions
# -----------------------------------------------------------------------------
def _reset_state():
    global _LAST_TURN, _LAST_EPISODE_STEP, _MOVE_COUNT
    _LAST_TURN = None
    _LAST_EPISODE_STEP = None
    _MOVE_COUNT = 3


def _read_deck():
    return list(DECK)


def _me(obs):
    return obs.current.players[obs.current.yourIndex]


def _opp(obs):
    return obs.current.players[1 - obs.current.yourIndex]


def _all_my(obs):
    ps = _me(obs)
    return [p for p in list(ps.active) + list(ps.bench) if p is not None]


def _all_opp(obs):
    ps = _opp(obs)
    return [p for p in list(ps.active) + list(ps.bench) if p is not None]


def _active(obs):
    ps = _me(obs)
    return ps.active[0] if ps.active else None


def _opp_active(obs):
    ps = _opp(obs)
    return ps.active[0] if ps.active else None


def _hand(obs):
    return [c for c in (_me(obs).hand or []) if c is not None]


def _hand_ids(obs):
    return [c.id for c in _hand(obs)]


def _discard(obs):
    return [c for c in (_me(obs).discard or []) if c is not None]


def _discard_ids(obs):
    return [c.id for c in _discard(obs)]


def _count_in_play(obs, cid):
    return sum(1 for p in _all_my(obs) if p.id == cid)


def _count_line(obs, ids):
    return sum(1 for p in _all_my(obs) if p.id in ids)


def _energy_count(p):
    if p is None:
        return 0
    cards = getattr(p, "energyCards", None)
    if cards is not None:
        return len(cards)
    return len(getattr(p, "energies", []) or [])


def _damage(p):
    if p is None:
        return 0
    return max(0, int(getattr(p, "maxHp", 0) or 0) - int(getattr(p, "hp", 0) or 0))


def _remaining_hp(p):
    return int(getattr(p, "hp", 0) or 0) if p is not None else 0


def _prize_value(p):
    if p is None:
        return 0
    data = CARD_DB.get(p.id)
    if data is None:
        return 1
    return 3 if getattr(data, "megaEx", False) else (2 if getattr(data, "ex", False) else 1)


def _has_ability(p):
    data = CARD_DB.get(p.id) if p else None
    return bool(getattr(data, "skills", []) or []) if data else False


def _bench_free(obs):
    ps = _me(obs)
    return max(0, int(ps.benchMax) - len(ps.bench))


def _zone_card(obs, area, index, player_index=None):
    if area is None or index is None:
        return None
    yi = obs.current.yourIndex
    pi = yi if player_index is None else int(player_index)
    if area == AreaType.DECK:
        arr = obs.select.deck or []
    elif area == AreaType.HAND:
        arr = obs.current.players[pi].hand or []
    elif area == AreaType.DISCARD:
        arr = obs.current.players[pi].discard or []
    elif area == AreaType.ACTIVE:
        arr = obs.current.players[pi].active or []
    elif area == AreaType.BENCH:
        arr = obs.current.players[pi].bench or []
    elif area == AreaType.PRIZE:
        arr = obs.current.players[pi].prize or []
    elif area == AreaType.STADIUM:
        arr = obs.current.stadium or []
    elif area == AreaType.LOOKING:
        arr = obs.current.looking or []
    else:
        arr = []
    return arr[index] if isinstance(index, int) and 0 <= index < len(arr) else None


def _source_card(obs, opt):
    yi = obs.current.yourIndex
    pi = opt.playerIndex if getattr(opt, "playerIndex", None) is not None else yi
    if opt.type == OptionType.PLAY:
        return _zone_card(obs, AreaType.HAND, opt.index, pi)
    return _zone_card(obs, getattr(opt, "area", None), getattr(opt, "index", None), pi)


def _target_card(obs, opt):
    area = getattr(opt, "inPlayArea", None)
    index = getattr(opt, "inPlayIndex", None)
    if opt.type == OptionType.ABILITY and area is None:
        area = getattr(opt, "area", None)
        index = getattr(opt, "index", None)
    return _zone_card(obs, area, index, obs.current.yourIndex)


def _option_card_for_context(obs, opt):
    return _source_card(obs, opt)


def _stadium_id(obs):
    stadium = obs.current.stadium or []
    return stadium[0].id if stadium and stadium[0] is not None else 0


def _has_ready_grimmsnarl(obs):
    return any(p.id == GRIMMSNARL and _energy_count(p) >= 2 for p in _all_my(obs))


def _needs_impidimp(obs):
    return _count_in_play(obs, IMPIDIMP) + _count_in_play(obs, MORGREM) + _count_in_play(obs, GRIMMSNARL) < 2


def _needs_snorunt(obs):
    return _count_in_play(obs, SNORUNT) + _count_in_play(obs, FROSLASS) < 1


def _needs_munkidori(obs):
    return _count_in_play(obs, MUNKIDORI) < 2


def _munkidori_without_energy(obs):
    return [p for p in _all_my(obs) if p.id == MUNKIDORI and _energy_count(p) == 0]


def _mature_impidimp_exists(obs):
    return any(p.id == IMPIDIMP and not getattr(p, "appearThisTurn", False) for p in _all_my(obs))


def _mature_morgrem_exists(obs):
    return any(p.id == MORGREM and not getattr(p, "appearThisTurn", False) for p in _all_my(obs))


def _opponent_tools(obs):
    return sum(len(getattr(p, "tools", []) or []) for p in _all_opp(obs))


def _ko_target_exists(obs, damage=180):
    return any(_remaining_hp(p) <= damage for p in _all_opp(obs))


def _boss_ko_exists(obs, damage=180):
    return any(_remaining_hp(p) <= damage for p in (_opp(obs).bench or []) if p is not None)


# -----------------------------------------------------------------------------
# 4) Card-specific search and recovery priorities
# -----------------------------------------------------------------------------
def _search_priority(obs, cid, effect_id):
    hand = Counter(_hand_ids(obs))
    discard = Counter(_discard_ids(obs))
    inplay = Counter(p.id for p in _all_my(obs))

    if effect_id == SPIKEMUTH:
        if cid == GRIMMSNARL:
            score = 180
            if _mature_morgrem_exists(obs): score += 1250
            if _mature_impidimp_exists(obs) and hand[RARE_CANDY]: score += 1100
            if hand[GRIMMSNARL]: score -= 700
            return score
        if cid == MORGREM:
            score = 220
            if _mature_impidimp_exists(obs): score += 1200
            if hand[MORGREM]: score -= 500
            return score
        if cid == IMPIDIMP:
            return 1350 if _needs_impidimp(obs) else 80

    if effect_id == POKE_PAD:
        if cid == FROSLASS:
            return 1280 if inplay[SNORUNT] else (650 if _needs_snorunt(obs) else 120)
        if cid == MORGREM:
            return 1220 if inplay[IMPIDIMP] else 180
        if cid == IMPIDIMP:
            return 1180 if _needs_impidimp(obs) else 140
        if cid == MUNKIDORI:
            return 1100 if _needs_munkidori(obs) else 120
        if cid == SNORUNT:
            return 900 if _needs_snorunt(obs) else 100

    if effect_id == NIGHT_STRETCHER:
        grim_count = _count_in_play(obs, GRIMMSNARL)
        line_count = _count_line(obs, MARNIE_LINE)
        attacker_emergency = grim_count == 0 or not _has_ready_grimmsnarl(obs)
        if cid == GRIMMSNARL:
            return 1500 if attacker_emergency and (inplay[MORGREM] or (_mature_impidimp_exists(obs) and hand[RARE_CANDY])) else (1020 if inplay[MORGREM] else 520)
        if cid == MORGREM:
            return 1380 if attacker_emergency and inplay[IMPIDIMP] else (820 if inplay[IMPIDIMP] else 260)
        if cid == IMPIDIMP:
            return 1280 if line_count < 2 and _bench_free(obs) > 0 else 180
        if cid == DARK:
            return 1120 if not attacker_emergency and _munkidori_without_energy(obs) else (820 if not attacker_emergency else 420)
        if cid == MUNKIDORI:
            return 900 if _needs_munkidori(obs) and line_count >= 2 else 320
        if cid == FROSLASS:
            return 760 if inplay[SNORUNT] and line_count >= 2 else 220
        if cid == SNORUNT:
            return 620 if _needs_snorunt(obs) and line_count >= 2 else 160

    if effect_id == PETREL:
        if cid == UNFAIR_STAMP: return 1180 if obs.current.turn >= 3 and len(_hand(obs)) <= 6 else 520
        if cid == RARE_CANDY:
            return 1120 if _mature_impidimp_exists(obs) else 520
        if cid == NIGHT_STRETCHER:
            useful = any(x in discard for x in (DARK, GRIMMSNARL, MUNKIDORI, IMPIDIMP, MORGREM, FROSLASS))
            return 1080 if useful else 500
        if cid == POKE_PAD:
            return 1040 if (_needs_munkidori(obs) or _needs_impidimp(obs) or _needs_snorunt(obs)) else 600
        if cid == SPIKEMUTH:
            return 1000 if _stadium_id(obs) != SPIKEMUTH else 400
        if cid == TOOL_SCRAPPER:
            return 980 if _opponent_tools(obs) else 100
        if cid == BOSS:
            return 960 if _boss_ko_exists(obs) else 450
        if cid == LILLIE:
            return 900 if len(_hand(obs)) <= 4 else 350
        if cid == POFFIN:
            return 860 if _bench_free(obs) and (_needs_impidimp(obs) or _needs_snorunt(obs)) else 300
        if cid == POKEGEAR: return 500
        if cid == DAWN: return 780 if obs.current.turn <= 4 else 350

    if effect_id == DAWN:
        if cid == GRIMMSNARL: return 1000 if (inplay[MORGREM] or inplay[IMPIDIMP]) else 650
        if cid == MORGREM: return 950 if inplay[IMPIDIMP] else 550
        if cid == FROSLASS: return 900 if inplay[SNORUNT] else 500
        if cid == MUNKIDORI: return 850 if _needs_munkidori(obs) else 400
        if cid == IMPIDIMP: return 820 if _needs_impidimp(obs) else 350
        if cid == SNORUNT: return 760 if _needs_snorunt(obs) else 300

    if effect_id == POKEGEAR:
        if cid == BOSS: return 1000 if _boss_ko_exists(obs) else 500
        if cid == PETREL: return 920
        if cid == DAWN: return 880 if obs.current.turn <= 4 else 450
        if cid == LILLIE: return 850 if len(_hand(obs)) <= 5 else 400

    # Generic search / reveal priority.
    generic = {
        GRIMMSNARL: 950, MORGREM: 880, MUNKIDORI: 850, IMPIDIMP: 820,
        FROSLASS: 780, SNORUNT: 700, RARE_CANDY: 900, DARK: 650,
        UNFAIR_STAMP: 840, NIGHT_STRETCHER: 800, POKE_PAD: 760,
        PETREL: 740, SPIKEMUTH: 720, BOSS: 700, LILLIE: 650,
    }
    return generic.get(cid, 100)


def _setup_score(obs, card, bench=False):
    if card is None:
        return -10_000
    cid = card.id
    if not bench:
        return {IMPIDIMP: 1000, MUNKIDORI: 700, SNORUNT: 500}.get(cid, 0)
    if cid == MUNKIDORI:
        return 950 if _count_in_play(obs, MUNKIDORI) == 0 else -50
    if cid == IMPIDIMP:
        return 900 if _count_line(obs, MARNIE_LINE) < 2 else -40
    if cid == SNORUNT:
        return 720 if _count_in_play(obs, SNORUNT) + _count_in_play(obs, FROSLASS) == 0 else -80
    return -100


def _switch_score(obs, p):
    if p is None:
        return -10000
    e = _energy_count(p)
    if p.id == GRIMMSNARL:
        return 3000 + 400 * min(e, 3) + _remaining_hp(p)
    if p.id == MORGREM:
        return 2200 + 250 * e
    if p.id == IMPIDIMP:
        return 1900 + 200 * e
    if p.id == MUNKIDORI:
        return 1300 + 100 * e
    if p.id == FROSLASS:
        return 900
    if p.id == SNORUNT:
        return 700
    return 100


def _damage_target_score(p, amount, active=False):
    if p is None:
        return -10000
    hp = _remaining_hp(p)
    prize = _prize_value(p)
    score = prize * 220 + max(0, 1400 - 4 * hp)
    if hp <= amount:
        score += 14000 + prize * 1800
    elif hp <= amount + 30:
        score += 1800
    if active:
        score += 120
    if _has_ability(p):
        score += 240
    return score


def _discard_score(obs, card):
    if card is None:
        return -10000
    cid = card.id
    hand = Counter(_hand_ids(obs))
    if cid == DARK:
        return 900 if hand[DARK] >= 2 else 150
    if cid == SPIKEMUTH:
        return 850 if _stadium_id(obs) == SPIKEMUTH or hand[SPIKEMUTH] >= 2 else 200
    if cid == IMPIDIMP:
        return 700 if _count_line(obs, MARNIE_LINE) >= 2 else 100
    if cid == SNORUNT:
        return 650 if _count_in_play(obs, SNORUNT) + _count_in_play(obs, FROSLASS) >= 1 else 100
    if cid == MORGREM:
        return 600 if hand[MORGREM] >= 2 else 100
    if cid == GRIMMSNARL:
        return 550 if hand[GRIMMSNARL] >= 2 else 50
    if cid == LILLIE:
        return 500 if hand[LILLIE] >= 2 else 150
    if cid in (POKEGEAR, TOOL_SCRAPPER):
        return 450
    if cid in (MUNKIDORI, RARE_CANDY, UNFAIR_STAMP, BOSS):
        return 50
    return 300


# -----------------------------------------------------------------------------
# 5) Context-specific decisions: search, evolve, attach, move damage, discard
# -----------------------------------------------------------------------------
def _score_non_main(obs, opt):
    global _MOVE_COUNT
    ctx = obs.select.context
    effect = getattr(obs.select, "effect", None)
    effect_id = effect.id if effect is not None else 0
    card = _option_card_for_context(obs, opt)

    if ctx == SelectContext.IS_FIRST:
        return 1000 if opt.type == OptionType.YES else 0
    if ctx in (SelectContext.ACTIVATE, SelectContext.FIRST_EFFECT):
        return 1000 if opt.type == OptionType.YES else 0
    if ctx == SelectContext.SETUP_ACTIVE_POKEMON:
        return _setup_score(obs, card, False)
    if ctx == SelectContext.SETUP_BENCH_POKEMON:
        return _setup_score(obs, card, True)
    if ctx == SelectContext.TO_BENCH:
        return _setup_score(obs, card, True)
    if ctx in (SelectContext.TO_ACTIVE, SelectContext.SWITCH):
        if effect_id == BOSS:
            return _damage_target_score(card, 180, False)
        return _switch_score(obs, card)
    if ctx == SelectContext.TO_HAND:
        return _search_priority(obs, card.id if card else 0, effect_id)
    if ctx == SelectContext.EVOLVE:
        target = _target_card(obs, opt)
        score = 1000
        if card and card.id == GRIMMSNARL: score += 3000
        if target and target.id == IMPIDIMP:
            score += 1000 + 250 * _energy_count(target)
            if target is _active(obs): score += 700
            if getattr(target, "appearThisTurn", False): score -= 5000
        return score
    if ctx == SelectContext.ATTACH_TO and effect_id == GRIMMSNARL:
        # Select every Darkness Energy offered by Punk Up.
        return 1000
    if ctx == SelectContext.ATTACH_FROM and effect_id == GRIMMSNARL:
        p = card
        if p is None or p.id not in MARNIE_LINE:
            return -1000
        active_bonus = 900 if p is _active(obs) else 0
        target_energy = 2
        deficit = target_energy - _energy_count(p)
        stage_bonus = {GRIMMSNARL: 700, MORGREM: 450, IMPIDIMP: 300}.get(p.id, 0)
        return 2000 * deficit + active_bonus + stage_bonus
    if ctx == SelectContext.REMOVE_DAMAGE_COUNTER:
        p = card
        return 1000 * min(3, _damage(p) // 10) + 300 * _prize_value(p) + (100 if p and p.id == MUNKIDORI else 0)
    if ctx == SelectContext.REMOVE_DAMAGE_COUNTER_COUNT:
        return int(getattr(opt, "number", 0) or 0) * 1000
    if ctx == SelectContext.DAMAGE_COUNTER:
        # Adrena-Brain target.
        p = card
        own_index = obs.current.yourIndex
        active = bool(p is not None and p in (_opp(obs).active or []))
        return _damage_target_score(p, 10 * _MOVE_COUNT, active)
    if ctx == SelectContext.DAMAGE:
        # Shadow Bullet's 30 bench damage.
        p = card
        return _damage_target_score(p, 30, False)
    if ctx == SelectContext.DISCARD:
        return _discard_score(obs, card)
    if ctx == SelectContext.DISCARD_TOOL_CARD:
        # Prefer tools on high-prize opponent Pokémon.
        target = card
        return 1000 + 400 * _prize_value(target)
    if ctx == SelectContext.DISCARD_ENERGY:
        # Usually a retreat payment. Prefer excess energy on lower-value lines.
        p = card
        return {GRIMMSNARL: 900, MORGREM: 800, MUNKIDORI: 500, FROSLASS: 400, SNORUNT: 300, IMPIDIMP: 250}.get(p.id if p else 0, 100)
    if ctx == SelectContext.SKILL_ORDER:
        return 1000
    if ctx in (SelectContext.DRAW_COUNT, SelectContext.DAMAGE_COUNTER_COUNT):
        return int(getattr(opt, "number", 0) or 0) * 1000
    if ctx == SelectContext.REMOVE_DAMAGE_COUNTER_COUNT:
        return int(getattr(opt, "number", 0) or 0) * 1000
    if ctx in (SelectContext.EVOLVES_FROM, SelectContext.EVOLVES_TO):
        return 1000
    if opt.type == OptionType.YES:
        return 1000
    if opt.type == OptionType.NO:
        return 0
    if card is not None:
        return 100
    return 0


# -----------------------------------------------------------------------------
# 6) Main-phase planner: compare every currently legal action
# -----------------------------------------------------------------------------
def _main_score(obs, opt):
    card = _source_card(obs, opt)
    target = _target_card(obs, opt)
    cid = card.id if card is not None else 0
    typ = opt.type
    hand_n = len(_hand(obs))
    turn = int(obs.current.turn)
    ready = _has_ready_grimmsnarl(obs)

    if typ == OptionType.ABILITY:
        if card is not None and card.id == MUNKIDORI:
            return 112000
        if card is not None and card.id == SPIKEMUTH:
            useful = _needs_impidimp(obs) or _mature_impidimp_exists(obs) or _mature_morgrem_exists(obs)
            return 106000 if useful else 18000
        return 102000

    if typ == OptionType.EVOLVE:
        if cid == GRIMMSNARL:
            return 101000 + (1800 if target is _active(obs) else 0)
        if cid == MORGREM:
            return 97000 + (1200 if target is _active(obs) else 0)
        if cid == FROSLASS:
            return 92000
        return 70000

    if typ == OptionType.ATTACH:
        if cid != DARK:
            return 1000
        if target is not None and target.id == MUNKIDORI and _energy_count(target) == 0:
            return 96000
        if target is not None and target.id == GRIMMSNARL:
            need = max(0, 2 - _energy_count(target))
            return 90000 + 3500 * need if need else 24000
        if target is not None and target.id in (IMPIDIMP, MORGREM):
            need = max(0, 2 - _energy_count(target))
            return 82000 + 3000 * need if need else 18000
        if target is not None and target.id in (FROSLASS, SNORUNT):
            return 30000
        return 20000

    if typ == OptionType.PLAY:
        if cid == UNFAIR_STAMP:
            return 95000
        if cid == RARE_CANDY:
            return 94000
        if cid == BOSS:
            return 94000 if _boss_ko_exists(obs) else (32000 if ready else 12000)
        if cid == POFFIN:
            useful = _bench_free(obs) > 0 and (_needs_impidimp(obs) or _needs_snorunt(obs))
            return (93000 if turn <= 5 else 62000) if useful else 8000
        if cid == POKE_PAD:
            useful = _needs_munkidori(obs) or _needs_impidimp(obs) or _needs_snorunt(obs) or _count_in_play(obs, SNORUNT)
            return 88000 if useful else 38000
        if cid == PETREL:
            return 87000 if not obs.current.supporterPlayed else 10000
        if cid == NIGHT_STRETCHER:
            useful = any(x in _discard_ids(obs) for x in (DARK, GRIMMSNARL, MUNKIDORI, IMPIDIMP, MORGREM, FROSLASS, SNORUNT))
            return 86000 if useful else 18000
        if cid == DAWN:
            return 85000 if turn <= 5 and not obs.current.supporterPlayed else 22000
        if cid == POKEGEAR:
            return 83000 if not obs.current.supporterPlayed else 16000
        if cid == LILLIE:
            if obs.current.supporterPlayed:
                return 8000
            return 99000 if hand_n <= 3 else (90000 if hand_n <= 5 else (48000 if hand_n <= 7 else 9000))
        if cid == SPIKEMUTH:
            return 81000 if _stadium_id(obs) != SPIKEMUTH else 12000
        if cid == TOOL_SCRAPPER:
            return 80000 if _opponent_tools(obs) else 5000
        if cid == MUNKIDORI:
            return 79000 if _bench_free(obs) and _needs_munkidori(obs) else 22000
        if cid == IMPIDIMP:
            return 78000 if _bench_free(obs) and _needs_impidimp(obs) else 18000
        if cid == SNORUNT:
            return 70000 if _bench_free(obs) and _needs_snorunt(obs) else 12000
        return 10000

    if typ == OptionType.RETREAT:
        active = _active(obs)
        if active is not None and active.id != GRIMMSNARL and any(p.id == GRIMMSNARL and _energy_count(p) >= 2 for p in (_me(obs).bench or [])):
            return 52000
        if active is not None and active.id in (FROSLASS, SNORUNT, MUNKIDORI) and any(p.id in (IMPIDIMP, MORGREM) for p in (_me(obs).bench or [])):
            return 26000
        return 5000

    if typ == OptionType.ATTACK:
        aid = int(getattr(opt, "attackId", 0) or 0)
        if aid == SHADOW_BULLET:
            return 30000 + (8000 if _ko_target_exists(obs, 180) else 0)
        if aid == CORKSCREW_MORGREM:
            return 18000
        if aid == FILCH:
            return 16000 if len(_hand(obs)) <= 6 else 9000
        if aid == CORKSCREW_IMPIDIMP:
            return 8000
        return 10000

    if typ == OptionType.END:
        return 0
    return -1000


# -----------------------------------------------------------------------------
# 7) Select the best-scoring legal option(s)
# -----------------------------------------------------------------------------
def _select_scored(obs, score_fn):
    global _MOVE_COUNT
    sel = obs.select
    opts = list(sel.option or [])
    n = len(opts)
    min_count = max(0, min(int(sel.minCount), n))
    max_count = max(min_count, min(int(sel.maxCount), n))
    if n == 0 or max_count == 0:
        return []
    scored = [(score_fn(obs, o), i) for i, o in enumerate(opts)]
    scored.sort(key=lambda x: (-x[0], x[1]))

    if sel.context == SelectContext.MAIN:
        chosen = [scored[0][1]] if max_count else []
    else:
        # Optional searches/bench placement may deliberately choose fewer cards.
        positive = [i for score, i in scored if score > 0]
        chosen = positive[:max_count]
        if len(chosen) < min_count:
            chosen = [i for _, i in scored[:min_count]]
        chosen = sorted(chosen)

    if sel.context == SelectContext.REMOVE_DAMAGE_COUNTER_COUNT and chosen:
        opt = opts[chosen[0]]
        _MOVE_COUNT = max(1, int(getattr(opt, "number", 1) or 1))
    return chosen


# -----------------------------------------------------------------------------
# 8) Final legality guard: never return an invalid index list
# -----------------------------------------------------------------------------
def _legalize(action, sel):
    n = len(sel.option or [])
    min_count = max(0, min(int(sel.minCount), n))
    max_count = max(min_count, min(int(sel.maxCount), n))
    out = []
    seen = set()
    for x in action or []:
        if isinstance(x, int) and 0 <= x < n and x not in seen:
            out.append(x); seen.add(x)
        if len(out) >= max_count:
            break
    if len(out) < min_count:
        for i in range(n):
            if i not in seen:
                out.append(i); seen.add(i)
            if len(out) >= min_count:
                break
    return sorted(out[:max_count])


# -----------------------------------------------------------------------------
# 9) Kaggle entry point and safe fallback
# -----------------------------------------------------------------------------
def _agent_impl(obs_dict):
    global _LAST_TURN, _LAST_EPISODE_STEP
    if not obs_dict or obs_dict.get("select") is None:
        _reset_state()
        return _read_deck()
    obs = to_observation_class(obs_dict)
    turn = int(obs.current.turn)
    step = int(obs_dict.get("step", 0) or 0)
    if _LAST_EPISODE_STEP is not None and step < _LAST_EPISODE_STEP:
        _reset_state()
    _LAST_EPISODE_STEP = step
    if _LAST_TURN is None or turn != _LAST_TURN:
        _LAST_TURN = turn
    try:
        if obs.select.context == SelectContext.MAIN:
            action = _select_scored(obs, _main_score)
        else:
            action = _select_scored(obs, _score_non_main)
        return _legalize(action, obs.select)
    except Exception:
        return _legalize([], obs.select)


# Keep agent as the final callable in the raw execution namespace.
def agent(obs_dict):
    return _agent_impl(obs_dict)
