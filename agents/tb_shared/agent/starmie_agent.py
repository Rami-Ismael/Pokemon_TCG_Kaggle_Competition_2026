"""Starmie / Froslass — rule pilot (gold-medal pattern).

Based on ashleysandlin Limitless list + community write-up:
  generic scoring, matchup modules, finish-mode cg search, PrizeTracker for deck search.

Deck: env STARMIE_DECK, deck.csv in cwd, or repo default.
"""

from __future__ import annotations

import os
import random
import sys

try:
    ROOT = __file__
except NameError:
    ROOT = None
CG_PATH = "/kaggle_simulations/agent"
for p in ([os.path.dirname(os.path.abspath(ROOT))] if ROOT else []) + [CG_PATH]:
    if p and p not in sys.path and os.path.isdir(p):
        sys.path.insert(0, p)

from cg.api import (
    AreaType,
    CardType,
    LogType,
    OptionType,
    SelectContext,
    all_card_data,
    to_observation_class,
)

try:
    from cg.api import all_attack
    ALL_ATTACKS = {a.attackId: a for a in all_attack()}
except Exception:
    ALL_ATTACKS = {}

try:
    from agent.finish_search import try_cg_search
    from agent.prize_tracker import PrizeTracker
except ImportError:
    from finish_search import try_cg_search
    from prize_tracker import PrizeTracker

# ── Card IDs ──

STARYU = 1030
MEGA_STARMIE_EX = 1031
SNORUNT = 860
MEGA_FROSLASS_EX = 861
FROSLASS_BASIC = 104
MUNKIDORI = 112
CINDERACE = 666

WATER_ENERGY = 3
DARK_ENERGY = 7
LEGACY_ENERGY = 12
IGNITION_ENERGY = 17
MIST_ENERGY = 11

LILLIE = 1227
SALVATORE = 1189
WALLY = 1229
HILDA = 1225
BOSS = 1182
BLACK_BELT = 1211
POFFIN = 1086
ENERGY_SEARCH = 1119
POKEGEAR = 1122
MEGA_SIGNAL = 1145
SWITCH = 1123
NIGHT_STRETCHER = 1097
GRAVITY_MOUNTAIN = 1252
CRUSHING_HAMMER = 1120
HARLEQUIN = 1223
HERO_CAPE = 1159
ULTRA_BALL = 1121
RISKY_RUINS = 1260
POKE_PAD = 1152

JETTING_BLOW = 1487
NEBULA_BEAM = 1488
RESENTFUL_REFRAIN = 1240
TURBO_FLARE = 965
MIND_BEND = 141
FROST_SMASH = 131
JETTING_BENCH_DAMAGE = 50

LUCARIO_LINE = {673, 674, 676, 677, 678}
DRAGAPULT_LINE = {119, 120, 121}
IONO_LINE = {265, 268, 269, 270, 271}
CRUSTLE_LINE = {344, 345}
ARCHALUDON_LINE = {169, 190}

STARMIE_LINE = {STARYU, MEGA_STARMIE_EX}
FROSLASS_LINE = {SNORUNT, MEGA_FROSLASS_EX, FROSLASS_BASIC}
MUNKIDORI_LINE = {MUNKIDORI}
CINDERACE_LINE = {CINDERACE}

CARD_DB = {c.cardId: c for c in all_card_data()}

_prize_tracker: PrizeTracker | None = None
_finish_line: list[int] | None = None


def _resolve_deck_path() -> str:
    env = os.environ.get("STARMIE_DECK")
    if env and os.path.exists(env):
        return env
    if os.path.exists("deck.csv"):
        return "deck.csv"
    try:
        here = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        here = None
    if here:
        packaged = os.path.join(here, "deck.csv")
        if os.path.exists(packaged):
            return packaged
        repo_default = os.path.join(here, "..", "agent_decks", "starmie_froslass_ashleysandlin.csv")
        if os.path.exists(repo_default):
            return repo_default
    return "/kaggle_simulations/agent/deck.csv"


with open(_resolve_deck_path(), "r", encoding="utf-8") as file:
    _csv = file.read().split("\n")
my_deck = [int(_csv[i]) for i in range(60)]

_USE_FROSLASS = any(c in my_deck for c in FROSLASS_LINE)
_USE_CINDERACE = CINDERACE in my_deck
_USE_HAMMER = CRUSHING_HAMMER in my_deck
_USE_MUNKIDORI = MUNKIDORI in my_deck
_USE_RISKY_RUINS = RISKY_RUINS in my_deck

_prize_tracker = PrizeTracker(my_deck)


# ── Board helpers ──


def my_index(obs):
    return obs.current.yourIndex


def my_state(obs):
    return obs.current.players[my_index(obs)]


def opp_state(obs):
    return obs.current.players[1 - my_index(obs)]


def active_pokemon(obs, player_index=None):
    pi = player_index if player_index is not None else my_index(obs)
    ps = obs.current.players[pi]
    for p in ps.active or []:
        if p is not None:
            return p
    return None


def bench_pokemon(obs, player_index=None):
    pi = player_index if player_index is not None else my_index(obs)
    return [p for p in (obs.current.players[pi].bench or []) if p is not None]


def all_my_pokemon(obs):
    out = []
    a = active_pokemon(obs)
    if a:
        out.append(a)
    out.extend(bench_pokemon(obs))
    return out


def opp_active(obs):
    return active_pokemon(obs, 1 - my_index(obs))


def opp_bench(obs):
    return bench_pokemon(obs, 1 - my_index(obs))


def hand_ids(obs):
    return [c.id for c in (my_state(obs).hand or []) if c is not None]


def energy_count(pokemon):
    if pokemon is None:
        return 0
    return len(getattr(pokemon, "energyCards", None) or getattr(pokemon, "energies", None) or [])


def water_on(pokemon):
    if pokemon is None:
        return 0
    n = 0
    for c in getattr(pokemon, "energyCards", None) or getattr(pokemon, "energies", None) or []:
        if c and c.id == WATER_ENERGY:
            n += 1
    return n


def prize_value(pokemon):
    if pokemon is None:
        return 0
    data = CARD_DB.get(pokemon.id)
    if data is None:
        return 1
    return max(1, int(getattr(data, "prize", 1) or 1))


def damage_on(pokemon):
    if pokemon is None:
        return 0
    hp = getattr(pokemon, "hp", 0) or 0
    max_hp = getattr(pokemon, "maxHp", hp) or hp
    return max(0, max_hp - hp)


def option_card(obs, opt):
    if getattr(opt, "cardId", None):
        return None
    idx = getattr(opt, "index", None)
    area = getattr(opt, "area", None)
    pi = getattr(opt, "playerIndex", my_index(obs))
    if idx is None or area is None:
        return None
    ps = obs.current.players[pi]
    match area:
        case AreaType.HAND:
            hand = ps.hand or []
            return hand[idx] if 0 <= idx < len(hand) else None
        case AreaType.DISCARD:
            disc = ps.discard or []
            return disc[idx] if 0 <= idx < len(disc) else None
        case AreaType.DECK:
            deck = obs.select.deck or []
            return deck[idx] if obs.select and 0 <= idx < len(deck) else None
        case AreaType.ACTIVE:
            act = ps.active or []
            return act[idx] if 0 <= idx < len(act) else None
        case AreaType.BENCH:
            bench = ps.bench or []
            return bench[idx] if 0 <= idx < len(bench) else None
    return None


def option_target(obs, opt):
    area = getattr(opt, "inPlayArea", None)
    idx = getattr(opt, "inPlayIndex", 0) or 0
    pi = getattr(opt, "playerIndex", my_index(obs))
    if area is None:
        return None
    ps = obs.current.players[pi]
    match area:
        case AreaType.ACTIVE:
            act = ps.active or []
            return act[idx] if 0 <= idx < len(act) else None
        case AreaType.BENCH:
            bench = ps.bench or []
            return bench[idx] if 0 <= idx < len(bench) else None
    return None


def visible_opp_ids(obs):
    ids = set()
    for p in [opp_active(obs)] + opp_bench(obs):
        if p:
            ids.add(p.id)
    return ids


def detect_matchup(obs):
    ids = visible_opp_ids(obs)
    if ids & LUCARIO_LINE:
        return "lucario"
    if ids & CRUSTLE_LINE:
        return "crustle"
    if ids & IONO_LINE:
        return "iono"
    if ids & DRAGAPULT_LINE:
        return "dragapult"
    if ids & ARCHALUDON_LINE:
        return "archaludon"
    if ids & STARMIE_LINE:
        return "mirror"
    return "generic"


def attack_damage(attack_id: int) -> int:
    atk = ALL_ATTACKS.get(attack_id)
    return int(getattr(atk, "damage", 0) or 0) if atk else 0


def _prize_counts(obs) -> tuple[int, int]:
    return len(my_state(obs).prize or []), len(opp_state(obs).prize or [])


def _bench_is_empty(obs) -> bool:
    return len(bench_pokemon(obs)) == 0


def _main_has_basic_play(obs) -> bool:
    for cid in hand_ids(obs):
        if cid == STARYU or (_USE_FROSLASS and cid == SNORUNT):
            return True
        if cid == POFFIN and _bench_is_empty(obs):
            return True
    return False


def can_use_jetting(obs) -> bool:
    active = active_pokemon(obs)
    return bool(active and active.id == MEGA_STARMIE_EX and energy_count(active) >= 1)


def remaining_hp(pokemon) -> int:
    if pokemon is None:
        return 0
    return int(getattr(pokemon, "hp", 0) or 0)


def can_jetting_ko(active, target):
    if active is None or target is None:
        return False
    return attack_damage(JETTING_BLOW) >= remaining_hp(target)


def jetting_bench_ko_count(obs) -> int:
    """Bench KOs from Jetting Blow after chip (Risky Ruins / Froslass)."""
    count = 0
    for bench_mon in opp_bench(obs):
        if damage_on(bench_mon) + JETTING_BENCH_DAMAGE >= remaining_hp(bench_mon):
            count += 1
    return count


def prized_penalty(card_id: int) -> float:
    if _prize_tracker is None:
        return 0.0
    prized = _prize_tracker.is_prized(card_id)
    if prized is True:
        return -1e9
    return 0.0


# ── Scoring ──

_SETUP_ACTIVE = {
    STARYU: (20000, "setup Active Staryu"),
    SNORUNT: (15000, "setup Active Snorunt"),
}
_SETUP_BENCH = {
    STARYU: (20000, "setup bench Staryu"),
    SNORUNT: (12000, "setup bench Snorunt"),
}


def score_setup(obs, opt):
    card = option_card(obs, opt)
    cid = card.id if card else getattr(opt, "cardId", None)
    ctx = obs.select.context

    if ctx == SelectContext.MULLIGAN:
        return (10000, "no mulligan") if opt.type == OptionType.NO else (0, "mulligan")
    if ctx == SelectContext.IS_FIRST:
        return (10000, "choose second") if opt.type == OptionType.NO else (0, "go first")
    if ctx == SelectContext.SETUP_ACTIVE_POKEMON:
        return _SETUP_ACTIVE.get(cid, (0, "unknown setup Active"))
    if ctx == SelectContext.SETUP_BENCH_POKEMON:
        if cid in _SETUP_BENCH:
            return _SETUP_BENCH[cid]
        return -10000, "skip non-basic setup bench"
    return 0, "non-setup"


def apply_tempo(obs, opt, score: int, reason: str) -> tuple[int, str]:
    """When Jetting is live, do not spend the turn on draw/items."""
    ctx = obs.select.context
    if ctx != SelectContext.MAIN:
        return score, reason
    if not can_use_jetting(obs):
        return score, reason
    if opt.type == OptionType.ATTACK:
        return score, reason
    if opt.type == OptionType.RETREAT and score >= 15000:
        return score, reason
    if opt.type == OptionType.PLAY and score >= 40000:
        return score, reason
    if opt.type == OptionType.PLAY:
        card = option_card(obs, opt)
        cid = card.id if card else getattr(opt, "cardId", None)
        if cid == BOSS and score >= 20000:
            return score, reason
    if opt.type in {OptionType.EVOLVE, OptionType.ATTACH}:
        return min(score, 5000), f"tempo: attack ready ({reason})"
    if opt.type == OptionType.PLAY:
        return min(score, 5000), f"tempo: attack ready ({reason})"
    return score, reason


def score_play(obs, opt):
    card = option_card(obs, opt)
    cid = card.id if card else getattr(opt, "cardId", 0)
    ids = hand_ids(obs)
    matchup = detect_matchup(obs)

    if cid in STARMIE_LINE:
        if cid == STARYU:
            if _bench_is_empty(obs):
                return 50000, "play Staryu (empty bench)"
            return 12000, "play Staryu"
        return 8000, "play Starmie line"
    if _USE_FROSLASS and cid in FROSLASS_LINE:
        if cid == SNORUNT:
            if _bench_is_empty(obs):
                return 50000, "play Snorunt (empty bench)"
            if matchup not in ("lucario", "iono"):
                return 4000, "deprio Froslass off-matchup"
            return 11000, "play Snorunt"
        if matchup == "lucario":
            return 9000, "play Froslass line vs Lucario"
        if matchup == "iono":
            return 7000, "play Froslass line vs Iono"
        return 4000, "deprio Froslass off-matchup"
    if _USE_MUNKIDORI and cid in MUNKIDORI_LINE:
        if len(bench_pokemon(obs)) < 4:
            return 8500, "play Munkidori"
        return 5000, "play Munkidori"
    if _USE_CINDERACE and cid in CINDERACE_LINE:
        if energy_count(active_pokemon(obs)) == 0 and len(bench_pokemon(obs)) < 4:
            return 11000, "play Cinderace for Turbo Flare"
        return 5000, "play Cinderace"

    if cid == POFFIN and len(my_state(obs).bench or []) < 5:
        if _bench_is_empty(obs):
            return 50000, "Poffin (empty bench)"
        return 15000, "Poffin"
    if cid == MEGA_SIGNAL:
        if can_use_jetting(obs):
            return 5000, "Mega Signal skip: Jetting live"
        return 14000, "Mega Signal"
    if cid == LILLIE and not obs.current.supporterPlayed:
        if can_use_jetting(obs):
            return 5000, "Lillie skip: Jetting live"
        return 13000, "Lillie"
    if cid == SALVATORE and not obs.current.supporterPlayed:
        if can_use_jetting(obs):
            return 5000, "Salvatore skip: Jetting live"
        return 12500, "Salvatore"
    if cid == HILDA and not obs.current.supporterPlayed:
        if can_use_jetting(obs):
            return 5000, "Hilda skip: Jetting live"
        return 11000, "Hilda"
    if cid == WALLY and not obs.current.supporterPlayed:
        active = active_pokemon(obs)
        if active and active.id in (MEGA_STARMIE_EX, MEGA_FROSLASS_EX) and damage_on(active) > 0:
            return 16000, "Wally heal mega"
        return 4000, "Wally"
    if cid == BOSS and not obs.current.supporterPlayed:
        return _score_boss_play(obs)
    if cid == BLACK_BELT and not obs.current.supporterPlayed:
        return 10000, "Black Belt"
    if _USE_HAMMER and cid == CRUSHING_HAMMER:
        if can_use_jetting(obs):
            return 3000, "Hammer skip: Jetting live"
        return _score_crushing_hammer(obs)
    if cid == HARLEQUIN and not obs.current.supporterPlayed:
        return 9000, "Harlequin disruption"
    if cid == ULTRA_BALL:
        if _bench_is_empty(obs):
            return -5000, "Ultra Ball: bench empty"
        return 9500, "Ultra Ball"
    if cid == HERO_CAPE:
        return 6000, "Hero's Cape"
    if cid == ENERGY_SEARCH:
        return 9000, "Energy Search"
    if cid == POKEGEAR and not obs.current.supporterPlayed:
        return 8500, "Pokegear"
    if cid == SWITCH:
        return 7000, "Switch"
    if cid == NIGHT_STRETCHER:
        return 7500, "Night Stretcher"
    if cid == GRAVITY_MOUNTAIN:
        if matchup in ("lucario", "crustle", "dragapult"):
            return 11000, "Gravity Mountain"
        return 5000, "Gravity Mountain"
    if _USE_RISKY_RUINS and cid == RISKY_RUINS:
        return _score_risky_ruins(obs)
    if cid == POKE_PAD and len(my_state(obs).bench or []) < 5:
        return 8000, "Poke Pad"
    if cid == WATER_ENERGY:
        return 3000, "play Water"
    if cid == DARK_ENERGY and _USE_MUNKIDORI:
        return 3500, "play Dark"
    return 1000, "generic play"


def apply_matchup_overrides(obs, opt, score: int, reason: str) -> tuple[int, str]:
    matchup = detect_matchup(obs)
    aid = getattr(opt, "attackId", 0) or 0
    card = option_card(obs, opt)
    cid = card.id if card else getattr(opt, "cardId", None)

    if matchup == "lucario":
        if opt.type == OptionType.PLAY and cid == GRAVITY_MOUNTAIN:
            return 16000, "Lucario: Gravity Mountain"
        if opt.type == OptionType.PLAY and cid == BOSS and not obs.current.supporterPlayed:
            return max(score, 22000), "Lucario: Boss gust"
        if opt.type == OptionType.ATTACK and aid == JETTING_BLOW:
            return score + 4000, "Lucario: Jetting spread"
        if opt.type == OptionType.ATTACK and aid == NEBULA_BEAM:
            return score + 3000, "Lucario: Nebula vs ex"

    if matchup == "dragapult":
        if opt.type == OptionType.ATTACK and aid == JETTING_BLOW:
            bench_basics = sum(
                1 for b in opp_bench(obs)
                if b.id in DRAGAPULT_LINE or remaining_hp(b) <= 70
            )
            return score + 4000 + bench_basics * 1500, "Dragapult: Jetting basics"
        if opt.type == OptionType.PLAY and cid == BOSS and not obs.current.supporterPlayed:
            return max(score, 20000), "Dragapult: Boss gust"

    if matchup == "iono":
        if opt.type == OptionType.ATTACK and aid == RESENTFUL_REFRAIN:
            hand_sz = len(opp_state(obs).hand or [])
            if hand_sz >= 3:
                return max(score, 36000), "Iono: Resentful"
        if opt.type == OptionType.PLAY and cid == BOSS and not obs.current.supporterPlayed:
            return max(score, 20000), "Iono: Boss"

    if matchup == "archaludon":
        if opt.type == OptionType.ATTACK and aid in (JETTING_BLOW, NEBULA_BEAM):
            return score + 5000, "Archaludon: press Water advantage"
        if opt.type == OptionType.PLAY and cid == BOSS and not obs.current.supporterPlayed:
            return max(score, 22000), "Archaludon: Boss gust"

    if matchup == "mirror":
        if opt.type == OptionType.ATTACK and aid == JETTING_BLOW:
            return score + 2000, "Mirror: Jetting race"

    return score, reason


def _score_risky_ruins(obs):
    """Stadium chips bench basics — sets up Jetting Blow KOs on 70 HP targets."""
    my_prize = len(my_state(obs).prize or [])
    if my_prize >= 5 and len(opp_bench(obs)) <= 3:
        return 13500, "Risky Ruins early"
    if jetting_bench_ko_count(obs) > 0:
        return 16000, "Risky Ruins + Jetting KO setup"
    return 7000, "Risky Ruins"


def _score_boss_play(obs):
    remaining = len(my_state(obs).prize or [])
    active = active_pokemon(obs)
    attacks = []
    if active and active.id == MEGA_STARMIE_EX:
        attacks = [JETTING_BLOW, NEBULA_BEAM]
    best = -500
    reason = "save Boss"
    for target in opp_bench(obs):
        for aid in attacks:
            dmg = attack_damage(aid)
            if aid == JETTING_BLOW:
                dmg = 120
            if dmg >= remaining_hp(target):
                pv = prize_value(target)
                if pv >= remaining:
                    return 25000, "LETHAL Boss"
                s = 5000 + pv * 500
                if s > best:
                    best = s
                    reason = "Boss bench KO"
    return best, reason


def _score_crushing_hammer(obs):
    """Strip opponent energy so they can't stabilize (ladder hybrid plan)."""
    opp = opp_active(obs)
    attached = energy_count(opp)
    if attached >= 2:
        return 17000, "Crushing Hammer heavy attach"
    if attached == 1:
        return 14000, "Crushing Hammer strip"
    if len(opp_bench(obs)) >= 2:
        return 8000, "Crushing Hammer bench pressure"
    return 3000, "Crushing Hammer"


def score_evolve(obs, opt):
    card = option_card(obs, opt)
    target = option_target(obs, opt)
    cid = card.id if card else None
    tid = target.id if target else None
    if cid == MEGA_STARMIE_EX and tid == STARYU:
        active = active_pokemon(obs)
        if active and active.id == STARYU and energy_count(active) >= 1:
            return 35000, "evolve Staryu with energy"
        return 25000, "evolve Starmie ex"
    if cid == MEGA_FROSLASS_EX and tid in (SNORUNT, FROSLASS_BASIC):
        return 15000, "evolve Froslass ex"
    if cid == FROSLASS_BASIC and tid == SNORUNT:
        return 13000, "evolve Froslass"
    return 8000, "evolve"


def score_attach(obs, opt):
    target = option_target(obs, opt)
    if target is None:
        return 1000, "attach"
    tid = target.id
    e = energy_count(target)
    score = 1000
    if tid == MEGA_STARMIE_EX:
        score = 15000 + max(0, 3 - e) * 3000
        if e == 0:
            score += 5000
    elif tid == STARYU:
        score = 10000 + max(0, 1 - e) * 4000
    elif tid in FROSLASS_LINE:
        score = 4000
    elif tid in MUNKIDORI_LINE:
        score = 6000 + max(0, 1 - e) * 3000
    elif tid in CINDERACE_LINE:
        score = 7000 + max(0, 2 - e) * 2500
    if getattr(opt, "inPlayArea", None) == AreaType.ACTIVE:
        score += 2000
    return score, "attach"


def score_attack(obs, opt):
    aid = getattr(opt, "attackId", 0) or 0
    active = active_pokemon(obs)
    opp = opp_active(obs)
    my_prize, opp_prize = _prize_counts(obs)
    behind = my_prize > opp_prize
    matchup = detect_matchup(obs)

    if aid == JETTING_BLOW:
        if opp and can_jetting_ko(active, opp):
            score = 55000 + prize_value(opp) * 5000
            if behind:
                score += 5000
            if my_prize <= prize_value(opp):
                score += 10000
            return score, "Jetting KO"
        bench_ko = sum(1 for b in opp_bench(obs) if remaining_hp(b) <= 50)
        bench_ko += jetting_bench_ko_count(obs)
        score = 42000 + bench_ko * 3000
        if behind:
            score += 3000
        if _USE_HAMMER and bench_ko == 0:
            weak_bench = sum(1 for b in opp_bench(obs) if remaining_hp(b) <= 70)
            score += weak_bench * 2000
        if _USE_RISKY_RUINS:
            chip_setup = sum(
                1 for b in opp_bench(obs)
                if remaining_hp(b) <= 70 and damage_on(b) > 0
            )
            score += chip_setup * 1500
        return score, "Jetting Blow"
    if aid == NEBULA_BEAM:
        if opp and attack_damage(NEBULA_BEAM) >= remaining_hp(opp):
            score = 52000 + prize_value(opp) * 5000
            if behind:
                score += 5000
            return score, "Nebula KO"
        if my_prize <= 2:
            return 38000, "Nebula pressure"
        return 30000, "Nebula Beam"
    if aid == TURBO_FLARE:
        if opp and attack_damage(TURBO_FLARE) >= opp.hp:
            return 22000, "Turbo Flare KO"
        empty_bench = sum(1 for _ in range(5 - len(bench_pokemon(obs))))
        if empty_bench > 0 and any(p.id == STARYU for p in bench_pokemon(obs)):
            return 15000, "Turbo Flare energy to Staryu"
        return 11000, "Turbo Flare"
    if aid == RESENTFUL_REFRAIN:
        hand_sz = len(opp_state(obs).hand or [])
        dmg = 50 * hand_sz
        if opp and dmg >= remaining_hp(opp):
            return 50000, "Resentful KO"
        if can_use_jetting(obs) and hand_sz < 4:
            return 5000, "Resentful skip: prefer Jetting"
        if matchup == "iono" and hand_sz >= 4:
            return 35000, "Resentful vs Iono hand"
        return 8000 + hand_sz * 500, "Resentful Refrain"
    if aid == MIND_BEND:
        if opp and attack_damage(MIND_BEND) >= opp.hp:
            return 20000, "Mind Bend KO"
        return 7000, "Mind Bend"
    if aid == FROST_SMASH:
        if opp and attack_damage(FROST_SMASH) >= opp.hp:
            return 19000, "Frost Smash KO"
        return 6500, "Frost Smash"
    return 5000, "attack"


def _benched_mega_starmie_ready(obs):
    for pokemon in bench_pokemon(obs):
        if pokemon.id == MEGA_STARMIE_EX and energy_count(pokemon) >= 1:
            return pokemon
    return None


def score_retreat(obs, opt):
    active = active_pokemon(obs)
    ready = _benched_mega_starmie_ready(obs)
    if ready and active and active.id != MEGA_STARMIE_EX:
        return 22000, "retreat to ready Mega Starmie"
    if active and damage_on(active) > active.hp // 2:
        if ready:
            return 20000, "retreat wounded to Mega Starmie"
        return 8000, "retreat wounded"
    return -5000, "no retreat"


def score_target(obs, opt):
    card = option_card(obs, opt)
    cid = card.id if card else getattr(opt, "cardId", 0)
    ctx = obs.select.context
    matchup = detect_matchup(obs)

    if ctx in {SelectContext.TO_HAND, SelectContext.TO_DECK, SelectContext.TO_DECK_BOTTOM}:
        base = 5000
        if cid == WATER_ENERGY:
            base = 12000
        elif cid in STARMIE_LINE:
            base = 11000
        elif cid in FROSLASS_LINE:
            base = 9000
        elif cid == MUNKIDORI:
            base = 8000
        elif cid == MEGA_STARMIE_EX:
            base = 13000
        elif cid == MEGA_FROSLASS_EX:
            base = 10000
        return base + prized_penalty(cid), "deck pick"

    if ctx == SelectContext.ATTACH_TO:
        return (12000, "Water to attacker") if cid == WATER_ENERGY else (1000, "attach pick")

    if ctx in {SelectContext.TO_FIELD, SelectContext.TO_BENCH}:
        if cid == STARYU:
            return 15000, "bench Staryu"
        if _USE_FROSLASS and cid == SNORUNT:
            return 12000, "bench Snorunt"
        if _USE_MUNKIDORI and cid == MUNKIDORI:
            return 10000, "bench Munkidori"
        if cid == MEGA_STARMIE_EX:
            return 8000, "bench Starmie ex"
        if _USE_CINDERACE and cid == CINDERACE:
            return 7000, "bench Cinderace"
        return 5000, "to field"

    if ctx in {SelectContext.SWITCH, SelectContext.TO_ACTIVE}:
        yi = my_index(obs)
        pi = getattr(opt, "playerIndex", yi)
        if pi != yi and card:
            pv = prize_value(card)
            return 20000 + pv * 3000, "Boss target"
        if cid == MEGA_STARMIE_EX:
            e = energy_count(card) if card else 0
            return 25000 + e * 2000, "promote Mega Starmie ex"
        if cid == STARYU and water_on(card) >= 1:
            return 14000, "promote Staryu"
        if _USE_CINDERACE and cid == CINDERACE and energy_count(card) >= 1:
            return 10000, "promote Cinderace"
        return 5000, "promote"

    if ctx == SelectContext.HEAL:
        return damage_on(card) * 50, "heal"

    if ctx == SelectContext.DAMAGE:
        hp = getattr(card, "hp", 999) if card else 999
        return 10000 - hp, "damage target"

    return 1000 + prized_penalty(cid), "generic target"


def score_discard(obs, opt):
    card = option_card(obs, opt)
    cid = card.id if card else getattr(opt, "cardId", 0)
    if cid in (MEGA_STARMIE_EX, MEGA_FROSLASS_EX):
        return -5000, "keep mega"
    if cid == WATER_ENERGY and hand_ids(obs).count(WATER_ENERGY) <= 2:
        return -2000, "keep Water"
    return 3000, "discard"


def score_option(obs, opt):
    ctx = obs.select.context

    if ctx in {
        SelectContext.IS_FIRST,
        SelectContext.MULLIGAN,
        SelectContext.SETUP_ACTIVE_POKEMON,
        SelectContext.SETUP_BENCH_POKEMON,
    }:
        return score_setup(obs, opt)

    if opt.type in {OptionType.YES, OptionType.NO} and ctx == SelectContext.IS_FIRST:
        return score_setup(obs, opt)

    if ctx == SelectContext.MAIN:
        match opt.type:
            case OptionType.PLAY:
                score, reason = score_play(obs, opt)
            case OptionType.EVOLVE:
                score, reason = score_evolve(obs, opt)
            case OptionType.ATTACH:
                score, reason = score_attach(obs, opt)
            case OptionType.ATTACK:
                score, reason = score_attack(obs, opt)
            case OptionType.RETREAT:
                score, reason = score_retreat(obs, opt)
            case OptionType.END:
                if _bench_is_empty(obs) and _main_has_basic_play(obs):
                    score, reason = -50000, "empty bench: must bench basic"
                elif can_use_jetting(obs):
                    score, reason = -50000, "must Jetting Blow"
                else:
                    score, reason = 0, "end turn"
            case _:
                score, reason = 1000, "generic MAIN"
        return apply_matchup_overrides(obs, opt, *apply_tempo(obs, opt, score, reason))

    if ctx == SelectContext.ATTACK or opt.type == OptionType.ATTACK:
        score, reason = score_attack(obs, opt)
        return apply_matchup_overrides(obs, opt, score, reason)

    if ctx in {
        SelectContext.TO_HAND,
        SelectContext.TO_DECK,
        SelectContext.TO_DECK_BOTTOM,
        SelectContext.ATTACH_TO,
        SelectContext.TO_FIELD,
        SelectContext.TO_BENCH,
        SelectContext.SWITCH,
        SelectContext.TO_ACTIVE,
        SelectContext.HEAL,
        SelectContext.DAMAGE,
    }:
        return score_target(obs, opt)

    if opt.type == OptionType.DISCARD:
        return score_discard(obs, opt)

    match opt.type:
        case OptionType.PLAY:
            score, reason = score_play(obs, opt)
        case OptionType.EVOLVE:
            score, reason = score_evolve(obs, opt)
        case OptionType.ATTACH:
            score, reason = score_attach(obs, opt)
        case OptionType.ATTACK:
            score, reason = score_attack(obs, opt)
        case OptionType.RETREAT:
            score, reason = score_retreat(obs, opt)
        case OptionType.TARGET:
            score, reason = score_target(obs, opt)
        case OptionType.DISCARD:
            score, reason = score_discard(obs, opt)
        case OptionType.END:
            score, reason = (-100, "end")
        case _:
            score, reason = (1000, "other")
    score, reason = apply_tempo(obs, opt, score, reason)
    return apply_matchup_overrides(obs, opt, score, reason)


def choose_options(obs, obs_dict: dict):
    global _finish_line
    options = obs.select.option or []
    if not options:
        return []

    my_prize = len(my_state(obs).prize or [])
    if my_prize <= 3 and obs_dict.get("search_begin_input"):
        finish = try_cg_search(obs_dict, options, budget_ms=400)
        if finish is not None:
            _finish_line = finish
            return finish

    scored = []
    for i, opt in enumerate(options):
        try:
            score, _ = score_option(obs, opt)
        except Exception:
            score = -999999
        scored.append((score, i))
    scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)

    selected = []
    min_c = obs.select.minCount
    max_c = obs.select.maxCount
    for score, i in scored:
        if len(selected) >= max_c:
            break
        if score < 0 and len(selected) >= min_c:
            continue
        selected.append(i)
    if len(selected) < min_c:
        selected = [i for _, i in scored[:min_c]]
    return selected


def _agent_impl(obs_dict: dict) -> list[int]:
    global _finish_line
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        _finish_line = None
        return my_deck

    if _prize_tracker is not None:
        _prize_tracker.update(obs, obs_dict)

    if not obs.select.option:
        return []

    # Replay verified finish line when search committed a sequence
    if _finish_line is not None:
        if len(_finish_line) <= len(obs.select.option):
            out = _finish_line
            _finish_line = None
            return out

    return choose_options(obs, obs_dict)


def _legal_fallback(obs_dict: dict) -> list[int]:
    sel = obs_dict.get("select")
    if sel is None:
        return my_deck
    n = len(sel.get("option", []))
    min_c = int(sel.get("minCount") or 0)
    max_c = int(sel.get("maxCount") or 0)
    if n == 0 or max_c == 0:
        return []
    k = min_c if min_c > 0 else min(1, max_c)
    k = min(k, max_c, n)
    return list(range(k))


def _is_legal(out, obs_dict: dict) -> bool:
    sel = obs_dict.get("select")
    if sel is None:
        return isinstance(out, list) and len(out) == 60
    if not isinstance(out, list):
        return False
    n = len(sel.get("option", []))
    min_c = int(sel.get("minCount") or 0)
    max_c = int(sel.get("maxCount") or 0)
    if len(out) != len(set(out)):
        return False
    if not all(isinstance(i, int) and 0 <= i < n for i in out):
        return False
    return min_c <= len(out) <= max_c


try:
    from agent.starmie_bench_guard import apply_bench_guard
except ImportError:
    from starmie_bench_guard import apply_bench_guard


def agent(obs_dict: dict) -> list[int]:
    try:
        raw = _agent_impl(obs_dict)
        bench_on = os.environ.get("STARMIE_BENCH_GUARD", "1") != "0"
        out = apply_bench_guard(obs_dict, raw) if bench_on else raw
        if not _is_legal(out, obs_dict):
            return _legal_fallback(obs_dict)
    except Exception:
        return _legal_fallback(obs_dict)
    return out
