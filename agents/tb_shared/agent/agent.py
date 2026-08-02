"""PTCG AI Battle Challenge (cabt Engine) — agent.

CONFIRMED INTERFACE (cabt Engine 0.1.0 docs, June 2026 — see data/CABT_API.md):

    def agent(obs_dict: dict) -> list[int]

The simulator calls `agent` once per decision point. It returns a LIST of
0-based indices into obs_dict["select"]["option"]. The number of indices
returned must be between select.minCount and select.maxCount (inclusive),
and indices must be distinct and in range.

obs_dict has three top-level keys:
    - "logs":    list of events since the agent's last decision.
    - "current": board state (State) or None during initial deck selection.
    - "select":  SelectData or None during initial deck selection.

DECK-SELECTION PHASE:
    When obs_dict["select"] is None (equivalently current is None), the engine
    is asking for a deck. The agent must return a list of 60 card IDs (not
    option indices). We load these from deck.csv next to this module.

SELECT PHASE key fields:
    select.type      -> SelectType (MAIN, CARD, ENERGY, ATTACK, YES_NO, ...)
    select.context   -> SelectContext (why we are choosing; 49 values)
    select.minCount  -> min indices to return (can be 0)
    select.maxCount  -> max indices to return
    select.option[i] -> Option; .type is an OptionType (PLAY, ATTACH, EVOLVE,
                        ABILITY, DISCARD, RETREAT, ATTACK, END, CARD, ENERGY,
                        YES, NO, NUMBER, ...)

GUARANTEE: never crash; always return a legal selection (competition contract).
_legal_fallback provides a safe default for any unhandled selection.

Determinism: a seeded RNG keeps nightly win-rate comparisons fair.

ARCHITECTURE (pluggable scorers):
    `Agent.act` owns the never-crash scaffold — the deck-selection short-circuit,
    the try/except, and `_legal_fallback`. The per-decision *brain* is a pluggable
    `OptionScorer`:

        class OptionScorer:
            def choose(self, obs_dict, select, current, options) -> list[int]: ...

    The default brain is `HeuristicScorer` (the rule-based policy below). Later
    phases inject alternate scorers (e.g. `SearchScorer`, `LearnedScorer`) via
    `Agent(scorer=...)`; all share the same legality + fallback scaffold, so any
    brain still degrades to a legal move. Injecting a scorer changes *which* legal
    move is chosen, never the never-crash contract.

NOTE: full validation needs the official cabt engine + a downloaded deck (Kaggle
token required). Until then, correctness is checked offline against synthetic
observations in scripts/smoke_test.py.
"""

from __future__ import annotations

import csv
import os
import random
from typing import Any

# --- OptionType IDs (from cabt api.OptionType) --------------------------------
OPT_NUMBER, OPT_YES, OPT_NO, OPT_CARD = 0, 1, 2, 3
OPT_TOOL_CARD, OPT_ENERGY_CARD, OPT_ENERGY = 4, 5, 6
OPT_PLAY, OPT_ATTACH, OPT_EVOLVE, OPT_ABILITY = 7, 8, 9, 10
OPT_DISCARD, OPT_RETREAT, OPT_ATTACK, OPT_END = 11, 12, 13, 14
OPT_SKILL, OPT_SPECIAL_CONDITION = 15, 16

# --- SelectType IDs (from cabt api.SelectType) --------------------------------
SEL_MAIN, SEL_CARD, SEL_ATTACHED_CARD, SEL_CARD_OR_ATTACHED = 0, 1, 2, 3
SEL_ENERGY, SEL_SKILL, SEL_ATTACK, SEL_EVOLVE = 4, 5, 6, 7
SEL_COUNT, SEL_YES_NO, SEL_SPECIAL_CONDITION = 8, 9, 10

# --- SelectContext IDs (from cabt api.SelectContext) --------------------------
CTX_MAIN = 0
CTX_SETUP_ACTIVE_POKEMON = 1
CTX_SETUP_BENCH_POKEMON = 2
CTX_SWITCH = 3
CTX_TO_ACTIVE = 4
CTX_TO_BENCH = 5
CTX_TO_HAND = 7
CTX_DISCARD = 8
CTX_TO_DECK = 9
CTX_TO_DECK_BOTTOM = 10
CTX_DAMAGE_COUNTER = 13
CTX_DAMAGE_COUNTER_ANY = 14
CTX_DAMAGE = 15
CTX_REMOVE_DAMAGE_COUNTER = 16
CTX_HEAL = 17
CTX_ATTACH_FROM = 21
CTX_ATTACH_TO = 22
CTX_ATTACK = 35
CTX_DRAW_COUNT = 38
CTX_DAMAGE_COUNTER_COUNT = 39
CTX_REMOVE_DAMAGE_COUNTER_COUNT = 40
CTX_IS_FIRST = 41
CTX_MULLIGAN = 42
CTX_ACTIVATE = 43

DECK_SIZE = 60
_DEFAULT_DECK_PATH = os.path.join(os.path.dirname(__file__), "deck.csv")

# Current baseline deck roles, from report/deck_concept_v1.md.
CARD_WATER_ENERGY = 3
CARD_KYOGRE = 721
CARD_SNOVER = 722
CARD_MEGA_ABOMASNOW_EX = 723
CARD_BLACK_KYUREM_EX = 179
CARD_VELUZA = 159
CARD_CHIEN_PAO = 209
CARD_STARYU = 1030
CARD_MEGA_STARMIE_EX = 1031
CARD_MEGA_SIGNAL = 1145
CARD_MAXIMUM_BELT = 1158
CARD_CYRANO = 1205
CARD_LILLIE = 1227
CARD_WAITRESS = 1235

BASIC_POKEMON_FALLBACK = {
    CARD_BLACK_KYUREM_EX, CARD_CHIEN_PAO, CARD_KYOGRE, CARD_SNOVER,
    CARD_STARYU, CARD_VELUZA,
}

_ATTACK_DATA: "dict[int, Any] | None" = None
_CARD_DATA: "dict[int, Any] | None" = None


def _option_type(opt: Any) -> "int | None":
    """Return the OptionType id of an option, tolerating dict or object form."""
    if isinstance(opt, dict):
        return opt.get("type")
    return getattr(opt, "type", None)


def _get(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def load_deck(path: str = _DEFAULT_DECK_PATH) -> "list[int]":
    """Load 60 card IDs (one per line) from deck.csv. Returns [] if missing."""
    if not os.path.exists(path):
        return []
    ids = []
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if not row:
                continue
            cell = row[0].strip()
            if cell:
                ids.append(int(cell))
    return ids


def _attack_data(attack_id):
    global _ATTACK_DATA
    if _ATTACK_DATA is None:
        _ATTACK_DATA = {}
        try:
            from cg.api import all_attack

            _ATTACK_DATA = {a.attackId: a for a in all_attack()}
        except Exception:
            _ATTACK_DATA = {}
    return _ATTACK_DATA.get(attack_id)


def _card_data(card_id):
    global _CARD_DATA
    if _CARD_DATA is None:
        _CARD_DATA = {}
        try:
            from cg.api import all_card_data

            _CARD_DATA = {c.cardId: c for c in all_card_data()}
        except Exception:
            _CARD_DATA = {}
    return _CARD_DATA.get(card_id)


class OptionScorer:
    """Pluggable per-decision brain.

    Given a select phase (already past the deck-selection short-circuit and
    inside `Agent.act`'s try/except), return the chosen 0-based option indices.

    Contract:
        choose(obs_dict, select, current, options) -> list[int]
            obs_dict : the full observation (logs/current/select).
            select   : obs_dict["select"] (never None here).
            current  : obs_dict["current"] or {} (defensive default).
            options  : select["option"] or [] (the legal choices).
        The return must be distinct, in-range indices within
        [select.minCount, select.maxCount]. Raising is allowed: `Agent.act`
        catches it and substitutes a legal fallback, so a buggy brain never
        breaks the never-crash contract.

    Subclasses: `HeuristicScorer` (default). Later phases add `SearchScorer`
    and `LearnedScorer` implementing this same interface.
    """

    def choose(self, obs_dict, select, current, options):
        raise NotImplementedError


class HeuristicScorer(OptionScorer):
    """Rule-based PTCG decision policy over the cabt observation interface.

    A legal, deterministic policy that develops board state before attacking.
    This is the default brain behind `Agent`'s never-crash scaffold; it owns
    only scoring/selection — the deck short-circuit and `_legal_fallback` live
    on `Agent`.
    """

    # T7 v1: setup first. The initial attack-first policy measured 7.5% into
    # random because it skipped optional bench/setup choices and under-developed.
    MAIN_PRIORITY = (
        OPT_EVOLVE, OPT_PLAY, OPT_ATTACH, OPT_ABILITY,
        OPT_ATTACK, OPT_RETREAT, OPT_END,
    )

    def __init__(self, rng=None) -> None:
        # Kept for parity / future stochastic tie-breaks; current logic is
        # deterministic and does not consume the RNG.
        self._rng = rng if rng is not None else random.Random(0)

    def choose(self, obs_dict, select, current, options):
        if not options:
            return []
        sel_type = select.get("type")
        context = select.get("context")
        min_count = int(select.get("minCount", 1) or 0)
        if sel_type == SEL_MAIN:
            return [self._pick_main(options, current)]
        if sel_type == SEL_CARD:
            return self._pick_cards(select, current)
        if sel_type == SEL_COUNT:
            return self._pick_count(options)
        if sel_type == SEL_YES_NO:
            return self._pick_yes_no(options, context, min_count)
        if sel_type == SEL_ATTACK:
            return [self._best_attack_index(options, current)]
        return self._take_min(options, min_count, select.get("maxCount"))

    def _pick_main(self, options, current):
        """Pick a MAIN action by setup-aware priority."""
        by_type: dict[int | None, list[int]] = {}
        for i, opt in enumerate(options):
            by_type.setdefault(_option_type(opt), []).append(i)
        # Never attack/end with an empty bench if we can still bench a Basic.
        bench_count, bench_max = self._bench_counts(current)
        if (
            bench_count == 0
            and bench_max > 0
            and OPT_PLAY in by_type
            and (_get(current, "players", []) or [])
        ):
            from agent.deck_tech import tech_for_deck
            from agent.smart_bench import should_play_basic_to_bench

            tech = tech_for_deck(self._visible_card_ids(current))
            play_idx = self._best_option_index(OPT_PLAY, by_type[OPT_PLAY], options, current)
            card = self._hand_card_for_option(options[play_idx], current)
            card_id = int(_get(card, "id", 0) or 0)
            if should_play_basic_to_bench(card_id, current, tech):
                return play_idx
        for pref in self.MAIN_PRIORITY:
            if pref in by_type:
                return self._best_option_index(pref, by_type[pref], options, current)
        return 0

    def _best_option_index(self, opt_type, indices, options, current):
        if opt_type == OPT_PLAY:
            return max(indices, key=lambda i: self._play_score(options[i], current))
        if opt_type == OPT_ATTACH:
            return max(indices, key=lambda i: self._attach_score(options[i], current))
        if opt_type == OPT_ATTACK:
            return max(indices, key=lambda i: self._attack_score(options[i], current))
        if opt_type == OPT_RETREAT:
            return indices[0]
        return indices[0]

    def _pick_cards(self, select, current):
        options = select.get("option") or []
        context = select.get("context")
        min_count = int(select.get("minCount", 1) or 0)
        max_count = int(select.get("maxCount", len(options)) or 0)
        if not options or max_count <= 0:
            return []

        if context == CTX_SETUP_BENCH_POKEMON:
            ranked = sorted(range(len(options)),
                            key=lambda i: self._card_option_score(options[i], current, select),
                            reverse=True)
            from agent.deck_tech import tech_for_deck
            from agent.smart_bench import setup_bench_target_count

            deck_ids = [int(_get(o, "id", 0) or 0) for o in options]
            tech = tech_for_deck(deck_ids)
            target = setup_bench_target_count(
                ranked, options, current, tech,
                min_count, max_count,
            )
            bench_count, _ = self._bench_counts(current)
            if min_count <= 0 and bench_count == 0 and ranked:
                target = max(1, target)
            return ranked[:target]

        if context in (CTX_SETUP_ACTIVE_POKEMON, CTX_TO_ACTIVE, CTX_SWITCH):
            return [max(range(len(options)),
                        key=lambda i: self._promotion_score(options[i], current, select))]

        if context in (CTX_TO_HAND, CTX_ATTACH_FROM, CTX_ATTACH_TO, CTX_HEAL,
                       CTX_REMOVE_DAMAGE_COUNTER):
            ranked = sorted(range(len(options)),
                            key=lambda i: self._card_option_score(options[i], current, select),
                            reverse=True)
        elif context in (CTX_DAMAGE_COUNTER, CTX_DAMAGE_COUNTER_ANY, CTX_DAMAGE):
            ranked = sorted(range(len(options)),
                            key=lambda i: self._damage_target_score(options[i], current, select),
                            reverse=True)
        elif context in (CTX_DISCARD, CTX_TO_DECK, CTX_TO_DECK_BOTTOM):
            ranked = sorted(range(len(options)),
                            key=lambda i: self._card_option_score(options[i], current, select))
        else:
            ranked = list(range(len(options)))

        if context in (CTX_TO_HAND, CTX_ATTACH_TO) and max_count > 0:
            # These are usually beneficial optional searches/attachments. The
            # prior v1 policy declined them whenever minCount was zero.
            count = min(max_count, len(options))
        else:
            count = min(max(min_count, 0), max_count, len(options))
        return ranked[:count]

    def _pick_count(self, options):
        best = max(range(len(options)), key=lambda i: _get(options[i], "number", 0) or 0)
        return [best]

    def _pick_yes_no(self, options, context, min_count):
        prefer_yes = context in {
            CTX_MULLIGAN, CTX_ACTIVATE, 44, 45, 46,
        }
        if context == CTX_IS_FIRST:
            prefer_yes = True
        if min_count <= 0 and context not in {CTX_ACTIVATE, 44, 45, 46}:
            prefer_yes = False
        idx = self._yes_no_index(options, prefer_yes=prefer_yes)
        if idx is not None:
            return [idx]
        return self._take_min(options, min_count, 1)

    def _best_attack_index(self, options, current=None):
        return max(range(len(options)), key=lambda i: self._attack_score(options[i], current))

    def _play_score(self, opt, current):
        card = self._hand_card_for_option(opt, current)
        card_id = _get(card, "id", 0) or 0
        data = _card_data(card_id)
        score = self._card_id_score(card_id, current)
        is_basic = (
            bool(_get(data, "basic", False)) if data is not None
            else card_id in BASIC_POKEMON_FALLBACK
        )
        if is_basic:
            from agent.deck_tech import tech_for_deck
            from agent.smart_bench import bench_counts as sb_bench_counts
            from agent.smart_bench import should_play_basic_to_bench

            bench_count, bench_max = sb_bench_counts(current)
            tech = tech_for_deck(self._visible_card_ids(current))
            if bench_count == 0:
                score += 6100
            elif bench_count < bench_max and should_play_basic_to_bench(card_id, current, tech):
                score += 2600
            else:
                score -= 4000
        if card_id == CARD_MEGA_SIGNAL:
            score += 1500
        if card_id in (CARD_LILLIE, CARD_WAITRESS, CARD_CYRANO):
            score += 900
        if card_id == CARD_MAXIMUM_BELT:
            score += 300
        return score

    def _attack_score(self, opt, current=None):
        attack_id = _get(opt, "attackId", 0) or 0
        attack = _attack_data(attack_id)
        damage = int(_get(attack, "damage", 0) or 0) if attack is not None else 0
        score = damage * 10 + attack_id / 10000.0
        text = (_get(attack, "text", "") or "") if attack is not None else ""
        if damage <= 0:
            # Variable-damage attacks are legal but risky for a deterministic
            # first-pass pilot. Keep them below known fixed 130/200 attacks.
            if "100 damage for each" in text:
                score += 500
            elif "20 damage for each" in text:
                score += 150
        if "Discard" in text:
            score -= 80
        opp_active = self._opponent_active(current or {})
        if opp_active is not None and damage > 0:
            hp = int(_get(opp_active, "hp", 0) or 0)
            if hp and damage >= hp:
                score += 5000
            else:
                score += max(0, 220 - hp)
        return score

    def _attach_score(self, opt, current):
        target_area = _get(opt, "inPlayArea")
        target_index = _get(opt, "inPlayIndex", 0) or 0
        score = 0
        if target_area == 4:  # ACTIVE
            score += 40
            pokemon = self._pokemon_at(current, target_area, target_index)
            energy_count = len(_get(pokemon, "energies", []) or []) if pokemon else 0
            score += max(0, 4 - energy_count) * 10
            score += self._pokemon_role_bonus(_get(pokemon, "id", 0) or 0) / 20
        elif target_area == 5:  # BENCH
            pokemon = self._pokemon_at(current, target_area, target_index)
            energy_count = len(_get(pokemon, "energies", []) or []) if pokemon else 0
            score += 20 + max(0, 3 - energy_count) * 8
            score += self._pokemon_role_bonus(_get(pokemon, "id", 0) or 0) / 25
        return score

    def _card_option_score(self, opt, current, select=None):
        pokemon = self._card_option_pokemon(opt, current)
        if pokemon is not None:
            context = _get(select, "context") if select is not None else None
            if context in (CTX_HEAL, CTX_REMOVE_DAMAGE_COUNTER):
                return self._heal_target_score(opt, current, pokemon)
            return self._promotion_score(opt, current, select)
        card = self._card_from_option(opt, current, select)
        card_id = _get(card, "id", 0) or _get(opt, "cardId", 0) or 0
        return self._card_id_score(card_id, current)

    def _promotion_score(self, opt, current, select=None):
        pokemon = self._card_option_pokemon(opt, current)
        if pokemon is None:
            card = self._card_from_option(opt, current, select)
            card_id = _get(card, "id", 0) or _get(opt, "cardId", 0) or 0
            return self._card_id_score(card_id, current)
        hp = int(_get(pokemon, "hp", 0) or 0)
        max_hp = int(_get(pokemon, "maxHp", hp) or hp)
        energy_count = len(_get(pokemon, "energies", []) or [])
        card_id = _get(pokemon, "id", 0) or 0
        score = hp + max_hp + 45 * energy_count + self._pokemon_role_bonus(card_id)
        if _get(opt, "area") == 4:
            score += 30
        return score

    def _damage_target_score(self, opt, current, select=None):
        pokemon = self._card_option_pokemon(opt, current)
        if pokemon is None:
            return 0
        your_index = _get(current, "yourIndex", 0) or 0
        player_index = _get(opt, "playerIndex", your_index)
        opponent_bonus = 100000 if player_index != your_index else -100000
        hp = int(_get(pokemon, "hp", 0) or 0)
        max_hp = int(_get(pokemon, "maxHp", hp) or hp)
        damage = max(0, max_hp - hp)
        energy_count = len(_get(pokemon, "energies", []) or [])
        remaining_counters = int(_get(select, "remainDamageCounter", 0) or 0) if select else 0
        ko_bonus = 15000 if remaining_counters and remaining_counters * 10 >= hp else 0
        active_bonus = 1200 if _get(opt, "area") == 4 else 0
        card_id = _get(pokemon, "id", 0) or 0
        prize_bonus = self._target_prize_bonus(card_id)
        return (
            opponent_bonus + ko_bonus + active_bonus + prize_bonus +
            damage * 12 + energy_count * 250 + max(0, 400 - hp)
        )

    def _heal_target_score(self, opt, current, pokemon):
        your_index = _get(current, "yourIndex", 0) or 0
        player_index = _get(opt, "playerIndex", your_index)
        if player_index != your_index:
            return -100000
        hp = int(_get(pokemon, "hp", 0) or 0)
        max_hp = int(_get(pokemon, "maxHp", hp) or hp)
        damage = max(0, max_hp - hp)
        energy_count = len(_get(pokemon, "energies", []) or [])
        card_id = _get(pokemon, "id", 0) or 0
        return damage * 20 + energy_count * 80 + self._pokemon_role_bonus(card_id)

    def _pokemon_role_bonus(self, card_id):
        if card_id == CARD_MEGA_ABOMASNOW_EX:
            return 900
        if card_id == CARD_BLACK_KYUREM_EX:
            return 820
        if card_id == CARD_MEGA_STARMIE_EX:
            return 880
        if card_id == CARD_KYOGRE:
            return 500
        if card_id == CARD_CHIEN_PAO:
            return 440
        if card_id == CARD_VELUZA:
            return 400
        if card_id == CARD_STARYU:
            return 260
        if card_id == CARD_SNOVER:
            return 180
        card = _card_data(card_id)
        if card is not None:
            if bool(_get(card, "megaEx", False)):
                return 700
            if bool(_get(card, "ex", False)):
                return 500
        return 0

    def _target_prize_bonus(self, card_id):
        card = _card_data(card_id)
        if card is None:
            return self._pokemon_role_bonus(card_id)
        if bool(_get(card, "megaEx", False)):
            return 3000
        if bool(_get(card, "ex", False)):
            return 1800
        return 0

    def _card_id_score(self, card_id, current):
        if card_id == CARD_MEGA_ABOMASNOW_EX:
            return 1000 if self._has_snover_in_play(current) else 420
        if card_id == CARD_MEGA_STARMIE_EX:
            return 980 if self._has_pokemon_in_play(current, CARD_STARYU) else 440
        if card_id == CARD_STARYU:
            return 870
        if card_id == CARD_SNOVER:
            return 850
        if card_id == CARD_KYOGRE:
            return 720
        if card_id == CARD_BLACK_KYUREM_EX:
            return 760
        if card_id == CARD_CHIEN_PAO:
            return 700
        if card_id == CARD_VELUZA:
            return 660
        if card_id == CARD_MEGA_SIGNAL:
            return 680
        if card_id in (CARD_LILLIE, CARD_WAITRESS, CARD_CYRANO):
            return 620
        if card_id == CARD_MAXIMUM_BELT:
            return 500
        if card_id == CARD_WATER_ENERGY:
            return 120 if not self._energy_attached(current) else 60
        return card_id % 1000

    def _has_snover_in_play(self, current):
        return self._has_pokemon_in_play(current, CARD_SNOVER)

    def _has_pokemon_in_play(self, current, card_id):
        your = self._your_player(current)
        for pokemon in (_get(your, "active", []) or []) + (_get(your, "bench", []) or []):
            if pokemon is not None and _get(pokemon, "id") == card_id:
                return True
        return False

    def _energy_attached(self, current):
        return bool(_get(current, "energyAttached", False))

    def _your_player(self, current):
        players = _get(current, "players", []) or []
        if not players:
            return {}
        your_index = _get(current, "yourIndex", 0) or 0
        return players[your_index] if 0 <= your_index < len(players) else {}

    def _visible_card_ids(self, current) -> list[int]:
        ids: list[int] = []
        your = self._your_player(current)
        for poke in (_get(your, "active", []) or []) + (_get(your, "bench", []) or []):
            if poke is not None:
                cid = int(_get(poke, "id", 0) or 0)
                if cid:
                    ids.append(cid)
        for card in _get(your, "hand", []) or []:
            if card is not None:
                cid = int(_get(card, "id", 0) or 0)
                if cid:
                    ids.append(cid)
        return ids

    def _bench_counts(self, current):
        your = self._your_player(current)
        bench = _get(your, "bench", []) or []
        bench_max = int(_get(your, "benchMax", 5) or 5)
        return len(bench), bench_max

    def _opponent_active(self, current):
        players = _get(current, "players", []) or []
        your_index = _get(current, "yourIndex", 0) or 0
        opp_index = 1 - your_index
        if not (0 <= opp_index < len(players)):
            return None
        active = _get(players[opp_index], "active", []) or []
        return active[0] if active else None

    def _card_option_pokemon(self, opt, current):
        area = _get(opt, "area")
        index = _get(opt, "index", 0) or 0
        return self._pokemon_at(current, area, index, _get(opt, "playerIndex"))

    def _pokemon_at(self, current, area, index, player_index=None):
        players = _get(current, "players", []) or []
        if not players:
            return None
        your_index = _get(current, "yourIndex", 0) or 0
        pidx = your_index if player_index is None else player_index
        if pidx < 0 or pidx >= len(players):
            return None
        player = players[pidx]
        if area == 4:
            active = _get(player, "active", []) or []
            return active[index] if 0 <= index < len(active) else None
        if area == 5:
            bench = _get(player, "bench", []) or []
            return bench[index] if 0 <= index < len(bench) else None
        return None

    def _card_from_option(self, opt, current, select=None):
        area = _get(opt, "area")
        index = _get(opt, "index", 0) or 0
        player_index = _get(opt, "playerIndex")
        players = _get(current, "players", []) or []
        your_index = _get(current, "yourIndex", 0) or 0
        pidx = your_index if player_index is None else player_index
        if area == 1 and select is not None:
            deck = _get(select, "deck", []) or []
            return deck[index] if 0 <= index < len(deck) else None
        if area == 12:
            looking = _get(current, "looking", []) or []
            return looking[index] if 0 <= index < len(looking) else None
        if not (0 <= pidx < len(players)):
            return None
        player = players[pidx]
        if area == 2:
            hand = _get(player, "hand", []) or []
            return hand[index] if 0 <= index < len(hand) else None
        if area == 3:
            discard = _get(player, "discard", []) or []
            return discard[index] if 0 <= index < len(discard) else None
        if area == 6:
            prize = _get(player, "prize", []) or []
            return prize[index] if 0 <= index < len(prize) else None
        return None

    def _hand_card_for_option(self, opt, current):
        index = _get(opt, "index", 0) or 0
        your = self._your_player(current)
        hand = _get(your, "hand", []) or []
        return hand[index] if 0 <= index < len(hand) else None

    def _take_min(self, options, min_count, max_count):
        """Return a legal count of indices. For optional YES/NO, decline (NO)."""
        n = len(options)
        if min_count <= 0:
            if self._is_yes_no(options):
                no_idx = self._yes_no_index(options, prefer_yes=False)
                return [no_idx] if no_idx is not None else []
            return []
        count = max(min_count, 1)
        if isinstance(max_count, int):
            count = min(count, max_count)
        count = min(count, n)
        return list(range(count))

    @staticmethod
    def _is_yes_no(options):
        types = {_option_type(o) for o in options}
        return types <= {OPT_YES, OPT_NO} and OPT_YES in types

    @staticmethod
    def _yes_no_index(options, prefer_yes):
        want = OPT_YES if prefer_yes else OPT_NO
        for i, o in enumerate(options):
            if _option_type(o) == want:
                return i
        return None


class Agent:
    """Never-crash scaffold around a pluggable `OptionScorer`.

    `Agent` owns the competition contract: the deck-selection short-circuit, the
    try/except guard, and `_legal_fallback`. It delegates *which legal option to
    pick* to its `scorer` (default `HeuristicScorer`). Swapping the scorer (e.g.
    a future search or learned policy) changes decisions but never the guarantee
    that a legal selection is always returned.

    Bench-critical decisions (empty bench, setup bench) always route through
    RuleCore so MCTS/Learned cannot skip basic development (see bench_guard.py).
    """

    def __init__(self, seed=0, deck_path: str = _DEFAULT_DECK_PATH,
                 scorer: "OptionScorer | None" = None) -> None:
        self._rng = random.Random(seed)
        self._deck_path = deck_path
        self._scorer = scorer if scorer is not None else HeuristicScorer(self._rng)
        self._bench_scorer = None

    def _bench_scorer_for(self, deck_path: str):
        if self._bench_scorer is None:
            from agent.deck_tech import tech_for_deck
            from agent.agent import load_deck

            deck_ids = load_deck(deck_path)
            if tech_for_deck(deck_ids).name == "mega_lucario_ex":
                from agent.lucario_policy import LucarioScorer

                self._bench_scorer = LucarioScorer(rng=self._rng, deck_path=deck_path)
            else:
                from agent.rule_core import RuleCoreScorer

                self._bench_scorer = RuleCoreScorer(rng=self._rng, deck_path=deck_path)
        return self._bench_scorer

    def __call__(self, obs_dict):
        return self.act(obs_dict)

    def act(self, obs_dict):
        select = obs_dict.get("select")
        # Deck-selection phase: select is None -> return 60 card IDs.
        if select is None:
            deck = load_deck(self._deck_path)
            if len(deck) == DECK_SIZE:
                return deck
            return deck[:DECK_SIZE] if deck else []
        try:
            current = obs_dict.get("current") or {}
            options = select.get("option") or []
            from agent.bench_guard import bench_critical

            deck_path = self._deck_path
            if not os.path.exists(deck_path) and os.path.exists("deck.csv"):
                deck_path = "deck.csv"
            if bench_critical(obs_dict, select, current, options):
                return self._bench_scorer_for(deck_path).choose(
                    obs_dict, select, current, options
                )
            return self._scorer.choose(obs_dict, select, current, options)
        except Exception:
            return self._legal_fallback(select)

    def _legal_fallback(self, select):
        """Always-legal default: the first minCount (>=1 if required) indices."""
        options = select.get("option") or []
        n = len(options)
        if n == 0:
            return []
        min_count = int(select.get("minCount", 1) or 0)
        if min_count <= 0:
            return []
        max_count = select.get("maxCount")
        count = min_count
        if isinstance(max_count, int):
            count = min(count, max_count)
        return list(range(min(count, n)))


def build_agent(
    seed=0,
    deck_path: str = _DEFAULT_DECK_PATH,
    scorer: "OptionScorer | None" = None,
) -> Agent:
    """Factory. The cabt harness expects a callable agent(obs_dict)->list[int]."""
    return Agent(seed=seed, deck_path=deck_path, scorer=scorer)


_DEFAULT_AGENT = build_agent(seed=0)


def agent(obs_dict):
    """Engine entry point: agent(obs_dict) -> list[int] of option indices."""
    return _DEFAULT_AGENT.act(obs_dict)


if __name__ == "__main__":
    main_obs = {
        "logs": [], "current": {"turn": 3},
        "select": {
            "type": SEL_MAIN, "context": 0, "minCount": 1, "maxCount": 1,
            "option": [
                {"type": OPT_END}, {"type": OPT_PLAY, "index": 0},
                {"type": OPT_ATTACK, "attackId": 12},
            ],
        },
    }
    out = agent(main_obs)
    assert out == [1], f"expected PLAY setup before ATTACK, got {out}"
    assert isinstance(agent({"logs": [], "current": None, "select": None}), list)
    print("OK: agent picks ATTACK in MAIN; deck phase returns a list.")
