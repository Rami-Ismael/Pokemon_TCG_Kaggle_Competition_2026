"""
Improved Probabilistic Agent — clean reimplementation
=====================================================

A from-scratch, readable rewrite of Ivan Ternovskiy's Kaggle notebook
"Improved Probabilistic agent" (public score 967.7) for the Pokémon TCG
AI Battle Challenge.

WHAT THIS ALGORITHM ACTUALLY IS
-------------------------------
The notebook is titled "Probabilistic Expectimax", but the control flow is
NOT expectimax (it never enumerates chance nodes weighted by probability).
It is **flat root Monte-Carlo search with a UCB1 bandit** — i.e. MCTS with
the tree-expansion step removed. Selection (UCB1) + simulation (rollout) +
backpropagation (running mean) are present; expansion is not. Only the root
actions are ever evaluated.

THE THREE-LAYER SHAPE
---------------------
1. A large hand-written HEURISTIC (`HeuristicPolicy`) ranks every legal move.
2. SEARCH re-ranks only the top-K of that list by actually playing each move
   out on the real engine and scoring the resulting board.
3. If search is unavailable / times out / errors, the agent degrades to the
   pure heuristic. It is designed to never crash.

FIX vs. THE ORIGINAL
--------------------
The original calls `search_begin(obs, your_deck=yd)`, but the current engine
requires 7 positional args. On this build the original's search path throws
`TypeError`, is swallowed by a bare `except`, and the agent silently runs as
a PURE HEURISTIC. This reimplementation calls `search_begin` with the full
signature so the Monte-Carlo re-ranker actually executes.
"""
from __future__ import annotations

import math
import os
import random
import time
from collections import defaultdict

from cg.api import (
    AreaType, Card, CardType, EnergyType, Observation, OptionType,
    Pokemon, SelectContext, all_card_data, to_observation_class,
)

# Probe for the engine-as-forward-model search API. If it is missing we still
# play — just without the Monte-Carlo re-ranker.
_SEARCH_OK = False
try:
    from cg.api import search_begin, search_step, search_end, search_release
    _SEARCH_OK = True
except Exception:
    pass

# --- Tunables -------------------------------------------------------------
# Search default-off, matching agent_core_improved: until 2026-08-11 the bandit
# here was dead code (HeuristicPolicy.choose() truncated to maxCount=1, so it
# never saw a second candidate), so every benchmark/Glicko number this agent
# has ever posted is the PURE HEURISTIC's. Turning search on by default would
# silently change a pool regular, so it stays an env-var override for A/B
# runs. (An earlier version of this comment cited the sibling's "live search
# lost 35.0%" as evidence the search path is weak — that number is VOID: it
# was measured with the perspective-inversion bug fixed 2026-08-12 in
# simulate_action. Live-search strength at the fixed sign is an open
# question, not a settled negative.)
USE_SEARCH = os.environ.get("USE_SEARCH", "0") != "0"
SEARCH_TIME_BUDGET = 1.5     # seconds per MAIN decision (match cap is 600s)
SEARCH_MAX_CANDIDATES = 8    # only re-rank the heuristic's top-8 moves
LOW_DECK_COUNT = 10          # "running low on deck" threshold for card economy
MEGA_BRAVE = 983             # attack id for Mega Lucario ex's big attack


# --- Deck: 60 hardcoded card IDs -----------------------------------------
# Mega Lucario ex line (677 Riolu -> 678 Mega Lucario ex), plus Lunatone/
# Solrock (675/676) and Makuhita/Hariyama (673/674). Card 6 is Basic Fighting
# Energy (exempt from the 4-copy limit, hence ~17 copies).
DECK = [
    673, 673, 674, 674, 675, 675, 676, 676,
    676, 677, 677, 677, 678, 678, 678, 678,
    1102, 1102, 1102, 1102, 1123, 1123, 1141, 1141,
    1141, 1141, 1142, 1142, 1142, 1142, 1152, 1152,
    6, 1159, 1182, 1182, 1192, 1192, 1192, 1192,
    1227, 1227, 1227, 1227, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6,
    6, 1182, 677, 1252,
]


class C:
    """Named card IDs, so the scoring code reads as strategy, not magic numbers."""
    KYOGRE, SNOVER, MEGA_ABOMASNOW_EX = 721, 722, 723
    MAKUHITA, HARIYAMA = 673, 674
    LUNATONE, SOLROCK = 675, 676
    RIOLU, MEGA_LUCARIO_EX = 677, 678
    BASIC_FIGHTING_ENERGY = 6
    DUSK_BALL, SWITCH, PREMIUM_POWER_PRO, FIGHTING_GONG = 1102, 1123, 1141, 1142
    POKE_PAD, HERO_CAPE, BOSS_ORDERS = 1152, 1159, 1182
    CARMINE, LILLIE_DETERMINATION, GRAVITY_MOUNTAIN = 1192, 1227, 1252
    LUMIOSE_CITY, LILLIES_PEARL, LEGACY_ENERGY = 1267, 1172, 12


# Load our deck: prefer a real deck.csv (local dev convenience / actual
# Kaggle sandbox file) if one exists, otherwise the hardcoded DECK above --
# never write to cwd just to read the same data back, and never crash import
# just because no deck.csv happens to be sitting next to us.
DECK_PATH = "deck.csv"
if not os.path.exists(DECK_PATH):
    DECK_PATH = "/kaggle_simulations/agent/deck.csv"
try:
    with open(DECK_PATH, "r", encoding="utf-8") as f:
        my_deck = [int(line) for line in f.read().splitlines() if line.strip()]
    if len(my_deck) != 60:
        my_deck = list(DECK)
except Exception:
    my_deck = list(DECK)

# card_table[id] -> static CardData (weakness, ex/megaEx flags, stage, ...)
card_table = {card.cardId: card for card in all_card_data()}


class AttackPlan:
    """The single attack the heuristic committed to this turn.

    Populated once per MAIN decision by `HeuristicPolicy._plan_attack` and then
    read by many scorers so that energy attach, retreat, switch and gust all
    line up behind the SAME chosen attacker/target.
    """
    def __init__(self, attacker=-1, target=-1, attack_index=-1, remain_hp=-1, needs_energy=False):
        self.attacker = attacker          # board index of our attacker
        self.target = target              # board index of the opponent target
        self.attack_index = attack_index  # which of the attacker's 2 attacks
        self.remain_hp = remain_hp        # target HP left after the hit
        self.needs_energy = needs_energy  # must attach 1 energy to enable it


# Per-turn scratch state (the original used module globals; kept for fidelity).
plan = AttackPlan()
pre_turn = -1
ability_used = False

# Fallback-layer diagnostics: counts how often each of the three never-crash
# layers actually triggers, so a silently-broken search or heuristic shows up
# as a nonzero rate instead of vanishing into "the agent didn't crash."
_DIAG = defaultdict(int)


def diag_reset() -> None:
    _DIAG.clear()


def diag_snapshot() -> dict:
    total = max(1, _DIAG.get("decisions", 0))
    out = dict(_DIAG)
    out["fallback_rate"] = (
        _DIAG.get("search_fallback", 0)
        + _DIAG.get("empty_ordered_fallback", 0)
        + _DIAG.get("heuristic_fallback", 0)
        + _DIAG.get("parse_fallback", 0)
    ) / total
    return out


def get_card(obs: Observation, area: AreaType, index: int, player_index: int):
    """Resolve (area, index, player) -> the Card/Pokemon object it points at.

    Options refer to cards indirectly by location; this is the single lookup
    that turns those coordinates into a concrete object.
    """
    player = obs.current.players[player_index]
    if area == AreaType.DECK: return obs.select.deck[index]
    if area == AreaType.HAND: return player.hand[index]
    if area == AreaType.DISCARD: return player.discard[index]
    if area == AreaType.ACTIVE: return player.active[index]
    if area == AreaType.BENCH: return player.bench[index]
    if area == AreaType.PRIZE: return player.prize[index]
    if area == AreaType.STADIUM: return obs.current.stadium[index]
    if area == AreaType.LOOKING: return obs.current.looking[index]
    return None


# =========================================================================
# LAYER: target valuation — "how much do I want this Pokemon gone / fed?"
# =========================================================================
def prize_count(pokemon: Pokemon) -> int:
    """Prizes the opponent takes for KO-ing this Pokemon (1 / 2 ex / 3 megaEx),
    reduced by Legacy Energy and Lillie's Pearl mitigations."""
    data = card_table[pokemon.id]
    count = 3 if data.megaEx else 2 if data.ex else 1
    for card in pokemon.energyCards:
        if card.id == C.LEGACY_ENERGY:
            count -= 1
    for card in pokemon.tools:
        if card.id == C.LILLIES_PEARL and "Lillie" in data.name:
            count -= 1
    return max(0, count)


def target_score(pokemon: Pokemon) -> int:
    """Priority value of KO-ing this opponent Pokemon.

    Prizes dominate (x2000); attached energy and tools are worth denying;
    evolved stages are juicier; a few specific threats get hand-tuned bumps.
    """
    data = card_table[pokemon.id]
    score = prize_count(pokemon) * 2000 + len(pokemon.energies) * 300 + len(pokemon.tools) * 200
    if data.stage2:
        score += 500
    elif data.stage1:
        score += 250
    if pokemon.id in {144, 322, 323, 337}:  # low-value / utility mons
        score -= 200
    if pokemon.id == C.SNOVER:
        score += 950
    elif pokemon.id == C.MEGA_ABOMASNOW_EX:
        score += 250
    if pokemon.id == C.RIOLU:
        score += 800
    elif pokemon.id == C.MEGA_LUCARIO_EX:
        score += 100
    return score + pokemon.hp


# =========================================================================
# LAYER: HeuristicPolicy — the actual brain (ranks every legal option)
# =========================================================================
class HeuristicPolicy:
    """Hand-written policy. `choose()` returns a RANKED LIST of every legal
    option index (not a single pick). Search re-orders the head of that list;
    the tail is a free crash-safe fallback.
    """

    def __init__(self, obs: Observation):
        self.obs = obs
        self.state = obs.current
        self.select = obs.select
        self.context = self.select.context
        self.my_index = self.state.yourIndex
        self.op_index = 1 - self.my_index
        self.me = self.state.players[self.my_index]
        self.opponent = self.state.players[self.op_index]

        # cheap board census, filled by _count_cards / _scan_main_options
        self.field_counts = defaultdict(int)
        self.hand_counts = defaultdict(int)
        self.discard_counts = defaultdict(int)
        self.has_ready_lucario_line = False
        self.has_ready_hariyama_line = False
        self.can_switch = self.can_gust = self.can_attack = self.can_use_mega_brave = False
        self.stadium_id = self.state.stadium[0].id if self.state.stadium else 0

        self._count_cards()
        self._scan_main_options()

    # ---- top-level entry ------------------------------------------------
    def choose(self) -> list[int]:
        """Score every legal option, return ALL indices sorted best-first.
        Never truncate here: maxCount is enforced at the agent() boundary,
        and search needs the full ranking (maxCount is 1 on MAIN decisions,
        so truncating here starves the bandit down to a single candidate)."""
        if not self.select.option or self.select.maxCount == 0:
            return []
        if self.context == SelectContext.MAIN:
            self._plan_attack()
        scores = [self._score_option(o) for o in self.select.option]
        ranked = [i for i, _ in sorted(enumerate(scores), key=lambda kv: kv[1], reverse=True)]
        self._remember_lunatone_ability(ranked)
        return ranked

    # ---- board census ---------------------------------------------------
    def _count_cards(self) -> None:
        for pokemon in self.me.active + self.me.bench:
            if pokemon is None:
                continue
            self.field_counts[pokemon.id] += 1
            if pokemon.id in {C.MAKUHITA, C.HARIYAMA} and len(pokemon.energies) >= 3:
                self.has_ready_hariyama_line = True
            if pokemon.id in {C.RIOLU, C.MEGA_LUCARIO_EX} and len(pokemon.energies) >= 2:
                self.has_ready_lucario_line = True
        for card in self.me.hand:
            self.hand_counts[card.id] += 1
        for card in self.me.discard:
            self.discard_counts[card.id] += 1

    def _scan_main_options(self) -> None:
        """Precompute which meta-moves are legal this turn (switch/gust/attack)."""
        if self.context != SelectContext.MAIN:
            return
        for option in self.select.option:
            if option.type == OptionType.PLAY:
                card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
                if card.id == C.SWITCH: self.can_switch = True
                elif card.id == C.BOSS_ORDERS: self.can_gust = True
            elif option.type == OptionType.EVOLVE:
                card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
                if card.id == C.HARIYAMA: self.can_gust = True
            elif option.type == OptionType.RETREAT:
                self.can_switch = True
            elif option.type == OptionType.ATTACK:
                self.can_attack = True
                if option.attackId == MEGA_BRAVE:
                    self.can_use_mega_brave = True

    # ---- board helpers --------------------------------------------------
    def _my_board(self): return self.me.active + self.me.bench
    def _opponent_board(self): return self.opponent.active + self.opponent.bench
    def _opponent_has(self, ids): return any(p is not None and p.id in ids for p in self._opponent_board())
    def _opponent_is_water_deck(self): return self._opponent_has({C.KYOGRE, C.SNOVER, C.MEGA_ABOMASNOW_EX})
    def _opponent_is_crustle_wall(self): return self._opponent_has({344, 345})


    # ---- attack planning ------------------------------------------------
    def _can_evolve_board_index(self, board_index: int) -> bool:
        """Is there a legal EVOLVE option targeting this board slot this turn?"""
        for option in self.select.option:
            if option.type != OptionType.EVOLVE:
                continue
            target_index = option.inPlayIndex + (1 if option.inPlayArea == AreaType.BENCH else 0)
            if target_index == board_index:
                return True
        return False

    def _base_attack(self, pokemon: Pokemon, attack_index: int):
        """(energy_required, base_damage, base_score) for a Pokemon's attack,
        or None if that attack doesn't exist / can't fire. Damage numbers are
        card-specific and hand-encoded (this is the deck's own attack table)."""
        energy_required = base_damage = base_score = 0
        if pokemon.id == C.MEGA_LUCARIO_EX:
            if attack_index == 0:
                energy_required, base_damage = 1, 130
                # bonus damage scales with discarded Fighting Energy (capped)
                base_score += 60 * min(3, self.discard_counts[C.BASIC_FIGHTING_ENERGY])
            else:
                energy_required, base_damage = 2, 270
            # avoid feeding a water opponent prizes when they're about to close
            if self._opponent_is_water_deck() and len(self.opponent.prize) <= 3:
                base_score -= 500
        elif attack_index == 1:
            return None
        elif pokemon.id == C.HARIYAMA:
            energy_required, base_damage = 3, 210
        elif pokemon.id == C.MAKUHITA:
            return None
        elif pokemon.id == C.SOLROCK and self.field_counts[C.LUNATONE] >= 1:
            energy_required, base_damage = 1, 70
        if base_damage <= 0:
            return None
        return energy_required, base_damage, base_score

    def _base_attack_after_evolution(self, pokemon: Pokemon, board_index: int, attack_index: int):
        """Look ahead: if Makuhita can evolve into Hariyama this turn, plan the
        Hariyama attack (with a penalty for the evolution cost)."""
        if pokemon.id == C.MAKUHITA and attack_index == 0 and self._can_evolve_board_index(board_index):
            return 3, 210, -100
        return self._base_attack(pokemon, attack_index)

    def _plan_attack(self) -> None:
        """Pick THE attack for this turn: iterate (attacker, attack, target),
        score each by prizes taken / damage dealt, and store the winner in the
        module-global `plan`. Everything else this turn lines up behind it."""
        global plan
        best_score = -1
        plan = AttackPlan()
        if self.state.turn < 2:  # no attacking on the opening turn
            return

        for attacker_index, my_pokemon in enumerate(self._my_board()):
            if my_pokemon is None:
                continue
            # can only attack from the bench if we can switch it up first
            if attacker_index != 0 and not self.can_switch:
                break

            for attack_index in range(2):
                attack = self._base_attack_after_evolution(my_pokemon, attacker_index, attack_index)
                if attack is None:
                    continue
                energy_required, base_damage, base_score = attack
                energy_count = len(my_pokemon.energies)
                # second attack from the active needs the Mega Brave option live
                if attack_index == 1 and attacker_index == 0 and energy_count >= 2 and not self.can_use_mega_brave:
                    break

                # can we reach the energy requirement by attaching one this turn?
                needs_energy = False
                if energy_count < energy_required:
                    if self.hand_counts[C.BASIC_FIGHTING_ENERGY] >= 1 and not self.state.energyAttached:
                        energy_count += 1
                        needs_energy = energy_count >= energy_required
                    if not needs_energy:
                        continue

                for target_index, op_pokemon in enumerate(self._opponent_board()):
                    if op_pokemon is None:
                        continue
                    # can only hit the bench if we can gust it into the active
                    if target_index != 0 and not self.can_gust:
                        break
                    # don't waste Mega Lucario into an immune Crustle wall slot
                    if self._opponent_is_crustle_wall() and my_pokemon.id == C.MEGA_LUCARIO_EX and op_pokemon.id == 345:
                        continue

                    # apply weakness / resistance
                    damage = base_damage
                    op_data = card_table[op_pokemon.id]
                    if op_data.weakness == EnergyType.FIGHTING:
                        damage *= 2
                    elif op_data.resistance == EnergyType.FIGHTING:
                        damage -= 30

                    # value: full target_score if it's a KO, else prorated by dmg
                    score = target_score(op_pokemon)
                    prize = prize_count(op_pokemon) if op_pokemon.hp <= damage else 0
                    if prize == 0:
                        score *= damage / op_pokemon.hp
                    if len(self.opponent.prize) <= prize:  # lethal for the game
                        score = 500000

                    # positional bonuses: prefer active attacker & active target
                    score += base_score + (220 if attacker_index == 0 else 0) \
                        + (300 if target_index == 0 else 0) + energy_count
                    if score > best_score:
                        best_score = score
                        plan = AttackPlan(attacker_index, target_index, attack_index,
                                          op_pokemon.hp - damage, needs_energy)


    # ---- energy targeting ----------------------------------------------
    def _energy_target_score(self, pokemon: Pokemon, active: bool) -> int:
        """How much do we want to attach energy to THIS Pokemon? Feeds both the
        ATTACH scorer and ATTACH_FROM card choices."""
        energy_count = len(pokemon.energies)
        score = 8000 + (10 if active else 0)
        if pokemon.id in {C.MAKUHITA, C.HARIYAMA}:
            if pokemon.id == C.HARIYAMA:
                score += 1
            if self._opponent_is_crustle_wall():
                score += 260 if energy_count < 3 else 30
            else:
                score += 100 if energy_count < 3 else 0
                score -= 50 if self.has_ready_hariyama_line else 0
        elif pokemon.id == C.LUNATONE:
            score -= 100
        elif pokemon.id == C.SOLROCK:
            score += 20 if energy_count < 1 else -100
        elif pokemon.id in {C.RIOLU, C.MEGA_LUCARIO_EX}:
            if pokemon.id == C.MEGA_LUCARIO_EX:
                score += 1
            score += 100 if energy_count < 2 else 0
            score -= 50 if self.has_ready_lucario_line else 0
        return score

    # ---- the scorer dispatch: one function per decision family ----------
    def _score_option(self, option) -> float:
        """Route each legal option to the scorer for its OptionType."""
        if option.type == OptionType.NUMBER: return option.number
        if option.type == OptionType.YES: return 100 if self.context == SelectContext.IS_FIRST else 1
        if option.type == OptionType.NO: return 0
        if option.type == OptionType.CARD: return self._score_card_choice(option)
        if option.type == OptionType.PLAY: return self._score_play(option)
        if option.type == OptionType.ATTACH: return self._score_attach(option)
        if option.type == OptionType.EVOLVE: return self._score_evolve(option)
        if option.type == OptionType.ABILITY: return self._score_ability(option)
        if option.type == OptionType.RETREAT: return 2000 if plan.attacker >= 1 else -1
        if option.type == OptionType.ATTACK:
            # fire the attack that matches the plan (Mega Brave <-> attack_index 1)
            return 1100 if (option.attackId == MEGA_BRAVE) == (plan.attack_index == 1) else 1000
        return 0


    def _score_card_choice(self, option) -> float:
        """CARD selections in many sub-contexts: switch target, setup, draw/keep,
        energy-from choices. Each context has its own preferences."""
        card = get_card(self.obs, option.area, option.index, option.playerIndex)
        if card is None:
            return 0
        if self.context in {SelectContext.SWITCH, SelectContext.TO_ACTIVE}:
            return self._score_active_choice(option, card)
        if self.context == SelectContext.SETUP_ACTIVE_POKEMON:
            return (2 if card.id == C.SOLROCK and self.state.firstPlayer == self.my_index
                    else 4 if card.id == C.SOLROCK
                    else 3 if card.id == C.RIOLU
                    else 1 if card.id == C.MAKUHITA else 0)
        if self.context == SelectContext.TO_HAND:
            # draw/search: avoid dupes, tune each line by board state & matchup
            score = 200 - self.hand_counts[card.id] * 100
            if card.id == C.MAKUHITA:
                score += (80 if self.field_counts[card.id] < 2 else -20) if self._opponent_is_crustle_wall() \
                    else (-10 if self.field_counts[card.id] >= 1 else 10)
            elif card.id == C.HARIYAMA:
                score += (120 if self.field_counts[C.MAKUHITA] >= 1 else -5) if self._opponent_is_crustle_wall() \
                    else (20 if self.field_counts[C.MAKUHITA] >= 1 else -20)
            elif card.id == C.LUNATONE:
                score += -250 if self.field_counts[card.id] >= 1 else 60
            elif card.id == C.SOLROCK:
                score += -250 if self.field_counts[card.id] >= 1 else 50
            elif card.id == C.RIOLU:
                score += (-150 if self.field_counts[C.RIOLU] + self.field_counts[C.MEGA_LUCARIO_EX] >= 2
                          else -3 if self.field_counts[C.RIOLU] + self.field_counts[C.MEGA_LUCARIO_EX] >= 1 else 40)
            elif card.id == C.MEGA_LUCARIO_EX:
                score += 40 if self.field_counts[C.RIOLU] >= 1 else -15
            elif card.id == C.BASIC_FIGHTING_ENERGY:
                score += 30 if not ability_used or not self.state.energyAttached else -1
            return score
        if self.context == SelectContext.ATTACH_FROM and isinstance(card, Pokemon):
            return self._energy_target_score(card, option.area == AreaType.ACTIVE)
        return 0

    def _score_active_choice(self, option, card) -> float:
        """Which Pokemon should be in the Active Spot (switch / promote)."""
        if not isinstance(card, Pokemon):
            return 0
        if option.playerIndex != self.my_index:
            # choosing the OPPONENT's new active (gust): match the plan target
            return 100 if option.index == plan.target - 1 else 0
        score = len(card.energies) * 2
        if option.index == plan.attacker - 1:
            score += 100
        if card.id == C.MEGA_LUCARIO_EX:
            score += 8 if self._opponent_is_water_deck() and len(self.opponent.prize) <= 3 else 20
        elif card.id == C.HARIYAMA and len(card.energies) >= 2:
            score += 45 if self._opponent_is_crustle_wall() else 15
        elif card.id == C.MAKUHITA and len(card.energies) >= 2:
            score += 35 if self._opponent_is_crustle_wall() else 10
        elif card.id == C.SOLROCK:
            score += 5
        elif card.id == C.RIOLU:
            score += 4
        return score


    def _score_play(self, option) -> float:
        """Playing a card from hand: Pokemon, trainers, tools, stadiums."""
        card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        data = card_table[card.id]
        if data.cardType == CardType.POKEMON:
            # avoid over-committing duplicate support Pokemon
            if card.id in {C.LUNATONE, C.SOLROCK} and self.field_counts[card.id] >= 1:
                return -1
            if card.id == C.RIOLU and self.field_counts[C.RIOLU] + self.field_counts[C.MEGA_LUCARIO_EX] >= 2:
                return -1
            return 20000
        if card.id == C.SWITCH:
            return 6000 if plan.attacker > 0 else -1
        if card.id == C.PREMIUM_POWER_PRO:
            if self.state.supporterPlayed and plan.remain_hp <= 0:
                return -1
            if not self.can_attack:
                return 3050 if (not self.state.supporterPlayed and self.hand_counts[C.CARMINE] > 0
                                and self.hand_counts[C.LILLIE_DETERMINATION] == 0
                                and not self.me.deckCount <= LOW_DECK_COUNT) else -1
            return 5000
        if card.id == C.BOSS_ORDERS:
            return 3200 if plan.target >= 1 else -1
        if card.id == C.CARMINE:
            if self._opponent_is_crustle_wall() and any(c.id in {C.HARIYAMA, C.MAKUHITA} for c in self.me.hand):
                return -1
            return -1 if self.me.deckCount <= LOW_DECK_COUNT else 3000
        if card.id == C.LILLIE_DETERMINATION:
            return -1 if self.me.deckCount <= LOW_DECK_COUNT else 3100
        if card.id == C.GRAVITY_MOUNTAIN:
            return 3500 if any(p is not None and card_table[p.id].stage2 for p in self._opponent_board()) \
                else (1200 if self.stadium_id else -1)
        return 10000

    def _score_attach(self, option) -> float:
        """Attaching an energy or a tool to one of our Pokemon."""
        card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        pokemon = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        if not isinstance(pokemon, Pokemon):
            return 0
        if card.id == C.HERO_CAPE:
            score = 7000
            if self._opponent_is_water_deck():
                return 12200 if pokemon.id == C.RIOLU else 12800 if pokemon.id == C.MEGA_LUCARIO_EX else score
            if pokemon.id == C.RIOLU: score += 100
            elif pokemon.id == C.MEGA_LUCARIO_EX: score += 200
            return score
        score = self._energy_target_score(pokemon, option.inPlayArea == AreaType.ACTIVE)
        board_index = option.inPlayIndex if option.inPlayArea == AreaType.ACTIVE else option.inPlayIndex + 1
        if board_index == plan.attacker and plan.needs_energy:
            score += 200  # prioritize enabling the planned attack
        return score

    def _score_evolve(self, option) -> float:
        pokemon = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        if not isinstance(pokemon, Pokemon):
            return 0
        # don't evolve the active Makuhita we intend to attack with this turn
        if pokemon.id == C.MAKUHITA and plan.target == 0 and not self._opponent_is_crustle_wall():
            return -1
        return 9000 + len(pokemon.energies)

    def _score_ability(self, option) -> float:
        card = get_card(self.obs, option.area, option.index, self.my_index)
        if card.id == C.LUMIOSE_CITY: return 1
        if card.id == C.LUNATONE and self.me.deckCount <= LOW_DECK_COUNT: return -1
        return 30000

    def _remember_lunatone_ability(self, ranked: list[int]) -> None:
        """Track once-per-turn Lunatone ability use across successive calls."""
        global ability_used
        if self.context != SelectContext.MAIN or not ranked:
            return
        option = self.select.option[ranked[0]]
        if option.type != OptionType.ABILITY:
            return
        card = get_card(self.obs, option.area, option.index, self.my_index)
        if card is not None and card.id == C.LUNATONE:
            ability_used = True


# =========================================================================
# LAYER: evaluate_state — the static board evaluation (value function)
# =========================================================================
def evaluate_state(obs) -> float:
    """Score a board from our perspective. Strict magnitude tiers:
    terminal (1e7) >> prize difference (1e4) >> board strength (1e2).
    Prizes are the win condition, so they dominate. Stall decks (Crustle,
    Snorlax) are special-cased because 'hit things' misreads them."""
    st = obs.current
    if st is None:
        return 0.0
    me, op = st.players[st.yourIndex], st.players[1 - st.yourIndex]

    prize_diff = len(op.prize) - len(me.prize)
    if len(me.prize) == 0:
        return 9999999.0   # we won
    if len(op.prize) == 0:
        return -9999999.0  # we lost
    val = prize_diff * 10000.0

    op_board = [op.active[0] if op.active else None] + list(op.bench)
    is_crustle = any(p is not None and p.id in {344, 345} for p in op_board)
    is_snorlax = any(p is not None and p.id == 143 for p in op_board)
    is_stall = is_crustle or is_snorlax

    # 1. our board strength
    for p in [me.active[0] if me.active else None] + list(me.bench):
        if p is None:
            continue
        val += len(p.energies) * 200.0
        if is_crustle:
            if p.id == C.HARIYAMA: val += 1500.0
            elif p.id == C.MAKUHITA: val += 800.0
        else:
            if p.id == C.MEGA_LUCARIO_EX: val += 500.0
            elif p.id == C.HARIYAMA: val += 300.0
            elif p.id in {C.RIOLU, C.MAKUHITA}: val += 100.0

    # 1.5 hold the anti-stall attackers in hand until needed
    if is_crustle:
        for c in me.hand:
            if c.id == C.HARIYAMA: val += 1000.0
            elif c.id == C.MAKUHITA: val += 500.0

    if me.active and me.active[0] is not None:
        val += me.active[0].hp * 2.0
        if len(me.active[0].energies) >= 2:
            val += 500.0

    # 2. predictive threat map: assume opponent attaches 1 energy next turn
    op_max_damage = 0
    for p in op_board:
        if p is None:
            continue
        val -= p.hp * 1.5
        assumed_energies = len(p.energies) + 1
        if p.id == C.MEGA_LUCARIO_EX: op_dmg = 270 if assumed_energies >= 2 else 130 if assumed_energies >= 1 else 0
        elif p.id == C.HARIYAMA: op_dmg = 210 if assumed_energies >= 3 else 0
        elif p.id == C.KYOGRE: op_dmg = 130 if assumed_energies >= 3 else 0
        elif p.id == C.MEGA_ABOMASNOW_EX: op_dmg = 240 if assumed_energies >= 4 else 0
        elif p.id == 121: op_dmg = 130 if assumed_energies >= 2 else 0  # Dragapult ex
        else: op_dmg = assumed_energies * 40
        op_max_damage = max(op_max_damage, op_dmg)

    # 3. lethal-threat penalty: are we about to lose our active?
    if me.active and me.active[0] is not None:
        my_active = me.active[0]
        if op_max_damage >= my_active.hp:
            prize_risk = 2 if my_active.id == C.MEGA_LUCARIO_EX else 1
            val -= prize_risk * 4000.0
        elif op_max_damage > 0:
            val -= op_max_damage * 1.5

    # 4. anti-stall resource conservation
    deck_c = getattr(me, "deckCount", 60)
    if is_stall:
        val += deck_c * 30.0
        val += getattr(me, "handCount", len(me.hand)) * 2.0
    else:
        val += getattr(me, "handCount", len(me.hand)) * 10.0
    if deck_c < 5:
        val -= 10000.0
    return val


# =========================================================================
# LAYER: rollout — play our own turn out with the heuristic, one turn deep
# =========================================================================
def rollout_turn(sid, cur_obs, your_index):
    """Drive the engine forward using the heuristic until OUR turn ends
    (or the game ends, or a 20-step guard trips). We never simulate the
    opponent — the search horizon is exactly one of our turns."""
    steps = 0
    while steps < 20:
        if cur_obs.current.result is not None and cur_obs.current.result != -1:
            break                                   # game over
        if cur_obs.current.yourIndex != your_index:
            break                                   # our turn ended -> stop
        if cur_obs.select.context != SelectContext.MAIN:
            sub = HeuristicPolicy(cur_obs).choose()
            sel = sub[: max(1, cur_obs.select.minCount)]
        else:
            nxt = HeuristicPolicy(cur_obs).choose()
            if not nxt:
                break
            sel = [nxt[0]]
            if cur_obs.select.option[nxt[0]].type == OptionType.END:
                search_step(sid, sel)
                break
        # search_step returns a SearchState directly (.observation/.searchId);
        # any engine failure raises, so guard with try/except.
        try:
            step = search_step(sid, sel)
        except Exception:
            break
        if step is None or step.observation is None:
            break
        cur_obs, sid = step.observation, step.searchId
        steps += 1
    return cur_obs


def _plausible_opponent(obs):
    """Build best-effort predictions for the opponent's hidden zones so the
    engine's search_begin has legal inputs. We do NOT know these cards, so we
    fill them with a generic Basic (Riolu) — this is a determinization of our
    OWN uncertainty, not real opponent belief modelling."""
    st = obs.current
    op = st.players[1 - st.yourIndex]
    filler = C.RIOLU  # any Basic Pokemon id is a legal placeholder
    op_deck = [filler] * getattr(op, "deckCount", 0)
    op_prize = [filler] * len(op.prize)
    op_hand = [filler] * getattr(op, "handCount", len(getattr(op, "hand", []) or []))
    # opponent active only needed if it's face-down; pass its id if visible
    op_active = []
    if op.active and op.active[0] is not None:
        op_active = [op.active[0].id]
    return op_deck, op_prize, op_hand, op_active


def simulate_action(obs, action) -> float:
    """Determinize + roll out a single root action, return the eval of the
    resulting board. THE FIXED search_begin CALL lives here.

    The original passed `search_begin(obs, your_deck=yd)`; the current engine
    needs all 7 args, so here we supply our sampled deck order plus plausible
    opponent zones."""
    st = obs.current
    my_p = st.players[st.yourIndex]

    # sample a plausible order for OUR remaining deck (resolves our draw luck)
    n_deck = getattr(my_p, "deckCount", 0)
    your_deck = random.sample(my_deck, n_deck) if n_deck else []
    your_prize = [C.BASIC_FIGHTING_ENERGY] * len(my_p.prize)
    op_deck, op_prize, op_hand, op_active = _plausible_opponent(obs)

    # NOTE: in the current engine BOTH search_begin and search_step return a
    # SearchState directly (fields: .observation, .searchId); there is no
    # ApiResult wrapper and no .error field — failures raise exceptions. The
    # original code assumed a `.state.searchId` wrapper, which no longer exists.
    began = False
    try:
        root = search_begin(
            obs,
            your_deck=your_deck,
            your_prize=your_prize,
            opponent_deck=op_deck,
            opponent_prize=op_prize,
            opponent_hand=op_hand,
            opponent_active=op_active,
        )
        began = True
        step = search_step(root.searchId, [action])
        if step is None or step.observation is None:
            return -float("inf")
        cur = rollout_turn(step.searchId, step.observation, st.yourIndex)
        val = evaluate_state(cur)
        # PERSPECTIVE FIX (2026-08-12): when the rollout ends because the turn
        # passed (e.g. an attack), the engine renders the final observation for
        # the OPPONENT — yourIndex flips — and evaluate_state is view-relative.
        # Without negation every turn-ending line is scored as the opponent's
        # advantage, inverting the leaf on exactly the most common rollouts.
        # evaluate_state is not perfectly antisymmetric (hand/deck economy terms
        # differ), but its dominant tiers (terminal 1e7, prizes 1e4) are, so
        # negation restores the sign of what actually decides comparisons.
        if cur.current is not None and cur.current.yourIndex != st.yourIndex:
            val = -val
        return val
    except Exception:
        return -float("inf")
    finally:
        # LEAK FIX (2026-08-12): search_end() is what lets the native engine
        # reuse the states this simulation allocated ("Memory used during the
        # search will be reused in the next search"). Without it every sim
        # leaks engine states permanently — invisible on a laptop, an OOM on
        # the evaluator's ~197 MiB envelope. agent_core_improved always did
        # this; this lineage never did.
        if began:
            try:
                search_end()
            except Exception:
                pass


# =========================================================================
# LAYER: flat_monte_carlo_search — UCB1 bandit over the top-K root actions
# =========================================================================
def flat_monte_carlo_search(obs, base_order=None, override_margin=0.0):
    """Re-rank the top-K MAIN moves of a base ranking by Monte-Carlo simulation.

    This is MCTS with the expansion step removed: SELECT the next candidate to
    try with UCB1, SIMULATE it with a determinized one-turn rollout, and
    BACKPROP the value into a running mean. No tree is built beyond the root.

    `base_order` is the candidate-ranking seam: a full best-first list of
    option indices. Defaults to this file's own heuristic; external arms
    (e.g. an IL policy) may pass their own ranking and reuse the bandit,
    rollout, and evaluator unchanged.

    `override_margin` is the veto seam: search may displace the base
    ranking's top action only if its mean simulated value beats that
    action's mean by at least this much (in leaf-value units). 0.0 (the
    default) reproduces the unconditional-override behavior; large values
    turn search into a blunder veto that mostly defers to the base policy.

    Returns a full permutation of option indices (best first), or None if
    search is unavailable / not applicable so the caller can fall back."""
    if not (_SEARCH_OK and USE_SEARCH):
        return None
    select = obs.select
    if select is None or select.context != SelectContext.MAIN:
        return None
    t0 = time.time()

    if base_order is None:
        base_order = HeuristicPolicy(obs).choose()
    candidates = base_order[:SEARCH_MAX_CANDIDATES]
    if not candidates:
        return None
    if len(candidates) == 1:
        return [candidates[0]] + [i for i in base_order if i != candidates[0]]

    visits = {a: 0 for a in candidates}
    total_val = {a: 0.0 for a in candidates}

    try:
        # Phase 1 — try each candidate once (initial exploration).
        for a in candidates:
            if time.time() - t0 > SEARCH_TIME_BUDGET:
                break
            val = simulate_action(obs, a)
            if val != -float("inf"):
                visits[a] += 1
                total_val[a] += val

        # Phase 2 — spend the remaining budget via UCB1.
        while time.time() - t0 < SEARCH_TIME_BUDGET:
            total_visits = sum(visits.values())
            if total_visits == 0:
                break
            valid_scores = [total_val[a] / visits[a] for a in candidates if visits[a] > 0]
            if not valid_scores:
                break
            # min-max normalize means into [0,1] so the sqrt(2)-style constant
            # behaves; then use a SMALLER C (0.5): fewer sims -> exploit more.
            min_s, max_s = min(valid_scores), max(valid_scores)
            if max_s == min_s:
                max_s = min_s + 1.0

            best_ucb, best_a = -float("inf"), candidates[0]
            for a in candidates:
                if visits[a] == 0:            # always try an unvisited arm first
                    best_a = a
                    break
                avg = total_val[a] / visits[a]
                norm_avg = (avg - min_s) / (max_s - min_s)
                ucb = norm_avg + 0.5 * math.sqrt(math.log(total_visits) / visits[a])
                if ucb > best_ucb:
                    best_ucb, best_a = ucb, a

            val = simulate_action(obs, best_a)
            if val != -float("inf"):
                visits[best_a] += 1
                total_val[best_a] += val

        # winner = highest empirical mean; keep the rest of the heuristic order
        mean = lambda a: total_val[a] / visits[a] if visits[a] > 0 else -float("inf")
        best_action = max(candidates, key=mean)
        base_top = base_order[0]
        if override_margin > 0.0 and best_action != base_top:
            # An override must DEMONSTRATE the margin. If the base policy's
            # top action was never successfully simulated, the gap cannot be
            # measured — defer to the base policy rather than overriding on
            # one-sided evidence.
            if (visits.get(base_top, 0) == 0
                    or mean(best_action) - mean(base_top) < override_margin):
                best_action = base_top
        return [best_action] + [i for i in base_order if i != best_action]
    except Exception:
        return None


# =========================================================================
# LAYER: agent — the never-crash entry point Kaggle calls each decision
# =========================================================================
def agent(obs_dict: dict) -> list[int]:
    """Convert the raw dict, run search (falling back to the heuristic), and
    always return a legal list of option indices. Three fallback layers:
    search -> heuristic -> trivial range()."""
    try:
        obs = to_observation_class(obs_dict)
    except Exception:
        _DIAG["decisions"] += 1
        _DIAG["parse_fallback"] += 1
        return my_deck if obs_dict.get("select") is None else [0]
    if obs.select is None:
        return my_deck  # deck-submission phase

    global pre_turn, ability_used, plan
    if pre_turn != obs.current.turn:      # reset per-turn scratch state
        pre_turn = obs.current.turn
        ability_used = False
        plan = AttackPlan()

    _DIAG["decisions"] += 1
    try:
        ordered = flat_monte_carlo_search(obs)
        if ordered is None:
            _DIAG["search_fallback"] += 1
            ordered = HeuristicPolicy(obs).choose()
        n = len(obs.select.option)
        ordered = [i for i in ordered if 0 <= i < n]
        if not ordered:
            _DIAG["empty_ordered_fallback"] += 1
            return list(range(min(max(1, obs.select.minCount), n)))
        k = max(min(obs.select.maxCount, n), min(max(1, obs.select.minCount), n))
        return ordered[:k]
    except Exception:
        _DIAG["heuristic_fallback"] += 1
        n = len(obs.select.option)
        return list(range(min(max(1, obs.select.minCount), n)))
