"""
Improved Probabilistic agent — faithful, readable reimplementation of
Ivan Ternovskiy's "Improved Probabilistic agent" (Kaggle score 967.7,
author: aristophanivan).

WHAT IT ACTUALLY IS (the title lies):
  "Probabilistic Expectimax" is not expectimax. It is
  DETERMINIZED FLAT MONTE-CARLO SEARCH WITH A UCB1 BANDIT — i.e. MCTS at
  depth 1 with the expansion step removed. There is no tree, no opponent
  model, and a one-turn rollout horizon.

THE SHAPE:
  1. A hand-written heuristic `AdvancedPolicy` proposes a *ranked* list of
     every legal move (best first). This is the actual brain and must be
     good on its own.
  2. A flat UCB1 bandit (SEARCH_ALGO) re-ranks only the top 8 candidates by
     playing each one out with the REAL game engine (cg.api used as a forward
     model) and scoring the resulting board with evaluate_state.
  3. Every non-MAIN decision skips search and uses the raw heuristic.
  4. Three independent fallback layers guarantee the agent never crashes:
       search fails -> heuristic ranking
       heuristic fails -> [0, ...] (legal lowest indices)
       parse fails    -> the deck (initial) or [0]

This module is self-contained and import-safe: `agent(obs_dict)` is the
entry point the competition harness calls.

Run a local self-play sanity check from the repo root with:
    uv run python -c "from kaggle_environments import make; \
        import sys; sys.path.insert(0,'agents/mega_lucario'); \
        import agent_core_improved as a; \
        print(make('cabt').run([a.agent, a.agent])[-1][0]['status'])"
"""

from __future__ import annotations

import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

# --- make `cg` importable both in the bundle (cg/ next to this file) and
# --- locally (data/external/cg-lib/cg). Harmless in the Kaggle sandbox.
_HERE = Path(__file__).resolve().parent
for _cand in (
    _HERE,
    _HERE.parent.parent / "data" / "external" / "cg-lib",
    Path("/kaggle_simulations/agent"),
):
    if (_cand / "cg" / "api.py").exists():
        sys.path.insert(0, str(_cand))
        break

from cg.api import (  # noqa: E402
    AreaType,
    Card,
    CardType,
    EnergyType,
    Observation,
    OptionType,
    Pokemon,
    SelectContext,
    all_card_data,
    to_observation_class,
)

# Engine search API is optional: if it is unavailable the agent silently
# degrades to the pure heuristic (never-crash contract).
_SEARCH_OK = False
try:
    from cg.api import search_begin, search_step, search_end  # noqa: F401

    _SEARCH_OK = True
except Exception:  # pragma: no cover - depends on environment
    pass

# BC-policy prior for SEARCH_ALGO's UCB1 term (see bc_prior.py) is likewise
# optional: if the checkpoint or its ML dependencies aren't available,
# SEARCH_ALGO falls back to plain UCB1, identical to before this existed.
_BC_PRIOR_OK = False
try:
    import bc_prior

    _BC_PRIOR_OK = True
except Exception:  # pragma: no cover - depends on environment
    pass

# Printed at import time (not just under __main__) so this shows up in real
# match stdout/logs -- the only way to tell after the fact whether a given
# submission's BC prior actually loaded, vs. silently degrading to plain
# UCB1 per the except-and-pass above.
print(f"bc_prior available: {_BC_PRIOR_OK}")


# ---------------------------------------------------------------------------
# 1. Deck configuration (the exact 60-card Mega Lucario ex list)
# ---------------------------------------------------------------------------
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
assert len(DECK) == 60, f"deck must be 60 cards, got {len(DECK)}"

# Card-id constants (mirrors the original `class C`).
class C:
    KYOGRE, SNOVER, MEGA_ABOMASNOW_EX = 721, 722, 723
    MAKUHITA, HARIYAMA = 673, 674
    LUNATONE, SOLROCK = 675, 676
    RIOLU, MEGA_LUCARIO_EX = 677, 678
    BASIC_FIGHTING_ENERGY = 6
    DUSK_BALL, SWITCH, PREMIUM_POWER_PRO, FIGHTING_GONG = 1102, 1123, 1141, 1142
    POKE_PAD, HERO_CAPE, BOSS_ORDERS = 1152, 1159, 1182
    CARMINE, LILLIE_DETERMINATION, GRAVITY_MOUNTAIN = 1192, 1227, 1252
    LUMIOSE_CITY, LILLIES_PEARL, LEGACY_ENERGY = 1267, 1172, 12


MEGA_BRAVE = 983
LOW_DECK_COUNT = 10

# `my_deck` is what the agent returns when first asked for the deck. Prefer the
# bundled deck.csv; fall back to the hardcoded DECK so the agent is runnable
# even without the file (e.g. quick local tests).
DECK_PATH = "deck.csv"
if not os.path.exists(DECK_PATH):
    DECK_PATH = "/kaggle_simulations/agent/deck.csv"
try:
    with open(DECK_PATH, "r", encoding="utf-8") as _f:
        my_deck = [int(line) for line in _f.read().splitlines() if line.strip()][:60]
    if len(my_deck) != 60:
        my_deck = list(DECK)
except Exception:
    my_deck = list(DECK)

all_card = all_card_data()
card_table = {card.cardId: card for card in all_card}


# ---------------------------------------------------------------------------
# 2. Tiny helpers
# ---------------------------------------------------------------------------
class AttackPlan:
    """State of the planned attack for the current turn (mirrors the source)."""

    def __init__(self, attacker=-1, target=-1, attack_index=-1, remain_hp=-1, needs_energy=False):
        self.attacker = attacker
        self.target = target
        self.attack_index = attack_index
        self.remain_hp = remain_hp
        self.needs_energy = needs_energy


# Module-level turn state (original pattern).
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
    """Safely extract a Card or Pokemon from a zone by (area, index)."""
    player = obs.current.players[player_index]
    mapping = {
        AreaType.DECK: lambda: obs.select.deck[index],
        AreaType.HAND: lambda: player.hand[index],
        AreaType.DISCARD: lambda: player.discard[index],
        AreaType.ACTIVE: lambda: player.active[index],
        AreaType.BENCH: lambda: player.bench[index],
        AreaType.PRIZE: lambda: player.prize[index],
        AreaType.STADIUM: lambda: obs.current.stadium[index],
        AreaType.LOOKING: lambda: obs.current.looking[index],
    }
    getter = mapping.get(area)
    return getter() if getter is not None else None


def prize_count(pokemon: Pokemon) -> int:
    """Prize cards a Pokemon yields when KO'd, with Legacy Energy / Lillie's Pearl."""
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
    """Tactical worth of targeting a given opponent Pokemon (used in attack planning)."""
    data = card_table[pokemon.id]
    score = prize_count(pokemon) * 2000 + len(pokemon.energies) * 300 + len(pokemon.tools) * 200
    if data.stage2:
        score += 500
    elif data.stage1:
        score += 250
    if pokemon.id in {144, 322, 323, 337}:  # Squawkabilly ex, Noctowl, Fan Rotom, Archaludon ex
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


# ---------------------------------------------------------------------------
# 3. Static evaluation (the search target function)
# ---------------------------------------------------------------------------
def evaluate_state(obs) -> float:
    """Board value used to score a simulated end-of-turn state.

    Strict lexicographic priority by magnitude:
        terminal win/loss (1e7)  >>  prize difference (1e4)  >>  board/energy (1e2)
    Prizes dominate because they ARE the win condition. Stall decks (Crustle,
    Snorlax) get special-cased because the normal "hit things" evaluation
    misreads them.
    """
    st = obs.current
    if st is None:
        return 0.0
    me, op = st.players[st.yourIndex], st.players[1 - st.yourIndex]

    prize_diff = len(op.prize) - len(me.prize)
    if len(me.prize) == 0:
        return 9999999.0
    if len(op.prize) == 0:
        return -9999999.0
    val = prize_diff * 10000.0

    op_active = [op.active[0] if op.active else None] + list(op.bench)
    is_crustle = any(p is not None and p.id in {344, 345} for p in op_active)
    is_snorlax = any(p is not None and p.id == 143 for p in op_active)
    is_stall = is_crustle or is_snorlax

    # 1. My board strength
    my_active = [me.active[0] if me.active else None] + list(me.bench)
    for p in my_active:
        if p is None:
            continue
        val += len(p.energies) * 200.0
        if is_crustle:
            if p.id == C.HARIYAMA:
                val += 1500.0
            elif p.id == C.MAKUHITA:
                val += 800.0
        else:
            if p.id == C.MEGA_LUCARIO_EX:
                val += 500.0
            elif p.id == C.HARIYAMA:
                val += 300.0
            elif p.id in {C.RIOLU, C.MAKUHITA}:
                val += 100.0

    # 1.5 Hand conservation (crucial against stall)
    if is_crustle:
        for c in me.hand:
            if c.id == C.HARIYAMA:
                val += 1000.0
            elif c.id == C.MAKUHITA:
                val += 500.0

    if me.active and me.active[0] is not None:
        val += me.active[0].hp * 2.0
        if len(me.active[0].energies) >= 2:
            val += 500.0

    # 2. Predictive threat mapping (assume opponent attaches 1 energy next turn)
    op_max_damage = 0
    for p in op_active:
        if p is None:
            continue
        val -= p.hp * 1.5
        assumed_energies = len(p.energies) + 1
        if p.id == C.MEGA_LUCARIO_EX:
            op_dmg = 270 if assumed_energies >= 2 else 130 if assumed_energies >= 1 else 0
        elif p.id == C.HARIYAMA:
            op_dmg = 210 if assumed_energies >= 3 else 0
        elif p.id == C.KYOGRE:
            op_dmg = 130 if assumed_energies >= 3 else 0
        elif p.id == C.MEGA_ABOMASNOW_EX:
            op_dmg = 240 if assumed_energies >= 4 else 0
        elif p.id == 121:  # Dragapult ex
            op_dmg = 130 if assumed_energies >= 2 else 0
        else:
            op_dmg = assumed_energies * 40
        op_max_damage = max(op_max_damage, op_dmg)

    # 3. Lethal threat penalty
    if me.active and me.active[0] is not None:
        my_active = me.active[0]
        if op_max_damage >= my_active.hp:
            prize_risk = 2 if my_active.id == C.MEGA_LUCARIO_EX else 1
            val -= prize_risk * 4000.0
        elif op_max_damage > 0:
            val -= op_max_damage * 1.5

    # 4. Anti-stall deck/hand conservation
    deck_c = getattr(me, "deckCount", 60)
    if is_stall:
        val += deck_c * 30.0
        val += getattr(me, "handCount", len(me.hand)) * 2.0
    else:
        val += getattr(me, "handCount", len(me.hand)) * 10.0

    if deck_c < 5:
        # Graded, not a cliff: -10000 flat here used to equal a full prize
        # swing regardless of how close to actually decking out (0 cards)
        # the state was, so a one-turn rollout landing at deck_c=4 read as
        # catastrophic as deck_c=0 and swamped every other signal in this
        # function -- which made UCB1 flee otherwise-correct plays late
        # game. Scale with actual proximity to the real deck-out loss.
        val -= (5 - deck_c) * 300.0
    return val


# ---------------------------------------------------------------------------
# 4. AdvancedPolicy — the heuristic brain
# ---------------------------------------------------------------------------
class AdvancedPolicy:
    """Per-context heuristic that returns a *ranked* list of legal option indices."""

    def __init__(self, obs: Observation):
        self.obs = obs
        self.state = obs.current
        self.select = obs.select
        self.context = self.select.context
        self.my_index = self.state.yourIndex
        self.op_index = 1 - self.my_index
        self.me = self.state.players[self.my_index]
        self.opponent = self.state.players[self.op_index]

        self.field_counts = defaultdict(int)
        self.hand_counts = defaultdict(int)
        self.discard_counts = defaultdict(int)
        self.has_ready_lucario_line = False
        self.has_ready_hariyama_line = False
        self.can_switch = self.can_gust = self.can_attack = self.can_use_mega_brave = False
        self.stadium_id = self.state.stadium[0].id if self.state.stadium else 0

        self._count_cards()
        self._scan_main_options()

    # -- setup ------------------------------------------------------------
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
        if self.context != SelectContext.MAIN:
            return
        for option in self.select.option:
            if option.type == OptionType.PLAY:
                card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
                if card.id == C.SWITCH:
                    self.can_switch = True
                elif card.id == C.BOSS_ORDERS:
                    self.can_gust = True
            elif option.type == OptionType.EVOLVE:
                card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
                if card.id == C.HARIYAMA:
                    self.can_gust = True
            elif option.type == OptionType.RETREAT:
                self.can_switch = True
            elif option.type == OptionType.ATTACK:
                self.can_attack = True
                if option.attackId == MEGA_BRAVE:
                    self.can_use_mega_brave = True

    # -- board helpers ----------------------------------------------------
    def _my_board(self):
        return self.me.active + self.me.bench

    def _opponent_board(self):
        return self.opponent.active + self.opponent.bench

    def _opponent_has(self, ids: set) -> bool:
        return any(p is not None and p.id in ids for p in self._opponent_board())

    def _opponent_is_water_deck(self) -> bool:
        return self._opponent_has({C.KYOGRE, C.SNOVER, C.MEGA_ABOMASNOW_EX})

    def _opponent_is_crustle_wall(self) -> bool:
        return self._opponent_has({344, 345})

    def _can_evolve_board_index(self, board_index: int) -> bool:
        for option in self.select.option:
            if option.type != OptionType.EVOLVE:
                continue
            target_index = option.inPlayIndex + (1 if option.inPlayArea == AreaType.BENCH else 0)
            if target_index == board_index:
                return True
        return False

    # -- attack planning (looks ahead at damage, incl. post-evolution) ---
    def _base_attack(self, pokemon: Pokemon, attack_index: int):
        energy_required = base_damage = base_score = 0
        if pokemon.id == C.MEGA_LUCARIO_EX:
            if attack_index == 0:
                energy_required, base_damage = 1, 130
                base_score += 60 * min(3, self.discard_counts[C.BASIC_FIGHTING_ENERGY])
            else:
                energy_required, base_damage = 2, 270
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
        if pokemon.id == C.MAKUHITA and attack_index == 0 and self._can_evolve_board_index(board_index):
            return 3, 210, -100
        return self._base_attack(pokemon, attack_index)

    def _plan_attack(self) -> None:
        global plan
        best_score = -1
        plan = AttackPlan()
        if self.state.turn < 2:
            return
        for attacker_index, my_pokemon in enumerate(self._my_board()):
            if my_pokemon is None:
                continue
            if attacker_index != 0 and not self.can_switch:
                break
            for attack_index in range(2):
                attack = self._base_attack_after_evolution(my_pokemon, attacker_index, attack_index)
                if attack is None:
                    continue
                energy_required, base_damage, base_score = attack
                energy_count = len(my_pokemon.energies)
                if attack_index == 1 and attacker_index == 0 and energy_count >= 2 and not self.can_use_mega_brave:
                    break
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
                    if target_index != 0 and not self.can_gust:
                        break
                    if self._opponent_is_crustle_wall() and my_pokemon.id == C.MEGA_LUCARIO_EX and op_pokemon.id == 345:
                        continue
                    damage = base_damage
                    op_data = card_table[op_pokemon.id]
                    if op_data.weakness == EnergyType.FIGHTING:
                        damage *= 2
                    elif op_data.resistance == EnergyType.FIGHTING:
                        damage -= 30
                    score = target_score(op_pokemon)
                    prize = prize_count(op_pokemon) if op_pokemon.hp <= damage else 0
                    if prize == 0:
                        score *= damage / op_pokemon.hp
                    if len(self.opponent.prize) <= prize:
                        score = 500000
                    score += base_score + (220 if attacker_index == 0 else 0) + (
                        300 if target_index == 0 else 0
                    ) + energy_count
                    if score > best_score:
                        best_score = score
                        plan = AttackPlan(
                            attacker_index, target_index, attack_index,
                            op_pokemon.hp - damage, needs_energy,
                        )

    # -- energy-target scoring -------------------------------------------
    def _energy_target_score(self, pokemon: Pokemon, active: bool) -> int:
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

    # -- the dispatcher ---------------------------------------------------
    def rank_all(self) -> list[int]:
        """Full heuristic ranking of every legal option, untruncated.

        `choose()` truncates this to `maxCount` for its callers (the game
        engine expects at most `maxCount` picks); SEARCH_ALGO needs the
        full ranking to draw its top-N candidates from, since at MAIN
        decisions maxCount is always 1 and truncating first would leave
        it only one candidate to "search" over.
        """
        if not self.select.option or self.select.maxCount == 0:
            return []
        if self.context == SelectContext.MAIN:
            self._plan_attack()
        scores = [self._score_option(option) for option in self.select.option]
        ranked = [i for i, _ in sorted(enumerate(scores), key=lambda item: item[1], reverse=True)]
        self._remember_lunatone_ability(ranked)
        return ranked

    def choose(self) -> list[int]:
        return self.rank_all()[: self.select.maxCount]

    def _score_option(self, option) -> float:
        if option.type == OptionType.NUMBER:
            return option.number
        if option.type == OptionType.YES:
            return 100 if self.context == SelectContext.IS_FIRST else 1
        if option.type == OptionType.NO:
            return 0
        if option.type == OptionType.CARD:
            return self._score_card_choice(option)
        if option.type == OptionType.PLAY:
            return self._score_play(option)
        if option.type == OptionType.ATTACH:
            return self._score_attach(option)
        if option.type == OptionType.EVOLVE:
            return self._score_evolve(option)
        if option.type == OptionType.ABILITY:
            return self._score_ability(option)
        if option.type == OptionType.RETREAT:
            return 2000 if plan.attacker >= 1 else -1
        if option.type == OptionType.ATTACK:
            return 1100 if (option.attackId == MEGA_BRAVE) == (plan.attack_index == 1) else 1000
        return 0

    def _score_card_choice(self, option) -> float:
        card = get_card(self.obs, option.area, option.index, option.playerIndex)
        if card is None:
            return 0
        if self.context in {SelectContext.SWITCH, SelectContext.TO_ACTIVE}:
            return self._score_active_choice(option, card)
        if self.context == SelectContext.SETUP_ACTIVE_POKEMON:
            return (
                2 if card.id == C.SOLROCK and self.state.firstPlayer == self.my_index
                else 4 if card.id == C.SOLROCK
                else 3 if card.id == C.RIOLU
                else 1 if card.id == C.MAKUHITA
                else 0
            )
        if self.context == SelectContext.TO_HAND:
            score = 200 - self.hand_counts[card.id] * 100
            if card.id == C.MAKUHITA:
                score += (80 if self.field_counts[card.id] < 2 else -20) if self._opponent_is_crustle_wall() else (
                    -10 if self.field_counts[card.id] >= 1 else 10
                )
            elif card.id == C.HARIYAMA:
                score += (120 if self.field_counts[C.MAKUHITA] >= 1 else -5) if self._opponent_is_crustle_wall() else (
                    20 if self.field_counts[C.MAKUHITA] >= 1 else -20
                )
            elif card.id == C.LUNATONE:
                score += -250 if self.field_counts[card.id] >= 1 else 60
            elif card.id == C.SOLROCK:
                score += -250 if self.field_counts[card.id] >= 1 else 50
            elif card.id == C.RIOLU:
                score += -150 if (self.field_counts[C.RIOLU] + self.field_counts[C.MEGA_LUCARIO_EX] >= 2) else (
                    -3 if (self.field_counts[C.RIOLU] + self.field_counts[C.MEGA_LUCARIO_EX] >= 1) else 40
                )
            elif card.id == C.MEGA_LUCARIO_EX:
                score += 40 if self.field_counts[C.RIOLU] >= 1 else -15
            elif card.id == C.BASIC_FIGHTING_ENERGY:
                score += 30 if not ability_used or not self.state.energyAttached else -1
            return score
        if self.context == SelectContext.ATTACH_FROM and isinstance(card, Pokemon):
            return self._energy_target_score(card, option.area == AreaType.ACTIVE)
        return 0

    def _score_active_choice(self, option, card: Pokemon) -> float:
        if not isinstance(card, Pokemon):
            return 0
        if option.playerIndex != self.my_index:
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
        card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        data = card_table[card.id]
        if data.cardType == CardType.POKEMON:
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
                return (
                    3050
                    if (not self.state.supporterPlayed and self.hand_counts[C.CARMINE] > 0
                        and self.hand_counts[C.LILLIE_DETERMINATION] == 0
                        and not self.me.deckCount <= LOW_DECK_COUNT)
                    else -1
                )
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
            return 3500 if any(p is not None and card_table[p.id].stage2 for p in self._opponent_board()) else (
                1200 if self.stadium_id else -1
            )
        return 10000

    def _score_attach(self, option) -> float:
        card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        pokemon = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        if not isinstance(pokemon, Pokemon):
            return 0
        if card.id == C.HERO_CAPE:
            score = 7000
            if self._opponent_is_water_deck():
                return 12200 if pokemon.id == C.RIOLU else 12800 if pokemon.id == C.MEGA_LUCARIO_EX else score
            if pokemon.id == C.RIOLU:
                score += 100
            elif pokemon.id == C.MEGA_LUCARIO_EX:
                score += 200
            return score
        score = self._energy_target_score(pokemon, option.inPlayArea == AreaType.ACTIVE)
        board_index = option.inPlayIndex if option.inPlayArea == AreaType.ACTIVE else option.inPlayIndex + 1
        if board_index == plan.attacker and plan.needs_energy:
            score += 200
        return score

    def _score_evolve(self, option) -> float:
        pokemon = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        if not isinstance(pokemon, Pokemon):
            return 0
        if pokemon.id == C.MAKUHITA and plan.target == 0 and not self._opponent_is_crustle_wall():
            return -1
        return 9000 + len(pokemon.energies)

    def _score_ability(self, option) -> float:
        card = get_card(self.obs, option.area, option.index, self.my_index)
        if card.id == C.LUMIOSE_CITY:
            return 1
        if card.id == C.LUNATONE and self.me.deckCount <= LOW_DECK_COUNT:
            return -1
        return 30000

    def _remember_lunatone_ability(self, ranked: list[int]) -> None:
        global ability_used
        if self.context != SelectContext.MAIN or not ranked:
            return
        option = self.select.option[ranked[0]]
        if option.type != OptionType.ABILITY:
            return
        card = get_card(self.obs, option.area, option.index, self.my_index)
        if card is not None and card.id == C.LUNATONE:
            ability_used = True


# ---------------------------------------------------------------------------
# 5. Search layer — flat UCB1 bandit over the heuristic head
# ---------------------------------------------------------------------------
# Disabled by default: SEARCH_ALGO used to be dead code (AdvancedPolicy.choose()
# truncated its candidate list to maxCount, always 1 at real MAIN decisions --
# see rank_all() below). Fixing that and actually running the bandit live was
# benchmarked at 10 games/pair vs makthanithin_improved_prob and lost 1/20
# (35.0% overall win%, Glicko 1670+/-37 vs 2193+/-72) against the pure-heuristic
# 50/50 baseline. Root cause: the bandit trusts evaluate_state's shallow
# one-turn board scorer over AdvancedPolicy's domain-tuned heuristic whenever
# they disagree, and that trade currently loses. Keep search off until
# evaluate_state is reworked to match the heuristic's logic; env-var override
# lets scripts/benchmark_agents.py-style runs turn it back on for that A/B.
USE_SEARCH = os.environ.get("USE_SEARCH", "0") != "0"
SEARCH_TIME_BUDGET = 1.5
SEARCH_MAX_CANDIDATES = 8

# PUCT-style prior weight for the BC policy (see bc_prior.py): 0 disables
# it (pure UCB1, the original behavior). Kept small and additive -- it
# biases *exploration order* toward moves the trained policy favors, it
# never overrides a candidate's actual simulated value. Env-var override
# lets scripts/benchmark_agents.py-style A/B runs toggle it without a
# second copy of this file.
USE_BC_PRIOR = os.environ.get("USE_BC_PRIOR", "1") != "0"
BC_PRIOR_C = 0.75


def _state_of(obj):
    """Unwrap a search result to its `SearchState`.

    The competition `cg.api` returns an `ApiResult` (with `.state` and
    `.error`); the local packaged `cg.api` returns a `SearchState` directly
    and raises on error. This helper makes the agent run on both.
    """
    if obj is None:
        return None
    inner = getattr(obj, "state", None)
    return inner if inner is not None else obj


def _ok(obj) -> bool:
    """True if the search result carried no error and a usable state."""
    if obj is None:
        return False
    return getattr(obj, "error", 0) == 0


def rollout_turn(sid, cur_obs, your_index):
    """Play out MY turn only (one-turn horizon), scoring with the heuristic.

    Stops at end of my turn / game over. Never simulates the opponent.
    """
    steps = 0
    while steps < 20:
        if cur_obs.current.result is not None and cur_obs.current.result != -1:
            break
        if cur_obs.current.yourIndex != your_index:
            break
        if cur_obs.select.context != SelectContext.MAIN:
            sub = AdvancedPolicy(cur_obs).choose()
            sel = sub[: max(1, cur_obs.select.minCount)]
        else:
            nxt = AdvancedPolicy(cur_obs).choose()
            if not nxt:
                break
            sel = [nxt[0]]
            if cur_obs.select.option[nxt[0]].type == OptionType.END:
                search_step(sid, sel)
                break
        ar = search_step(sid, sel)
        st = _state_of(ar)
        if not _ok(ar) or st is None:
            break
        cur_obs, sid = st.observation, st.searchId
        steps += 1
    return cur_obs


def _predict(obs):
    """Build the six card-ID predictions `search_begin` requires.

    `your_deck` is randomly sampled (not just padded) so repeated UCB1 visits
    to the same candidate action see different plausible draws -- that's the
    "Monte" in this flat Monte-Carlo search. We cannot see hidden opponent
    cards, so those five fields predict from our own known deck list padded
    to the required lengths. This is a *prediction*, not a belief model --
    the engine only needs the right lengths and a Basic Pokémon somewhere on
    the opponent side at setup. (Adapted local-cg call.)
    """
    st = obs.current
    me = st.players[st.yourIndex]
    op = st.players[1 - st.yourIndex]

    def pad(seq, n, filler=6):
        seq = list(seq)
        if len(seq) < n:
            seq = seq + [filler] * (n - len(seq))
        return seq[:n]

    opp_active = []
    if op.active and op.active[0] is None:
        opp_active = [673]  # a known Basic (Makuhita) stand-in prediction
    return dict(
        your_deck=random.sample(my_deck, getattr(me, "deckCount", 60)),
        your_prize=pad(my_deck, len(me.prize)),
        opponent_deck=pad(my_deck, getattr(op, "deckCount", 60)),
        opponent_prize=pad(my_deck, len(op.prize)),
        opponent_hand=pad(my_deck, getattr(op, "handCount", 0)),
        opponent_active=opp_active,
    )


def simulate_action(obs, action):
    """Determinized one-step forward model.

    The determinization samples MY OWN deck order only — it does NOT model the
    opponent's hidden state. That is the surprising, load-bearing finding from
    the teardown: the strongest public agent does no opponent belief sampling.
    The opponent predictions here are placeholders built from our known deck;
    they are NOT a belief model (see teardown "Layer 4").

    `rollout_turn` below scores its steps by instantiating `AdvancedPolicy` on
    *hypothetical* simulated observations, which mutates the module-global
    `plan`/`ability_used` (they're turn-scoped globals, not instance state) as
    a side effect of scoring. Left alone, that overwrites the real values
    computed for the actual current decision with values from a board state
    that never happens, corrupting the next real MAIN sub-decision this turn.
    Save/restore them around the rollout so simulation never leaks into
    real play.
    """
    global plan, ability_used
    saved_plan, saved_ability_used = plan, ability_used
    try:
        preds = _predict(obs)
        sbi = search_begin(agent_observation=obs, **preds)
        sb_state = _state_of(sbi)
        if not _ok(sbi) or sb_state is None:
            return -float("inf")
        ar = search_step(sb_state.searchId, [action])
        ar_state = _state_of(ar)
        if not _ok(ar) or ar_state is None:
            search_end()
            return -float("inf")
        cur = rollout_turn(ar_state.searchId, ar_state.observation, obs.current.yourIndex)
        result = evaluate_state(cur)
        # PERSPECTIVE FIX (2026-08-12): rollout_turn exits on yourIndex flip
        # (turn-ending actions, e.g. attacks), and the engine renders that
        # final observation for the OPPONENT. evaluate_state is view-relative,
        # so without negation every turn-ending line was scored as the
        # opponent's advantage. The 2026-08-11 "live search lost 35% to its
        # own heuristic" benchmark was measured WITH this inversion — treat
        # that number as void, not as evidence about the leaf's quality.
        if cur.current is not None and cur.current.yourIndex != obs.current.yourIndex:
            result = -result
        search_end()
        return result
    finally:
        plan, ability_used = saved_plan, saved_ability_used


def SEARCH_ALGO(obs_dict, obs):
    """Flat UCB1 bandit. No tree, no expansion — MCTS at depth 1.

    1. Rank every legal action with the heuristic.
    2. Keep the top 8 candidates.
    3. Simulate each once (phase 1 exploration).
    4. UCB1 (C=0.5) over values min-max normalized into [0,1] spends the rest
       of the 1.5 s budget (phase 2).
    5. Return [best] + the rest of the heuristic ranking.
    """
    if not (_SEARCH_OK and USE_SEARCH):
        return None
    select = obs.select
    if select is None or select.context != SelectContext.MAIN:
        return None
    t0 = time.time()

    base_order = AdvancedPolicy(obs).rank_all()
    candidates = base_order[:SEARCH_MAX_CANDIDATES]
    if not candidates:
        return None
    if len(candidates) == 1:
        return [candidates[0]] + [i for i in base_order if i != candidates[0]]

    visits = {a: 0 for a in candidates}
    total_val = {a: 0.0 for a in candidates}

    # BC-policy prior (see bc_prior.py): a PUCT-style bonus that biases
    # exploration toward candidates the trained policy favors. Computed
    # once (the distribution doesn't change during this search). None
    # whenever the prior is disabled/unavailable/errors -- in that case the
    # loop below falls back to the exact original pure-UCB1 formula.
    bc_probs = (
        bc_prior.candidate_probs(obs_dict, candidates)
        if (_BC_PRIOR_OK and USE_BC_PRIOR)
        else None
    )

    try:
        # Phase 1: exploration — try each candidate once.
        for a in candidates:
            if time.time() - t0 > SEARCH_TIME_BUDGET:
                break
            val = simulate_action(obs, a)
            if val != -float("inf"):
                visits[a] += 1
                total_val[a] += val

        # Phase 2: UCB1 (+ optional BC-policy PUCT prior) — spend the
        # remaining budget on the most promising.
        while time.time() - t0 < SEARCH_TIME_BUDGET:
            total_visits = sum(visits.values())
            if total_visits == 0:
                break
            valid_scores = [total_val[a] / visits[a] for a in candidates if visits[a] > 0]
            if not valid_scores:
                break
            min_s, max_s = min(valid_scores), max(valid_scores)
            if max_s == min_s:
                max_s = min_s + 1.0

            best_ucb, best_a = -float("inf"), candidates[0]
            for a in candidates:
                if visits[a] == 0:
                    best_a = a
                    break
                avg = total_val[a] / visits[a]
                norm_avg = (avg - min_s) / (max_s - min_s)
                ucb = norm_avg + 0.5 * math.sqrt(math.log(total_visits) / visits[a])
                if bc_probs is not None:
                    prior = bc_probs.get(a, 0.0)
                    ucb += BC_PRIOR_C * prior * math.sqrt(total_visits) / (1 + visits[a])
                if ucb > best_ucb:
                    best_ucb, best_a = ucb, a

            val = simulate_action(obs, best_a)
            if val != -float("inf"):
                visits[best_a] += 1
                total_val[best_a] += val

        best_action = max(
            candidates,
            key=lambda a: total_val[a] / visits[a] if visits[a] > 0 else -float("inf"),
        )
        return [best_action] + [i for i in base_order if i != best_action]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 6. Entry point — the never-crash contract
# ---------------------------------------------------------------------------
def agent(obs_dict: dict) -> list[int]:
    """Competition entry point. Returns a ranked list of legal option indices.

    Three independent fallbacks guarantee no crash:
      search fails   -> heuristic ranking
      heuristic fails -> lowest legal indices [0, ...]
      parse fails    -> the deck (initial) or [0]
    """
    try:
        obs = to_observation_class(obs_dict)
    except Exception:
        _DIAG["decisions"] += 1
        _DIAG["parse_fallback"] += 1
        return my_deck if obs_dict.get("select") is None else [0]
    if obs.select is None:
        return my_deck

    global pre_turn, ability_used, plan
    if pre_turn != obs.current.turn:
        pre_turn = obs.current.turn
        ability_used = False
        plan = AttackPlan()

    _DIAG["decisions"] += 1
    try:
        ordered = SEARCH_ALGO(obs_dict, obs)
        if ordered is None:
            _DIAG["search_fallback"] += 1
            ordered = AdvancedPolicy(obs).choose()
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


if __name__ == "__main__":
    # Quick standalone import check.
    print(f"deck loaded: {len(my_deck)} cards")
    print(f"card table: {len(card_table)} cards")
    print(f"search available: {_SEARCH_OK}")
    print(f"bc_prior available: {_BC_PRIOR_OK}")
