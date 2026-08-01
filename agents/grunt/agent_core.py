"""Grunt: a maximally offensive, greedy one-ply agent.

Deck-agnostic "MaxDamage" baseline -- the same spirit as the
"MaxBasePower" heuristic common in Pokémon-battling AI literature, but
improved to actually apply this engine's damage equation instead of
comparing raw base power:

  effective_damage = base_damage * 2   if defender.weakness == attacker.energyType
                      base_damage - 30 if defender.resistance == attacker.energyType
                      (both can't apply: weakness and resistance are never
                      the same type on one card)

Whenever an ATTACK option is on the menu, Grunt always takes the attack
with the highest `effective_damage` against the opponent's current active
Pokémon (see `_score_attack`) -- it never holds back an attack to develop
the board first. That single-minded, no-lookahead rule is the "greedy
one-ply search" this agent amounts to: it evaluates only the immediate
next state of each candidate move, never sequences of moves or future
turns.

The second piece of type-chart reasoning fires when Grunt has to choose
which Pokémon becomes active -- a forced switch (KO'd active), a
voluntary retreat, initial setup, or picking the target of an
opponent-switching effect. `_matchup_score` ranks candidates by the same
weakness/resistance chart: reward a candidate that is super-effective
against the opposing active and is not itself weak to it (see
`_score_switch_card`).

Every other decision Grunt has to make (attaching Energy, evolving,
playing cards, abilities, ...) does not involve damage or matchups, so it
falls back to a simple, generic development priority -- it is not the
focus of this agent and is intentionally coarse.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]

for _cand in (_REPO / "data" / "external" / "cg-lib", _HERE):
    if (_cand / "cg" / "api.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from cg.api import (  # noqa: E402
    AreaType,
    CardType,
    EnergyType,
    OptionType,
    Pokemon,
    SelectContext,
    all_attack,
    all_card_data,
    to_observation_class,
)

CARD_TABLE = {c.cardId: c for c in all_card_data()}
ATTACK_TABLE = {a.attackId: a for a in all_attack()}

# `my_deck` is intentionally NOT pre-declared (see agents/il_agent/agent_core.py's
# comment on why): it is injected externally, either by main.py or by a
# benchmark harness reading the sibling deck.csv.

# Any ATTACK option beats every non-attack option, no matter how small its
# damage -- this constant just needs to clear the largest plausible
# non-attack score plus the largest plausible effective damage.
_ATTACK_BASE = 1_000_000.0
_END_SCORE = -1_000_000.0

_SWITCH_CONTEXTS = {
    SelectContext.SWITCH,
    SelectContext.TO_ACTIVE,
    SelectContext.SETUP_ACTIVE_POKEMON,
}


def _get_card(obs, area, index: int, player_index: int):
    """Safely fetch a Card/Pokemon from a zone; mirrors the sample agent's helper."""
    ps = obs.current.players[player_index]
    try:
        if area == AreaType.HAND:
            return ps.hand[index] if ps.hand else None
        if area == AreaType.DISCARD:
            return ps.discard[index]
        if area == AreaType.ACTIVE:
            return ps.active[index] if ps.active else None
        if area == AreaType.BENCH:
            return ps.bench[index]
        if area == AreaType.PRIZE:
            return ps.prize[index]
        if area == AreaType.DECK:
            return obs.select.deck[index] if obs.select and obs.select.deck else None
        if area == AreaType.STADIUM:
            return obs.current.stadium[index] if obs.current.stadium else None
        if area == AreaType.LOOKING:
            return obs.current.looking[index] if obs.current.looking else None
    except IndexError:
        return None
    return None


def _active(player_state):
    """The active Pokémon, or None (empty Active Spot or face-down card)."""
    return player_state.active[0] if player_state.active else None


def _effective_damage(base_damage: int, attacker_type: EnergyType, defender_id: int) -> int:
    """Apply the type chart: weakness doubles damage, resistance cuts 30 flat."""
    defender = CARD_TABLE.get(defender_id)
    if defender is None:
        return max(0, base_damage)
    damage = base_damage
    if defender.weakness is not None and defender.weakness == attacker_type:
        damage *= 2
    if defender.resistance is not None and defender.resistance == attacker_type:
        damage -= 30
    return max(0, damage)


def _matchup_score(mine_id: int, theirs_id: int) -> float:
    """How good a type matchup `mine_id` is against `theirs_id` (mine attacking theirs)."""
    mine = CARD_TABLE.get(mine_id)
    theirs = CARD_TABLE.get(theirs_id)
    if mine is None or theirs is None:
        return 0.0
    score = 0.0
    if theirs.weakness is not None and theirs.weakness == mine.energyType:
        score += 2.0  # I hit them for double damage
    if theirs.resistance is not None and theirs.resistance == mine.energyType:
        score -= 1.0  # my attacks land soft
    if mine.weakness is not None and mine.weakness == theirs.energyType:
        score -= 2.0  # they hit me for double damage
    if mine.resistance is not None and mine.resistance == theirs.energyType:
        score += 1.0  # their attacks land soft on me
    return score


def _score_attack(my_state, op_state, attack_id) -> float:
    attacker = _active(my_state)
    defender = _active(op_state)
    attack = ATTACK_TABLE.get(attack_id)
    if attacker is None or defender is None or attack is None:
        # Can't evaluate precisely -- still strongly prefer attacking over
        # any non-attack option, just without a damage-based tiebreak.
        return _ATTACK_BASE
    attacker_data = CARD_TABLE.get(attacker.id)
    attacker_type = attacker_data.energyType if attacker_data is not None else EnergyType.COLORLESS
    return _ATTACK_BASE + _effective_damage(attack.damage, attacker_type, defender.id)


def _score_switch_card(obs, o, my_index: int, my_state, op_state) -> float:
    candidate = _get_card(obs, o.area, o.index, o.playerIndex)
    if not isinstance(candidate, Pokemon):
        return 0.0

    if o.playerIndex == my_index:
        # Choosing my own next active Pokémon: best type matchup against
        # the current opposing active first, then healthier / more
        # energy-ready as a tiebreak.
        opponent = _active(op_state)
        score = _matchup_score(candidate.id, opponent.id) if opponent is not None else 0.0
        score += candidate.hp / max(candidate.maxHp, 1)
        score += 0.05 * len(candidate.energies)
        return score

    # Choosing an opposing Pokémon (e.g. an opponent-switching effect):
    # bring out whichever is the worst matchup for them against my
    # current active.
    my_active = _active(my_state)
    if my_active is None:
        return 0.0
    return _matchup_score(my_active.id, candidate.id)


def _score_option(obs, o, context, my_index: int, my_state, op_state) -> float:
    if o.type == OptionType.ATTACK:
        return _score_attack(my_state, op_state, o.attackId)

    if o.type == OptionType.CARD and context in _SWITCH_CONTEXTS:
        return _score_switch_card(obs, o, my_index, my_state, op_state)

    if o.type == OptionType.NUMBER:
        return float(o.number or 0)

    if o.type == OptionType.YES:
        return 1.0

    if o.type == OptionType.EVOLVE:
        # A generic development action, not the focus of this agent:
        # evolving raises HP/attack potential of a Pokémon already in play.
        return 500.0

    if o.type == OptionType.ATTACH:
        # Prefer feeding the active Pokémon Energy over the bench -- it's
        # the one that can turn Energy into damage this turn.
        return 400.0 + (50.0 if o.inPlayArea == AreaType.ACTIVE else 0.0)

    if o.type == OptionType.PLAY:
        card = _get_card(obs, AreaType.HAND, o.index, my_index)
        data = CARD_TABLE.get(card.id) if card is not None else None
        if data is not None and data.cardType == CardType.POKEMON:
            return 350.0  # more board presence = more future attackers
        return 150.0

    if o.type == OptionType.ABILITY:
        return 100.0

    if o.type == OptionType.RETREAT:
        # Offense-minded: don't voluntarily give up the turn to retreat.
        return -50.0

    if o.type == OptionType.END:
        return _END_SCORE

    # NO / DISCARD / generic CARD selections outside a switch context /
    # anything else not covered above: no informed preference either way.
    return 0.0


def agent(obs_dict: dict) -> list[int]:
    """Main Agent Function. Returns a list of option indices (see cg.api.Option)."""
    select = None
    try:
        if obs_dict.get("select") is None:
            return my_deck

        obs = to_observation_class(obs_dict)
        select = obs.select
        if not select.option:
            return []

        state = obs.current
        context = select.context
        my_index = state.yourIndex
        my_state = state.players[my_index]
        op_state = state.players[1 - my_index]

        scores = [
            _score_option(obs, o, context, my_index, my_state, op_state)
            for o in select.option
        ]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        min_count = select.minCount or 0
        max_count = select.maxCount
        if min_count == 0:
            # Optional selection: only take options with a real (positive)
            # preference, otherwise decline by selecting nothing.
            return [i for i in ranked if scores[i] > 0][:max_count]

        picked = ranked[:max_count]
        if len(picked) < min_count:
            for i in ranked:
                if i not in picked:
                    picked.append(i)
                if len(picked) >= min_count:
                    break
        return picked
    except Exception:
        # Never-crash contract: an uncaught exception here means INVALID,
        # which is an instant loss -- always fall back to a legal choice.
        try:
            if select is None:
                return []
            n = len(select.option)
            lo = max(select.minCount or 0, 0)
            hi = min(select.maxCount if select.maxCount is not None else n, n)
            k = lo if lo > 0 else min(1, hi)
            return list(range(min(k, hi)))
        except Exception:
            return []
