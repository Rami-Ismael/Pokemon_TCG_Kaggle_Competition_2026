"""Shared empty-bench guard (R7) — one implementation, deck-specific priorities.

When bench_count == 0, force the best legal basic onto bench (setup TO_BENCH /
SETUP_BENCH, or MAIN PLAY) before the rule pilot skips, ends, or attacks.
Does not force a second bench Pokémon.

Used by dragapult_bench_guard.py and alakazam_bench_guard.py (thin wrappers).
"""

from __future__ import annotations

from typing import Sequence

from cg.api import (
    AreaType,
    CardType,
    Observation,
    OptionType,
    SelectContext,
    all_card_data,
    to_observation_class,
)

card_table = {c.cardId: c for c in all_card_data()}


def is_basic_pokemon_id(card_id: int) -> bool:
    data = card_table.get(card_id)
    if data is None:
        return False
    return data.cardType == CardType.POKEMON and bool(getattr(data, "basic", False))


def bench_count(obs: Observation, my_index: int) -> int:
    return len(obs.current.players[my_index].bench or [])


def _get_card(obs: Observation, area: AreaType, index: int, player_index: int):
    ps = obs.current.players[player_index]
    match area:
        case AreaType.DECK:
            return obs.select.deck[index]
        case AreaType.HAND:
            return ps.hand[index]
        case AreaType.DISCARD:
            return ps.discard[index]
        case AreaType.ACTIVE:
            return ps.active[index]
        case AreaType.BENCH:
            return ps.bench[index]
        case AreaType.PRIZE:
            return ps.prize[index]
        case AreaType.STADIUM:
            return obs.current.stadium[index]
        case AreaType.LOOKING:
            return obs.current.looking[index]
        case _:
            return None


def _card_id_for_option(obs: Observation, option, my_index: int) -> int | None:
    if option.type == OptionType.CARD:
        card = _get_card(obs, option.area, option.index, option.playerIndex)
        return card.id if card is not None else None
    if option.type == OptionType.PLAY:
        card = _get_card(obs, AreaType.HAND, option.index, my_index)
        return card.id if card is not None else None
    return None


def mandatory_bench_indices(
    obs: Observation,
    *,
    bench_priority: Sequence[int] = (),
) -> list[int]:
    """Option indices that bench a basic when we have zero bench Pokémon."""
    if obs.select is None:
        return []
    state = obs.current
    my_index = state.yourIndex
    if bench_count(obs, my_index) > 0:
        return []

    context = obs.select.context
    out: list[int] = []
    for i, option in enumerate(obs.select.option):
        card_id = _card_id_for_option(obs, option, my_index)
        if card_id is None or not is_basic_pokemon_id(card_id):
            continue
        if context == SelectContext.SETUP_BENCH_POKEMON and option.type == OptionType.CARD:
            out.append(i)
        elif context == SelectContext.MAIN and option.type == OptionType.PLAY:
            out.append(i)
        elif context == SelectContext.TO_BENCH and option.type == OptionType.CARD:
            out.append(i)
    return out


def _best_mandatory_index(
    obs: Observation,
    indices: list[int],
    bench_priority: Sequence[int],
) -> int:
    my_index = obs.current.yourIndex
    priority = tuple(bench_priority)

    def rank(i: int) -> tuple[int, int]:
        card_id = _card_id_for_option(obs, obs.select.option[i], my_index) or 9999
        try:
            return (priority.index(card_id), i)
        except ValueError:
            return (len(priority), i)

    return min(indices, key=rank)


def _main_would_skip_bench(obs: Observation, selection: list[int]) -> bool:
    """MAIN only: override guard when pick is END/ATTACK/empty skip, not setup tempo."""
    if obs.select is None or obs.select.context != SelectContext.MAIN:
        return True
    if not selection:
        return True
    my_index = obs.current.yourIndex
    opts = obs.select.option
    for i in selection:
        if not (0 <= i < len(opts)):
            continue
        opt = opts[i]
        if opt.type in (OptionType.END, OptionType.ATTACK):
            return True
        if opt.type == OptionType.PLAY:
            cid = _card_id_for_option(obs, opt, my_index)
            if cid is not None and is_basic_pokemon_id(cid):
                return False  # already benching a basic — handled upstream
            return False  # non-basic play (Rare Candy, supporter, …) — keep policy
        if opt.type in (
            OptionType.EVOLVE,
            OptionType.ABILITY,
            OptionType.ATTACH,
            OptionType.RETREAT,
            OptionType.DISCARD,
        ):
            return False
    return True


def apply_bench_guard(
    obs_dict: dict,
    selection: list[int],
    bench_priority: Sequence[int],
    *,
    respect_main_tempo: bool = False,
) -> list[int]:
    """Override selection when empty bench and a mandatory basic bench is legal.

    respect_main_tempo: when True (Alakazam), only override MAIN on END/ATTACK.
    Dragapult uses False — full guard that fixed no_active on ladder.
    """
    if obs_dict.get("select") is None:
        return selection
    if not isinstance(selection, list):
        selection = list(selection) if selection else []

    obs = to_observation_class(obs_dict)
    mandatory = mandatory_bench_indices(obs, bench_priority=bench_priority)
    if not mandatory:
        return selection
    if selection and any(i in mandatory for i in selection):
        return selection
    if respect_main_tempo and obs.select.context == SelectContext.MAIN:
        if not _main_would_skip_bench(obs, selection):
            return selection
    return [_best_mandatory_index(obs, mandatory, bench_priority)]
