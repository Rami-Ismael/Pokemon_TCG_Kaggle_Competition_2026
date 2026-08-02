"""Empty-bench guard for Archaludon ex / Cinderace pilot (R7).

Archaludon deck basics: Duraludon (169), Relicanth (57). Cinderace (666) is not Basic
in engine data — bench via evolution only.

Extends shared guard with SETUP_BENCH / TO_BENCH / TO_FIELD contexts the v5 pilot skips.
"""

from __future__ import annotations

from cg.api import OptionType, SelectContext, to_observation_class

try:
    from empty_bench_guard import (
        _best_mandatory_index,
        _card_id_for_option,
        apply_bench_guard as _apply_core,
        bench_count,
        mandatory_bench_indices as _mandatory_core,
    )
except ImportError:
    from agent.empty_bench_guard import (
        _best_mandatory_index,
        _card_id_for_option,
        apply_bench_guard as _apply_core,
        bench_count,
        mandatory_bench_indices as _mandatory_core,
    )

DURALUDON = 169
RELICANTH = 57

# Engine-verified basics for this list (see tests/test_empty_bench_guard.py).
_BENCH_PRIORITY = (DURALUDON, RELICANTH)
_BENCH_BASICS = frozenset(_BENCH_PRIORITY)

_CARD_CONTEXTS = frozenset({
    SelectContext.SETUP_BENCH_POKEMON,
    SelectContext.TO_BENCH,
    SelectContext.TO_FIELD,
})


def mandatory_bench_indices(obs) -> list[int]:
    """Legal option indices that bench a basic when our bench is empty."""
    found = _mandatory_core(obs, bench_priority=_BENCH_PRIORITY)
    if found:
        return found
    if bench_count(obs, obs.current.yourIndex) > 0:
        return []
    if obs.select is None or obs.select.context not in _CARD_CONTEXTS:
        return []

    my_index = obs.current.yourIndex
    out: list[int] = []
    for i, option in enumerate(obs.select.option):
        if option.type != OptionType.CARD:
            continue
        card_id = _card_id_for_option(obs, option, my_index)
        if card_id in _BENCH_BASICS:
            out.append(i)
    return out


def apply_bench_guard(obs_dict: dict, selection: list[int]) -> list[int]:
    """Force a basic onto bench before END/attack/item tempo when bench is empty."""
    if obs_dict.get("select") is None:
        return selection
    if not isinstance(selection, list):
        selection = list(selection) if selection else []

    obs = to_observation_class(obs_dict)
    if bench_count(obs, obs.current.yourIndex) > 0:
        return selection

    mandatory = mandatory_bench_indices(obs)
    if not mandatory:
        return selection
    if selection and any(i in mandatory for i in selection):
        return selection

    # Full MAIN override (Dragapult pattern) — v5 often prefers items over bench PLAY.
    return [_best_mandatory_index(obs, mandatory, _BENCH_PRIORITY)]
