"""Empty-bench guard for Dragapult ex Crispin pilot (R7).

Wraps empty_bench_guard.py with Dreepy-line deck priorities.
"""

from __future__ import annotations

try:
    from empty_bench_guard import (
        apply_bench_guard as _apply_core,
        bench_count,
        mandatory_bench_indices as _mandatory_core,
    )
except ImportError:
    from agent.empty_bench_guard import (
        apply_bench_guard as _apply_core,
        bench_count,
        mandatory_bench_indices as _mandatory_core,
    )

# Dreepy line first, then other basics in the sample list.
_BENCH_PRIORITY = (119, 235, 140, 184, 1071)


def mandatory_bench_indices(obs):
    return _mandatory_core(obs, bench_priority=_BENCH_PRIORITY)


def apply_bench_guard(obs_dict: dict, selection: list[int]) -> list[int]:
    return _apply_core(obs_dict, selection, _BENCH_PRIORITY, respect_main_tempo=False)
