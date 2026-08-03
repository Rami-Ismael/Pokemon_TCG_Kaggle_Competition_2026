"""Empty-bench guard for Alakazam best5 pilot (R7, tempo-safe MAIN).

Wraps empty_bench_guard.py with Abra-line deck priorities.
Default OFF in alakazam_agent (ALAKAZAM_BENCH_GUARD=0) — guard regressed local gate.
"""

from __future__ import annotations

try:
    from empty_bench_guard import apply_bench_guard as _apply_core
except ImportError:
    from agent.empty_bench_guard import apply_bench_guard as _apply_core

_BENCH_PRIORITY = (741, 305, 343, 858, 142, 140)


def apply_bench_guard(obs_dict: dict, selection: list[int]) -> list[int]:
    return _apply_core(obs_dict, selection, _BENCH_PRIORITY, respect_main_tempo=True)
