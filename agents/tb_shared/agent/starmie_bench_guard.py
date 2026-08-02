"""Empty-bench guard for Starmie / Froslass pilot (R7)."""

from __future__ import annotations

try:
    from empty_bench_guard import apply_bench_guard as _apply
except ImportError:
    from agent.empty_bench_guard import apply_bench_guard as _apply

# Staryu and Snorunt basics — prefer Staryu line for attackers.
_BENCH_PRIORITY = (1030, 860)


def apply_bench_guard(obs_dict: dict, selection: list[int]) -> list[int]:
    return _apply(obs_dict, selection, _BENCH_PRIORITY)
