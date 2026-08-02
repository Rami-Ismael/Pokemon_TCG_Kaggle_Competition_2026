"""Smart bench depth — one backup minimum, at most two voluntary basics.

Bench Pokémon are prize liabilities when the opponent runs gust, spread, or
bench-targeting attacks. We always keep **at least one** backup (empty bench →
instant loss on active KO) but avoid filling all five slots.

Target depth:
  0 → must bench (mandatory)
  1 → default steady state; only add a 2nd when it is a distinct, high-value line
  2+ → do not voluntarily PLAY more basics onto bench
"""

from __future__ import annotations

from agent.agent import BASIC_POKEMON_FALLBACK, HeuristicScorer, _card_data, _get

MAX_VOLUNTARY_BENCH = 2
_MIN_SETUP_PRIORITY_FOR_SECOND = 30.0
_EARLY_SECOND_BENCH_TURN = 15


def bench_counts(current) -> tuple[int, int]:
    your = _get(current, "players", []) or []
    your_index = _get(current, "yourIndex", 0) or 0
    if not (0 <= your_index < len(your)):
        return 0, 5
    player = your[your_index]
    bench = _get(player, "bench", []) or []
    bench_max = int(_get(player, "benchMax", 5) or 5)
    return len(bench), bench_max


def active_pokemon_id(current) -> int:
    your = _get(current, "players", []) or []
    your_index = _get(current, "yourIndex", 0) or 0
    if not (0 <= your_index < len(your)):
        return 0
    active = _get(your[your_index], "active", []) or []
    if not active:
        return 0
    return int(_get(active[0], "id", 0) or 0)


def field_count(card_id: int, current) -> int:
    your = _get(current, "players", []) or []
    your_index = _get(current, "yourIndex", 0) or 0
    if not (0 <= your_index < len(your)):
        return 0
    player = your[your_index]
    count = 0
    for poke in (_get(player, "active", []) or []) + (_get(player, "bench", []) or []):
        if poke and int(_get(poke, "id", 0) or 0) == card_id:
            count += 1
    return count


def is_basic_pokemon(card_id: int) -> bool:
    data = _card_data(card_id)
    if data is not None:
        return bool(_get(data, "basic", False))
    return card_id in BASIC_POKEMON_FALLBACK


def setup_priority(card_id: int, tech) -> float:
    if tech is None:
        return 0.0
    return float(dict(getattr(tech, "setup_priority", ()) or {}).get(card_id, 0.0))


def worthwhile_second_bench(
    card_id: int,
    *,
    bench_count: int,
    turn: int,
    active_id: int,
    field_dup_count: int,
    tech,
) -> bool:
    """True when a 2nd basic on bench is worth the snipe risk."""
    if bench_count != 1:
        return False
    if field_dup_count > 0:
        return False
    if card_id == active_id:
        return False
    if turn > _EARLY_SECOND_BENCH_TURN:
        return False

    priority = setup_priority(card_id, tech)
    if priority >= _MIN_SETUP_PRIORITY_FOR_SECOND:
        return True

    if tech is not None:
        wall = getattr(tech, "non_ex_wall_bases", frozenset())
        wall_line = getattr(tech, "wall_line", frozenset())
        if card_id in wall and active_id not in wall_line:
            return True
    return False


def worthwhile_second_bench_obs(card_id: int, current, tech) -> bool:
    bench_count, _ = bench_counts(current)
    return worthwhile_second_bench(
        card_id,
        bench_count=bench_count,
        turn=int(_get(current, "turn", 0) or 0),
        active_id=active_pokemon_id(current),
        field_dup_count=field_count(card_id, current),
        tech=tech,
    )


def should_play_basic_to_bench(card_id: int, current, tech) -> bool:
    bench_count, bench_max = bench_counts(current)
    if bench_max <= 0 or bench_count >= bench_max:
        return False
    if bench_count == 0:
        return is_basic_pokemon(card_id)
    if bench_count >= MAX_VOLUNTARY_BENCH:
        return False
    return worthwhile_second_bench_obs(card_id, current, tech)


def setup_bench_target_count(
    ranked_indices: list[int],
    options,
    current,
    tech,
    min_count: int,
    max_count: int,
) -> int:
    """How many setup-bench picks to take when minCount may be 0."""
    if max_count <= 0 or not ranked_indices:
        return 0
    if min_count > 0:
        return min(min_count, max_count, MAX_VOLUNTARY_BENCH)

    hs = HeuristicScorer()
    picks = 1
    if max_count >= 2 and len(ranked_indices) >= 2:
        c0 = _option_card_id(options[ranked_indices[0]], current, hs)
        c1 = _option_card_id(options[ranked_indices[1]], current, hs)
        if (
            c0
            and c1
            and c0 != c1
            and setup_priority(c0, tech) >= _MIN_SETUP_PRIORITY_FOR_SECOND
            and setup_priority(c1, tech) >= _MIN_SETUP_PRIORITY_FOR_SECOND
            and field_count(c0, current) == 0
            and field_count(c1, current) == 0
        ):
            picks = 2
    return min(picks, max_count, MAX_VOLUNTARY_BENCH)


def _option_card_id(opt, current, hs: HeuristicScorer) -> int:
    card = hs._hand_card_for_option(opt, current)
    return int(_get(card, "id", 0) or 0)
