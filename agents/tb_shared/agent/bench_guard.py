"""Bench-depth guard — route development-critical decisions to RuleCore.

Kaggle replays (e.g. vs Yuki_Kaneko): Lucario RL agent had Makuhita active,
**empty bench**, Riolu in hand → active KO → instant loss → μ drop.

All submission scorers (MCTS, Learned, Search) must not skip the *first* backup
basic. Additional bench depth (2nd Pokémon) is handled by smart_bench scoring,
not forced here — we cap exposure because opponents can gust/snipe bench.
"""

from __future__ import annotations

from agent.agent import (
    CTX_SETUP_BENCH_POKEMON,
    OPT_PLAY,
    SEL_CARD,
    SEL_MAIN,
    HeuristicScorer,
    _get,
    _option_type,
)
from agent.smart_bench import bench_counts, should_play_basic_to_bench


def bench_critical(obs_dict, select, current, options) -> bool:
    """True → caller must use RuleCore/Heuristic, not MCTS/Learned."""
    if not options:
        return False
    sel_type = select.get("type")
    if sel_type == SEL_CARD and select.get("context") == CTX_SETUP_BENCH_POKEMON:
        return True
    if sel_type != SEL_MAIN:
        return False
    if not (_get(current, "players", []) or []):
        return False

    bench_count, _ = bench_counts(current)
    if bench_count > 0:
        return False

    # Empty bench: must develop before END/ATTACK/MCTS noise.
    return bool(_mandatory_play_indices(options, current))


def _mandatory_play_indices(options, current) -> list[int]:
    hs = HeuristicScorer()
    out: list[int] = []
    for i, opt in enumerate(options):
        if _option_type(opt) != OPT_PLAY:
            continue
        card = hs._hand_card_for_option(opt, current)
        card_id = int(_get(card, "id", 0) or 0)
        if should_play_basic_to_bench(card_id, current, tech=None):
            out.append(i)
    return out
