"""The tree evaluator must never run the critic on filler-visible states.

Background (2026-08-12, flat-bandit line, same critic family): post-turn
search observations are rendered for the opponent, whose "hand" is the
determinization filler we fabricated for search_begin. On those states the
critic collapses to a near-constant (std 0.037 vs 0.178 on real states), so
its value is bias, not signal. The fix in search_prior_mcts: opponent-view
nodes get priors from their own observation but INHERIT their value from the
nearest root-view ancestor plus objective prize progress.

These tests pin the two load-bearing pieces without the engine:
  1. evaluate(want_value=False) must not touch the critic at all.
  2. _inherited_value applies the seat-indexed prize delta with clamping.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from pokemon_tcg.search_prior_mcts import (  # noqa: E402
    ILPriorEvaluator,
    PRIZE_WEIGHT,
    _inherited_value,
)


class _ExplodingCritic:
    """Fails the test if the evaluator ever calls it."""

    def __call__(self, **kwargs):
        raise AssertionError("critic was called on a want_value=False node")


class _Seat:
    def __init__(self, prizes: int):
        self.prize = list(range(prizes))


class _State:
    def __init__(self, my_prizes: int, opp_prizes: int, your_index: int = 0):
        # seat-indexed players list, root player at index 0 by default
        if your_index == 0:
            self.players = [_Seat(my_prizes), _Seat(opp_prizes)]
        else:
            self.players = [_Seat(opp_prizes), _Seat(my_prizes)]


def test_want_value_false_skips_critic_entirely():
    # policy=None is never reached: an un-encodable obs falls back before the
    # policy forward, which is exactly the path we want — the assertion is
    # that the CRITIC is not consulted even on the fallback path.
    ev = ILPriorEvaluator(policy=None, critic=_ExplodingCritic())
    value, priors = ev.evaluate({}, [[0], [1]], want_value=False)
    assert value is None
    assert len(priors) == 2


def test_inherited_value_no_progress_is_ancestor_value():
    lv = _State(my_prizes=3, opp_prizes=3)
    now = _State(my_prizes=3, opp_prizes=3)
    assert _inherited_value(0.2, lv, now, your_index=0) == 0.2


def test_inherited_value_taking_a_prize_helps_root():
    # root (seat 0) took one prize: own prize pool shrank 3 -> 2
    lv = _State(my_prizes=3, opp_prizes=3)
    now = _State(my_prizes=2, opp_prizes=3)
    assert _inherited_value(0.0, lv, now, your_index=0) == PRIZE_WEIGHT


def test_inherited_value_opponent_prize_hurts_root():
    lv = _State(my_prizes=3, opp_prizes=3)
    now = _State(my_prizes=3, opp_prizes=2)
    assert _inherited_value(0.0, lv, now, your_index=0) == -PRIZE_WEIGHT


def test_inherited_value_seat_indexed_not_view_relative():
    # same physical situation with root sitting in seat 1
    lv = _State(my_prizes=3, opp_prizes=3, your_index=1)
    now = _State(my_prizes=2, opp_prizes=3, your_index=1)
    assert _inherited_value(0.0, lv, now, your_index=1) == PRIZE_WEIGHT


def test_inherited_value_clamps_to_unit_range():
    lv = _State(my_prizes=6, opp_prizes=6)
    now = _State(my_prizes=1, opp_prizes=6)  # five prizes of progress
    assert _inherited_value(0.9, lv, now, your_index=0) == 1.0
    now_bad = _State(my_prizes=6, opp_prizes=1)
    assert _inherited_value(-0.9, lv, now_bad, your_index=0) == -1.0
