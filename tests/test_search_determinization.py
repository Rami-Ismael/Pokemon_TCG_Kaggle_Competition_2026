"""Regression test for the search-loop determinization bug in agent_core_improved.

Background: SEARCH_ALGO (agents/mega_lucario/agent_core_improved.py) is a flat
UCB1 bandit that revisits the same candidate action multiple times and
averages `evaluate_state()` across those visits. That only produces useful
signal if repeated visits of the *same* action can see a *different*
simulated future -- i.e. `_predict()`'s `your_deck` field must be a random
sample of the deck, not a fixed truncation.

A regression once replaced `random.sample(my_deck, n)` (the behavior of the
public `makthanithin_improved_prob` agent this module ports) with
`pad(my_deck, n)`, a deterministic prefix of the declared decklist. That
silently collapsed every UCB1 revisit of one candidate to an identical
value -- wasted search budget with no crash and no offline signal, since
`simulate_action` never raises and `evaluate_state` still returns a number.

This test does not require the engine's search API or a live game: it
constructs a minimal `State` directly (the same dataclasses `cg.api` uses)
and asserts `_predict()`'s `your_deck` output varies across repeated calls
on the identical input state.

Run directly: uv run python tests/test_search_determinization.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "agents" / "mega_lucario"))
_cg_cands = [REPO / "data" / "external" / "cg-lib", REPO / "agents" / "mega_lucario"]
if sys.platform != "darwin":
    # cg-lib is gitignored; the beginner-guide copy is the only in-repo cg on a
    # fresh clone. Its libcg.so is a Linux build (crashes on macOS), so it is a
    # non-darwin-only last resort.
    _cg_cands.append(REPO / "notebooks" / "beginner-guide" / "sample-agent-output")
for _cand in _cg_cands:
    if (_cand / "cg" / "api.py").exists():
        sys.path.insert(0, str(_cand))
        break

from cg.api import Observation, PlayerState, SelectData, SelectType, SelectContext, State  # noqa: E402

import agent_core_improved as ag  # noqa: E402


def _make_observation(deck_count: int = 40, prize_count: int = 6) -> Observation:
    """A minimal, legal-shaped mid-game Observation with nothing on board.

    `_predict()` only reads `obs.current` (players' deckCount/prize/hand/active),
    so the board/select details are left empty/placeholder.
    """
    me = PlayerState(
        active=[], bench=[], benchMax=5, deckCount=deck_count, discard=[],
        prize=[None] * prize_count, handCount=5, hand=[],
        poisoned=False, burned=False, asleep=False, paralyzed=False, confused=False,
    )
    op = PlayerState(
        active=[], bench=[], benchMax=5, deckCount=40, discard=[],
        prize=[None] * 6, handCount=7, hand=None,
        poisoned=False, burned=False, asleep=False, paralyzed=False, confused=False,
    )
    state = State(
        turn=4, turnActionCount=0, yourIndex=0, firstPlayer=0,
        supporterPlayed=False, stadiumPlayed=False, energyAttached=False,
        retreated=False, result=-1, stadium=[], looking=None, players=[me, op],
    )
    select = SelectData(
        type=SelectType.CARD, context=SelectContext.MAIN, minCount=1, maxCount=1,
        remainDamageCounter=0, remainEnergyCost=0, option=[], deck=None,
        contextCard=None, effect=None,
    )
    return Observation(select=select, logs=[], current=state)


def test_predict_your_deck_is_randomized():
    """Repeated `_predict()` calls on the SAME state must not always match.

    A fixed-order `pad()` truncation (the regression) returns the identical
    tuple on every call. A `random.sample()` draw almost never does, given
    the deck's duplicate-heavy composition and deck_count=40.
    """
    obs = _make_observation(deck_count=40)
    samples = [tuple(ag._predict(obs)["your_deck"]) for _ in range(10)]
    assert len(set(samples)) > 1, (
        "_predict()['your_deck'] returned the identical card sequence on "
        "every call for a fixed input state -- this means the deck "
        "prediction is a deterministic truncation again, not a random "
        "sample, which silently starves SEARCH_ALGO's UCB1 revisits of any "
        "new information. See agent_core_improved.py's _predict()."
    )


def test_predict_your_deck_not_fixed_prefix():
    """`your_deck` must not equal the naive fixed-prefix-of-my_deck truncation."""
    deck_count = 40
    obs = _make_observation(deck_count=deck_count)
    fixed_prefix = tuple(ag.my_deck[:deck_count])
    matches = sum(
        tuple(ag._predict(obs)["your_deck"]) == fixed_prefix for _ in range(10)
    )
    assert matches < 10, (
        "_predict()['your_deck'] matched the fixed pad()-style deck prefix "
        "on every call -- looks like the pad() regression, not random.sample()."
    )


def test_predict_your_deck_is_well_formed():
    """Whatever the sampling strategy, the output must stay a valid subset."""
    deck_count = 40
    obs = _make_observation(deck_count=deck_count)
    for _ in range(5):
        your_deck = ag._predict(obs)["your_deck"]
        assert len(your_deck) == deck_count
        remaining = list(ag.my_deck)
        for card_id in your_deck:
            assert card_id in remaining, f"predicted card {card_id} not in my_deck"
            remaining.remove(card_id)


if __name__ == "__main__":
    test_predict_your_deck_is_randomized()
    test_predict_your_deck_not_fixed_prefix()
    test_predict_your_deck_is_well_formed()
    print("ALL CHECKS PASSED")
