"""The lastview leaf must never hand the critic a filler-visible observation.

Pins the 2026-08-12 fix in agents/search_arms/bc_alldays52_ucb1_criticleaf:
post-turn search observations are rendered for the opponent, whose hand is
our determinization filler, and the critic collapses to a near-constant on
them (std 0.037 vs 0.178 on real states). _simulate_lastview therefore
scores the last ROOT-VIEW observation plus objective end facts.

The arm module loads the real BC checkpoint and critic at import, so these
tests SKIP when the worktree has no model symlinks (see the
worktree-cg-lib-symlink memory / run-ptcg-battle doctor).
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
_CG = REPO / "data" / "external" / "cg-lib"
if (_CG / "cg" / "api.py").exists() and str(_CG) not in sys.path:
    sys.path.insert(0, str(_CG))

_BC_MODEL = REPO / "agents" / "bc_alldays52_jun16_aug07_seed42" / "model"
_CRITIC = REPO / "models" / "critic_outcome_bcalldays52trunk_2026-08-01_to_2026-08-07_seed42"
if not (_BC_MODEL.exists() and (_CRITIC / "critic_state.pt").exists()):
    pytest.skip("BC/critic checkpoints not linked in this worktree",
                allow_module_level=True)

from cg.api import SelectContext  # noqa: E402


@pytest.fixture()
def arm(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "criticleaf_under_test",
        REPO / "agents" / "search_arms" / "bc_alldays52_ucb1_criticleaf" / "agent_core.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _seat(prizes):
    return types.SimpleNamespace(prize=list(range(prizes)), deckCount=0, hand=[])


def _obs(your_index, prizes=(2, 2), result=-1, main=True):
    opt = [types.SimpleNamespace(type="NOT_END")]
    return types.SimpleNamespace(
        current=types.SimpleNamespace(
            yourIndex=your_index, result=result,
            players=[_seat(prizes[0]), _seat(prizes[1])],
        ),
        select=types.SimpleNamespace(
            context=SelectContext.MAIN if main else None,
            option=opt, minCount=1, maxCount=1,
        ),
    )


def _wire(arm, script, critic_value=0.25, critic_log=None):
    """Script the engine: each search_step pops the next observation."""
    steps = list(script)
    arm._imp.search_begin = lambda *a, **k: types.SimpleNamespace(searchId=1)
    def search_step(sid, sel):
        return types.SimpleNamespace(searchId=sid + 1, observation=steps.pop(0))
    arm._imp.search_step = search_step
    arm._imp.search_end = lambda: None
    arm._imp._plausible_opponent = lambda o: ([], [], [], [])

    class FakePolicy:
        def __init__(self, obs): pass
        def choose(self): return [0]
    arm._imp.HeuristicPolicy = FakePolicy

    log = critic_log if critic_log is not None else []
    def fake_leaf(obs):
        log.append(obs)
        return critic_value
    # _rollout_and_score resolves _critic_leaf as a module global
    arm._critic_leaf = fake_leaf
    # rebind the global seen by _rollout_and_score/_simulate_lastview
    arm._simulate_lastview.__globals__["_critic_leaf"] = fake_leaf
    return log


def test_first_action_flips_critic_scores_root_obs(arm):
    root = _obs(your_index=0)
    flipped = _obs(your_index=1)                 # attack ended the turn
    log = _wire(arm, [flipped])
    val = arm._simulate_lastview(root, 0)
    assert len(log) == 1 and log[0] is root      # critic saw the ROOT view
    assert val == 0.25                            # no prize delta


def test_critic_scores_last_root_view_not_flipped_final(arm):
    root = _obs(your_index=0)
    mid = _obs(your_index=0)                      # still our turn
    flipped = _obs(your_index=1)
    log = _wire(arm, [mid, flipped])
    arm._simulate_lastview(root, 0)
    assert len(log) == 1 and log[0] is mid        # last root-view state
    assert flipped not in log                     # filler state never encoded


def test_prize_progress_added_objectively(arm):
    root = _obs(your_index=0, prizes=(2, 2))
    flipped = _obs(your_index=1, prizes=(1, 2))   # we took one prize
    _wire(arm, [flipped], critic_value=0.1)
    val = arm._simulate_lastview(root, 0)
    assert val == pytest.approx(0.1 + arm.PRIZE_WEIGHT)


def test_terminal_win_short_circuits_critic(arm):
    root = _obs(your_index=0, prizes=(2, 2))
    won = _obs(your_index=1, prizes=(0, 2))       # our prize pool empty = win
    log = _wire(arm, [won])
    val = arm._simulate_lastview(root, 0)
    assert val == 10.0
    assert log == []                              # critic never consulted
