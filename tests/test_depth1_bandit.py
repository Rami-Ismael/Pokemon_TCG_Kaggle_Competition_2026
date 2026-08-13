"""Regression tests for the depth-1 bandit fixes of 2026-08-11/12.

Pins four behaviors in agents/improved_probabilistic/main.py, each of which
was a real, battle-measured bug when absent:

  1. FULL RANKING: HeuristicPolicy.choose() must return every option index
     (truncating to maxCount=1 starved the bandit to a single candidate and
     made search dead code — found 2026-08-11).
  2. PERSPECTIVE: simulate_action must negate the leaf when the rollout ends
     in the opponent's view (the missing negation inverted the leaf on every
     turn-ending line and voided two battle results — found 2026-08-12).
  3. LEAK CONTRACT: every simulation that reaches search_begin must be
     followed by exactly one search_end, success or failure (engine states
     are otherwise never reused — an OOM on the evaluator's ~197 MiB).
  4. MARGIN GATE: override_margin defers to the base ranking's top action on
     sub-margin gaps AND when that action has zero successful simulations
     (an unmeasured gap must not permit an override).

The bandit/simulate tests stub the engine seams (search_begin/step,
rollout_turn, evaluate_state, HeuristicPolicy) — they test OUR control flow,
not the engine. Test 1 uses one real cabt game because choose()'s scoring
needs a genuine observation.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for _cand in (REPO / "data" / "external" / "cg-lib",):
    if (_cand / "cg" / "api.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from cg.api import SelectContext  # noqa: E402


def _load_main():
    """Fresh private module instance per test — module state stays isolated."""
    spec = importlib.util.spec_from_file_location(
        "improved_prob_under_test",
        REPO / "agents" / "improved_probabilistic" / "main.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.USE_SEARCH = True
    return mod


def _fake_obs(n_options: int, your_index: int = 0):
    """Minimal MAIN-decision observation for the bandit's control flow."""
    opt = [types.SimpleNamespace(type=None) for _ in range(n_options)]
    select = types.SimpleNamespace(
        context=SelectContext.MAIN, option=opt, minCount=1, maxCount=1)
    current = types.SimpleNamespace(yourIndex=your_index)
    return types.SimpleNamespace(select=select, current=current)


def _stub_sim(mod, values: dict[int, float]):
    """Replace simulate_action with a fixed per-action value table."""
    calls = []
    def sim(obs, a):
        calls.append(a)
        return values.get(a, -float("inf"))
    mod.simulate_action = sim
    return calls


# --- 4 + full-permutation: bandit mechanics ---------------------------------

def test_bandit_reranks_to_value_argmax_and_returns_full_permutation():
    mod = _load_main()
    mod.SEARCH_TIME_BUDGET = 0.05
    _stub_sim(mod, {0: 1.0, 1: 5.0, 2: 2.0, 3: 0.0})
    obs = _fake_obs(4)
    ordered = mod.flat_monte_carlo_search(obs, base_order=[0, 1, 2, 3])
    assert ordered[0] == 1                      # value argmax wins
    assert sorted(ordered) == [0, 1, 2, 3]      # full permutation
    assert ordered[1:] == [0, 2, 3]             # tail keeps base order


def test_margin_gate_defers_on_submargin_gap():
    mod = _load_main()
    mod.SEARCH_TIME_BUDGET = 0.05
    _stub_sim(mod, {0: 1.0, 1: 1.05})           # gap 0.05 < margin 0.1
    ordered = mod.flat_monte_carlo_search(
        _fake_obs(2), base_order=[0, 1], override_margin=0.1)
    assert ordered[0] == 0                       # defers to base top


def test_margin_gate_overrides_on_decisive_gap():
    mod = _load_main()
    mod.SEARCH_TIME_BUDGET = 0.05
    _stub_sim(mod, {0: 1.0, 1: 1.5})             # gap 0.5 >= margin 0.1
    ordered = mod.flat_monte_carlo_search(
        _fake_obs(2), base_order=[0, 1], override_margin=0.1)
    assert ordered[0] == 1


def test_margin_gate_defers_when_base_top_never_simulated():
    mod = _load_main()
    mod.SEARCH_TIME_BUDGET = 0.05
    _stub_sim(mod, {1: 1.0})                     # base top (0) always fails
    ordered = mod.flat_monte_carlo_search(
        _fake_obs(2), base_order=[0, 1], override_margin=0.1)
    assert ordered[0] == 0                       # unmeasured gap -> no override


def test_zero_margin_reproduces_unconditional_override():
    mod = _load_main()
    mod.SEARCH_TIME_BUDGET = 0.05
    _stub_sim(mod, {0: 1.0, 1: 1.01})
    ordered = mod.flat_monte_carlo_search(
        _fake_obs(2), base_order=[0, 1], override_margin=0.0)
    assert ordered[0] == 1


# --- 2 + 3: simulate_action perspective and leak contract -------------------

def _wire_sim_stubs(mod, *, final_your_index, begin_raises=False,
                    step_raises=False, leaf_value=100.0):
    """Stub the engine seams around simulate_action; return the call log."""
    log = {"end": 0}
    root_obs_current = types.SimpleNamespace(
        yourIndex=0,
        players=[types.SimpleNamespace(prize=[1], deckCount=0, hand=[]),
                 types.SimpleNamespace(prize=[1], deckCount=0, hand=[])],
    )
    final_obs = types.SimpleNamespace(
        current=types.SimpleNamespace(yourIndex=final_your_index))

    def search_begin(*a, **k):
        if begin_raises:
            raise ValueError("begin failed")
        return types.SimpleNamespace(searchId=7)
    def search_step(sid, sel):
        if step_raises:
            raise ValueError("step failed")
        return types.SimpleNamespace(searchId=8, observation=final_obs)
    def search_end():
        log["end"] += 1
    mod.search_begin = search_begin
    mod.search_step = search_step
    mod.search_end = search_end
    mod.rollout_turn = lambda sid, o, idx: final_obs
    mod.evaluate_state = lambda o: leaf_value
    mod._plausible_opponent = lambda o: ([], [], [], [])

    obs = types.SimpleNamespace(current=root_obs_current)
    return obs, log


def test_simulate_negates_when_rollout_ends_in_opponent_view():
    mod = _load_main()
    obs, log = _wire_sim_stubs(mod, final_your_index=1, leaf_value=100.0)
    assert mod.simulate_action(obs, 0) == -100.0
    assert log["end"] == 1


def test_simulate_keeps_sign_in_root_view():
    mod = _load_main()
    obs, log = _wire_sim_stubs(mod, final_your_index=0, leaf_value=100.0)
    assert mod.simulate_action(obs, 0) == 100.0
    assert log["end"] == 1


def test_simulate_ends_search_even_when_step_raises():
    mod = _load_main()
    obs, log = _wire_sim_stubs(mod, final_your_index=0, step_raises=True)
    assert mod.simulate_action(obs, 0) == -float("inf")
    assert log["end"] == 1                       # began -> must end


def test_simulate_no_end_when_begin_raises():
    mod = _load_main()
    obs, log = _wire_sim_stubs(mod, final_your_index=0, begin_raises=True)
    assert mod.simulate_action(obs, 0) == -float("inf")
    assert log["end"] == 0                       # nothing began -> nothing to end


# --- 1: choose() must not truncate (needs a real observation) ---------------

def test_choose_returns_every_option_index_on_real_main_decision():
    from cg.api import to_observation_class
    from kaggle_environments import make

    mod = _load_main()
    mod.USE_SEARCH = False                       # pure-heuristic game, fast
    env = make("cabt")
    trace = env.run([mod.agent, mod.agent])
    checked = 0
    for state in trace:
        od = state[0]["observation"]
        if od is None:
            continue
        try:
            obs = to_observation_class(od)
        except Exception:
            continue
        sel = obs.select
        if (sel is None or sel.context != SelectContext.MAIN
                or not sel.option or len(sel.option) <= (sel.maxCount or 1)):
            continue
        ranked = mod.HeuristicPolicy(obs).choose()
        assert sorted(ranked) == list(range(len(sel.option))), (
            f"choose() must rank ALL {len(sel.option)} options, got {len(ranked)}")
        checked += 1
        if checked >= 5:
            break
    assert checked > 0, "no multi-option MAIN decision found in a full game"
