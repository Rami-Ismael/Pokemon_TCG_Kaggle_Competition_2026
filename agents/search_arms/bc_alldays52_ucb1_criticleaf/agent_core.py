"""UCB1 depth-1 arm: IL ranker + CENTERED CRITIC LEAF + margin-gated override.

Builds on `bc_alldays52_ucb1_rerank` (same IL ranker, same repaired bandit,
same fallback discipline) and changes exactly two things through the seams
added for this purpose:

    leaf   -> the retrained outcome critic
              (critic_outcome_bcalldays52trunk_2026-08-01_to_2026-08-07_seed42:
              trunk = bc_alldays52, MC target, days 08-01..08-07; on eval day
              2026-08-09 it beats the base-rate constant, AUC 0.741; Platt +
              centering constants from its calibration.json), replacing the
              hand-written evaluate_state. The leaf returns a centered value
              in [-1, 1] from the VIEWER's perspective; simulate_action's
              2026-08-12 perspective fix negates it when the rollout ends in
              the opponent's view.
    margin -> search may displace the IL top-1 only if its mean simulated
              value beats the IL pick's mean by OVERRIDE_MARGIN (env var,
              default 0.0 = unconditional override; in centered-leaf units
              where the full range is [-1, 1]).

Experiment card: notes/experiments/2026-08-12-critic-leaf-margin-gate.md
(the follow-up pre-registered in notes/experiments/2026-08-11-il-ucb1-depth1-
rerank.md, updated after the 08-12 perspective-bug discovery).
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]

_CG = _REPO / "data" / "external" / "cg-lib"
if (_CG / "cg" / "api.py").exists() and str(_CG) not in sys.path:
    sys.path.insert(0, str(_CG))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The rerank arm already wires the private BC core + private search module
# with loud-failure checkpoint loading; reuse it and patch its private
# instances (no sys.modules pollution — _load never registers names).
_base = _load(
    "bc_alldays52_ucb1_rerank_for_criticleaf",
    _REPO / "agents" / "search_arms" / "bc_alldays52_ucb1_rerank" / "agent_core.py",
)
_bc, _imp = _base._bc, _base._imp
my_deck = list(_base.my_deck)

# --- critic leaf ------------------------------------------------------------
import json  # noqa: E402

import torch  # noqa: E402

from pokemon_tcg.il_dataset import encode_observation  # noqa: E402
from pokemon_tcg.offline_critic import load_critic  # noqa: E402
from pokemon_tcg.search_prior_mcts import _obs_to_dict  # noqa: E402

_CRITIC_DIR = Path(os.environ.get(
    "CRITIC_DIR",
    _REPO / "models" / "critic_outcome_bcalldays52trunk_2026-08-01_to_2026-08-07_seed42",
))
# Fail loudly: a missing critic silently degrading to evaluate_state would
# benchmark the wrong leaf under this arm's name.
assert (_CRITIC_DIR / "critic_state.pt").exists(), f"critic missing: {_CRITIC_DIR}"
_cal = json.loads((_CRITIC_DIR / "calibration.json").read_text())
_PLATT_A, _PLATT_B = float(_cal["platt_a"]), float(_cal["platt_b"])
_CENTER = float(_cal["center"])
_critic = load_critic(_CRITIC_DIR, device="cpu")
_critic.eval()

OVERRIDE_MARGIN = float(os.environ.get("OVERRIDE_MARGIN", "0.0"))

LEAF_DIAG = {"calls": 0, "encode_none": 0}


@torch.inference_mode()
def _critic_leaf(obs) -> float:
    """Centered critic value in [-1, 1] from the VIEWER's perspective.

    Same transform as search_prior_mcts.PriorCriticEvaluator._leaf_value:
    platt-scale the raw win probability, then center. Returns 0.0 (neutral)
    when the state cannot be encoded — matching that evaluator's fallback.
    simulate_action handles the perspective negation."""
    LEAF_DIAG["calls"] += 1
    obs_dict = _obs_to_dict(obs)
    feats = encode_observation(obs_dict) if obs_dict is not None else None
    if feats is None:
        LEAF_DIAG["encode_none"] += 1
        return 0.0
    feats.pop("n_real_options", None)
    batch = {k: v.unsqueeze(0) for k, v in feats.items()}
    v01 = float(_critic(**batch))
    v01 = min(1.0 - 1e-6, max(1e-6, v01))
    z = math.log(v01 / (1.0 - v01))
    v01 = 1.0 / (1.0 + math.exp(-(_PLATT_A * z + _PLATT_B)))
    return max(-1.0, min(1.0, 2.0 * (v01 - _CENTER)))


# --- leaf wiring ------------------------------------------------------------
# LEAF_MODE (env):
#   'final'    -> critic on the rollout's final observation, negated on
#                 perspective flip by simulate_action. MEASURED BROKEN
#                 2026-08-12: post-turn observations are rendered for the
#                 opponent, whose hand is our determinization FILLER — the
#                 critic collapses to a near-constant (−0.637 ± 0.037) on
#                 those states, so every turn-ending line gets the same flat
#                 bonus and candidate comparisons carry no board signal.
#                 Kept only as the measured-broken control.
#   'lastview' -> (default) critic on the LAST ROOT-VIEW observation of the
#                 rollout (in-distribution, no fabricated zones visible, no
#                 negation), plus the objective view-independent facts from
#                 the true end state: terminal result and prize deltas
#                 (players[] is seat-indexed, so prize counts are valid in
#                 any view). One prize of net progress = +0.5 in leaf units.
LEAF_MODE = os.environ.get("LEAF_MODE", "lastview")
PRIZE_WEIGHT = 0.5

import random as _random  # noqa: E402


def _simulate_lastview(obs, action) -> float:
    """Determinize + roll out one root action; score from the ROOT player's
    perspective without ever encoding a filler-visible observation.

    LEAK NOTE: search_end() in the finally is what lets the native engine
    reuse this simulation's states; without it every sim leaks engine memory
    permanently (fine on a laptop, an OOM on the evaluator's ~197 MiB)."""
    st = obs.current
    r = st.yourIndex
    my_p = st.players[r]
    n_deck = getattr(my_p, "deckCount", 0)
    your_deck = _random.sample(_imp.my_deck, n_deck) if n_deck else []
    od, op, oh, oa = _imp._plausible_opponent(obs)
    began = False
    try:
        root = _imp.search_begin(
            obs, your_deck=your_deck,
            your_prize=[6] * len(my_p.prize),
            opponent_deck=od, opponent_prize=op,
            opponent_hand=oh, opponent_active=oa,
        )
        began = True
        step = _imp.search_step(root.searchId, [action])
        if step is None or step.observation is None:
            return -float("inf")
        return _rollout_and_score(obs, step, r)
    except Exception:
        return -float("inf")
    finally:
        if began:
            try:
                _imp.search_end()
            except Exception:
                pass


def _rollout_and_score(obs, step, r) -> float:
    st = obs.current
    # rollout (same policy/guards as _imp.rollout_turn) tracking the last
    # observation rendered for the ROOT player
    cur, sid = step.observation, step.searchId
    last_view = cur if (cur.current and cur.current.yourIndex == r) else None
    steps = 0
    while steps < 20 and cur.current is not None:
        if cur.current.result is not None and cur.current.result != -1:
            break
        if cur.current.yourIndex != r:
            break
        if cur.select.context != _base.SelectContext.MAIN:
            sub = _imp.HeuristicPolicy(cur).choose()
            sel = sub[: max(1, cur.select.minCount)]
        else:
            nxt = _imp.HeuristicPolicy(cur).choose()
            if not nxt:
                break
            sel = [nxt[0]]
            if cur.select.option[nxt[0]].type == _imp.OptionType.END:
                try:
                    _imp.search_step(sid, sel)
                except Exception:
                    pass
                break
        try:
            nstep = _imp.search_step(sid, sel)
        except Exception:
            break
        if nstep is None or nstep.observation is None:
            break
        cur, sid = nstep.observation, nstep.searchId
        if cur.current is not None and cur.current.yourIndex == r:
            last_view = cur
        steps += 1

    fin = cur.current
    if fin is None:
        return -float("inf")
    # objective terminal check: an empty prize pool means that seat won
    if len(fin.players[r].prize) == 0:
        return 10.0
    if len(fin.players[1 - r].prize) == 0:
        return -10.0
    # critic on the last root-view state (root obs if the very first action
    # already ended the turn — then prizes carry the action's whole effect)
    base_obs = last_view if last_view is not None else obs
    v = _critic_leaf(base_obs)
    net_prizes = (
        (len(st.players[r].prize) - len(fin.players[r].prize))
        - (len(st.players[1 - r].prize) - len(fin.players[1 - r].prize))
    )
    return v + PRIZE_WEIGHT * net_prizes


if LEAF_MODE == "lastview":
    _imp.simulate_action = _simulate_lastview
else:
    # measured-broken control: critic on the final (possibly filler-visible)
    # observation, sign-corrected by simulate_action's perspective negation
    _imp.evaluate_state = _critic_leaf

ARM_DIAG = _base.ARM_DIAG  # searched/changed_top1/etc. live in the base arm


def agent(obs_dict: dict) -> list[int]:
    try:
        if obs_dict.get("select") is None:
            return my_deck
        obs = _base.to_observation_class(obs_dict)
        select = obs.select
        if (
            select is not None
            and select.context == _base.SelectContext.MAIN
            and select.option
            and len(select.option) >= 2
        ):
            ARM_DIAG["main_seen"] += 1
            base_order = _base.il_ranking(obs_dict, select)
            if base_order is None:
                ARM_DIAG["il_rank_none"] += 1
                return _bc.agent(obs_dict)
            ordered = _imp.flat_monte_carlo_search(
                obs, base_order=base_order, override_margin=OVERRIDE_MARGIN)
            if ordered is None:
                ARM_DIAG["search_none"] += 1
                return _bc.agent(obs_dict)
            ARM_DIAG["searched"] += 1
            if ordered[0] != base_order[0]:
                ARM_DIAG["changed_top1"] += 1
            n = len(select.option)
            ordered = [i for i in ordered if 0 <= i < n]
            if not ordered:
                return _bc.agent(obs_dict)
            mx = select.maxCount if select.maxCount is not None else 1
            mn = select.minCount or 0
            k = max(min(mx, n), min(max(1, mn), n))
            return ordered[:k]
        return _bc.agent(obs_dict)
    except Exception:
        return _bc.agent(obs_dict)


for _hook in ("diag_snapshot", "diag_first", "diag_reset"):
    if hasattr(_bc, _hook):
        globals()[_hook] = getattr(_bc, _hook)
