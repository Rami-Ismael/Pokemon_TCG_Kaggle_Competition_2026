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


# Patch the private search instance's leaf. evaluate_state is called in
# exactly one live place (simulate_action, post-rollout) so this swap is the
# whole leaf change; terminal states are also fine through the critic (it
# was trained on terminal-adjacent states) and the perspective negation
# still applies downstream of this call.
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
