"""IL-prior MCTS agent: our trained imitation policy as the prior of the
engine's official Search API, with the offline critic as leaf evaluator.

Same `agent(obs_dict) -> list[int]` interface as agents/il_agent/agent_core.py
and the same never-crash contract: ANY exception in the search/model path
falls back to `_safe_choice` (always-legal), never raises (INVALID = loss).

Phase-3 experiment wiring (see reports/search_api_phase2.md for the budget):
- policy prior: models/il_agent (or MCTS_IL_MODEL_DIR)
- value head:  models/critic_search_prior (or MCTS_IL_CRITIC_DIR); if the
  critic dir is missing the agent still runs prior-only (leaf value 0) —
  that degraded mode is reported at import time, not hidden.
- SEARCH_COUNT: MCTS_IL_SEARCH_COUNT env, default 30 (Phase-2a safe point).

CPU is forced (evaluator parity) and torch threads are capped at 1 BEFORE
model load, matching the evaluator envelope.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]

for _cand in (_REPO / "data" / "external" / "cg-lib", _HERE):
    if (_cand / "cg" / "api.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))
for _cand in (_HERE, _REPO / "src"):
    if (_cand / "pokemon_tcg" / "il_dataset.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

try:
    import torch

    torch.set_num_threads(1)  # evaluator envelope; BEFORE any heavy torch work

    from pokemon_tcg.il_model import PTCGImitationPolicy
    from pokemon_tcg.offline_critic import load_critic
    from pokemon_tcg.search_prior_mcts import ILPriorEvaluator, mcts_choose

    _ML_AVAILABLE = True
except Exception:
    _ML_AVAILABLE = False

_DEVICE = "cpu"  # evaluator parity; never resolve_device() here on purpose
SEARCH_COUNT = int(os.environ.get("MCTS_IL_SEARCH_COUNT", "30"))

_MODEL_DIR = os.environ.get("MCTS_IL_MODEL_DIR") or str(_REPO / "models" / "il_agent")
_CRITIC_DIR = os.environ.get("MCTS_IL_CRITIC_DIR") or str(
    _REPO / "models" / "critic_search_prior"
)

_EVALUATOR = None
if _ML_AVAILABLE:
    try:
        _policy = PTCGImitationPolicy.from_pretrained(_MODEL_DIR)
        _policy.to(_DEVICE).eval()
        _policy.requires_grad_(False)
        _critic = None
        if Path(_CRITIC_DIR).exists():
            _critic = load_critic(_CRITIC_DIR, device=_DEVICE)
        else:
            print(
                f"[mcts_il_agent] critic dir {_CRITIC_DIR} missing -> prior-only search",
                file=sys.stderr,
            )
        _EVALUATOR = ILPriorEvaluator(_policy, _critic, device=_DEVICE)
    except Exception as e:  # degraded, not crashed
        print(f"[mcts_il_agent] model load failed ({e!r}) -> _safe_choice only", file=sys.stderr)
        _EVALUATOR = None

_RNG = random.Random(0)

# Fallback bookkeeping for the run-fallback-diagnostic skill: every decision
# NOT chosen by the search path increments a cause counter here.
from collections import Counter  # noqa: E402

FALLBACK_COUNTS: Counter = Counter()
DECISION_COUNT = 0

# `my_deck` intentionally not pre-declared (injection contract; see
# agents/il_agent/agent_core.py).


def _safe_choice(select: dict) -> list[int]:
    n = len(select.get("option") or [])
    lo = max(select.get("minCount") or 0, 0)
    hi = min(select["maxCount"] if select.get("maxCount") is not None else n, n)
    k = lo if lo > 0 else min(1, hi)
    return list(range(min(k, hi)))


def agent(obs_dict: dict) -> list[int]:
    select = obs_dict.get("select")
    if select is None:
        return my_deck  # noqa: F821  (injected)

    global DECISION_COUNT
    DECISION_COUNT += 1
    if _EVALUATOR is None:
        FALLBACK_COUNTS["model_unavailable"] += 1
        return _safe_choice(select)
    try:
        return mcts_choose(
            obs_dict,
            my_deck,  # noqa: F821
            _EVALUATOR,
            search_count=SEARCH_COUNT,
            rng=_RNG,
        )
    except Exception as e:
        FALLBACK_COUNTS[f"search_exception:{type(e).__name__}"] += 1
        return _safe_choice(select)
