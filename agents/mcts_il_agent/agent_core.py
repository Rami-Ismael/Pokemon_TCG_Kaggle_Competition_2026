"""IL-prior MCTS agent: our trained imitation policy as the prior of the
engine's official Search API, with the offline critic as leaf evaluator.

Same `agent(obs_dict) -> list[int]` interface as agents/il_agent/agent_core.py
and the same never-crash contract: ANY exception in the search/model path
falls back to `_safe_choice` (always-legal), never raises (INVALID = loss).

Experiment wiring (budget from the search-API timing measurements):
- policy prior: models/il_agent (or MCTS_IL_MODEL_DIR)
- value head:  models/critic_outcome_day_2026-07-26_seed42 (or
  MCTS_IL_CRITIC_DIR), used ONLY if the dir carries a calibration.json
  (Platt + leaf centering, fit by scripts/fit_critic_calibration.py); a
  missing or uncalibrated critic means prior-only search (leaf value 0),
  reported at import time, not hidden. The old default,
  models/critic_search_prior, is worse than a constant by its own
  train_metadata.json and must be requested explicitly if ever needed.
- SEARCH_COUNT: MCTS_IL_SEARCH_COUNT env, default 30 (the measured safe
  point inside the ladder's ~1 s/turn budget).

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
    _ML_IMPORT_ERROR = None
except Exception as _import_exc:
    _ML_AVAILABLE = False
    _ML_IMPORT_ERROR = _import_exc

_DEVICE = "cpu"  # evaluator parity; never resolve_device() here on purpose
SEARCH_COUNT = int(os.environ.get("MCTS_IL_SEARCH_COUNT", "30"))

# STRICT LOAD — same contract as agents/il_agent/agent_core.py: fail loudly
# everywhere EXCEPT inside the Kaggle evaluator. Outside the evaluator, a
# silently degraded agent is a measurement hazard: it "runs", scores badly,
# and — because benchmark_agents.py persists Glicko by default — writes a
# _safe_choice rating into the COMPOUNDING ratings file under this agent's
# name (the same failure shape as the 2026-08-04 deck-selection incident).
# Inside the evaluator an uncaught exception is INVALID (= instant loss), so
# degradation stays correct there. PTCG_STRICT_LOAD forces either way
# (set 0 to rehearse the degraded path locally).
_IN_EVALUATOR = (
    Path("/kaggle_simulations/agent").exists() or (_HERE / "model").exists()
)
_STRICT_ENV = os.environ.get("PTCG_STRICT_LOAD")
_STRICT_LOAD = (
    (not _IN_EVALUATOR) if _STRICT_ENV is None
    else _STRICT_ENV.strip().lower() not in ("0", "false", "no", "")
)

# Bundle-local first (submissions ship model/ next to this file, same pattern
# as agents/il_agent), then the dev-repo checkpoint. An EXPLICIT env override
# that doesn't exist always fails loudly — silently resolving it to the dev
# checkpoint would benchmark the WRONG model under this agent's name.
_ENV_MODEL_DIR = os.environ.get("MCTS_IL_MODEL_DIR")
_MODEL_DIR = _ENV_MODEL_DIR or (
    str(_HERE / "model")
    if (_HERE / "model").exists()
    else str(_REPO / "models" / "il_agent")
)
if _ENV_MODEL_DIR and not Path(_ENV_MODEL_DIR).exists():
    raise FileNotFoundError(
        f"MCTS_IL_MODEL_DIR={_ENV_MODEL_DIR} does not exist; refusing to fall "
        f"back to the dev checkpoint (that would run the WRONG model). In a "
        f"git worktree, symlink the checkpoint dir from the main checkout — "
        f"see .claude/skills/run-fallback-diagnostic/SKILL.md."
    )
if _STRICT_LOAD and not Path(_MODEL_DIR).exists():
    raise FileNotFoundError(
        f"no checkpoint at {_MODEL_DIR} (and no bundled model/ beside "
        f"{_HERE}). Refusing to run as a silent _safe_choice fallback, which "
        f"would look like a weak model rather than a missing one — and would "
        f"poison the compounding Glicko file. In a git worktree: "
        f"ln -s <main-checkout>/models/il_agent models/il_agent  "
        f"(see .claude/skills/run-fallback-diagnostic). To rehearse the "
        f"degraded path on purpose, set PTCG_STRICT_LOAD=0."
    )
if _STRICT_LOAD and not _ML_AVAILABLE:
    raise ImportError(
        f"torch/pokemon_tcg imports failed ({_ML_IMPORT_ERROR!r}); refusing "
        f"to run as _safe_choice-only under strict load. "
        f"Set PTCG_STRICT_LOAD=0 to rehearse the degraded path."
    )

# Critic default. models/critic_search_prior — the old default — records
# `beats_constant_baseline: false` in its own train_metadata.json (worse than
# predicting a constant; memory `shipped-mcts-critic-worse-than-constant`),
# and critic_trainday is LOST (dangling symlinks to a deleted worktree, no HF
# copy). The default is now the retrained, audited outcome critic
# (AUC 0.76, shuffled-label control clean; HF backup Rami/ptcg-s2v2-arms).
# A missing critic dir is a legitimate, loudly-reported prior-only arm —
# the submission bundle ships prior-only on purpose — so it does NOT raise.
_ENV_CRITIC_DIR = os.environ.get("MCTS_IL_CRITIC_DIR")
_CRITIC_DIR = _ENV_CRITIC_DIR or (
    str(_HERE / "critic")
    if (_HERE / "critic").exists()
    else str(_REPO / "models" / "critic_outcome_day_2026-07-26_seed42")
)
if _ENV_CRITIC_DIR and not Path(_ENV_CRITIC_DIR).exists():
    raise FileNotFoundError(
        f"MCTS_IL_CRITIC_DIR={_ENV_CRITIC_DIR} does not exist. Unset it for "
        f"prior-only search, or point it at a real critic dir."
    )

_EVALUATOR = None
if _ML_AVAILABLE:
    try:
        _policy = PTCGImitationPolicy.from_pretrained(_MODEL_DIR)
        _policy.to(_DEVICE).eval()
        _policy.requires_grad_(False)
        _critic = None
        if not Path(_CRITIC_DIR).exists():
            print(
                f"[mcts_il_agent] critic dir {_CRITIC_DIR} missing -> prior-only search",
                file=sys.stderr,
            )
        elif not (Path(_CRITIC_DIR) / "calibration.json").exists():
            # An uncentered leaf is wrong by construction (turn-parity bias,
            # memory `search-leaf-value-must-be-centered`) — running it
            # silently would measure the artifact, so the critic is dropped.
            print(
                f"[mcts_il_agent] {_CRITIC_DIR} has no calibration.json -> "
                f"prior-only search. Fit it with "
                f"scripts/fit_critic_calibration.py --critic-dir {_CRITIC_DIR}",
                file=sys.stderr,
            )
        else:
            _critic = load_critic(_CRITIC_DIR, device=_DEVICE)
        # from_critic_dir reads calibration.json (Platt + centering).
        # MCTS_IL_CENTER_LEAF=0 keeps Platt but pins center=0.5 — the
        # uncentered CONTROL arm, never the configuration to ship.
        _EVALUATOR = ILPriorEvaluator.from_critic_dir(
            _policy, _critic, _CRITIC_DIR, device=_DEVICE,
            center_leaf=os.environ.get("MCTS_IL_CENTER_LEAF", "1") != "0",
        )
    except Exception as e:  # degraded, not crashed — but only where allowed
        if _STRICT_LOAD:
            raise
        print(f"[mcts_il_agent] model load failed ({e!r}) -> _safe_choice only", file=sys.stderr)
        _EVALUATOR = None

_RNG = random.Random(0)

# Fallback bookkeeping for the run-fallback-diagnostic skill: every decision
# NOT chosen by the search path increments a cause counter here. Exposes the
# same diag_snapshot()/diag_reset() interface the benchmark harness collects
# (hasattr(mod, "diag_snapshot")); always on — the tiny counter cost is paid
# by the dev harness only, this module is not the submission bundle.
from collections import Counter  # noqa: E402

FALLBACK_COUNTS: Counter = Counter()
DECISION_COUNT = 0


def diag_reset() -> None:
    global DECISION_COUNT
    FALLBACK_COUNTS.clear()
    DECISION_COUNT = 0


def diag_snapshot() -> dict:
    out = dict(FALLBACK_COUNTS)
    fallbacks = sum(FALLBACK_COUNTS.values())
    out["enabled"] = True
    out["decisions"] = DECISION_COUNT
    out["fallbacks"] = fallbacks
    out["fallback_rate"] = fallbacks / max(1, DECISION_COUNT)
    return out


def diag_first() -> dict[str, dict]:
    return {}

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
