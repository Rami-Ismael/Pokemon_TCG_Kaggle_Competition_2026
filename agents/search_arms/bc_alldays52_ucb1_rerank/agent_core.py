"""UCB1 depth-1 re-rank arm: IL policy as the candidate ranker.

Arm definition (vs the `bc_alldays52_jun16_aug07_seed42` control, which is the
same checkpoint with no search):

    ranker      -> the IL policy's full logits argsort (NOT the hand heuristic,
                   and NOT a PUCT bonus -- this is what distinguishes the arm
                   from the 900-game search+BC-prior null, where the heuristic
                   stayed the ranker)
    search      -> flat_monte_carlo_search from the improved_probabilistic
                   lineage (repaired 2026-08-11: the bandit actually runs now),
                   via its `base_order` seam; top-8 candidates, 1.5 s budget
    rollout     -> the lineage's deck-matched Mega Lucario heuristic, one turn
    leaf        -> evaluate_state (KNOWN SUSPECT: named the broken part in the
                   search-prior pool negative; a negative result here scopes to
                   "the leaf loses the trades", not "search can't help IL")
    checkpoint  -> agents/bc_alldays52_jun16_aug07_seed42/model (byte-identical
                   to models/bc_alldays52_jun16_aug07_seed42)
    deck        -> the control's own deck.csv (Mega Lucario ex mirror; matches
                   the search lineage's hardcoded DECK, verified 2026-08-11)

Non-MAIN decisions, single-option MAIN decisions, IL declines, and every
failure path fall through to the unmodified BC agent -- the search layer is
strictly additive, so arm-vs-control differences attribute to it alone.

Experiment card: notes/experiments/2026-08-11-il-ucb1-depth1-rerank.md
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]

# cg-lib: the worktree-tracked canonical arm64 copy (benchmark_agents.py rule)
_CG = _REPO / "data" / "external" / "cg-lib"
if (_CG / "cg" / "api.py").exists() and str(_CG) not in sys.path:
    sys.path.insert(0, str(_CG))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- private BC core with an EXPLICIT checkpoint (fail loudly, never fall
# back to models/il_agent -- the 2026-08-03 negative-control lesson) ---------
_BC_DIR = _REPO / "agents" / "bc_alldays52_jun16_aug07_seed42"
_CKPT = _BC_DIR / "model"
_prev = os.environ.get("IL_MODEL_DIR")
os.environ["IL_MODEL_DIR"] = str(_CKPT)
try:
    _bc = _load("bc_alldays52_core_ucb1_rerank_arm", _BC_DIR / "agent_core.py")
finally:
    if _prev is None:
        os.environ.pop("IL_MODEL_DIR", None)
    else:
        os.environ["IL_MODEL_DIR"] = _prev

my_deck = [
    int(x) for x in (_HERE / "deck.csv").read_text().splitlines() if x.strip()
][:60]
_bc.my_deck = list(my_deck)

assert _bc._ML_AVAILABLE, "torch / encoder unavailable -- IL ranker cannot run"
assert _bc._load_model() is not None, f"BC checkpoint failed to load from {_CKPT}"

# --- private search module, bandit ON for this instance only ---------------
_imp = _load(
    "improved_prob_search_ucb1_rerank_arm",
    _REPO / "agents" / "improved_probabilistic" / "main.py",
)
_imp.USE_SEARCH = True
_imp.my_deck = list(my_deck)  # determinization samples OUR remaining deck order
assert _imp._SEARCH_OK, "cg search API unavailable -- bandit cannot run"

import torch  # noqa: E402  (after _ML_AVAILABLE assert; BC core imported it)

from cg.api import SelectContext, to_observation_class  # noqa: E402

# Counters a smoke test / battle report can read to prove the mechanism is
# live (this repo's standing silent-fallback lesson).
ARM_DIAG = {
    "main_seen": 0,       # MAIN decisions with >=2 options
    "searched": 0,        # bandit ran and returned a ranking
    "changed_top1": 0,    # bandit's pick != IL top-1 (the mechanism metric)
    "il_rank_none": 0,    # IL ranking unavailable -> BC fallback
    "decline_defer": 0,   # IL's argmax was the decline pseudo-action -> BC
    "search_none": 0,     # bandit unavailable/failed -> BC fallback
}


def il_ranking(obs_dict: dict, select) -> list[int] | None:
    """Full best-first ranking of the real option indices from the IL logits.

    Returns None to signal fall-back-to-BC: model/encoder failure, or the
    policy's own argmax being the decline pseudo-action (search only re-ranks
    real options, so forcing an action there would silently override the
    teacher's decline behavior instead of its ranking)."""
    model = _bc._load_model()
    if model is None:
        return None
    out = _bc._score_options(model, obs_dict, frozenset())
    if out is None:
        return None
    logits, n_real = out
    add_decline = (select.minCount or 0) == 0
    upto = n_real + (1 if add_decline else 0)
    top = int(logits[:upto].argmax().item())
    if add_decline and top == n_real:
        ARM_DIAG["decline_defer"] += 1
        return None
    return [int(i) for i in torch.argsort(logits[:n_real], descending=True)]


def agent(obs_dict: dict) -> list[int]:
    try:
        if obs_dict.get("select") is None:
            return my_deck
        obs = to_observation_class(obs_dict)
        select = obs.select
        if (
            select is not None
            and select.context == SelectContext.MAIN
            and select.option
            and len(select.option) >= 2
        ):
            ARM_DIAG["main_seen"] += 1
            base = il_ranking(obs_dict, select)
            if base is None:
                ARM_DIAG["il_rank_none"] += 1
                return _bc.agent(obs_dict)
            ordered = _imp.flat_monte_carlo_search(obs, base_order=base)
            if ordered is None:
                ARM_DIAG["search_none"] += 1
                return _bc.agent(obs_dict)
            ARM_DIAG["searched"] += 1
            if ordered[0] != base[0]:
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


# Fallback-diagnostic hooks pass through to the BC core (its counters cover
# the model path; ARM_DIAG above covers the search layer).
for _hook in ("diag_snapshot", "diag_first", "diag_reset"):
    if hasattr(_bc, _hook):
        globals()[_hook] = getattr(_bc, _hook)
