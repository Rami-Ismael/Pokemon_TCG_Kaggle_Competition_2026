"""Imitation-learning agent: transformer policy trained via behavior cloning.

Same `agent(obs_dict) -> list[int]` interface as
agents/mega_lucario/agent_core.py. Loads a PTCGImitationPolicy checkpoint
(see scripts/train_il.py) and scores the legal option set (Pattern B),
including a synthetic DECLINE slot when `minCount == 0` and, for
`maxCount > 1`, autoregressive re-masked picks (mirrors training exactly --
see src/pokemon_tcg/il_dataset.py's `iter_decisions`).

Never-crash contract: every path that touches the model is wrapped so any
exception (OOV id past a defensive clamp bug, a malformed observation, a
future engine field, etc.) falls back to `_safe_choice`, which always
returns a legal index list respecting minCount/maxCount instead of raising.
An uncaught exception here means INVALID means an instant loss.

This extends to the ML stack itself not being importable at all: whether
torch/transformers are available in the evaluator sandbox is, as of this
writing, unverified (the working reference agents in this repo depend on
none of it). `import torch` and the `pokemon_tcg` package are therefore
wrapped too -- if they fail, `_ML_AVAILABLE` is False and `agent()` never
touches them, running purely on `cg.api` (which every reference agent in
this repo already depends on and which is bundled either way) and always
returning a legal, non-random `_safe_choice`. Degraded, not crashed.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]

for _cand in (_REPO / "data" / "external" / "cg-lib", _HERE):
    if (_cand / "cg" / "api.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))
# `pokemon_tcg`: prefer a bundle-local copy (submissions/il_agent/pokemon_tcg/,
# same flat-bundle pattern as `cg` above) so this works with no dev-repo
# `src/` layout at all; fall back to the dev repo's src/ for local testing
# from agents/il_agent/ where no such bundle-local copy exists.
for _cand in (_HERE, _REPO / "src"):
    if (_cand / "pokemon_tcg" / "il_dataset.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))
        break

from cg.api import to_observation_class  # noqa: E402

try:
    import torch

    from pokemon_tcg.device import resolve_device
    from pokemon_tcg.il_dataset import MAX_OPTIONS, encode_observation
    from pokemon_tcg.il_model import PTCGImitationPolicy

    _ML_AVAILABLE = True
except Exception:
    _ML_AVAILABLE = False
    MAX_OPTIONS = 48  # unused when _ML_AVAILABLE is False; kept for readability

MODEL_DIR = os.environ.get("IL_MODEL_DIR", str(_HERE / "model"))
if not Path(MODEL_DIR).exists():
    # Local dev checkout: agents/il_agent/ has no bundled `model/`, the
    # checkpoint lives at the repo-level models/il_agent/ instead. A real
    # submission bundle (submissions/il_agent/) ships its own sibling
    # `model/` dir, so this branch is dev-only.
    MODEL_DIR = str(_REPO / "models" / "il_agent")
# The evaluator is CPU-only; forcing CPU here (rather than calling
# resolve_device() bare) means this path is exercised identically whether
# run on the MPS laptop or the real CPU-only evaluator -- see 2.7.
_DEVICE = resolve_device(override="cpu") if _ML_AVAILABLE else "cpu"
# Hard time cap per decision, well inside the 600s/2000s match budgets --
# a belt-and-suspenders guard, not the primary latency control (measured
# inference is ~ms-scale, see notes/phase0_discovery_report.md).
_MAX_DECISION_SECONDS = 5.0

# `my_deck` is intentionally NOT pre-declared here (matches agents/mega_lucario/agent_core.py):
# it must be injected externally (main.py, or a benchmark harness's hasattr(mod, "my_deck")
# check) -- pre-declaring it as `[]` would make that detection think it's already set and
# skip injection, leaving a silently-empty (illegal) deck.

_model = None
_model_load_attempted = False


def _load_model():
    global _model, _model_load_attempted
    if not _ML_AVAILABLE:
        return None
    if _model_load_attempted:
        return _model
    _model_load_attempted = True
    try:
        _model = PTCGImitationPolicy.from_pretrained(MODEL_DIR)
        _model.to(_DEVICE)
        _model.eval()
    except Exception:
        _model = None
    return _model


def _safe_choice(select) -> list[int]:
    """A legal index list respecting minCount/maxCount.

    Used whenever the model shouldn't be trusted for this decision.
    """
    n = len(select.option)
    lo = max(select.minCount or 0, 0)
    hi = min(select.maxCount if select.maxCount is not None else n, n)
    k = lo if lo > 0 else min(1, hi)
    k = min(k, hi)
    return list(range(k))


def _score_options(model: PTCGImitationPolicy, obs_dict: dict, exclude: frozenset) -> torch.Tensor | None:
    feats = encode_observation(obs_dict, exclude=exclude)
    if feats is None:
        return None
    n_real = feats.pop("n_real_options")
    batch = {k: v.unsqueeze(0).to(_DEVICE) for k, v in feats.items()}
    with torch.no_grad():
        logits = model(**batch)["logits"][0]
    return logits, n_real


def _model_choice(model: PTCGImitationPolicy, obs_dict: dict, select) -> list[int] | None:
    """Autoregressive Pattern-B decoding. Returns None to signal "fall back"."""
    n_opts = len(select.option)
    max_count = select.maxCount if select.maxCount is not None else 1
    min_count = select.minCount or 0
    add_decline = min_count == 0

    picked: list[int] = []
    excluded: set[int] = set()
    deadline = time.monotonic() + _MAX_DECISION_SECONDS
    for _ in range(max(max_count, 1)):
        if time.monotonic() > deadline:
            return None
        out = _score_options(model, obs_dict, frozenset(excluded))
        if out is None:
            return None
        logits, n_real = out
        decline_idx = n_real if add_decline else None
        upto = n_real + (1 if add_decline else 0)
        choice = int(logits[:upto].argmax().item())
        if decline_idx is not None and choice == decline_idx:
            break
        if choice >= n_opts or choice in excluded:
            return None  # defensive: should be structurally impossible, never emit garbage
        picked.append(choice)
        excluded.add(choice)
        if len(picked) >= max_count:
            break
        if max_count == 1:
            break
    if len(picked) < min_count:
        return None
    return picked


def agent(obs_dict: dict) -> list[int]:
    """Main Agent Function. Returns a list of option indices (see cg.api.Option)."""
    select = None
    try:
        if obs_dict.get("select") is None:
            # Initial deck selection: this policy only models move choice, not deck-building
            return my_deck

        obs = to_observation_class(obs_dict)
        select = obs.select
        if not select.option:
            return []
        if len(select.option) > MAX_OPTIONS:
            return _safe_choice(select)

        model = _load_model()
        if model is None:
            return _safe_choice(select)

        result = _model_choice(model, obs_dict, select)
        if result is None:
            return _safe_choice(select)
        return result
    except Exception:
        # Never-crash contract (2.6): any unexpected failure anywhere above
        # falls back to a legal, non-raising choice rather than propagating
        # -- an uncaught exception here is an instant loss, not a retry.
        try:
            return _safe_choice(select) if select is not None else []
        except Exception:
            return []
