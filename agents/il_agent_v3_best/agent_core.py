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

import json
import os
import sys
import time
import traceback
from collections import Counter
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
    from pokemon_tcg.il_dataset import (
        ATTACK_VOCAB_SIZE,
        CARD_VOCAB_SIZE,
        MAX_OPTIONS,
        encode_observation,
    )
    from pokemon_tcg.il_model import PTCGImitationPolicy

    _ML_AVAILABLE = True
except Exception:
    _ML_AVAILABLE = False
    MAX_OPTIONS = 48  # unused when _ML_AVAILABLE is False; kept for readability

_ENV_MODEL_DIR = os.environ.get("IL_MODEL_DIR")
MODEL_DIR = _ENV_MODEL_DIR or str(_HERE / "model")
if not Path(MODEL_DIR).exists():
    if _ENV_MODEL_DIR:
        # An EXPLICIT override that doesn't exist must fail loudly, not
        # silently resolve to the dev checkpoint: an s2_arms wrapper whose
        # arm dir is missing (fresh git worktree, models/s2 not symlinked)
        # would otherwise benchmark Stage-1 weights under the arm's name --
        # this defeated a negative control on 2026-08-03. Import-time and
        # dev/benchmark-only, so the never-crash contract is intact: the
        # Kaggle evaluator never sets IL_MODEL_DIR (a submission bundle
        # ships its own sibling model/ dir, which exists).
        raise FileNotFoundError(
            f"IL_MODEL_DIR={_ENV_MODEL_DIR} does not exist; refusing to fall "
            f"back to the dev checkpoint (that would run the WRONG model). "
            f"In a git worktree, symlink the checkpoint dir from the main "
            f"checkout -- see .claude/skills/run-fallback-diagnostic/SKILL.md."
        )
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

# ---- Fallback tracking (opt-in via PTCG_FALLBACK_TRACK=1) --------------
# Counts how often each never-crash fallback path above actually fires,
# keyed by cause, so "the agent runs" is distinguishable from "the model is
# being bypassed on N% of decisions". OBSERVES the existing fallback sites
# only -- it must never add its own try/except around model logic, and with
# the flag unset (the submission bundle never sets it) every hook is a
# single falsy branch: a genuine no-op under Kaggle's per-step budget.
_TRACK = os.environ.get("PTCG_FALLBACK_TRACK", "") == "1"

# Reasons that mean "the model did not choose this decision"; their sum is
# the fallback_rate numerator. Colon-suffixed variants (policy_exception:
# ValueError, model_unavailable:load_failed) count under their base name.
# unknown_card / unknown_attack / nan_logits / model_load_error are NOT
# here on purpose: they are silent-degradation signals -- the model still
# chose -- reported alongside but never mixed into the rate.
_FALLBACK_REASONS = frozenset({
    "no_legal_actions", "too_many_options", "model_unavailable",
    "encode_none", "step_timeout", "model_action_illegal",
    "min_count_unmet", "policy_exception", "policy_exception_nested",
})

_DIAG = Counter()
_DIAG_FIRST: dict[str, dict] = {}


def _diag(reason: str, detail: dict | None = None) -> None:
    """Count one occurrence; keep full detail for occurrence #1 of each base reason.

    The deep copy (JSON round-trip) runs only on that first occurrence, so
    steady-state cost is one Counter increment. `default=str` keeps the
    copy total: this runs inside except handlers where a raise would turn a
    counted fallback into an INVALID loss.
    """
    if not _TRACK:
        return
    _DIAG[reason] += 1
    base = reason.split(":", 1)[0]
    if base not in _DIAG_FIRST:
        snap = {"reason": reason, "occurrence": int(_DIAG[reason])}
        snap.update(detail or {})
        _DIAG_FIRST[base] = json.loads(json.dumps(snap, default=str))


def diag_reset() -> None:
    _DIAG.clear()
    _DIAG_FIRST.clear()


def diag_snapshot() -> dict:
    """Counts + rate, same interface the benchmark harness already collects
    from the rule agents (hasattr(mod, 'diag_snapshot'))."""
    out = dict(_DIAG)
    decisions = _DIAG.get("decisions", 0)
    fallbacks = sum(
        n for k, n in _DIAG.items() if k.split(":", 1)[0] in _FALLBACK_REASONS
    )
    out["enabled"] = _TRACK
    out["fallbacks"] = fallbacks
    out["fallback_rate"] = fallbacks / max(1, decisions)
    return out


def diag_first() -> dict[str, dict]:
    """Full context (obs / chosen action / traceback) for occurrence #1 per reason."""
    return dict(_DIAG_FIRST)


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
        _diag("model_load_error", {
            "model_dir": MODEL_DIR, "traceback": traceback.format_exc(),
        })
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
    if _TRACK:
        # An id outside the trained vocab was silently clamped into the
        # shared last-row OOV bucket by il_dataset._clamp_id -- the model
        # still scores the option, but on a card it has never seen. This is
        # the only place that degradation is observable at inference.
        if (int(feats["slot_card_id"].max()) == CARD_VOCAB_SIZE - 1
                or int(feats["opt_ref_card_id"].max()) == CARD_VOCAB_SIZE - 1):
            _diag("unknown_card", {"obs": obs_dict})
        if int(feats["opt_attack"].max()) == ATTACK_VOCAB_SIZE - 1:
            _diag("unknown_attack", {"obs": obs_dict})
    n_real = feats.pop("n_real_options")
    batch = {k: v.unsqueeze(0).to(_DEVICE) for k, v in feats.items()}
    with torch.no_grad():
        logits = model(**batch)["logits"][0]
    if _TRACK and bool(torch.isnan(logits[:n_real]).any()):
        # Count-only: argmax over NaN logits still yields an in-range (legal)
        # index, so behavior is unchanged -- but the "choice" is garbage.
        _diag("nan_logits", {"obs": obs_dict})
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
            _diag("step_timeout", {
                "budget_s": _MAX_DECISION_SECONDS, "picked": picked, "obs": obs_dict,
            })
            return None
        out = _score_options(model, obs_dict, frozenset(excluded))
        if out is None:
            _diag("encode_none", {"obs": obs_dict})
            return None
        logits, n_real = out
        decline_idx = n_real if add_decline else None
        upto = n_real + (1 if add_decline else 0)
        choice = int(logits[:upto].argmax().item())
        if decline_idx is not None and choice == decline_idx:
            break
        if choice >= n_opts or choice in excluded:
            # defensive: should be structurally impossible, never emit garbage.
            # Structurally zero when masking is correct -- any count here is a
            # masking bug, not a fallback (tests/test_fallback_tracking.py).
            _diag("model_action_illegal", {
                "choice": choice, "n_opts": n_opts, "n_real": n_real,
                "excluded": sorted(excluded), "obs": obs_dict,
            })
            return None
        picked.append(choice)
        excluded.add(choice)
        if len(picked) >= max_count:
            break
        if max_count == 1:
            break
    if len(picked) < min_count:
        _diag("min_count_unmet", {
            "picked": picked, "min_count": min_count, "obs": obs_dict,
        })
        return None
    return picked


def agent(obs_dict: dict) -> list[int]:
    """Main Agent Function. Returns a list of option indices (see cg.api.Option)."""
    select = None
    try:
        if obs_dict.get("select") is None:
            # Initial deck selection: this policy only models move choice, not deck-building
            return my_deck

        if _TRACK:
            # Denominator for fallback_rate: every call that carries a select.
            # Counted before parsing so a parse crash still lands in it.
            _DIAG["decisions"] += 1
        obs = to_observation_class(obs_dict)
        select = obs.select
        if not select.option:
            _diag("no_legal_actions", {"obs": obs_dict})
            return []
        if len(select.option) > MAX_OPTIONS:
            _diag("too_many_options", {"n_opts": len(select.option), "obs": obs_dict})
            return _safe_choice(select)

        model = _load_model()
        if model is None:
            _diag(
                "model_unavailable:load_failed" if _ML_AVAILABLE
                else "model_unavailable:no_ml_stack",
                {"model_dir": MODEL_DIR},
            )
            return _safe_choice(select)

        result = _model_choice(model, obs_dict, select)
        if result is None:
            # cause already counted at the failing site inside _model_choice
            return _safe_choice(select)
        return result
    except Exception as e:
        # Never-crash contract (2.6): any unexpected failure anywhere above
        # falls back to a legal, non-raising choice rather than propagating
        # -- an uncaught exception here is an instant loss, not a retry.
        _diag("policy_exception:" + type(e).__name__, {
            "traceback": traceback.format_exc(), "obs": obs_dict,
        })
        try:
            return _safe_choice(select) if select is not None else []
        except Exception as e2:
            _diag("policy_exception_nested:" + type(e2).__name__, {
                "traceback": traceback.format_exc(),
            })
            return []
