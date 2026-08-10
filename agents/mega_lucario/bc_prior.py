"""Optional BC-policy prior for SEARCH_ALGO's UCB1 exploration term.

This is the integration the corpus discovery pass (Q28) argued
for: standalone, the trained imitation-learning policy loses to this
repo's own search agents (Rung 2: 3.3% win rate vs agent_core_improved).
But it already outputs a full softmax over the legal option set, not just
an argmax -- exactly what's needed to use it as a PUCT-style prior that
biases the search's exploration toward moves the policy favors, while the
actual rollout-based value estimates (from the real game engine, via
simulate_action) still decide the outcome. The search can only be helped
or left unchanged by a good prior; it can't be misled into taking an
illegal or unrolled-out action, since every candidate is still simulated
and scored by evaluate_state as before.

Self-contained and import-safe, matching every other agent module in this
repo: if torch/transformers/the checkpoint aren't available,
`candidate_probs` returns None and SEARCH_ALGO's caller degrades to plain
UCB1 with no prior term (see agent_core_improved.py) -- never raises.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]

for _cand in (_HERE, _REPO / "src", Path("/kaggle_simulations/agent")):
    if (_cand / "pokemon_tcg" / "il_dataset.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))
        break

_AVAILABLE = False
try:
    import torch

    from pokemon_tcg.il_dataset import encode_observation

    # The card-id vocabulary predates this integration; a checkpoint's
    # option_type_emb width can mismatch the current encoder version if
    # either changes independently, so tolerate a partial load rather than
    # refuse to run the search agent at all over a prior that's optional.
    from pokemon_tcg.il_model import PTCGImitationPolicy

    _AVAILABLE = True
except Exception:
    pass

MODEL_DIR = os.environ.get("IL_MODEL_DIR", str(_HERE / "model"))
if not Path(MODEL_DIR).exists():
    if os.environ.get("IL_MODEL_DIR"):
        # An EXPLICIT override that doesn't exist must fail loudly, not
        # silently resolve to the dev checkpoint (that would run a prior
        # from the WRONG model under the requested checkpoint's name) --
        # same fix as agents/il_agent/agent_core.py, 2026-08-03. Safe for
        # submissions: the Kaggle evaluator never sets IL_MODEL_DIR.
        raise FileNotFoundError(
            f"IL_MODEL_DIR={os.environ['IL_MODEL_DIR']} does not exist; "
            f"refusing to fall back to the dev checkpoint. In a git "
            f"worktree, symlink the checkpoint dir from the main checkout."
        )
    # Local dev checkout: agents/mega_lucario/ has no bundled `model/`, the
    # checkpoint lives at the repo-level models/il_agent/ instead. A real
    # submission bundle ships its own sibling `model/` dir (built the same
    # way as submissions/il_agent/, see scripts/build_*_submission.sh), so
    # this branch is dev-only -- same pattern as agents/il_agent/agent_core.py.
    MODEL_DIR = str(_REPO / "models" / "il_agent")

_model = None
_load_attempted = False


def _load():
    global _model, _load_attempted
    if not _AVAILABLE:
        return None
    if _load_attempted:
        return _model
    _load_attempted = True
    try:
        _model = PTCGImitationPolicy.from_pretrained(MODEL_DIR)
        _model.eval()
    except Exception:
        _model = None
    return _model


def candidate_probs(obs_dict: dict, candidates: list[int]) -> dict[int, float] | None:
    """Softmax over just `candidates` (renormalized), or None if unavailable.

    Deliberately scoped to the caller's own candidate set rather than the
    full option list: SEARCH_ALGO only ever needs a relative preference
    among the moves it's already decided to simulate.
    """
    model = _load()
    if model is None or not candidates:
        return None
    try:
        feats = encode_observation(obs_dict)
        if feats is None:
            return None
        n_real = feats.pop("n_real_options")
        if any(c >= n_real for c in candidates):
            return None  # a candidate points past what the encoder scored -- don't guess
        batch = {k: v.unsqueeze(0) for k, v in feats.items()}
        with torch.no_grad():
            logits = model(**batch)["logits"][0]
        sub = logits[candidates]
        if not torch.isfinite(sub).all():
            return None
        probs = torch.softmax(sub, dim=-1)
        return dict(zip(candidates, (float(p) for p in probs.tolist())))
    except Exception:
        return None
