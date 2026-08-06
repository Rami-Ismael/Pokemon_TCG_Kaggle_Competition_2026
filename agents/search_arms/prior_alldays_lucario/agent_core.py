"""Search+BC-prior arm: agent_core_improved with the flat-UCB1 search ON and
the PUCT-style exploration bonus served by the all-days BC checkpoint.

Arm definition (vs the `agent_core_improved` control, which is the same HEAD
source with its defaults USE_SEARCH=0):

    USE_SEARCH   = True   -> SEARCH_ALGO's flat UCB1 bandit runs at MAIN decisions
    USE_BC_PRIOR = True   -> bc_prior.candidate_probs adds the PUCT bonus
    checkpoint            -> models/il_alldays_0804 (127,748 steps, 3 ep over the
                             full hub corpus, eval_acc 0.7583 -- the strongest
                             BC clone by robustness evidence: it collapsed the
                             il_agent exploiter from 95% to 38.5%)
    deck                  -> agents/mega_lucario/deck.csv (frozen Mega Lucario ex,
                             same as the control)

The flags are module globals read at call time inside SEARCH_ALGO, so they are
set by post-import assignment on a private module instance -- no env vars are
mutated, and the control arm loaded in the same benchmark process keeps its
search-off defaults. The BC prior is likewise a private module instance under
a unique module name: `import bc_prior` inside the core would otherwise share
one sys.modules["bc_prior"] singleton (and hence one MODEL_DIR) across every
agent in the process.

Known caveat, stated up front: this arm pilots the Mega Lucario ex deck, whose
defining cards (Riolu 677, Mega Lucario ex 678) appear in 0/800 sampled BC
training perspectives -- the prior's embeddings for the arm's own key cards are
untrained. The measurement is still the submission-relevant one (that is the
deck the bundle ships), but any null result does not rule out the prior helping
on a deck the checkpoint actually saw.
"""
import importlib.util
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_CORE_DIR = _REPO / "agents" / "mega_lucario"
_CKPT = _REPO / "models" / "il_alldays_0804"

if str(_CORE_DIR) not in sys.path:
    sys.path.insert(0, str(_CORE_DIR))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Private bc_prior instance for this arm. IL_MODEL_DIR is read at bc_prior
# import time, so scope it around the exec and restore (s2_arms pattern).
_prev = os.environ.get("IL_MODEL_DIR")
os.environ["IL_MODEL_DIR"] = str(_CKPT)
try:
    _bc = _load("bc_prior_alldays_search_arm", _CORE_DIR / "bc_prior.py")
finally:
    if _prev is None:
        os.environ.pop("IL_MODEL_DIR", None)
    else:
        os.environ["IL_MODEL_DIR"] = _prev

# Fail loudly at load: a missing checkpoint would otherwise degrade silently
# to plain UCB1 and the benchmark would rate the wrong arm (this repo's
# standing silent-fallback lesson).
assert _bc._AVAILABLE, "torch / encoder unavailable -- BC prior cannot run"
assert _bc._load() is not None, f"BC prior checkpoint failed to load from {_CKPT}"

_mod = _load("agent_core_improved_search_prior_arm", _CORE_DIR / "agent_core_improved.py")
_mod.USE_SEARCH = True
_mod.USE_BC_PRIOR = True
_mod.bc_prior = _bc
_mod._BC_PRIOR_OK = True

# Count prior consultations so a smoke test can prove the PUCT bonus is live
# (candidate_probs returning None every call would be plain UCB1 in disguise).
PRIOR_CALLS = {"calls": 0, "nonnull": 0}
_orig_candidate_probs = _bc.candidate_probs


def _counted_candidate_probs(obs_dict, candidates):
    out = _orig_candidate_probs(obs_dict, candidates)
    PRIOR_CALLS["calls"] += 1
    if out is not None:
        PRIOR_CALLS["nonnull"] += 1
    return out


_bc.candidate_probs = _counted_candidate_probs

my_deck = [
    int(x) for x in (_CORE_DIR / "deck.csv").read_text().splitlines() if x.strip()
][:60]
_mod.my_deck = my_deck
agent = _mod.agent
