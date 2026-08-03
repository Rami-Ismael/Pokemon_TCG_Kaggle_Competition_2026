"""il_agent + deterministic-future features: benchmark wrapper.

Loads the UNMODIFIED agents/il_agent/agent_core.py as a fresh module, but
pointed at the feature-model checkpoint (models/il_agent_features) via
IL_MODEL_DIR. The checkpoint's own config says which feature groups it
consumes (PTCGILConfig.global_features/opt_features), so no encoder flags
are needed here -- the same code path serves both models.

Deck: this directory ships a copy of agents/il_agent/deck.csv, so a
benchmark between `il_feature` and `il_agent` compares POLICY ONLY on an
identical 60-card list (deck is bound to agent identity in the harness --
see scripts/benchmark_agents.py).

IL_MODEL_DIR is snapshotted and restored around the inner import: il_agent's
agent_core reads it once at import time, so leaking it would silently
redirect a later plain `il_agent` load in the same process to the WRONG
model -- exactly the failure mode the loud-missing-dir check exists for.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]

MODEL_DIR = _REPO / "models" / "il_agent_features"
if not MODEL_DIR.exists():
    raise FileNotFoundError(
        f"{MODEL_DIR} does not exist -- train the combined feature model first "
        "(see scripts/run_feature_ablation.py / notes/feature_ablation_candidates.md)."
    )

_saved = os.environ.get("IL_MODEL_DIR")
os.environ["IL_MODEL_DIR"] = str(MODEL_DIR)
try:
    _spec = importlib.util.spec_from_file_location(
        "il_feature_inner_core", _REPO / "agents" / "il_agent" / "agent_core.py"
    )
    _inner = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_inner)
finally:
    if _saved is None:
        os.environ.pop("IL_MODEL_DIR", None)
    else:
        os.environ["IL_MODEL_DIR"] = _saved

# Wire the deck into the INNER module (agent() resolves `my_deck` in its own
# globals), and mirror it here so the harness's hasattr check skips its own
# injection (which would only reach this wrapper's namespace).
my_deck = [
    int(x)
    for x in (_HERE / "deck.csv").read_text().splitlines()
    if x.strip()
][:60]
_inner.my_deck = my_deck

agent = _inner.agent
diag_snapshot = _inner.diag_snapshot
diag_first = _inner.diag_first
diag_reset = _inner.diag_reset
