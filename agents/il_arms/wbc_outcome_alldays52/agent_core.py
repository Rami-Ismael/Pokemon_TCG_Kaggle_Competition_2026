"""IL checkpoint arm: il_agent inference core pointed at
models/wbc_outcome_alldays52_jun16_aug07_seed42 -- the weighted-BC
(outcome-weighted cross-entropy, MARWIL/AWR one-step offline RL) stage of
the v5 pipeline, trained 1 epoch on the 52-day corpus (2026-06-16..08-07)
initialized from bc_alldays52_jun16_aug07_seed42, seed 42.

One identical wrapper backs every arm in agents/il_arms/, so nothing but the
checkpoint differs across the model axis -- wrapper implementation is held
fixed rather than being a second, invisible variable.

IL_MODEL_DIR is set only around the core's import and then restored, so a
later plain `il_agent` load still resolves the frozen 3-epoch PRIOR.

`my_deck` is seeded from agents/il_agent/deck.csv (the Mega Lucario ex
control). The benchmark's `<arm>@<deck-tag>` override replaces it on the
core module's own globals -- see load_agent() in scripts/benchmark_agents.py.
"""
import importlib.util
import os
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_CKPT = _REPO / "models" / "wbc_outcome_alldays52_jun16_aug07_seed42"
_CORE = _REPO / "agents" / "il_agent" / "agent_core.py"

_prev = os.environ.get("IL_MODEL_DIR")
os.environ["IL_MODEL_DIR"] = str(_CKPT)
try:
    _spec = importlib.util.spec_from_file_location("il_core_wbc_outcome_alldays52", _CORE)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
finally:
    if _prev is None:
        os.environ.pop("IL_MODEL_DIR", None)
    else:
        os.environ["IL_MODEL_DIR"] = _prev

_mod.my_deck = [
    int(x) for x in (_REPO / "agents" / "il_agent" / "deck.csv").read_text().split()
    if x.strip()
]
my_deck = _mod.my_deck
agent = _mod.agent

diag_snapshot = _mod.diag_snapshot
diag_first = _mod.diag_first
diag_reset = _mod.diag_reset
