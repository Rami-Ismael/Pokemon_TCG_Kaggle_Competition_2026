"""All-days imitation wrapper (plain-name scheme): il_agent inference core, checkpoint selfplay_g1/refs/u430080 (self-play gen 1).

Same pattern as agents/s2_arms/: loads agents/il_agent/agent_core.py as a
private module with IL_MODEL_DIR pointed at models/s2v2/e0_s43, env var
restored afterward so a later plain il_agent load still gets the PRIOR.
"""
import importlib.util
import os
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_CKPT = _REPO / "models" / "selfplay_g1" / "refs" / "u430080"
_CORE = _REPO / "agents" / "il_agent" / "agent_core.py"

_prev = os.environ.get("IL_MODEL_DIR")
os.environ["IL_MODEL_DIR"] = str(_CKPT)
try:
    _spec = importlib.util.spec_from_file_location("il_core_selfplay_g1_ref430k", _CORE)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
finally:
    if _prev is None:
        os.environ.pop("IL_MODEL_DIR", None)
    else:
        os.environ["IL_MODEL_DIR"] = _prev

my_deck = [int(x) for x in (_REPO / "agents" / "il_agent" / "deck.csv").read_text().split() if x.strip()]
_mod.my_deck = my_deck
agent = _mod.agent

# Fallback-diagnostic hook (run-fallback-diagnostic skill): re-export the
# inner core's counters so the wrapper is instrumented like agents/s2_arms/.
diag_snapshot = _mod.diag_snapshot
