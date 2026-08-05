"""IL checkpoint arm: il_agent inference core pointed at models/il_agent_hfstream_combined_3ep.

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
_CKPT = _REPO / "models" / "il_agent_hfstream_combined_3ep"
_CORE = _REPO / "agents" / "il_agent" / "agent_core.py"

_prev = os.environ.get("IL_MODEL_DIR")
os.environ["IL_MODEL_DIR"] = str(_CKPT)
try:
    _spec = importlib.util.spec_from_file_location("il_core_il_hfstream_comb_3ep", _CORE)
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

# Re-export the fallback tracker (same as agents/s2_arms/) so
# scripts/fallback_diagnostic.py can measure how often THIS arm answered
# with _safe_choice instead of the checkpoint. Without these, the
# diagnostic exits early and a silent-fallback arm looks unmeasured.
diag_reset = _mod.diag_reset
diag_snapshot = _mod.diag_snapshot
