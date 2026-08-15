"""Self-play arm: il_agent inference core pointed at the FINAL checkpoint of
the v5 lineage-only self-play run (no external agents in training; league =
own checkpoint lineage only; init = wbc_outcome_alldays52_jun16_aug07_seed42;
KL-anchored, il-prior = bc_alldays52_jun16_aug07_seed42; seed 42).

The final checkpoint dir is named u<step>_final with <step> unknown until the
run ends, so it is resolved by glob at import time -- newest *_final wins.
Importing this module before the run has finished raises rather than silently
wrapping a mid-run checkpoint.

One identical wrapper backs every arm in agents/il_arms/; only the checkpoint
differs across the model axis.
"""
import importlib.util
import os
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_RUN = Path(
    "/Users/ramiismael/projects/kaggle/Pokemon_TCG_Kaggle_Competition_2026"
    "/.claude/worktrees/pokemon-tcg-ppo-pipeline-0e0618/models"
    "/selfplay_lineageonly_alldays52_jun16_aug07_seed42"
)
_finals = sorted(_RUN.glob("u*_final"), key=lambda p: int(p.name[1:].split("_")[0]))
if not _finals:
    raise FileNotFoundError(
        f"no u*_final checkpoint under {_RUN} -- the lineage self-play run "
        "has not finished; refusing to wrap a mid-run checkpoint"
    )
_CKPT = _finals[-1]
_CORE = _REPO / "agents" / "il_agent" / "agent_core.py"

_prev = os.environ.get("IL_MODEL_DIR")
os.environ["IL_MODEL_DIR"] = str(_CKPT)
try:
    _spec = importlib.util.spec_from_file_location("il_core_selfplay_lineage_alldays52", _CORE)
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
