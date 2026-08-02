"""Ensure the local cabt `cg` package is importable (dragapult/iono/abomasnow agents)."""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

_booted = False


def ensure_cg_engine() -> str:
    """Add cg-lib to sys.path once; return the directory added."""
    global _booted
    if _booted:
        return ""

    root = Path(__file__).resolve().parents[1]
    env = os.environ.get("CG_LIB", "").strip()
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    candidates.extend([
        root / "cg-lib",
        root / "data" / "sim" / "sample_submission",
    ])
    for hit in glob.glob("/kaggle/input/**/cg-lib", recursive=True):
        candidates.append(Path(hit))
    for p in candidates:
        if (p / "cg" / "game.py").exists():
            sys.path.insert(0, str(p))
            _booted = True
            return str(p)
        if p.name == "cg" and (p / "game.py").exists():
            sys.path.insert(0, str(p.parent))
            _booted = True
            return str(p.parent)
    raise FileNotFoundError(
        "cg engine not found. Set CG_LIB or run scripts/fetch_sim_engine.py "
        "(Windows cg.dll) or place cg-lib/ with cg/ next to this repo."
    )
