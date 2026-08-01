"""Single source of truth for device selection.

Rule: MPS if available, else CPU. There is no CUDA branch -- training happens
on Apple Silicon (MPS) and the Kaggle evaluator is CPU-only, so a cuda path
would be dead code. The same source tree runs unmodified in both places;
the only environment sniffing is `torch.backends.mps.is_available()`.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch


def resolve_device(override: str | None = None) -> str:
    """Return 'mps' or 'cpu'. Override via arg, then PTCG_DEVICE env var, then auto-detect.

    `override='cpu'` (or PTCG_DEVICE=cpu) forces CPU even on an MPS-capable
    laptop -- use this to reproduce/time exactly what the Kaggle evaluator
    will do before trusting a submission.
    """
    choice = override or os.environ.get("PTCG_DEVICE")
    if choice:
        choice = choice.lower()
        if choice not in ("mps", "cpu"):
            raise ValueError(f"unsupported device override: {choice!r} (must be 'mps' or 'cpu')")
        if choice == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("PTCG_DEVICE=mps requested but MPS is not available on this machine")
        return choice
    return "mps" if torch.backends.mps.is_available() else "cpu"


def log_device_info(run_dir: Path, device: str, dtype: str = "float32") -> None:
    """Record the resolved device/dtype/torch version once, into the run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "device": device,
        "dtype": dtype,
        "torch_version": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (run_dir / "device_info.json").write_text(json.dumps(info, indent=2))
