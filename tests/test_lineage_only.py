"""Lineage-only guards on the training driver.

These are CLI-level guards, so they are tested by invoking the driver and
reading its argparse error -- the failure they prevent is a script or a merge
from another line of work quietly reintroducing external opponents, and that
arrives as command-line flags, not as a Python call.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY_BIN = "/Users/ramiismael/projects/kaggle/Pokemon_TCG_Kaggle_Competition_2026/.venv-ppo/bin/python"


def run(*flags):
    r = subprocess.run([PY_BIN, "scripts/train_ppo_puffer.py", "--skip-preflight", *flags],
                       cwd=REPO, capture_output=True, text=True, timeout=180)
    return r.returncode, (r.stderr + r.stdout)


def test_nonzero_externals_share_is_rejected():
    for bad in ("0.4,0.3,0.3", "0.5,0.3,0.2", "0.625,0.365,0.01"):
        rc, out = run("--mix", bad)
        assert rc != 0 and "externals share must be 0" in out, (
            f"--mix {bad} was ACCEPTED; externals can re-enter training")


def test_zero_externals_share_is_accepted():
    rc, out = run("--mix", "0.625,0.375,0", "--help")
    assert rc == 0, out


def test_field_pool_must_be_empty():
    rc, out = run("--field-pool", "wmh_grimmsnarl,plamen06_steel")
    assert rc != 0 and "must be empty" in out


def test_league_rejects_agent_names():
    rc, out = run("--league", "models/il_agent,wmh_grimmsnarl")
    assert rc != 0 and "not saved checkpoints" in out


def test_league_accepts_real_checkpoints():
    rc, out = run("--league", "models/il_agent,models/s2v2/e0_s43", "--help")
    assert rc == 0, out


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"{name}: OK")
