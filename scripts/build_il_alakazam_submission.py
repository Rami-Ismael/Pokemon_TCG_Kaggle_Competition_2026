"""Build the pure-BC-cloning + Alakazam-deck submission.

Same architecture as submissions/il_agent/ (agents/il_agent/agent_core.py,
no search, no heuristic -- the trained transformer scores options directly)
but with my_deck swapped to agents/ryotasueyoshi_alakazam/deck.csv instead of
the frozen Mega Lucario ex deck.

Why: the A0'' finding + reports/bc_standalone_deck_test.json
found the Alakazam deck's defining cards (Abra/Kadabra/Alakazam) appear in
16.1% of training-corpus perspectives vs 0% for Mega Lucario ex/Riolu, and
that this checkpoint scores 62.5% [38.6,81.5] on Alakazam vs 0.0% [0.0,19.4]
on Lucario against makthanithin_improved_prob -- a statistically
distinguishable gap, not noise.

Checkpoint: models/il_agent/ (the CURRENT checkpoint -- epoch 2, step 38562,
sha256 below -- NOT the preserved 827.8 winning checkpoint, which predates
this deck-embedding finding and was never evaluated on it).

Run from the repo root with:
    uv run python scripts/build_il_alakazam_submission.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "submissions" / "il_agent_alakazam"
CHECKPOINT = REPO / "models" / "il_agent"
ALAKAZAM_DECK = REPO / "agents" / "ryotasueyoshi_alakazam" / "deck.csv"
CG = REPO / "data" / "external" / "cg-lib" / "cg"
PTCG_SRC = REPO / "src" / "pokemon_tcg"
PTCG_FILES = ["config.py", "device.py", "il_dataset.py", "il_model.py"]

MAIN_PY = '''import os
import sys

if os.path.exists("/kaggle_simulations/agent"):
    _BUNDLE_DIR = "/kaggle_simulations/agent"
else:
    try:
        _BUNDLE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        _BUNDLE_DIR = os.getcwd()

sys.path.insert(0, _BUNDLE_DIR)

import agent_core

with open(os.path.join(_BUNDLE_DIR, "deck.csv")) as f:
    agent_core.my_deck = [
        int(x) for x in f.read().splitlines() if x.strip()
    ][:60]

agent = agent_core.agent
'''


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build() -> Path:
    safetensors = CHECKPOINT / "model.safetensors"
    if not safetensors.exists():
        raise FileNotFoundError(f"{safetensors} missing")
    print(f"checkpoint sha256: {sha256_of(safetensors)}")

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    (OUT / "model").mkdir()
    (OUT / "pokemon_tcg").mkdir()

    shutil.copyfile(REPO / "agents" / "il_agent" / "agent_core.py", OUT / "agent_core.py")
    (OUT / "main.py").write_text(MAIN_PY)
    shutil.copyfile(ALAKAZAM_DECK, OUT / "deck.csv")
    for name in ["config.json", "model.safetensors"]:
        shutil.copyfile(CHECKPOINT / name, OUT / "model" / name)
    for name in PTCG_FILES:
        shutil.copyfile(PTCG_SRC / name, OUT / "pokemon_tcg" / name)
    (OUT / "pokemon_tcg" / "__init__.py").touch()

    cg_dest = OUT / "cg"
    shutil.copytree(CG, cg_dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))

    tar_path = OUT / "submission.tar.gz"
    if tar_path.exists():
        tar_path.unlink()
    with tarfile.open(tar_path, "w:gz") as tar:
        for name in ["main.py", "agent_core.py", "deck.csv"]:
            tar.add(OUT / name, arcname=name)
        tar.add(OUT / "model", arcname="model")
        tar.add(OUT / "pokemon_tcg", arcname="pokemon_tcg")
        tar.add(cg_dest, arcname="cg")

    mib = tar_path.stat().st_size / 2**20
    print(f"built {tar_path} ({mib:.2f} MiB)")
    with tarfile.open(tar_path) as t:
        names = t.getnames()
    assert "main.py" in names
    print("top-level entries:", sorted(n for n in names if "/" not in n))
    return tar_path


def verify_bundle(tar_path: Path):
    from kaggle_environments import make

    scratch = Path(tempfile.mkdtemp(prefix="il_alakazam_"))
    with tarfile.open(tar_path) as t:
        t.extractall(scratch)

    for mod in list(sys.modules):
        if mod.split(".")[0] in ("agent_core", "cg", "pokemon_tcg"):
            sys.modules.pop(mod, None)

    cwd = os.getcwd()
    os.chdir(scratch)
    try:
        spec = importlib.util.spec_from_file_location("bundle_main", scratch / "main.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        print("bundled deck len:", len(mod.agent_core.my_deck))

        env = make("cabt")
        trace = env.run([mod.agent, mod.agent])
        final = trace[-1]
        print("bundled self-play:", len(trace), "steps | statuses",
              [final[i]["status"] for i in range(2)], "| rewards", [final[i]["reward"] for i in range(2)])
        assert all(final[i]["status"] == "DONE" for i in range(2))
    finally:
        os.chdir(cwd)
        shutil.rmtree(scratch)
    print("OK: bundled submission plays a full legal game using only bundled files")


if __name__ == "__main__":
    tar_path = build()
    verify_bundle(tar_path)
