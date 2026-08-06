"""Build the IL-prior MCTS (prior-only) submission bundle.

Configuration that won locally in the IL-prior MCTS benchmarks: official
Search API lookahead, SEARCH_COUNT=30, PTCGImitationPolicy (models/il_agent)
option logits as child priors, NO critic (prior-only — the seed-42 critic
failed calibration and erased the gain, so it deliberately does not ship;
agents/mcts_il_agent/agent_core.py falls back to prior-only when no critic
dir is present and says so on stderr). Deck: the same Grimmsnarl list as
il_agent (agents/mcts_il_agent/deck.csv).

Local evidence at build time: beats plain il_agent 67.2% [58.4, 75.0]
pooled (n=119 decided, two independent runs, 0 fallbacks in 15.8K
decisions); 209 ms/decision mean forced-CPU at N=30 vs the 600 s/match
overage bank.

Run from the repo root:
    uv run python scripts/build_mcts_il_submission.py
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
OUT = REPO / "submissions" / "mcts_il_agent"
CHECKPOINT = REPO / "models" / "il_agent"
DECK = REPO / "agents" / "mcts_il_agent" / "deck.csv"
CG = REPO / "data" / "external" / "cg-lib" / "cg"
PTCG_SRC = REPO / "src" / "pokemon_tcg"
PTCG_FILES = [
    "config.py",
    "device.py",
    "il_dataset.py",
    "il_model.py",
    "offline_critic.py",
    "search_api.py",
    "search_prior_mcts.py",
]

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

    shutil.copyfile(REPO / "agents" / "mcts_il_agent" / "agent_core.py", OUT / "agent_core.py")
    (OUT / "main.py").write_text(MAIN_PY)
    shutil.copyfile(DECK, OUT / "deck.csv")
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

    scratch = Path(tempfile.mkdtemp(prefix="mcts_il_"))
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
        assert "model" in mod.agent_core._MODEL_DIR.replace("\\", "/").split("/")[-1], (
            f"bundle must resolve the LOCAL model dir, got {mod.agent_core._MODEL_DIR}"
        )
        assert mod.agent_core._EVALUATOR is not None, "policy failed to load in bundle"
        assert mod.agent_core._EVALUATOR.critic is None, "bundle must be prior-only (no critic)"

        env = make("cabt")
        trace = env.run([mod.agent, mod.agent])
        final = trace[-1]
        snap = mod.agent_core.diag_snapshot()
        print("bundled self-play:", len(trace), "steps | statuses",
              [final[i]["status"] for i in range(2)], "| rewards", [final[i]["reward"] for i in range(2)])
        print("bundle fallback snapshot:", {k: v for k, v in snap.items() if k != "enabled"})
        assert all(final[i]["status"] == "DONE" for i in range(2))
        assert snap["fallbacks"] == 0, f"fallbacks fired in bundle: {snap}"
    finally:
        os.chdir(cwd)
        shutil.rmtree(scratch)
    print("OK: bundled submission plays a full legal game (prior-only, 0 fallbacks) using only bundled files")


if __name__ == "__main__":
    tar_path = build()
    verify_bundle(tar_path)
