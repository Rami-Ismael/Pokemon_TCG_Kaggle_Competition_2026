"""Build the search + BC-prior hybrid + Alakazam-deck submission.

agents/mega_lucario/agent_core_improved.py's SEARCH_ALGO (1-ply UCB1 rollout
search, real multi-candidate lists via rank_all() -- see notes/phase0_gate0_report.md
for why the OLD winning-827.8 source didn't) with bc_prior.py's PUCT-style
bonus term, forced ON via main.py (HEAD defaults USE_SEARCH=0 since it lost
to pure heuristic on the Lucario deck; Kaggle's evaluator sets no env vars,
so the override has to happen in-process before agent_core_improved imports).

Why build this at all given the weaker local result: notes/phase0_gate0_report.md's
A0'' section found this exact config (USE_SEARCH=1 USE_BC_PRIOR=1 on the
Alakazam deck) scored only 10.0% [1.8,40.4] vs makthanithin_improved_prob,
tied with/worse than the no-search control (20.0%) at n=10/arm -- weaker
evidence than the pure-BC submission (build_il_alakazam_submission.py,
62.5% [38.6,81.5]). Built and submitted anyway per explicit request, to get
a real ladder read on both rather than assume the local benchmark predicts
correctly (see notes/phase1_gate1_report.md's own finding that the
local-vs-ladder residual is only collectable after a submission).

Checkpoint (for the BC prior): models/il_agent/ (current, epoch 2 step
38562) -- same one used in the A0'' benchmark, NOT the preserved 827.8
checkpoint.

Run from the repo root with:
    uv run python scripts/build_search_prior_alakazam_submission.py
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
OUT = REPO / "submissions" / "search_prior_alakazam"
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

# HEAD's agent_core_improved.py defaults USE_SEARCH=0 (real search lost to
# pure heuristic on the Lucario deck, see notes/phase0_gate0_report.md).
# This submission is deliberately testing search+prior ON, on a deck the BC
# prior has real trained signal for -- force it before the module-level
# os.environ.get() read happens at import time.
os.environ.setdefault("USE_SEARCH", "1")
os.environ.setdefault("USE_BC_PRIOR", "1")

import agent_core_improved

with open(os.path.join(_BUNDLE_DIR, "deck.csv")) as f:
    agent_core_improved.my_deck = [
        int(x) for x in f.read().splitlines() if x.strip()
    ][:60]

agent = agent_core_improved.agent
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

    shutil.copyfile(REPO / "agents" / "mega_lucario" / "agent_core_improved.py", OUT / "agent_core_improved.py")
    shutil.copyfile(REPO / "agents" / "mega_lucario" / "bc_prior.py", OUT / "bc_prior.py")
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
        for name in ["main.py", "agent_core_improved.py", "bc_prior.py", "deck.csv"]:
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

    scratch = Path(tempfile.mkdtemp(prefix="search_prior_alakazam_"))
    with tarfile.open(tar_path) as t:
        t.extractall(scratch)

    for mod in list(sys.modules):
        if mod.split(".")[0] in ("agent_core_improved", "bc_prior", "cg", "pokemon_tcg"):
            sys.modules.pop(mod, None)
    os.environ.pop("USE_SEARCH", None)
    os.environ.pop("USE_BC_PRIOR", None)

    cwd = os.getcwd()
    os.chdir(scratch)
    try:
        spec = importlib.util.spec_from_file_location("bundle_main", scratch / "main.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        print("bundled deck len:", len(mod.agent_core_improved.my_deck))
        print("bundled USE_SEARCH:", mod.agent_core_improved.USE_SEARCH)
        print("bundled USE_BC_PRIOR:", mod.agent_core_improved.USE_BC_PRIOR)
        print("bundled BC_PRIOR_C:", mod.agent_core_improved.BC_PRIOR_C)

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
