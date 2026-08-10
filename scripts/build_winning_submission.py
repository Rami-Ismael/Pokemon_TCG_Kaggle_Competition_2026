"""Rebuild the exact bundle that earned Kaggle submission 55162376 (public 827.8).

Reads configs/winning_build.json for provenance (checkpoint dir, sha256, git SHA
of the agent source at submission time) so the next submission's provenance is a
diff against that file, not archaeology.

Rebuilds FROM models/il_agent_winning_827.8/ (the preserved, literal 827.8
artifact) -- NOT from agents/mega_lucario/agent_core_improved.py or
models/il_agent/, both of which have since diverged (candidate-truncation
fix, deck-out penalty regrade, global-state leak fix).

Layout (top-level main.py required by the Kaggle harness):
    submission.tar.gz
    ├── main.py
    ├── agent_core_improved.py
    ├── bc_prior.py
    ├── deck.csv
    ├── cg/
    ├── model/
    │   ├── config.json
    │   └── model.safetensors
    └── pokemon_tcg/

Run from the repo root with:
    uv run python scripts/build_winning_submission.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO / "configs" / "winning_build.json"
OUT = REPO / "submissions" / "winning_827_8_rebuild"
CG = REPO / "data" / "external" / "cg-lib" / "cg"
PTCG_SRC = REPO / "src" / "pokemon_tcg"
PTCG_FILES = ["config.py", "device.py", "il_dataset.py", "il_model.py"]


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(cfg: dict) -> Path:
    ckpt_dir = REPO / cfg["checkpoint_dir"]
    safetensors = ckpt_dir / "model.safetensors"
    if not safetensors.exists():
        raise FileNotFoundError(
            f"{safetensors} missing -- the preserved winning checkpoint is not "
            f"present in this checkout. It lives in the main repo's models/ dir "
            f"(gitignored) and is not tracked by git."
        )
    actual_sha = sha256_of(safetensors)
    expected_sha = cfg["checkpoint_sha256"]
    if actual_sha != expected_sha:
        raise ValueError(
            f"checkpoint sha256 mismatch!\n  expected: {expected_sha}\n  actual:   {actual_sha}\n"
            f"This is NOT the checkpoint that earned 827.8 -- refusing to build."
        )
    print(f"checkpoint sha256 verified: {actual_sha}")

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    (OUT / "model").mkdir()
    (OUT / "pokemon_tcg").mkdir()

    for name in ["main.py", "agent_core_improved.py", "bc_prior.py", "deck.csv"]:
        shutil.copyfile(ckpt_dir / name, OUT / name)
    for name in ["config.json", "model.safetensors"]:
        shutil.copyfile(ckpt_dir / name, OUT / "model" / name)
    for name in PTCG_FILES:
        shutil.copyfile(PTCG_SRC / name, OUT / "pokemon_tcg" / name)
    (OUT / "pokemon_tcg" / "__init__.py").touch()

    if not CG.exists():
        raise FileNotFoundError(f"{CG} missing -- cg-lib vendor dir not found")
    cg_dest = OUT / "cg"
    shutil.copytree(
        CG,
        cg_dest,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )

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
    assert "main.py" in names, "main.py must sit at the top level"
    print("top-level entries:", sorted(n for n in names if "/" not in n))
    return tar_path


def verify_bundle(tar_path: Path, cfg: dict):
    """Unpack and self-play using ONLY the bundled files (eval-sandbox view).

    No IL_MODEL_DIR / USE_SEARCH / USE_BC_PRIOR env overrides are set here --
    the real Kaggle evaluator doesn't set them either, so this exercises the
    exact hardcoded/default values baked into the winning source (see
    cfg['known_dead_path_warning']: USE_SEARCH and USE_BC_PRIOR are True by
    default but the code path they gate never executes at MAIN decisions).
    """
    from kaggle_environments import make

    scratch = Path(tempfile.mkdtemp(prefix="winning_sub_"))
    with tarfile.open(tar_path) as t:
        t.extractall(scratch)

    for mod in list(sys.modules):
        if mod.split(".")[0] in ("agent_core_improved", "bc_prior", "cg", "pokemon_tcg"):
            sys.modules.pop(mod, None)

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
        print(
            "bundled self-play:",
            len(trace),
            "steps | statuses",
            [final[i]["status"] for i in range(2)],
            "| rewards",
            [final[i]["reward"] for i in range(2)],
        )
        assert all(final[i]["status"] == "DONE" for i in range(2))
    finally:
        os.chdir(cwd)
        shutil.rmtree(scratch)
    print("OK: rebuilt bundle plays a full legal game using only bundled files")


if __name__ == "__main__":
    cfg = load_config()
    print(f"rebuilding submission {cfg['kaggle_submission_ref']} (public {cfg['public_score']})")
    tar_path = build(cfg)
    verify_bundle(tar_path, cfg)
    print("\nProvenance for the next submission description:")
    print(f"  checkpoint sha256: {cfg['checkpoint_sha256']}")
    print(f"  agent source git SHA: {cfg['agent_source_git_sha']} (see {CONFIG_PATH.relative_to(REPO)})")
