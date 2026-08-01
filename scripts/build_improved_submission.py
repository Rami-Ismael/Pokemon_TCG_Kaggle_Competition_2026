"""Build and verify the Improved Probabilistic agent submission bundle.

Layout (top-level main.py required by the harness):
    submission.tar.gz
    ├── main.py
    ├── agent_core_improved.py
    ├── deck.csv
    └── cg/

Run from the repo root with:
    uv run python scripts/build_improved_submission.py
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "submissions" / "mega_lucario_improved"
AGENT_MOD = REPO / "agents" / "mega_lucario" / "agent_core_improved.py"
CG = REPO / "data" / "external" / "cg-lib" / "cg"


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    # Copy the (tested) agent module into the bundle so main.py can import it.
    shutil.copyfile(AGENT_MOD, OUT / "agent_core_improved.py")

    tar_path = OUT / "submission.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(OUT / "main.py", arcname="main.py")
        tar.add(OUT / "agent_core_improved.py", arcname="agent_core_improved.py")
        tar.add(OUT / "deck.csv", arcname="deck.csv")
        for item in sorted(CG.rglob("*")):
            if not item.is_file():
                continue
            if "__pycache__" in item.parts or item.suffix in {".pyc", ".pyo"}:
                continue
            tar.add(item, arcname=str(Path("cg") / item.relative_to(CG)))
    print(f"built {tar_path} ({tar_path.stat().st_size / 2**20:.2f} MiB)")

    with tarfile.open(tar_path) as t:
        names = t.getnames()
    assert "main.py" in names, "main.py must sit at the top level"
    print("top-level entries:", sorted(n for n in names if "/" not in n))
    return tar_path


def verify_bundle(tar_path: Path):
    """Unpack and self-play using ONLY the bundled files (eval-sandbox view)."""
    from kaggle_environments import make

    scratch = Path(tempfile.mkdtemp(prefix="impr_sub_"))
    with tarfile.open(tar_path) as t:
        t.extractall(scratch)

    # Fresh module view: only the bundle on sys.path.
    for mod in ["agent_core_improved", "cg", "cg.api", "cg.sim", "cg.game", "cg.utils"]:
        sys.modules.pop(mod, None)

    cwd = os.getcwd()
    os.chdir(scratch)
    try:
        spec = importlib.util.spec_from_file_location("bundle_main", scratch / "main.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        print("bundled deck len:", len(mod.agent_core_improved.my_deck))

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
    print("OK: bundled submission plays a full legal game using only bundled files")


if __name__ == "__main__":
    tar_path = build()
    verify_bundle(tar_path)
    print("\nREADY TO SUBMIT:")
    print(f"  uv run kaggle competitions submit -c pokemon-tcg-ai-battle \\")
    print(f"      -f {tar_path} -m \"Improved Probabilistic agent (reimplementation)\"")
