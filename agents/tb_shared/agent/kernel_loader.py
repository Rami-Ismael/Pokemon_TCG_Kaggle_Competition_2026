"""Load official Kaggle sample agents from extracted kernel directories.

Each opponent lives under data/kaggle_ref/opponents/<slug>/{main.py,deck.csv}
after `scripts/fetch_official_rule_samples.ps1` + `scripts/extract_public_agents.py`.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import uuid
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPP_DIR = ROOT / "data" / "kaggle_ref" / "opponents"

_cache: dict[str, Callable[[dict], list[int]]] = {}


def load_kernel_agent(slug: str) -> Callable[[dict], list[int]]:
    """Load main.agent from an extracted official kernel (cached)."""
    if slug in _cache:
        return _cache[slug]

    agent_dir = (OPP_DIR / slug).resolve()
    main_py = agent_dir / "main.py"
    if not main_py.is_file():
        raise FileNotFoundError(
            f"official kernel not extracted: {main_py}\n"
            f"Run scripts/fetch_official_rule_samples.ps1 then scripts/extract_public_agents.py"
        )

    cwd = os.getcwd()
    sys.path.insert(0, str(agent_dir))
    try:
        os.chdir(agent_dir)
        mod_name = f"official_{slug}_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(mod_name, main_py)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {main_py}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        fn = getattr(module, "agent", None)
        if fn is None:
            raise AttributeError(f"{main_py} has no agent()")
        _cache[slug] = fn
        return fn
    finally:
        os.chdir(cwd)
        try:
            sys.path.remove(str(agent_dir))
        except ValueError:
            pass
