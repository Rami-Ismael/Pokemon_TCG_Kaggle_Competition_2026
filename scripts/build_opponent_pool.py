"""Build data/opponent_pool.csv -- the Rung-2 public-agent opponent roster.

WHY THIS EXISTS
---------------
Q25 asks whether the opponent pool in benchmark_agents.py is representative of
the ladder. Beating our own three baselines was never evidence about the ladder;
a round-robin against a curated field of *public* agents is. makimakiai maintains
exactly such a field on Kaggle -- 65 public agents + 4 official sample agents = 69
-- as a metadata-only roster (notebook refs + labels, no decklists or replay
dumps). This script harvests that roster into data/opponent_pool.csv, the single
source of truth for "who is in the Rung-2 pool", and records which entries we have
actually mirrored, safety-reviewed, and wired as runnable agents locally.

IMPORTANT: the CSV is a *registry*, not a set of runnable agents. Of the 69 roster
entries, only the subset with a `local_agent` value has code present in this repo
(agents/<local_agent>/agent_core.py), individually reviewed for safety before
wiring in -- see notebooks/reference/INDEX.md. Everything else is a reference-only
watchlist entry: benchmark_agents.py can name it, but cannot run it until someone
pulls the notebook and reviews the code. Do not fabricate runnable agents from
these refs.

SOURCE
------
makimakiai/ptcg-public-28-plus-sample-4-roster-update -- the `AGENT_SOURCES`
literal in its code cell. Apache 2.0 (roster metadata only; individual notebooks
carry their own licenses). Captured via the Kaggle API.

USAGE
-----
    uv run python scripts/build_opponent_pool.py            # pulls the notebook fresh
    uv run python scripts/build_opponent_pool.py --notebook path/to/roster.ipynb

Re-run whenever makimakiai bumps the roster version; commit the CSV diff so the
pool's provenance stays in git history.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.pokemon_tcg import config  # noqa: E402

ROSTER_KERNEL = "makimakiai/ptcg-public-28-plus-sample-4-roster-update"
OUT_CSV = config.DATA_DIR / "opponent_pool.csv"

# source_ref -> local agent key registered in scripts/benchmark_agents.py.
# These are the roster entries whose notebooks we have mirrored under
# notebooks/reference/ and wired as agents/<key>/agent_core.py after an
# individual safety review. Keep in sync with AGENT_FILES over there.
SOURCE_REF_TO_LOCAL_AGENT = {
    "kiyotah/a-sample-rule-based-agent-mega-lucario-ex-deck": "rule_baseline",
    "kiyotah/a-sample-rule-based-agent-dragapult-ex-deck": "kiyotah_dragapult",
    "kiyotah/a-sample-rule-based-agent-iono-s-deck": "kiyotah_iono",
    "kiyotah/a-sample-rule-based-agent-mega-abomasnow-ex-deck": "kiyotah_abomasnow",
    "dedquoc/the-pok-mon-rule-based-engine": "dedquoc_rule_engine",
    "ryotasueyoshi/rule-based-not-psychic-alakazam-best-5th": "ryotasueyoshi_alakazam",
    "makthanithin/improved-probabilistic-agent": "makthanithin_improved_prob",
    "mechi22/ptcg-1070-9-alakazam-rule-based-skeleton": "mechi22_alakazam",
    # Phase-1 archetype-coverage recruitment (Gate 1) -- wired for a *new deck*
    # or a distinct policy, not raw count; same-deck clones stay reference-only.
    "plamen06/pokemon-steel": "plamen06_steel",                       # Metal (Archaludon)
    "prvsiyan/ptcg-ai-battle-control-v11-meta-portfolio": "prvsiyan_control_v11",  # Fighting/Great Tusk/Crustle
    "prvsiyan/ptcg-ai-battle-visible-grim-belief-alakazam-v21": "prvsiyan_grimbelief_alakazam",  # Alakazam belief
    "prvsiyan/ptcg-ai-battle-visible-templates-alakazam-v19": "prvsiyan_templates_alakazam",     # Alakazam templates (same deck, diff policy)
    "pllinas/ptcg-alakazam-rising-tide-v21-improved": "pllinas_alakazam",          # Alakazam rising-tide (new deck)
    "biohack44/beating-the-day-2-new": "biohack44_day2",             # new deck
    "pixiux/ptcg-mega-lucario-ex-v63": "pixiux_lucario_v63",         # Mega Lucario variant (new deck)
    "makthanithin/pokemon-tcg-ai-battle-1084-5-baseline": "makthanithin_1084_baseline",  # scored target LB 1084.5
    "daniilkrasnovvv/pokemon-conservative-probabilistic-agent": "daniilkrasnovvv_conservative_prob",  # distinct policy
    # Wired by origin/main's benchmark-pool consolidation (strongest public
    # opponent, LB ~950); my batch-pull had held it back as a redundant-deck
    # clone, but since it's wired it maps here as runnable.
    "romanrozen/strong-start-baseline-agent-v10-lb-950": "romanrozen_strong_start",
}


def pull_notebook(dest_dir: Path) -> Path:
    """Pull the roster notebook via the Kaggle API into dest_dir; return its path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"pulling {ROSTER_KERNEL} ...")
    subprocess.run(
        ["kaggle", "kernels", "pull", ROSTER_KERNEL, "-p", str(dest_dir)],
        check=True,
    )
    nbs = list(dest_dir.glob("*.ipynb"))
    if not nbs:
        raise FileNotFoundError(f"no .ipynb landed in {dest_dir}")
    return nbs[0]


def extract_agent_sources(notebook: Path) -> list[dict]:
    """Return the AGENT_SOURCES list-of-dicts literal from the notebook's code cells."""
    nb = json.loads(notebook.read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if "AGENT_SOURCES" not in src:
            continue
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "AGENT_SOURCES" for t in node.targets
            ):
                return ast.literal_eval(node.value)
    raise ValueError(f"AGENT_SOURCES not found in {notebook}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--notebook",
        type=Path,
        default=None,
        help="path to an already-pulled roster .ipynb (default: pull fresh)",
    )
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        notebook = args.notebook or pull_notebook(Path(tmp))
        sources = extract_agent_sources(notebook)

    rows = []
    for entry in sources:
        source_ref = entry["source_ref"]
        local = SOURCE_REF_TO_LOCAL_AGENT.get(source_ref, "")
        rows.append(
            {
                "key": entry["key"],
                "source_ref": source_ref,
                "url": entry["url"],
                "label": entry["label"],
                "local_agent": local,
                "runnable": "true" if local else "false",
            }
        )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["key", "source_ref", "url", "label", "local_agent", "runnable"]
        )
        writer.writeheader()
        writer.writerows(rows)

    runnable = sum(1 for r in rows if r["runnable"] == "true")
    # Guard against a silent mapping drift: every source_ref we claim to have
    # wired must actually appear in the roster we just parsed.
    parsed_refs = {r["source_ref"] for r in rows}
    stale = [ref for ref in SOURCE_REF_TO_LOCAL_AGENT if ref not in parsed_refs]
    if stale:
        print(f"  [WARN] mapped source_refs absent from roster (stale?): {stale}")

    print(f"wrote {OUT_CSV}  ({len(rows)} roster entries, {runnable} runnable locally)")


if __name__ == "__main__":
    main()
