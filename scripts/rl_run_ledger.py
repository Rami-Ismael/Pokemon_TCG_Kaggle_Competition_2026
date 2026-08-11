"""What did each RL run actually train against, and when did it run?

Motivation (Rami, 2026-08-06): argparse defaults move, so reading today's
`--mix` help text onto an older run gets that run's opponent distribution
backwards. Concretely, `--mix` did not exist until e8298fc (2026-08-04
20:06:15-05:00); before it, PTCGGym's own default `(0.5, 0.3, 0.2)` applied and
runs drew EXTERNAL opponents 20% of the time, while every run after it drew
them 0% by default. The 2026-08-03 `ppo_puffer` generation and the 2026-08-05
`selfplay_g*` generation therefore trained in materially different rooms, and a
post-mortem that treats them as one experiment is comparing two things.

This script reconstructs, per run:
  * WHEN it ran -- from snapshot mtimes, plus the `init_from` chain, which
    orders generations even when mtimes are copy times rather than write times.
  * WHICH COMMIT was live then -- `git rev-list -1 --before=<date>` on the
    default branch.
  * WHAT THE DEFAULTS WERE at that commit -- the `--mix` default is read out of
    the driver source AS OF that commit, not as of today; absence of the flag
    resolves to the env default recorded in DEFAULTS_BEFORE_MIX_FLAG.
  * WHAT IT ACTUALLY PLAYED -- run_config.json when present (written at launch
    by train_ppo_puffer.py since 2026-08-06), else the reconstruction above,
    clearly labelled as inferred.

Going forward the reconstruction path should stay empty: every new run writes
run_config.json with its resolved args, git commit, and derived external-
opponent share. Archaeology is the fallback, not the plan.

Usage:
    uv run python scripts/rl_run_ledger.py
    uv run python scripts/rl_run_ledger.py --json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pokemon_tcg import config  # noqa: E402

# The env-side default that applied whenever the driver passed no `mix` at all.
# Source: PTCGGym.__init__(mix=(0.5, 0.3, 0.2)) in pokemon_tcg/puffer_env.py,
# which has carried that signature since the file was introduced.
DEFAULTS_BEFORE_MIX_FLAG = (0.5, 0.3, 0.2)

# Landmarks worth printing next to the runs they reframe. Dates are commit
# dates on the default branch, verified with `git log -- <path>`.
LANDMARKS = [
    ("2026-08-04T20:06:15-05:00", "e8298fc",
     "--mix added; default 0.625,0.375,0 drops the public pool to 0% "
     "(was 20% via the env default)"),
    ("2026-08-05T10:30:26-05:00", "2f5c244",
     "--deck-pool / --mirror-deck added; before this every run was "
     "single-deck with no way to vary it"),
    ("2026-08-03T17:05:59-05:00", "3033811",
     "KL anchor + promotion ratchet added"),
]


def git(*a: str) -> str:
    return subprocess.run(("git", *a), cwd=REPO, capture_output=True,
                          text=True).stdout.strip()


def commit_at(when: dt.datetime) -> dict:
    """The state of the MAINLINE at `when`.

    --first-parent is load-bearing. Without it `rev-list --before` returns any
    commit reachable from main, including one authored on a side branch that
    merged later: that picked 2024c5e2 for the 08-05 11:09 selfplay_g1 run, a
    commit that does NOT contain the 08-04 --mix change, and so reported 20%
    external opponents for a run whose mainline had already moved to 0%.
    Walking first-parent restricts the answer to merges that had actually
    landed, which is what a checkout of main that day would have had.
    """
    sha = git("rev-list", "-1", "--first-parent", f"--before={when.isoformat()}", "main")
    if not sha:
        return {"sha": None, "date": None, "subject": None}
    date, subject = git("show", "-s", "--format=%ad|%s", "--date=iso-strict", sha).split("|", 1)
    return {"sha": sha[:9], "date": date, "subject": subject}


def mix_default_at(sha: str | None) -> tuple[tuple[float, float, float], str]:
    """The mirror/league/pool shares in force at `sha`, and how we know.

    Reads the driver source as of that commit rather than trusting the current
    file -- that substitution is the exact error this module exists to prevent.
    """
    if sha is None:
        return DEFAULTS_BEFORE_MIX_FLAG, "unknown commit; assumed env default"
    src = git("show", f"{sha}:scripts/train_ppo_puffer.py")
    if not src:
        return DEFAULTS_BEFORE_MIX_FLAG, "driver absent at this commit"
    m = re.search(r'"--mix",\s*default="([\d.,\s]+)"', src)
    if not m:
        return (DEFAULTS_BEFORE_MIX_FLAG,
                "no --mix flag at this commit -> PTCGGym default (0.5,0.3,0.2)")
    shares = tuple(float(x) for x in m.group(1).split(","))
    return shares, f"--mix default at {sha}"


def scan_runs(models_dir: Path) -> list[dict]:
    runs: list[dict] = []
    for meta_dir in sorted(models_dir.glob("*")):
        if not meta_dir.is_dir():
            continue
        metas = sorted(meta_dir.glob("**/ppo_metadata.json"))
        if not metas:
            continue
        stamps = [p.stat().st_mtime for p in metas]
        first, last = min(stamps), max(stamps)
        payloads = []
        for p in metas:
            try:
                payloads.append(json.loads(p.read_text()))
            except (OSError, ValueError):
                pass
        init_from = next((m.get("init_from") for m in payloads if m.get("init_from")), None)
        run_tags = sorted({m.get("run_tag") or "" for m in payloads} - {""})
        max_step = max((m.get("global_step") or 0 for m in payloads), default=0)

        cfg_path = meta_dir / "run_config.json"
        recorded = None
        if cfg_path.exists():
            try:
                recorded = json.loads(cfg_path.read_text())
            except (OSError, ValueError):
                recorded = None

        started = dt.datetime.fromtimestamp(first)
        if recorded and recorded.get("started_unix"):
            started = dt.datetime.fromtimestamp(recorded["started_unix"])

        runs.append({
            "run": meta_dir.name,
            "snapshots": len(metas),
            "max_step": max_step,
            "run_tags": run_tags,
            "init_from": init_from,
            "first_snapshot": dt.datetime.fromtimestamp(first).isoformat(timespec="minutes"),
            "last_snapshot": dt.datetime.fromtimestamp(last).isoformat(timespec="minutes"),
            "span_hours": round((last - first) / 3600, 1),
            "started": started,
            "recorded_config": recorded,
        })
    runs.sort(key=lambda r: r["started"])
    return runs


def resolve(run: dict) -> dict:
    rec = run["recorded_config"]
    if rec:
        o = rec["opponents"]
        return {
            "source": "recorded at launch (run_config.json)",
            "commit": (rec.get("git") or {}).get("commit", "")[:9] or None,
            "mix": o["mix_mirror_league_pool"],
            "external_share": o["external_opponent_share"],
            "deck_k": rec["decks"]["k"],
            "mirror_deck": rec["decks"]["mirror_deck"],
            "note": "",
        }
    c = commit_at(run["started"])
    mix, how = mix_default_at(c["sha"])
    # Reconstruction cannot see an explicit --mix or --league in the launch
    # command, so it reports the DEFAULT that was live. Every historical run
    # script was checked for overrides; none passed --mix. Deck-pool support
    # did not exist before 2f5c244, and no run script passes --deck-pool.
    return {
        "source": "inferred from the commit live that day",
        "commit": c["sha"],
        "mix": list(mix),
        # league default is our own checkpoint(s) -> only mix[2] is external
        "external_share": round(mix[2], 4),
        "deck_k": 1,
        "mirror_deck": False,
        "note": how,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--models-dir", type=Path, default=config.MODELS_DIR,
                    help="where the run artifacts live. Defaults to this "
                         "checkout's models/, which in a WORKTREE is mostly "
                         "empty (models/ is gitignored and only partially "
                         "symlinked) -- point it at the main checkout's "
                         "models/ to see the full history")
    args = ap.parse_args()

    runs = scan_runs(args.models_dir)
    if not runs:
        print(f"no RL run artifacts under {args.models_dir} -- if this is a "
              f"worktree, pass --models-dir <main checkout>/models")
        return
    for r in runs:
        r["resolved"] = resolve(r)

    if args.json:
        for r in runs:
            r["started"] = r["started"].isoformat(timespec="minutes")
        print(json.dumps(runs, indent=2))
        return

    marks = sorted(LANDMARKS, key=lambda m: m[0])
    mark_i = 0
    print(f"{'run':20} {'ran (first snapshot)':20} {'span':>6} {'steps':>9} "
          f"{'commit':>10} {'mix (mir/lg/pool)':>18} {'ext':>5} {'decks':>6}  source")
    print("-" * 132)
    for r in runs:
        while mark_i < len(marks) and dt.datetime.fromisoformat(marks[mark_i][0]).replace(tzinfo=None) < r["started"]:
            when, sha, what = marks[mark_i]
            print(f"   -- {when[:16]}  {sha}  {what}")
            mark_i += 1
        res = r["resolved"]
        mix = "/".join(f"{x:g}" for x in res["mix"])
        deck = f"K={res['deck_k']}" + ("m" if res["mirror_deck"] else "")
        tag = "recorded" if res["source"].startswith("recorded") else "INFERRED"
        print(f"{r['run']:20} {r['first_snapshot']:20} {r['span_hours']:5.1f}h "
              f"{r['max_step']:9,} {str(res['commit']):>10} {mix:>18} "
              f"{res['external_share']:5.0%} {deck:>6}  {tag}")
    for when, sha, what in marks[mark_i:]:
        print(f"   -- {when[:16]}  {sha}  {what}")

    print()
    print("chain (init_from), which orders generations even when snapshot "
          "mtimes are copy times:")
    for r in runs:
        if r["init_from"]:
            print(f"  {r['run']:20} <- {r['init_from']}")
    inferred = [r for r in runs if r["resolved"]["source"].startswith("inferred")]
    if inferred:
        print(f"\n{len(inferred)} run(s) INFERRED: no run_config.json, so the "
              f"mix/deck columns are the DEFAULTS live at that commit, not a "
              f"record of the launch command. Runs from 2026-08-06 on record "
              f"their own config and should never appear here.")


if __name__ == "__main__":
    main()
