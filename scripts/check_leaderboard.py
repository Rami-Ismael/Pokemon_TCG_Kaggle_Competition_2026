"""Read the real Kaggle ladder for pokemon-tcg-ai-battle and print where we stand.

WHY THIS EXISTS
---------------
Local benchmarks in this repo (Glicko round-robin, head-to-head win rates)
have now twice ranked our agents in the OPPOSITE order of the real public
leaderboard:

  * 2026-08-02: improved_prob_main was "top-ranked among our own agents
    (Glicko 1720.2)" locally -> scored 701.6 on the ladder, while the agent
    it "beat" locally had scored 804.0.
  * 2026-08-03: ppo_u120832 beat s2_e1_s43 87.5% [69.0, 95.7] head-to-head
    locally (Glicko 1478 vs 1300, non-overlapping) -> scored 232.1 on the
    ladder vs s2_e1_s43's 395.0.

The local pool measures strength *within our pool*; the ladder measures
strength against ~6,000 heterogeneous teams. They are different quantities.
This script fetches the ground truth so no claim of "better" is ever made
from local numbers alone. See .claude/skills/leaderboard-check/SKILL.md for
the rules that consume this output.

WHAT IT DOES
------------
1. Reads our submission history (`kaggle competitions submissions`): newest,
   best-ever, pending.
2. Downloads the full public leaderboard CSV and finds our team row by the
   username in `kaggle config view` -> real rank, team score, field size,
   top score, top-8 cutoff.
3. Prints a plain-text report, flags divergences (newest submission worse
   than what it replaced; team score below our best-ever because older
   submissions no longer count), and appends a timestamped snapshot to
   reports/leaderboard_history.jsonl so ladder-vs-local residuals can be
   tracked over time.

Auth: requires `uv run kaggle auth login` (OAuth) once per machine. If this
script errors with 401/403, that is the fix -- not kaggle.json.

Usage:
    uv run python scripts/check_leaderboard.py
    uv run python scripts/check_leaderboard.py --submissions-only  # skip CSV download
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from pokemon_tcg.config import COMPETITION, REPORTS_DIR

SNAPSHOT_PATH = REPORTS_DIR / "leaderboard_snapshot.json"
HISTORY_PATH = REPORTS_DIR / "leaderboard_history.jsonl"


def _kaggle(*args: str) -> subprocess.CompletedProcess:
    exe = shutil.which("kaggle")
    if exe is None:
        sys.exit(
            "kaggle CLI not found on PATH. Run this script via "
            "`uv run python scripts/check_leaderboard.py` so the project "
            "venv is active."
        )
    proc = subprocess.run([exe, *args], capture_output=True, text=True)
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout).strip()
        hint = ""
        if any(tok in msg for tok in ("401", "403", "Unauthorized", "Forbidden")):
            hint = (
                "\nAuth fix: `uv run kaggle auth login` "
                "(OAuth; kaggle.json is not used here)."
            )
        sys.exit(f"kaggle {' '.join(args)} failed:\n{msg}{hint}")
    return proc


def _username() -> str:
    out = _kaggle("config", "view").stdout
    for line in out.splitlines():
        if "username" in line:
            return line.split(":", 1)[1].strip()
    sys.exit(
        "Could not read username from `kaggle config view`; "
        "run `uv run kaggle auth login`."
    )


def fetch_submissions() -> list[dict]:
    """Our submission history, newest first. Score is None while pending."""
    out = _kaggle("competitions", "submissions", "-c", COMPETITION, "-v").stdout
    # -v emits CSV; the first line can be a "Next Page Token" banner on some
    # CLI versions, so scan for the header row instead of assuming line 0.
    lines = out.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("ref,"))
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))
    subs = []
    for r in rows:
        score = r.get("publicScore", "").strip()
        subs.append(
            {
                "ref": r["ref"].strip(),
                "date": r["date"].strip(),
                "status": r["status"].strip(),
                "score": float(score) if score else None,
                "description": r.get("description", "").strip(),
            }
        )
    subs.sort(key=lambda s: s["date"], reverse=True)
    return subs


def fetch_leaderboard_row(username: str) -> tuple[dict | None, dict]:
    """(our team row or None, field stats) from the full public leaderboard CSV."""
    with tempfile.TemporaryDirectory() as td:
        _kaggle(
            "competitions", "leaderboard", "-c", COMPETITION, "--download", "-p", td
        )
        zips = list(Path(td).glob("*.zip"))
        if not zips:
            sys.exit("Leaderboard download produced no zip file.")
        with zipfile.ZipFile(zips[0]) as zf:
            zf.extractall(td)
        csvs = [p for p in Path(td).glob("*.csv")]
        if not csvs:
            sys.exit("Leaderboard zip contained no CSV.")
        # utf-8-sig: Kaggle's CSV starts with a BOM that would corrupt the
        # first fieldname ("Rank") under plain utf-8.
        with open(csvs[0], encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))

    ours = None
    uname = username.lower()
    for r in rows:
        raw_members = r.get("TeamMemberUserNames", "").split(",")
        members = [m.strip().lower() for m in raw_members]
        if uname in members:
            ours = r
            break

    by_rank = {int(r["Rank"]): float(r["Score"]) for r in rows}
    field = {
        "total_teams": len(rows),
        "top_score": by_rank.get(1),
        # Top 8 advance (competition prize structure); rank 8's score is the bar.
        "top8_cutoff": by_rank.get(8),
    }
    return ours, field


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--submissions-only",
        action="store_true",
        help="skip the full-leaderboard CSV download (no rank/field stats)",
    )
    args = ap.parse_args()

    username = _username()
    subs = fetch_submissions()
    scored = [s for s in subs if s["score"] is not None]
    pending = [s for s in subs if s["score"] is None and "ERROR" not in s["status"]]
    newest = subs[0] if subs else None
    best_ever = max(scored, key=lambda s: s["score"]) if scored else None

    print(f"=== Ladder truth: {COMPETITION} (user {username}) ===")

    ours = None
    field: dict = {}
    if not args.submissions_only:
        ours, field = fetch_leaderboard_row(username)
        if ours is None:
            print("!! Our team was not found on the public leaderboard CSV.")
        else:
            rank, score = int(ours["Rank"]), float(ours["Score"])
            n_active = int(ours.get("SubmissionCount") or 0)
            print(f"Rank {rank} / {field['total_teams']}  |  team score {score}")
            print(
                f"Top score {field['top_score']}  |  "
                f"top-8 cutoff {field['top8_cutoff']}"
                f"  (gap to cutoff: {field['top8_cutoff'] - score:+.1f})"
            )
            print(
                f"Leaderboard counts {n_active} submission(s) for the team -- the team "
                f"score rests on recent submissions only, not on best-ever."
            )

    if best_ever:
        print(
            f"Best-ever submission: {best_ever['score']} "
            f"({best_ever['ref']}, {best_ever['date'][:10]})"
        )
    if newest:
        print(
            f"Newest submission:    {newest['score']} "
            f"({newest['ref']}, {newest['date'][:16]}, {newest['status']})"
        )
    if pending:
        print(f"Pending scores: {', '.join(s['ref'] for s in pending)}")

    print()
    if newest and newest["score"] is not None and len(scored) >= 2:
        prev = next((s for s in scored if s["ref"] != newest["ref"]), None)
        if prev and newest["score"] < prev["score"]:
            print(
                f"DIVERGENCE CHECK: newest submission ({newest['score']}) "
                f"scored BELOW the one before it ({prev['score']}, "
                f"{prev['ref']}). If it was submitted as an improvement, "
                f"the ladder disagrees -- say so explicitly."
            )
    if ours is not None and best_ever and float(ours["Score"]) < best_ever["score"]:
        print(
            f"DISPLACEMENT: team score {float(ours['Score'])} < best-ever "
            f"{best_ever['score']}. Recent submissions displaced the best "
            f"agent from the active set; the team is currently represented "
            f"by something weaker than its best-known build."
        )

    snapshot = {
        "retrieved_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "competition": COMPETITION,
        "username": username,
        "rank": int(ours["Rank"]) if ours else None,
        "total_teams": field.get("total_teams"),
        "team_score": float(ours["Score"]) if ours else None,
        "team_submission_count": int(ours["SubmissionCount"]) if ours else None,
        "top_score": field.get("top_score"),
        "top8_cutoff": field.get("top8_cutoff"),
        "best_ever": (
            {k: best_ever[k] for k in ("ref", "score", "date")} if best_ever else None
        ),
        "newest": (
            {k: newest[k] for k in ("ref", "score", "date", "status")}
            if newest
            else None
        ),
        "pending_refs": [s["ref"] for s in pending],
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2) + "\n")
    with open(HISTORY_PATH, "a") as fh:
        fh.write(json.dumps(snapshot) + "\n")
    rel = SNAPSHOT_PATH.relative_to(REPORTS_DIR.parent)
    print(f"\nSnapshot -> {rel} (history appended)")


if __name__ == "__main__":
    main()
