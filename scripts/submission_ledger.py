"""Per-submission ledger: rich local metadata + score-over-time for every Kaggle submission.

WHY THIS EXISTS
---------------
Kaggle's submit-message field caps at 500 characters — nowhere near enough to
record what a submission actually is (approach, config, git SHA, local
evidence, what it displaces from the active set). And ladder scores are live
ratings that drift for hours after posting (55215267 read 232.1 -> 265.5 ->
290.8 -> 299.1 -> 319.0 across one day), so a single score column is also not
enough: each submission needs a *timeline* of readings.

`notes/scores.md` holds the human-curated one-row-per-submission table (Gate 5
protocol). `reports/leaderboard_history.jsonl` tracks the TEAM (rank, active
set). This ledger fills the gap between them: machine-readable, per-REF,
append-only, unlimited detail, many timestamped score readings per ref.

STORAGE
-------
`reports/submission_ledger.jsonl`, append-only. Two event kinds:

  {"event": "submit", "ref": "...", ...metadata, "approach": "<unlimited text>"}
  {"event": "score",  "ref": "...", "at_utc": "...", "score": 670.4, "status": "..."}

A later "submit" event for the same ref amends the earlier one (later-wins per
field on fold). Score events are deduped: `refresh` only appends when the
score or status changed since the last reading for that ref.

USAGE
-----
    uv run python scripts/submission_ledger.py refresh
        Pull `kaggle competitions submissions` (live), append a score reading
        for every ref whose score/status moved, auto-stub refs never logged.

    uv run python scripts/submission_ledger.py log --ref 55228113 \
        --deck "Mega Lucario ex" --git-sha e42f3bb \
        --config "USE_SEARCH=0, AdvancedPolicy heuristic" \
        --local-result "same-build resubmit, no local eval" \
        --displaces "expected to displace ppo_u120832 (265-319) from active set" \
        --approach-file notes/submission_55228113.md
        (--approach "inline text" also works; so does piping to --approach -)

    uv run python scripts/submission_ledger.py show
        Table: ref, date, first -> latest score, drift, #reads, summary.

    uv run python scripts/submission_ledger.py show --ref 55228113
        Full metadata + complete score timeline for one submission.

Auth: `uv run kaggle auth login` (OAuth) once per machine — kaggle.json is
not the mechanism. `refresh` makes ONE cheap submissions-list call; it does
NOT download the 6k-row leaderboard CSV (that's check_leaderboard.py's job).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from pokemon_tcg.config import REPORTS_DIR

# fetch_submissions lives in check_leaderboard.py; sys.path[0] is scripts/
# when invoked as `uv run python scripts/submission_ledger.py`.
from check_leaderboard import fetch_submissions

LEDGER_PATH = REPORTS_DIR / "submission_ledger.jsonl"

SUBMIT_FIELDS = (
    "deck",
    "git_sha",
    "checkpoint_sha256",
    "config",
    "local_result",
    "displaces",
    "expects",
    "approach",
)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _read_events() -> list[dict]:
    if not LEDGER_PATH.exists():
        return []
    events = []
    for i, line in enumerate(LEDGER_PATH.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            sys.exit(f"{LEDGER_PATH}:{i} is not valid JSON — fix or remove the line.")
    return events


def _append(events: list[dict]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LEDGER_PATH, "a") as fh:
        for ev in events:
            fh.write(json.dumps(ev, ensure_ascii=False) + "\n")


def _fold(events: list[dict]) -> dict[str, dict]:
    """ref -> {meta fields (later submit events win per field), readings: [...]}"""
    subs: dict[str, dict] = {}
    for ev in events:
        ref = ev["ref"]
        entry = subs.setdefault(ref, {"ref": ref, "readings": []})
        if ev["event"] == "submit":
            for k, v in ev.items():
                if k not in ("event", "ref") and v is not None:
                    entry[k] = v
        elif ev["event"] == "score":
            entry["readings"].append(ev)
    for entry in subs.values():
        entry["readings"].sort(key=lambda r: r["at_utc"])
    return subs


def cmd_refresh(_args: argparse.Namespace) -> None:
    kaggle_subs = fetch_submissions()  # newest first; live API call
    folded = _fold(_read_events())
    now = _now()
    new_events: list[dict] = []

    for s in kaggle_subs:
        ref = s["ref"]
        entry = folded.get(ref)
        if entry is None:
            # Never seen: stub it so the 500-char kaggle description (often the
            # only surviving record for old refs) is preserved verbatim.
            new_events.append(
                {
                    "event": "submit",
                    "ref": ref,
                    "stub": True,
                    "logged_at_utc": now,
                    "kaggle_date": s["date"],
                    "kaggle_description": s["description"],
                }
            )
            entry = {"readings": []}
        last = entry["readings"][-1] if entry["readings"] else None
        changed = (
            last is None
            or last.get("score") != s["score"]
            or last.get("status") != s["status"]
        )
        if changed:
            new_events.append(
                {
                    "event": "score",
                    "ref": ref,
                    "at_utc": now,
                    "score": s["score"],
                    "status": s["status"],
                }
            )

    _append(new_events)
    n_scores = sum(1 for e in new_events if e["event"] == "score")
    n_stubs = sum(1 for e in new_events if e["event"] == "submit")
    print(
        f"refresh: {len(kaggle_subs)} submissions on Kaggle; "
        f"{n_scores} new score reading(s), {n_stubs} stubbed ref(s) "
        f"-> {LEDGER_PATH.relative_to(REPORTS_DIR.parent)}"
    )
    if n_scores == 0:
        print("(no score moved since last refresh — that is itself a data point)")


def cmd_log(args: argparse.Namespace) -> None:
    approach = args.approach
    if args.approach_file:
        approach = Path(args.approach_file).read_text()
    elif approach == "-":
        approach = sys.stdin.read()

    ev = {"event": "submit", "ref": args.ref, "logged_at_utc": _now()}
    for field in SUBMIT_FIELDS:
        if field == "approach":
            if approach:
                ev["approach"] = approach
            continue
        val = getattr(args, field)
        if val is not None:
            ev[field] = val
    if args.link:
        ev["links"] = args.link

    if len([k for k in ev if k not in ("event", "ref", "logged_at_utc")]) == 0:
        sys.exit("log: nothing to record — pass at least one metadata flag.")
    _append([ev])
    print(f"logged submit metadata for ref {args.ref} -> {LEDGER_PATH.name}")
    print("(amendment semantics: a later `log` for the same ref overrides per field)")


def _fmt_score(x: float | None) -> str:
    return f"{x:.1f}" if x is not None else "pending"


def cmd_show(args: argparse.Namespace) -> None:
    subs = _fold(_read_events())
    if not subs:
        sys.exit(f"ledger empty — run `refresh` first ({LEDGER_PATH} missing).")

    if args.ref:
        entry = subs.get(args.ref)
        if entry is None:
            sys.exit(f"ref {args.ref} not in ledger. Known: {', '.join(sorted(subs))}")
        readings = entry.pop("readings")
        print(json.dumps(entry, indent=2, ensure_ascii=False))
        print(f"\nScore timeline ({len(readings)} reading(s)):")
        for r in readings:
            print(f"  {r['at_utc']}  {_fmt_score(r['score']):>8}  {r.get('status', '')}")
        return

    def sort_key(e: dict) -> str:
        return e.get("kaggle_date") or e.get("logged_at_utc") or ""

    rows = sorted(subs.values(), key=sort_key, reverse=True)
    print(f"{'ref':>10}  {'date':10}  {'first':>8}  {'latest':>8}  {'drift':>7}  {'#rd':>3}  summary")
    for e in rows:
        scored = [r for r in e["readings"] if r["score"] is not None]
        first = scored[0]["score"] if scored else None
        latest = scored[-1]["score"] if scored else None
        no_score_status = (
            e["readings"][-1].get("status", "") if not scored and e["readings"] else ""
        )
        drift = f"{latest - first:+.1f}" if scored and len(scored) > 1 else ""
        summary = (
            e.get("approach", "").strip().splitlines()[0]
            if e.get("approach")
            else e.get("kaggle_description", "")
        )[:60]
        date = (e.get("kaggle_date") or "")[:10]
        first_s = "ERROR" if "ERROR" in no_score_status else _fmt_score(first)
        latest_s = "ERROR" if "ERROR" in no_score_status else _fmt_score(latest)
        print(
            f"{e['ref']:>10}  {date:10}  {first_s:>8}  "
            f"{latest_s:>8}  {drift:>7}  {len(e['readings']):>3}  {summary}"
        )
    print(
        "\nDetail: `show --ref <ref>`. Drift is first->latest public reading "
        "(live ratings move ~±100 same-build)."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("refresh", help="pull live scores from Kaggle, append readings")

    lp = sub.add_parser("log", help="record rich metadata for one submission ref")
    lp.add_argument("--ref", required=True, help="Kaggle submission ref (e.g. 55228113)")
    lp.add_argument("--deck")
    lp.add_argument("--git-sha", dest="git_sha")
    lp.add_argument("--checkpoint-sha256", dest="checkpoint_sha256")
    lp.add_argument("--config", help="env flags / hyperparams, no length limit")
    lp.add_argument("--local-result", dest="local_result", help="local evidence at submit time")
    lp.add_argument("--displaces", help="what this pushes out of the active set")
    lp.add_argument("--expects", help="predicted ladder outcome, falsifiable")
    lp.add_argument("--approach", help="full approach text; '-' reads stdin")
    lp.add_argument("--approach-file", help="file whose contents become the approach text")
    lp.add_argument("--link", action="append", help="related file/PR/report (repeatable)")

    sp = sub.add_parser("show", help="table of all submissions, or one ref in full")
    sp.add_argument("--ref", help="show full metadata + score timeline for this ref")

    args = ap.parse_args()
    {"refresh": cmd_refresh, "log": cmd_log, "show": cmd_show}[args.cmd](args)


if __name__ == "__main__":
    main()
