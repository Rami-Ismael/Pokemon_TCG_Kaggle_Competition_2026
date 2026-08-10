"""Download the Kaggle episodes-index dataset -- the per-day CSVs of episode links.

Dataset: kaggle/pokemon-tcg-ai-battle-episodes-index
    One CSV per competition day; each row is one episode with its replay link
    and the per-episode player-rating fields (avg_score / min_score / ...).
    This is the ratings source referenced by prompts/rl_pipeline_v4.md Sec.8 --
    the CLI's own `competitions episodes` walk cannot recover those historical
    ratings, only each submission's CURRENT publicScore.

The Kaggle API rate-limits this account hard and account-wide: once tripped,
EVERY endpoint returns 429 and recovery takes tens of minutes
([[kaggle-replay-endpoint-rate-limit]]). So the download is written to be
left running unattended:

  * one whole-dataset request instead of N per-file ones (fewer chances to 429)
  * escalating backoff, capped at an hour, up to --max-wait-hours total
  * resumable -- an already-populated output dir is left alone without --force

Auth: OAuth via `kaggle auth login` (~/.kaggle/access_token), NOT kaggle.json.

Usage:
    uv run python scripts/download_episode_index.py
    uv run python scripts/download_episode_index.py --list-only
    uv run python scripts/download_episode_index.py --max-wait-hours 12
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATASET = "kaggle/pokemon-tcg-ai-battle-episodes-index"
DEFAULT_OUT = REPO / "data" / "external" / "episode_index"

# Escalating waits between 429s, in seconds; the last value repeats until the
# --max-wait-hours budget is spent.
BACKOFF = (60, 120, 300, 600, 1200, 1800, 3600)

# Unbroken 429s for longer than this mean the daily quota, not the rate limiter.
# The per-second limiter has always cleared well inside an hour.
QUOTA_TELL_S = 3600


def _next_utc_midnight() -> datetime:
    now = datetime.now(timezone.utc)
    return (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)


def kaggle_cmd() -> list[str]:
    """The kaggle CLI, wherever it lives.

    Worktrees have no .venv of their own, so fall back to the main checkout's
    interpreter before asking uv to resolve an environment.
    """
    candidates = [
        REPO / ".venv" / "bin" / "kaggle",
        REPO.parents[2] / ".venv" / "bin" / "kaggle" if len(REPO.parents) > 2 else None,
    ]
    for c in candidates:
        if c is not None and c.exists():
            return [str(c)]
    found = shutil.which("kaggle")
    if found:
        return [found]
    return ["uv", "run", "kaggle"]


def run_kaggle(args: list[str], timeout: int = 1800) -> tuple[int, str]:
    proc = subprocess.run(
        kaggle_cmd() + args,
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def is_rate_limited(output: str) -> bool:
    return "429" in output or "Too Many Requests" in output


def with_backoff(label: str, args: list[str], max_wait_hours: float,
                 timeout: int = 1800) -> str:
    """Run a kaggle command, waiting out 429s until the patience budget is gone.

    `timeout` is per attempt. A 20 GiB episode day needs far more than the
    default: one took >30 min under disk pressure and was killed mid-unzip,
    leaving a complete extraction plus its redundant archive behind.
    """
    deadline = time.monotonic() + max_wait_hours * 3600
    started = time.monotonic()
    attempt = 0
    while True:
        code, out = run_kaggle(args, timeout=timeout)
        if code == 0 and not is_rate_limited(out):
            return out
        if not is_rate_limited(out):
            raise SystemExit(f"{label} failed (exit {code}):\n{out.strip()}")

        wait = BACKOFF[min(attempt, len(BACKOFF) - 1)]
        remaining = deadline - time.monotonic()
        elapsed = time.monotonic() - started

        # Two different 429s wear the same face. The per-second limiter clears
        # after tens of minutes of quiet, so backoff works. The DAILY QUOTA
        # clears only at UTC midnight, so backoff is pure waste -- 10 hours of
        # retries bought nothing on 2026-08-08. Past QUOTA_TELL_S of unbroken
        # 429s, call it quota and say when it actually resets.
        if elapsed > QUOTA_TELL_S:
            reset = _next_utc_midnight()
            hrs = (reset - datetime.now(timezone.utc)).total_seconds() / 3600
            raise SystemExit(
                f"{label}: 429 for {elapsed / 60:.0f} min straight -- this is the "
                f"DAILY QUOTA, not the rate limiter. It resets at "
                f"{reset:%Y-%m-%d %H:%M} UTC (~{hrs:.1f}h). Backoff cannot help; "
                f"re-run after the rollover -- the backfill is resumable."
            )
        if remaining <= 0:
            raise SystemExit(
                f"{label}: still rate-limited after {max_wait_hours:g}h of retries. "
                f"Kaggle's limiter is account-wide; try again later."
            )
        wait = int(min(wait, remaining))
        attempt += 1
        print(
            f"[{time.strftime('%H:%M:%S')}] {label}: 429 (attempt {attempt}), "
            f"sleeping {wait}s; {remaining / 3600:.1f}h of patience left",
            flush=True,
        )
        time.sleep(wait)


def list_files(max_wait_hours: float) -> str:
    return with_backoff(
        "list", ["datasets", "files", DATASET, "--page-size", "500"], max_wait_hours
    )


def download(out: Path, max_wait_hours: float) -> None:
    out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET} -> {out}", flush=True)
    with_backoff(
        "download",
        ["datasets", "download", "-d", DATASET, "-p", str(out), "--force"],
        max_wait_hours,
    )
    zips = sorted(out.glob("*.zip"))
    if not zips:
        # Newer CLI versions unzip on their own.
        print("No archive left behind; CLI appears to have unzipped in place.", flush=True)
        return
    for z in zips:
        print(f"Unzipping {z.name} ({z.stat().st_size / 1e6:.1f} MB)", flush=True)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(out)
        z.unlink()


def summarize(out: Path) -> None:
    csvs = sorted(p for p in out.rglob("*.csv"))
    if not csvs:
        print(f"No CSVs under {out} -- check the download output above.")
        return

    total_bytes = sum(p.stat().st_size for p in csvs)
    print()
    print(f"{len(csvs)} CSV file(s), {total_bytes / 1e6:.1f} MB total, in {out}")

    grand_rows = 0
    for p in csvs:
        with p.open(newline="") as fh:
            reader = csv.reader(fh)
            try:
                header = next(reader)
            except StopIteration:
                print(f"  {p.name}: EMPTY")
                continue
            rows = sum(1 for _ in reader)
        grand_rows += rows
        print(f"  {p.relative_to(out)}: {rows:,} rows, {len(header)} cols")

    print(f"\nTotal rows: {grand_rows:,}")
    with csvs[0].open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        print(f"\nColumns ({csvs[0].name}):")
        for c in header:
            print(f"  - {c}")
        for i, row in zip(range(2), reader):
            print(f"  sample[{i}]: {dict(zip(header, row))}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-wait-hours", type=float, default=6.0,
                    help="total patience budget for waiting out 429s (default 6)")
    ap.add_argument("--list-only", action="store_true",
                    help="print the dataset's file listing and exit")
    ap.add_argument("--force", action="store_true",
                    help="re-download even if the output dir already has CSVs")
    args = ap.parse_args()

    if args.list_only:
        print(list_files(args.max_wait_hours))
        return

    existing = sorted(args.out.rglob("*.csv")) if args.out.exists() else []
    if existing and not args.force:
        print(f"{len(existing)} CSV(s) already in {args.out}; pass --force to re-download.")
        summarize(args.out)
        return

    download(args.out, args.max_wait_hours)
    summarize(args.out)


if __name__ == "__main__":
    main()
