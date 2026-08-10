"""Backfill every Kaggle episode day into the Hugging Face corpus, one day at a time.

The index dataset (scripts/download_episode_index.py) lists 51 daily datasets,
each a flat folder of `<episode_id>.json` replays plus a per-day `manifest.csv`
carrying the HISTORICAL per-episode ratings. That manifest is the thing the
`kaggle competitions episodes` walk could never recover -- it only ever saw each
submission's CURRENT publicScore ([[episode-service-retention-window]]).

Two facts govern the design:

  * Each day is capped at 20 GiB and the laptop has ~45 GB free. Days therefore
    run STRICTLY SERIALLY, and the raw JSON for day N must be gone before day
    N+1 starts. Deleting raw is gated behind --delete-raw and only ever runs
    after the Hub verify passes.
  * Kaggle's limiter is account-wide and slow to release, so every API call goes
    through the same escalating backoff the index downloader uses.

Per day: download -> merge manifest -> pack -> local verify -> upload -> hub
verify -> (optional) delete raw. Each step is skipped when already done, so an
interrupted run resumes by re-running the same command.

NOTE on what this corpus is: the data card says replays are "selected daily,
ranked by average agent rating, and capped at 20 GiB per day". These are the
TOP-RATED episodes of each day, not a random sample -- any distributional claim
drawn from them inherits that selection.

Usage:
    uv run python scripts/backfill_episode_days.py --plan
    uv run python scripts/backfill_episode_days.py --run --days 1
    uv run python scripts/backfill_episode_days.py --run --delete-raw
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from download_episode_index import (  # noqa: E402
    DEFAULT_OUT as INDEX_DIR,
    kaggle_cmd,
    with_backoff,
)

SPLITS_DIR = REPO / "data" / "episodes" / "splits"
GLOBAL_MANIFEST = REPO / "data" / "episodes" / "manifest.csv"
DAY_MANIFESTS = REPO / "data" / "episodes" / "day_manifests"
LEDGER = REPO / "data" / "episodes" / "backfill_ledger.json"

# 2026-07-27 is the established held-out day; everything else trains.
# Reassigning it would leak eval into training, so it is pinned here.
EVAL_DAYS = {"2026-07-27"}

# A day is 20 GiB raw plus its archive; refuse to start one without headroom.
MIN_FREE_GB = 30.0


def free_gb(path: Path = REPO) -> float:
    return shutil.disk_usage(path).free / 1e9


def wait_for_disk(floor: float, max_hours: float, poll_s: int = 300) -> bool:
    """Block until free disk clears `floor`, or give up after max_hours.

    Disk here is shared with every other worktree and training run on this
    laptop, so dips below the floor are usually transient -- a neighbouring job
    finishing returns tens of GB. Exiting on the first dip turns a pause into a
    dead run that needs a human to restart it; waiting rides it out.
    """
    if free_gb() >= floor:
        return True
    t0 = time.time()
    while time.time() - t0 < max_hours * 3600:
        print(f"[disk] {free_gb():.1f} GB free, need {floor:.0f}; waiting "
              f"({(time.time() - t0) / 60:.0f} min so far)", flush=True)
        time.sleep(poll_s)
        if free_gb() >= floor:
            print(f"[disk] {free_gb():.1f} GB free, resuming", flush=True)
            return True
    return False


def read_index() -> list[dict]:
    csvs = sorted(INDEX_DIR.rglob("*.csv")) if INDEX_DIR.exists() else []
    if not csvs:
        raise SystemExit(
            f"No index CSV under {INDEX_DIR}. "
            "Run scripts/download_episode_index.py first."
        )
    rows: list[dict] = []
    for p in csvs:
        with p.open(newline="") as fh:
            rows.extend(csv.DictReader(fh))
    rows.sort(key=lambda r: r["date"])
    return rows


def load_ledger() -> dict:
    if LEDGER.exists():
        return json.loads(LEDGER.read_text())
    return {}


def save_ledger(led: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(led, indent=2, sort_keys=True) + "\n")


def hub_day_rows() -> dict[tuple[str, str], int]:
    """(split, day) -> episodes actually stored on the Hub.

    Presence is NOT completeness. Most existing Hub days are top-20 leaderboard
    slices holding 0.6-5% of their day ([[episode-service-retention-window]]) --
    2026-07-21 has 28 of 4,612 episodes. Skipping on presence alone would leave
    those days permanently truncated, so this reads each shard's Parquet footer
    for the real row count and callers compare it against the index.
    """
    import ingest_episodes
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    try:
        hub = ingest_episodes.hub_days()
    except Exception as exc:  # network/auth -- surface, do not silently backfill twice
        raise SystemExit(f"Could not list the Hub repo: {exc}")

    fs = HfFileSystem()
    out: dict[tuple[str, str], int] = {}
    for key, entries in hub.items():
        rows = 0
        for e in entries:
            with fs.open(f"datasets/{ingest_episodes.HF_EPISODES_REPO}/{e.path}", "rb") as fh:
                rows += pq.ParquetFile(fh).metadata.num_rows  # footer only
        out[key] = rows
    return out


def classify(rows: list[dict], hub: dict[tuple[str, str], int]) -> dict[str, list]:
    """Split the index into complete / partial / absent days."""
    buckets: dict[str, list] = {"complete": [], "partial": [], "absent": []}
    for r in rows:
        day = r["date"]
        key = (split_for(day), day)
        have = hub.get(key, 0)
        want = int(r["episode_count"])
        state = "absent" if have == 0 else ("complete" if have >= want else "partial")
        buckets[state].append((r, have, want))
    return buckets


def clear_stale_day(split: str, day: str) -> None:
    """Drop a truncated day's shards, on the Hub and locally, before re-packing.

    Re-uploading a fuller day would otherwise leave orphaned shards behind
    whenever the new pack produces fewer files than the old partial one.
    """
    import ingest_episodes
    from huggingface_hub import HfApi

    api = HfApi()
    path = f"{split}/day={day}"
    try:
        api.delete_folder(
            path_in_repo=path,
            repo_id=ingest_episodes.HF_EPISODES_REPO,
            repo_type="dataset",
            commit_message=f"Replace truncated {path} with the full Kaggle day",
        )
        print(f"[clear] removed stale Hub folder {path}")
    except Exception as exc:
        print(f"[clear] Hub folder {path} not removed ({exc})")

    local = REPO / "data" / "episodes_packed" / split / f"day={day}"
    if local.exists():
        shutil.rmtree(local)
        print(f"[clear] removed stale local shards {local}")


def split_for(day: str) -> str:
    return "eval" if day in EVAL_DAYS else "train"


def download_day(slug: str, raw_dir: Path, expect: int, max_wait_hours: float) -> None:
    """Fetch one daily dataset, unzipped, into raw_dir.

    A leftover folder is reused ONLY when it already holds the day's full
    episode count. Anything short is a partial download or an old top-20 slice
    (train-2026-07-08 sat on disk with 87 of 5,198 episodes) and gets wiped --
    reusing it would silently upload a fraction of the day and mark it done.
    """
    have = len(list(raw_dir.glob("*.json"))) if raw_dir.exists() else 0
    # Reuse must accept the same publish gap the gate does. Demanding an exact
    # index match threw away 2026-07-24's complete 4,444-file download (Kaggle
    # publishes 4,444 of the 4,445 it claims) and re-requested 21.5 GB, which
    # is what walked into the daily quota.
    if have and publish_gap_ok(have, expect):
        print(f"[download] reusing complete {raw_dir.name} ({have:,} json"
              + ("" if have == expect else f", index claims {expect:,}") + ")")
        return have
    if have:
        print(f"[download] discarding partial {raw_dir.name}: "
              f"{have:,} json but the day has {expect:,}")
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with_backoff(
        f"download {slug}",
        ["datasets", "download", "-d", f"kaggle/{slug}", "-p", str(raw_dir), "--unzip"],
        max_wait_hours,
        timeout=7200,  # 20 GiB download + unzip; 1800s killed one mid-cleanup
    )
    # A kill during --unzip leaves the archive next to the extracted files,
    # duplicating hundreds of MB on a disk that is already the binding limit.
    for z in raw_dir.glob("*.zip"):
        z.unlink()
    n = len(list(raw_dir.glob("*.json")))
    size = sum(p.stat().st_size for p in raw_dir.glob("*.json"))
    dt = time.time() - t0
    print(
        f"[download] {n:,} episodes / {size / 1e9:.1f} GB in {dt / 60:.1f} min "
        f"({size * 8 / 1e6 / max(dt, 1):.0f} Mbps)"
    )
    return n


# Kaggle's own numbers disagree slightly: the 2026-07-24 day manifest lists
# 4,445 episodes but only 4,444 files were published -- episode 87841523 returns
# 404 from the dataset. The 20 GiB cap truncates the tail while the manifest
# records the intended selection. Tolerate a handful; a real broken download is
# short by hundreds, not by one.
def publish_gap_ok(published: int, expect: int) -> bool:
    return 0 <= expect - published <= max(5, int(0.005 * expect))


def archive_day_manifest(raw_dir: Path, day: str) -> None:
    """Keep the day's own manifest so completeness stays auditable after deletion.

    It is the only record of WHICH episode ids belong to the day; the global
    manifest merges every day together and cannot answer that.
    """
    src = raw_dir / "manifest.csv"
    if not src.exists():
        return
    dst = DAY_MANIFESTS / f"{day}.csv"
    DAY_MANIFESTS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def hub_rows_for(split: str, day: str) -> int:
    """Episodes stored on the Hub for one day, read from the Parquet footers.

    The listing MUST be fresh: classify() lists the whole repo at startup, and
    fsspec caches that. The day we just uploaded is by definition absent from
    that snapshot, so a cached glob reports it missing and the gate fails a day
    that is actually fine. Skip the instance cache and drop the dircache.
    """
    import ingest_episodes
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(skip_instance_cache=True)
    fs.invalidate_cache()
    base = f"datasets/{ingest_episodes.HF_EPISODES_REPO}/{split}/day={day}"
    try:
        shards = fs.glob(f"{base}/shard-*.parquet")
    except FileNotFoundError:
        return 0
    rows = 0
    for p in shards:
        with fs.open(p, "rb") as fh:
            rows += pq.ParquetFile(fh).metadata.num_rows
    return rows


def merge_day_manifest(raw_dir: Path) -> int:
    """Fold the day's per-episode ratings into data/episodes/manifest.csv.

    Keyed by episode_id; existing rows win only when the incoming row has no
    score, so a real historical rating always beats a publicScore stand-in.
    """
    day_manifest = raw_dir / "manifest.csv"
    if not day_manifest.exists():
        print(f"[manifest] no manifest.csv in {raw_dir.name}; ratings stay null")
        return 0

    fieldnames = ["episode_id", "create_time", "avg_score", "min_score",
                  "sum_score", "agent_count", "size_bytes"]
    existing: dict[str, dict] = {}
    if GLOBAL_MANIFEST.exists():
        with GLOBAL_MANIFEST.open(newline="") as fh:
            for row in csv.DictReader(fh):
                existing[row["episode_id"]] = row

    added = 0
    with day_manifest.open(newline="") as fh:
        for row in csv.DictReader(fh):
            eid = row.get("episode_id")
            if not eid:
                continue
            merged = {k: row.get(k, "") for k in fieldnames}
            prev = existing.get(eid)
            if prev and not merged.get("avg_score"):
                continue
            if not prev:
                added += 1
            existing[eid] = merged

    tmp = GLOBAL_MANIFEST.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for eid in sorted(existing, key=int):
            w.writerow(existing[eid])
    tmp.replace(GLOBAL_MANIFEST)
    print(f"[manifest] +{added:,} new rows, {len(existing):,} total")
    return added


INGEST_BACKOFF = (60, 300, 900, 1800)


def ingest_day(raw_dir: Path) -> None:
    """pack -> local verify -> upload -> hub verify, via the existing pipeline.

    Retried: over a multi-hour run the Hub upload WILL hit transient network
    faults (a xet CAS ConnectionError killed 2026-06-28 mid-upload). Every
    stage is idempotent -- pack reuses existing shards, upload overwrites by
    path, verify re-reads from the Hub -- so a retry resumes rather than
    duplicating work.
    """
    for attempt, wait in enumerate((*INGEST_BACKOFF, None), 1):
        env = dict(os.environ)
        if attempt > 1:
            # The xet chunked-upload path is the flaky part: five failures on
            # five DIFFERENT xorb hashes while huggingface.co and the CAS host
            # both answered fine. Retries fall back to classic LFS, which is
            # slower but does not chunk.
            env["HF_HUB_DISABLE_XET"] = "1"
        proc = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "ingest_episodes.py"),
             "run", "--split-dir", str(raw_dir)],
            cwd=REPO,
            env=env,
        )
        if proc.returncode == 0:
            return
        if wait is None:
            raise SystemExit(
                f"FAIL {raw_dir.name}: ingest failed {attempt} times. "
                f"Raw KEPT at {raw_dir}."
            )
        print(f"[ingest] attempt {attempt} failed (exit {proc.returncode}); "
              f"retrying in {wait}s", flush=True)
        time.sleep(wait)


def process_day(row: dict, max_wait_hours: float, delete_raw: bool,
                stale: bool = False, prune_packed: bool = False) -> None:
    day = row["date"]
    split = split_for(day)
    raw_dir = SPLITS_DIR / f"{split}-{day}"

    print(f"\n{'=' * 70}\n{day} ({split})  "
          f"{int(row['episode_count']):,} episodes / "
          f"{int(row['total_bytes']) / 1e9:.1f} GB  |  {free_gb():.1f} GB free\n{'=' * 70}",
          flush=True)

    expect = int(row["episode_count"])
    if stale:
        clear_stale_day(split, day)
    published = download_day(row["daily_dataset_slug"], raw_dir, expect, max_wait_hours)
    if not publish_gap_ok(published, expect):
        raise SystemExit(
            f"FAIL {raw_dir.name}: downloaded {published:,} episodes but the index "
            f"says {expect:,} -- too large a gap to be Kaggle's publish cap. "
            f"Raw kept for inspection; nothing packed or uploaded."
        )
    if published != expect:
        print(f"[gap] Kaggle published {published:,} of the {expect:,} episodes its "
              f"index claims for {day}; the rest are not in the dataset (404).")

    merge_day_manifest(raw_dir)
    archive_day_manifest(raw_dir, day)
    ingest_day(raw_dir)

    # Two different claims, kept separate:
    #   [complete] -- the Hub holds every episode KAGGLE PUBLISHED. This is the
    #                 one we control, and it is a hard gate on deleting raw.
    #   [gap]      -- Kaggle published fewer than its own index claims. Upstream,
    #                 not actionable here, and must never be silently folded into
    #                 the first number.
    on_hub = hub_rows_for(split, day)
    if on_hub != published:
        raise SystemExit(
            f"FAIL {split}/day={day}: Hub has {on_hub:,} episodes but {published:,} "
            f"were downloaded. Raw KEPT at {raw_dir}."
        )
    print(f"[complete] {split}/day={day}: {on_hub:,}/{published:,} published "
          f"episodes on the Hub" + ("" if published == expect else
                                    f" (index claimed {expect:,})"))

    led = load_ledger()
    led[day] = {"split": split, "episodes": on_hub, "published": published,
                "index_claimed": expect, "slug": row["daily_dataset_slug"],
                "raw_deleted": False}
    save_ledger(led)

    if delete_raw:
        freed = sum(p.stat().st_size for p in raw_dir.glob("*.json"))
        shutil.rmtree(raw_dir)
        led[day]["raw_deleted"] = True
        save_ledger(led)
        print(f"[cleanup] removed {raw_dir.name}, freed {freed / 1e9:.1f} GB "
              f"({free_gb():.1f} GB now free)")
    else:
        print(f"[cleanup] KEEPING {raw_dir.name} (pass --delete-raw to reclaim "
              f"{sum(p.stat().st_size for p in raw_dir.glob('*.json')) / 1e9:.1f} GB)")

    if prune_packed:
        packed = REPO / "data" / "episodes_packed" / split / f"day={day}"
        if packed.exists():
            freed = sum(p.stat().st_size for p in packed.glob("*.parquet"))
            shutil.rmtree(packed)
            print(f"[cleanup] pruned local shard cache {packed.name}, "
                  f"freed {freed / 1e6:.0f} MB")


def cmd_plan(rows: list[dict]) -> None:
    buckets = classify(rows, hub_day_rows())
    todo = buckets["absent"] + buckets["partial"]
    todo.sort(key=lambda t: t[0]["date"])

    print(f"index: {len(rows)} days, {rows[0]['date']} -> {rows[-1]['date']}")
    print(f"  complete on the Hub : {len(buckets['complete'])}")
    print(f"  partial (truncated) : {len(buckets['partial'])}")
    print(f"  absent              : {len(buckets['absent'])}")

    have = sum(h for _, h, _ in buckets["complete"] + buckets["partial"])
    want = sum(int(r["episode_count"]) for r in rows)
    print(f"\nepisodes: {have:,} on the Hub of {want:,} published "
          f"({100 * have / want:.1f}%)")

    eps = sum(w - h for _, h, w in todo)
    byt = sum(int(r["total_bytes"]) for r, _, _ in todo)
    print(f"to backfill: {len(todo)} days, +{eps:,} episodes, "
          f"{byt / 1e12:.2f} TB to download")
    print(f"free disk now: {free_gb():.1f} GB (need >= {MIN_FREE_GB:.0f} GB per day)")
    print()
    print(f"  {'day':<12} {'split':<6} {'state':<8} {'have':>7} {'want':>7} {'GB':>6}")
    for r, h, w in todo:
        state = "partial" if h else "absent"
        print(f"  {r['date']:<12} {split_for(r['date']):<6} {state:<8} "
              f"{h:>7,} {w:>7,} {int(r['total_bytes']) / 1e9:>6.1f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="show what would run, then exit")
    ap.add_argument("--run", action="store_true", help="actually backfill")
    ap.add_argument("--days", type=int, default=0,
                    help="process at most N days this invocation (0 = all)")
    ap.add_argument("--only", help="process exactly this date (YYYY-MM-DD)")
    ap.add_argument("--delete-raw", action="store_true",
                    help="delete a day's raw JSON once its Hub verify passes")
    ap.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB)
    ap.add_argument("--disk-wait-hours", type=float, default=6.0,
                    help="how long to wait for other jobs to release disk (default 6)")
    ap.add_argument("--prune-packed", action="store_true",
                    help="drop the local Parquet cache once a day is Hub-verified; "
                         "the Hub copy is authoritative and re-packing is cheap")
    ap.add_argument("--max-wait-hours", type=float, default=6.0)
    args = ap.parse_args()

    rows = read_index()
    if args.only:
        rows = [r for r in rows if r["date"] == args.only]
        if not rows:
            raise SystemExit(f"{args.only} is not in the index")

    if args.plan or not args.run:
        cmd_plan(rows)
        return

    buckets = classify(rows, hub_day_rows())
    todo = sorted(buckets["absent"] + buckets["partial"], key=lambda t: t[0]["date"])
    if args.days:
        todo = todo[: args.days]
    if not todo:
        print("Nothing to backfill; every indexed day is complete on the Hub.")
        return

    for i, (row, have, want) in enumerate(todo, 1):
        # The headroom gate is about the DOWNLOAD. A day whose raw is already
        # complete on disk (e.g. the run was interrupted after downloading)
        # only needs room for its ~300 MB of shards, and blocking it on the
        # full 28 GB deadlocks: the space it wants is the space it is holding.
        raw_dir = SPLITS_DIR / f"{split_for(row['date'])}-{row['date']}"
        on_disk = len(list(raw_dir.glob("*.json"))) if raw_dir.exists() else 0
        # Same tolerance as reuse and the gate: a day already on disk needs room
        # for its ~300 MB of shards, not for a download it will not make.
        floor = (5.0 if on_disk and publish_gap_ok(on_disk, int(row["episode_count"]))
                 else args.min_free_gb)
        if not wait_for_disk(floor, args.disk_wait_hours):
            print(f"\nSTOP: {free_gb():.1f} GB free, below {floor:.0f} GB after "
                  f"{args.disk_wait_hours:g}h of waiting. {len(todo) - i + 1} day(s) "
                  f"left. Free space and re-run to resume.")
            return
        print(f"\n### day {i}/{len(todo)}  (Hub has {have:,} of {want:,})", flush=True)
        process_day(row, args.max_wait_hours, args.delete_raw, stale=have > 0,
                    prune_packed=args.prune_packed)

    print(f"\nDone: {len(todo)} day(s) backfilled.")


if __name__ == "__main__":
    main()
