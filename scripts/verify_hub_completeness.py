"""Prove every episode of every day is on the Hub -- or name exactly what is missing.

The ingest pipeline's own verify answers "does the Hub match what we
downloaded?" (hashes, id sets, byte compares). This answers the different and
weaker-guarded question: **did we download the whole day?** A day can pass every
hash check and still be 1.7% of itself.

Two checks per day, strongest first:

  ids   -- the episode-id set on the Hub vs the id set in that day's own
           manifest.csv, archived by backfill_episode_days.py under
           data/episodes/day_manifests/. Catches missing AND unexpected ids.
  count -- Hub row count vs the index's episode_count. Always available, even
           for days ingested before manifest archiving existed.

Exits nonzero if any day is short, so it can gate a training run.

Usage:
    uv run python scripts/verify_hub_completeness.py
    uv run python scripts/verify_hub_completeness.py --day 2026-07-08
    uv run python scripts/verify_hub_completeness.py --ids   # slower, reads id columns
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from backfill_episode_days import (  # noqa: E402
    DAY_MANIFESTS,
    LEDGER,
    read_index,
    split_for,
)


def hub_shards(fs, repo_id: str, split: str, day: str) -> list[str]:
    """Shards for one day; a day the Hub has never seen is empty, not an error."""
    try:
        return sorted(fs.glob(f"datasets/{repo_id}/{split}/day={day}/shard-*.parquet"))
    except FileNotFoundError:
        return []


def day_manifest_ids(day: str) -> set[int] | None:
    p = DAY_MANIFESTS / f"{day}.csv"
    if not p.exists():
        return None
    ids: set[int] = set()
    with p.open(newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                ids.add(int(row["episode_id"]))
            except (KeyError, ValueError):
                continue
    return ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day", help="check only this date")
    ap.add_argument("--ids", action="store_true",
                    help="also compare episode-id SETS (reads the id column of every shard)")
    args = ap.parse_args()

    import ingest_episodes
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    repo_id = ingest_episodes.HF_EPISODES_REPO
    fs = HfFileSystem()
    rows = read_index()
    if args.day:
        rows = [r for r in rows if r["date"] == args.day]
        if not rows:
            print(f"{args.day} is not in the index", file=sys.stderr)
            return 2

    # What Kaggle actually served per day, recorded at download time.
    published: dict[str, int] = {}
    if LEDGER.exists():
        for day, rec in json.loads(LEDGER.read_text()).items():
            if isinstance(rec, dict) and rec.get("published") is not None:
                published[day] = rec["published"]

    print(f"repo: {repo_id}")
    print(f"{'day':<12} {'split':<6} {'on hub':>8} {'expected':>9} {'ids':<10} state")

    short, missing_total, hub_total, want_total = [], 0, 0, 0
    for r in rows:
        day, want = r["date"], int(r["episode_count"])
        split = split_for(day)
        shards = hub_shards(fs, repo_id, split, day)

        n, hub_ids = 0, set()
        for p in shards:
            with fs.open(p, "rb") as fh:
                pf = pq.ParquetFile(fh)
                n += pf.metadata.num_rows
                if args.ids:
                    hub_ids.update(pf.read(columns=["episode_id"])
                                     .column("episode_id").to_pylist())

        hub_total += n
        want_total += want

        id_note, id_bad = "-", False
        if args.ids:
            man = day_manifest_ids(day)
            if man is None:
                id_note = "no manifest"
            else:
                miss, extra = man - hub_ids, hub_ids - man
                id_bad = bool(miss or extra)
                id_note = "OK" if not id_bad else f"-{len(miss)}/+{len(extra)}"

        # A day is complete when the Hub holds everything KAGGLE PUBLISHED.
        # Kaggle's index sometimes claims more than it actually published (the
        # 20 GiB cap truncates the tail): 2026-07-24 lists 4,445 but serves
        # 4,444, with the last id returning 404. That gap is upstream and must
        # not be reported as ours.
        pub = published.get(day)
        target = pub if pub is not None else want
        if n == target and not id_bad:
            state = "complete" if target == want else f"complete (kaggle -{want - target})"
        elif n == 0:
            state = "ABSENT"
        else:
            state = "SHORT"
        if not state.startswith("complete"):
            short.append((day, split, n, target))
            missing_total += max(target - n, 0)

        print(f"{day:<12} {split:<6} {n:>8,} {want:>9,} {id_note:<10} {state}")

    print(f"\n{hub_total:,} of {want_total:,} episodes on the Hub "
          f"({100 * hub_total / want_total:.1f}%)")
    if short:
        print(f"{len(short)} day(s) incomplete, {missing_total:,} episodes missing:")
        for day, split, n, want in short:
            print(f"  {split}/day={day}: {n:,}/{want:,}")
        return 1
    print("COMPLETE: every indexed day has all its episodes on the Hub.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
