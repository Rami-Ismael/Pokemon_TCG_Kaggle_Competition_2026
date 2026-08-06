"""Rolling episode ingest for the behavior-cloning corpus -- ADR-001, action item 3.

Moves one split-day of Kaggle episodes through the pipeline that keeps the
laptop's disk small while the Hugging Face Hub copy stays authoritative:

    download (Kaggle episode-service, optional)
      -> pack   (zstd Parquet shards via scripts/pack_episodes.py)
      -> upload (private HF dataset repo, config.HF_EPISODES_REPO)
      -> verify (re-read every row FROM THE HUB, hash-check, compare vs local)

This script NEVER deletes raw episode folders. Concurrent sessions and
long-running processes read data/episodes/, so deletion is the user's call:
after the Hub verify passes, the script prints which folder became safe to
delete and how much disk that frees -- nothing more.

Subcommands
-----------
  status             local split-days vs Hub days: sizes, shard counts, pending work
  run                pack -> local verify -> upload -> hub verify for one local day
  verify-hub         stream a day's shards back from the Hub and prove faithfulness
  download           fetch one calendar day of episodes from Kaggle (public
                     episode-service endpoints, politely rate-limited, resumable)
  backfill-manifest  fetch per-episode scores only (no replays) for a local day

Typical new-day cycle once episodes for 2026-08-04 exist on Kaggle:

    uv run python scripts/ingest_episodes.py download --date 2026-08-04 --split train
    uv run python scripts/ingest_episodes.py run \
        --split-dir data/episodes/splits/train-2026-08-04

Auth: the HF token comes from `hf auth login` (huggingface_hub reads it
automatically). `download`/`backfill-manifest` shell out to the official
`kaggle` CLI (OAuth via `uv run kaggle auth login`), whose episode support
was verified against this archive: `kaggle competitions replay <episode_id>`
returns files BYTE-IDENTICAL to data/episodes/splits/. Listing walks
leaderboard -> team-submissions -> episodes. Request quota is unverified --
the defaults are deliberately polite; downloading a ~4,500-episode day takes
a few hours wall-clock, unattended and resumable.

Per-episode player-rating fields (avg_score/min_score/sum_score): the CLI
does not expose the historical per-episode ratings the original manifest
recorded, so `download` approximates them with each participating
submission's CURRENT publicScore, filled only when the scan saw BOTH
participants -- blank means unknown, never a guess.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from pokemon_tcg.config import (
    COMPETITION,
    EPISODES_DIR,
    EPISODES_PACKED_DIR,
    EPISODES_SPLITS_DIR,
    HF_EPISODES_REPO,
    PROJECT_ROOT,
)

PACK_SCRIPT = PROJECT_ROOT / "scripts" / "pack_episodes.py"
MANIFEST_PATH = EPISODES_DIR / "manifest.csv"
MANIFEST_FIELDS = [
    "episode_id", "create_time", "avg_score", "min_score",
    "sum_score", "agent_count", "size_bytes",
]

DAY_DIR_RE = re.compile(r"^(train|eval)-(\d{4}-\d{2}-\d{2})$")
HUB_SHARD_RE = re.compile(r"^(train|eval)/day=(\d{4}-\d{2}-\d{2})/shard-\d+\.parquet$")


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------

def _hf_api():
    from huggingface_hub import HfApi

    return HfApi()


def local_days() -> dict[tuple[str, str], Path]:
    """(split, day) -> raw folder, for every canonical split-day dir on disk."""
    out: dict[tuple[str, str], Path] = {}
    if not EPISODES_SPLITS_DIR.exists():
        return out
    for p in sorted(EPISODES_SPLITS_DIR.iterdir()):
        m = DAY_DIR_RE.match(p.name)
        if p.is_dir() and m:
            out[(m.group(1), m.group(2))] = p
    return out


def hub_days(api=None) -> dict[tuple[str, str], list]:
    """(split, day) -> list of shard tree entries (with .path and .size) on the Hub."""
    api = api or _hf_api()
    out: dict[tuple[str, str], list] = defaultdict(list)
    for entry in api.list_repo_tree(HF_EPISODES_REPO, repo_type="dataset", recursive=True):
        m = HUB_SHARD_RE.match(entry.path)
        if m:
            out[(m.group(1), m.group(2))].append(entry)
    return dict(out)


def _dir_stats(p: Path) -> tuple[int, int]:
    files = list(p.glob("*.json"))
    return len(files), sum(f.stat().st_size for f in files)


def _split_day_from_dir(split_dir: Path) -> tuple[str, str]:
    m = DAY_DIR_RE.match(split_dir.name)
    if not m:
        sys.exit(
            f"{split_dir.name!r} is not a canonical split-day folder "
            "(expected train-YYYY-MM-DD or eval-YYYY-MM-DD)"
        )
    return m.group(1), m.group(2)


# --------------------------------------------------------------------------
# status
# --------------------------------------------------------------------------

def cmd_status(args: argparse.Namespace) -> int:
    api = _hf_api()
    local = local_days()
    hub = hub_days(api)

    print(f"Hub repo: {HF_EPISODES_REPO} (private dataset)")
    print(f"{'day':<12} {'split':<6} {'local raw':>18} {'hub packed':>16}  state")
    all_keys = sorted(set(local) | set(hub), key=lambda k: (k[1], k[0]))
    pending = []
    for key in all_keys:
        split, day = key
        if key in local:
            n, b = _dir_stats(local[key])
            local_s = f"{n} eps / {b / 1e9:.1f} GB"
        else:
            local_s = "-"
        if key in hub:
            hb = sum(e.size for e in hub[key])
            hub_s = f"{len(hub[key])} shards / {hb / 1e6:.0f} MB"
            state = "uploaded" + ("" if key in local else " (local raw pruned)")
        else:
            hub_s = "-"
            state = "PENDING upload"
            pending.append(key)
        print(f"{day:<12} {split:<6} {local_s:>18} {hub_s:>16}  {state}")

    free = shutil.disk_usage(EPISODES_DIR).free
    print(f"\nlocal disk free: {free / 1e9:.1f} GB")
    if pending:
        print("next:")
        for split, day in pending:
            print(
                f"  uv run python scripts/ingest_episodes.py run "
                f"--split-dir {local[(split, day)]}"
            )
    else:
        print("everything local is on the Hub.")
    return 0


# --------------------------------------------------------------------------
# pack -> upload -> verify
# --------------------------------------------------------------------------

def _pack(split_dir: Path, out_day_dir: Path, shard_raw_gib: float, level: int) -> None:
    # Reuse only when the cached shards actually cover the raw folder: a
    # re-ingested day (e.g. a partial day topped up later) leaves stale
    # smaller shards here, and blind reuse made verify fail on id-set
    # mismatch after a 2,592-episode day was compared against a stale
    # 2-row shard (2026-08-05).
    shards = sorted(out_day_dir.glob("*.parquet")) if out_day_dir.exists() else []
    if shards:
        n_rows = sum(pq.ParquetFile(s).metadata.num_rows for s in shards)
        n_raw = len(list(split_dir.glob("*.json")))
        if n_rows == n_raw:
            print(f"[pack] reusing existing shards in {out_day_dir} ({n_rows} rows)")
            return
        print(
            f"[pack] stale shards in {out_day_dir} ({n_rows} rows vs "
            f"{n_raw} raw files); repacking"
        )
        for s in shards:
            s.unlink()
    subprocess.run(
        [
            sys.executable, str(PACK_SCRIPT), "pack",
            "--split-dir", str(split_dir),
            "--out-root", str(EPISODES_PACKED_DIR),
            "--shard-raw-gib", str(shard_raw_gib),
            "--level", str(level),
        ],
        check=True,
    )


def _verify_local(split_dir: Path, out_day_dir: Path, sample: int) -> None:
    subprocess.run(
        [
            sys.executable, str(PACK_SCRIPT), "verify",
            "--shard-dir", str(out_day_dir),
            "--split-dir", str(split_dir),
            "--sample", str(sample),
        ],
        check=True,
    )


def _upload(out_day_dir: Path, split: str, day: str) -> None:
    api = _hf_api()
    t0 = time.time()
    api.upload_folder(
        repo_id=HF_EPISODES_REPO,
        repo_type="dataset",
        folder_path=str(out_day_dir),
        path_in_repo=f"{split}/day={day}",
        allow_patterns=["*.parquet"],
        commit_message=f"Add {split} day={day} episode shards (ingest_episodes.py)",
    )
    total = sum(p.stat().st_size for p in out_day_dir.glob("*.parquet"))
    dt = time.time() - t0
    print(f"[upload] {total / 1e6:.1f} MB in {dt:.0f}s ({total * 8 / 1e6 / dt:.0f} Mbps)")


def verify_hub_day(split: str, day: str, local_dir: Path | None, sample: int) -> int:
    """Stream the day's shards back FROM THE HUB and prove faithfulness.

    Checks: every row's episode_json re-hashes to its stored sha256; if the
    local raw folder still exists, the id sets match exactly and a random
    sample of rows is byte-compared against the original files. Passing this
    is the gate after which deleting the local raw folder loses nothing.
    """
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    base = f"datasets/{HF_EPISODES_REPO}/{split}/day={day}"
    shard_paths = sorted(fs.glob(f"{base}/shard-*.parquet"))
    if not shard_paths:
        print(f"FAIL: no shards on the Hub under {base}")
        return 1

    rng = random.Random(0)
    ids: set[int] = set()
    n = 0
    sampled: list[tuple[int, bytes]] = []
    t0 = time.time()
    for sp in shard_paths:
        with fs.open(sp, "rb") as fh:
            pf = pq.ParquetFile(fh)
            for rb in pf.iter_batches(
                batch_size=32, columns=["episode_id", "sha256", "episode_json"]
            ):
                for row in rb.to_pylist():
                    data = row["episode_json"]
                    if hashlib.sha256(data).hexdigest() != row["sha256"]:
                        print(f"FAIL: Hub row hash mismatch, episode {row['episode_id']}")
                        return 1
                    ids.add(row["episode_id"])
                    n += 1
                    if len(sampled) < sample:
                        sampled.append((row["episode_id"], data))
                    elif rng.random() < sample / n:
                        sampled[rng.randrange(sample)] = (row["episode_id"], data)

    msg = f"[verify-hub] OK: {n} rows hash-verified across {len(shard_paths)} Hub shard(s)"
    if local_dir is not None and local_dir.exists():
        disk_ids = {int(p.stem) for p in local_dir.glob("*.json")}
        if ids != disk_ids:
            missing = sorted(disk_ids - ids)[:5]
            extra = sorted(ids - disk_ids)[:5]
            print(f"FAIL: id sets differ (missing {missing}... extra {extra}...)")
            return 1
        n_cmp = 0
        for eid, data in sampled:
            src = local_dir / f"{eid}.json"
            if src.read_bytes() != data:
                print(f"FAIL: byte mismatch vs local file, episode {eid}")
                return 1
            n_cmp += 1
        msg += f"; id set matches {len(disk_ids)} local files; {n_cmp} rows byte-identical"
    print(f"{msg} ({time.time() - t0:.0f}s)")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    split_dir = Path(args.split_dir).resolve()
    split, day = _split_day_from_dir(split_dir)
    out_day_dir = EPISODES_PACKED_DIR / split / f"day={day}"

    if (split, day) in hub_days() and not args.force:
        print(f"{split}/day={day} is already on the Hub -- use --force to re-upload.")
        return 0

    _pack(split_dir, out_day_dir, args.shard_raw_gib, args.level)
    _verify_local(split_dir, out_day_dir, args.sample)
    _upload(out_day_dir, split, day)
    rc = verify_hub_day(split, day, split_dir, args.sample)
    if rc != 0:
        return rc

    n, b = _dir_stats(split_dir)
    packed = sum(p.stat().st_size for p in out_day_dir.glob("*.parquet"))
    print(
        f"\ndone: {split}/day={day} -- {n} episodes, {b / 1e9:.2f} GB raw -> "
        f"{packed / 1e6:.1f} MB on the Hub ({b / max(packed, 1):.0f}x)."
    )
    print(
        "The Hub copy is verified faithful. Deleting the local raw folder would "
        f"free {b / 1e9:.2f} GB:\n  rm -rf {split_dir}\n"
        "NOT run automatically -- other sessions may be reading it; delete when "
        "you decide."
    )
    return 0


def cmd_verify_hub(args: argparse.Namespace) -> int:
    local_dir = EPISODES_SPLITS_DIR / f"{args.split}-{args.day}"
    return verify_hub_day(
        args.split, args.day, local_dir if local_dir.exists() else None, args.sample
    )


# --------------------------------------------------------------------------
# Kaggle CLI: download / backfill-manifest
# --------------------------------------------------------------------------

def _kaggle_json(*args: str):
    """Run a kaggle CLI subcommand with --format json and parse its stdout.

    Tolerates non-JSON preamble lines (e.g. the leaderboard's
    "Next Page Token = ..." banner) by parsing from the first bracket.
    """
    proc = subprocess.run(
        ["kaggle", *args, "--format", "json"], capture_output=True, text=True
    )
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout).strip()
        hint = ""
        if any(t in msg for t in ("401", "403", "Unauthorized", "Forbidden")):
            hint = "\nAuth fix: `uv run kaggle auth login` (OAuth; kaggle.json is not used)."
        sys.exit(f"kaggle {' '.join(args)} failed:\n{msg}{hint}")
    out = proc.stdout
    start = min((i for i in (out.find("["), out.find("{")) if i >= 0), default=-1)
    if start < 0:
        sys.exit(f"kaggle {' '.join(args)}: no JSON in output:\n{out[:500]}")
    # raw_decode parses the FIRST complete JSON value and ignores trailing
    # output (some subcommands print a next-page-token line after the JSON;
    # the CLI exposes no flag to follow it, so one page is what we get)
    value, _end = json.JSONDecoder().raw_decode(out[start:])
    return value


def _leaderboard_team_ids(top_n: int) -> list[int]:
    """Top team ids from the public leaderboard (one page, ~top 50).

    The CLI paginates with a next-page token that we deliberately don't
    chase: behavior cloning wants the strongest players, and they are on
    page one.
    """
    rows = _kaggle_json("competitions", "leaderboard", COMPETITION, "--show")
    ids = []
    for r in rows[:top_n]:
        try:
            ids.append(int(r["teamId"]))
        except (KeyError, TypeError, ValueError):
            continue
    if not ids:
        sys.exit(f"leaderboard listing had no teamId rows (shape: {str(rows)[:200]})")
    return ids


def _list_day_episodes(team_ids: list[int], date: str, sleep: float) -> dict[int, dict]:
    """episode_id -> manifest fields, via team-submissions -> episodes.

    Rating fields are approximated from each participating submission's
    CURRENT publicScore (see module docstring); avg/min/sum are filled only
    when the scan saw both participants of an episode.
    """
    ep_meta: dict[int, dict] = {}
    ep_scores: dict[int, list[float]] = defaultdict(list)
    for i, tid in enumerate(team_ids):
        for sub in _kaggle_json("competitions", "team-submissions", str(tid)):
            if sub.get("id") is None:
                continue
            try:
                pub = float(sub.get("publicScore"))
            except (TypeError, ValueError):
                pub = None
            for ep in _kaggle_json("competitions", "episodes", str(sub["id"])):
                create = str(ep.get("createTime", ""))
                if not create.startswith(date) or "COMPLETED" not in str(ep.get("state", "")):
                    continue
                eid = int(ep["id"])
                ep_meta[eid] = {"episode_id": eid, "create_time": create}
                if pub is not None:
                    ep_scores[eid].append(pub)
            time.sleep(sleep)
        if (i + 1) % 10 == 0:
            print(f"  scanned {i + 1}/{len(team_ids)} teams -> {len(ep_meta)} episodes on {date}")
    out: dict[int, dict] = {}
    for eid, meta in ep_meta.items():
        scores = ep_scores.get(eid, [])
        row = dict(meta)
        if len(scores) >= 2:
            row.update(
                avg_score=sum(scores) / len(scores),
                min_score=min(scores),
                sum_score=sum(scores),
            )
        else:
            row.update(avg_score="", min_score="", sum_score="")
        row.update(agent_count=2, size_bytes="")
        out[eid] = row
    return out


def _download_replay(eid: int, target: Path) -> Path:
    """`kaggle competitions replay` into target/, renamed to {eid}.json."""
    subprocess.run(
        ["kaggle", "competitions", "replay", str(eid), "-p", str(target), "-q"],
        check=True, capture_output=True, text=True,
    )
    got = target / f"episode-{eid}-replay.json"
    if not got.exists():
        candidates = list(target.glob(f"*{eid}*.json"))
        if not candidates:
            raise FileNotFoundError(f"replay CLI produced no file for episode {eid}")
        got = candidates[0]
    final = target / f"{eid}.json"
    got.rename(final)
    return final


def _merge_manifest(new_rows: dict[int, dict]) -> None:
    """Merge rows into manifest.csv by episode_id (atomic rewrite, keeps order)."""
    existing: dict[int, dict] = {}
    if MANIFEST_PATH.exists():
        with MANIFEST_PATH.open(newline="") as f:
            for row in csv.DictReader(f):
                existing[int(row["episode_id"])] = row
    for eid, row in new_rows.items():
        merged = existing.get(eid, {})
        # never blank out a field that already has a value (e.g. size_bytes)
        merged.update({k: v for k, v in row.items() if v != "" or k not in merged})
        existing[eid] = merged
    tmp = MANIFEST_PATH.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for eid in sorted(existing):
            w.writerow({k: existing[eid].get(k, "") for k in MANIFEST_FIELDS})
    os.replace(tmp, MANIFEST_PATH)
    print(f"[manifest] merged {len(new_rows)} rows -> {MANIFEST_PATH} ({len(existing)} total)")


def cmd_download(args: argparse.Namespace) -> int:
    team_ids = _leaderboard_team_ids(args.top_teams)
    print(f"listing episodes for {len(team_ids)} top teams on {args.date}...")
    day_eps = _list_day_episodes(team_ids, args.date, 0.2)
    print(f"{len(day_eps)} episodes found for {args.date}")
    if not day_eps:
        return 1

    target = EPISODES_SPLITS_DIR / f"{args.split}-{args.date}"
    target.mkdir(parents=True, exist_ok=True)
    for eid in day_eps:
        f = target / f"{eid}.json"
        if f.exists():
            day_eps[eid]["size_bytes"] = f.stat().st_size
    missing = [eid for eid in sorted(day_eps) if not (target / f"{eid}.json").exists()]
    todo = missing[: args.limit] if args.limit else missing
    est_h = len(todo) * (args.sleep + 1.5) / 3600
    print(f"{len(todo)} to download ({len(day_eps) - len(missing)} already on disk"
          + (f", {len(missing) - len(todo)} deferred by --limit" if len(todo) < len(missing) else "")
          + f"); ~{est_h:.1f} h at --sleep {args.sleep}s; resumable (re-run skips done files)")

    failures = 0
    for i, eid in enumerate(todo):
        try:
            f = _download_replay(eid, target)
            day_eps[eid]["size_bytes"] = f.stat().st_size
        except (subprocess.CalledProcessError, FileNotFoundError):
            failures += 1
            print(f"  episode {eid} failed; continuing")
            if failures > 20 and failures > 0.2 * (i + 1):
                sys.exit("too many failures -- aborting (auth? quota?); re-run to resume")
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(todo)} downloaded ({failures} failures)")
        time.sleep(args.sleep)

    # only episodes actually on disk enter the manifest -- consumers treat a
    # manifest row as "this episode exists in the archive"
    _merge_manifest(
        {eid: row for eid, row in day_eps.items() if (target / f"{eid}.json").exists()}
    )
    n_disk = len(list(target.glob("*.json")))
    print(
        f"\n{n_disk} episodes now in {target} ({failures} failures).\n"
        f"next: uv run python scripts/ingest_episodes.py run --split-dir {target}\n"
        "then, if this day should join training, add it to "
        "data/episodes/splits/splits.json by hand (held-out-DAY rule: a day is "
        "train or eval, never both)."
    )
    return 0


def _hub_day_episode_ids(split: str, date: str) -> set[int]:
    """Episode-id universe for one day, read from the Hub shards' episode_id
    column (a few KB per shard) — the raw folder may be long deleted."""
    from huggingface_hub import HfApi, hf_hub_download

    prefix = f"{split}/day={date}/"
    shards = [
        f for f in HfApi().list_repo_files(HF_EPISODES_REPO, repo_type="dataset")
        if f.startswith(prefix) and f.endswith(".parquet")
    ]
    if not shards:
        sys.exit(f"no Hub shards under {prefix} in {HF_EPISODES_REPO}")
    ids: set[int] = set()
    for rel in shards:
        path = hf_hub_download(HF_EPISODES_REPO, rel, repo_type="dataset")
        ids |= {int(x) for x in pq.read_table(path, columns=["episode_id"]).column(0).to_pylist()}
    return ids


def cmd_backfill_manifest(args: argparse.Namespace) -> int:
    local_dir = EPISODES_SPLITS_DIR / f"{args.split}-{args.date}"
    if args.from_hub:
        # The 08-03 cleanup deleted most raw day folders; the Hub copy is the
        # id universe of record (rl_pipeline_v2 §2.B0 — do NOT let the ~24
        # surviving local files masquerade as the day).
        universe = _hub_day_episode_ids(args.split, args.date)
        where = f"hub {args.split}/day={args.date}"
    else:
        if not local_dir.exists():
            sys.exit(f"{local_dir} does not exist (use --from-hub for a deleted day)")
        universe = {int(p.stem) for p in local_dir.glob("*.json") if p.exists()}
        where = local_dir.name

    team_ids = _leaderboard_team_ids(args.top_teams)
    print(f"listing episodes for {len(team_ids)} top teams on {args.date}...")
    day_eps = _list_day_episodes(team_ids, args.date, args.sleep)

    hits = {eid: row for eid, row in day_eps.items() if eid in universe}
    scored = sum(1 for row in hits.values() if row.get("avg_score") != "")
    for eid, row in hits.items():
        f = local_dir / f"{eid}.json"
        if f.exists():
            row["size_bytes"] = f.stat().st_size
    print(f"score rows found for {len(hits)}/{len(universe)} episodes in {where} "
          f"({scored} with both participants' ratings; "
          "current-publicScore approximation, see module docstring)")
    if hits:
        _merge_manifest(hits)
    return 0


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("status", help="local vs Hub state, pending work")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("run", help="pack -> verify -> upload -> hub-verify one local day")
    p.add_argument("--split-dir", required=True)
    p.add_argument("--shard-raw-gib", type=float, default=5.0,
                   help="raw GiB per shard (smaller -> more shards -> more "
                        "DataLoader-worker parallelism; default 5 ~ 4-5 shards/day)")
    p.add_argument("--level", type=int, default=9, help="zstd level")
    p.add_argument("--sample", type=int, default=50, help="rows to byte-compare")
    p.add_argument("--force", action="store_true", help="re-upload even if day is on the Hub")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("verify-hub", help="stream a day back from the Hub and verify")
    p.add_argument("--split", required=True, choices=["train", "eval"])
    p.add_argument("--day", required=True)
    p.add_argument("--sample", type=int, default=50)
    p.set_defaults(fn=cmd_verify_hub)

    p = sub.add_parser("download", help="fetch one day's episodes from Kaggle")
    p.add_argument("--date", required=True, help="YYYY-MM-DD (episode createTime prefix)")
    p.add_argument("--split", required=True, choices=["train", "eval"])
    p.add_argument("--top-teams", type=int, default=50,
                   help="gather episodes from the top-N leaderboard teams "
                        "(one leaderboard page ~ top 50)")
    p.add_argument("--sleep", type=float, default=1.0,
                   help="seconds between replay downloads (quota unverified -- stay polite)")
    p.add_argument("--limit", type=int, default=None, help="cap downloads (smoke tests)")
    p.set_defaults(fn=cmd_download)

    p = sub.add_parser("backfill-manifest", help="fetch scores (no replays) for a local day")
    p.add_argument("--date", required=True)
    p.add_argument("--split", required=True, choices=["train", "eval"])
    p.add_argument("--top-teams", type=int, default=50)
    p.add_argument("--sleep", type=float, default=0.2,
                   help="seconds between listing calls")
    p.add_argument("--from-hub", action="store_true",
                   help="take the episode-id universe from the day's Hub "
                        "shards instead of local files (for days whose raw "
                        "folders were deleted after Hub verification)")
    p.set_defaults(fn=cmd_backfill_manifest)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
