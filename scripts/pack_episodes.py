"""Pack raw episode JSONs into zstd Parquet shards (and back) — ADR-001.

The behavior-cloning corpus (~800GB raw target) cannot live on this laptop
(~16GB free). Per notes/adr001_hf_streaming_corpus.md, episodes are packed
into zstd-compressed Parquet shards (~50-100x smaller), pushed to a private
Hugging Face dataset repo, and streamed (or cache-read) at train time. This
script is the pack/unpack/verify tool that makes that loop safe:

* ``pack``   -- split-day folder of ``{episode_id}.json`` files -> per-day
  shards ``{out_root}/{split}/day={day}/shard-NNN.parquet``. One row per
  episode; ``episode_json`` holds the EXACT file bytes (``large_binary``, so
  packing is lossless and byte-reversible), plus a stored sha256 and, where
  ``manifest.csv`` covers the episode, its Kaggle scores (``avg_score`` etc.)
  so skill-filtering can run as a streaming ``.filter`` without parsing JSON.
* ``unpack`` -- reconstruct the original ``{episode_id}.json`` files from
  shards, verifying each row's sha256 on the way out.
* ``verify`` -- prove a packed day is a faithful substitute for the raw
  folder BEFORE any raw file is deleted: row count == file count, id sets
  match, every row's bytes re-hash to the stored sha256, and a random sample
  is byte-compared against the original files.

Shard/row-group sizing: rows are ~2.5-6.5MB JSON blobs, so row groups are
kept small (default 32 rows ~ 130MB raw ~ 2-3MB compressed) to bound memory
when streaming, and shards roll over at ~10GiB of raw input (~200MB
compressed) so a 20GB day yields ~2 files and DataLoader workers / Hub
uploads get useful parallelism.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = PROJECT_ROOT / "data" / "episodes_packed"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "episodes" / "manifest.csv"

SCHEMA = pa.schema(
    [
        ("episode_id", pa.int64()),
        ("day", pa.string()),
        ("split_folder", pa.string()),
        ("n_bytes", pa.int64()),
        ("sha256", pa.string()),
        ("create_time", pa.string()),
        ("avg_score", pa.float64()),
        ("min_score", pa.float64()),
        ("agent_count", pa.int32()),
        ("episode_json", pa.large_binary()),
    ]
)


def _load_manifest(path: Path) -> dict[int, dict]:
    """episode_id -> manifest row. Missing file or episode is fine (nulls)."""
    if not path.exists():
        return {}
    out: dict[int, dict] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                out[int(row["episode_id"])] = row
            except (KeyError, ValueError):
                continue
    return out


def _day_from_folder(split_dir: Path) -> str:
    """'train-2026-07-01' -> '2026-07-01'; fall back to the raw name."""
    name = split_dir.name
    parts = name.split("-", 1)
    return parts[1] if len(parts) == 2 else name


def _split_from_folder(split_dir: Path) -> str:
    return split_dir.name.split("-", 1)[0]  # 'train' / 'eval'


def cmd_pack(args: argparse.Namespace) -> int:
    split_dir = Path(args.split_dir).resolve()
    files = sorted(split_dir.glob("*.json"), key=lambda p: p.stem)
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"no *.json files in {split_dir}", file=sys.stderr)
        return 1

    day = args.day or _day_from_folder(split_dir)
    split_name = args.split_name or _split_from_folder(split_dir)
    out_dir = Path(args.out_root) / split_name / f"day={day}"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(Path(args.manifest))
    shard_raw_bytes = int(args.shard_raw_gib * (1 << 30))

    t0 = time.time()
    raw_total = 0
    n_rows = 0
    shard_idx = 0
    writer: pq.ParquetWriter | None = None
    shard_raw = 0
    batch: list[dict] = []
    shard_paths: list[Path] = []

    def open_writer() -> pq.ParquetWriter:
        path = out_dir / f"shard-{shard_idx:03d}.parquet"
        shard_paths.append(path)
        return pq.ParquetWriter(
            path, SCHEMA, compression="zstd", compression_level=args.level
        )

    def flush_batch() -> None:
        nonlocal batch, writer
        if not batch:
            return
        if writer is None:
            writer = open_writer()
        # one write_table call per <=row_group_size rows == one row group
        writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
        batch = []

    for path in files:
        data = path.read_bytes()
        m = manifest.get(int(path.stem), {})
        batch.append(
            {
                "episode_id": int(path.stem),
                "day": day,
                "split_folder": split_dir.name,
                "n_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "create_time": m.get("create_time"),
                "avg_score": float(m["avg_score"]) if m.get("avg_score") else None,
                "min_score": float(m["min_score"]) if m.get("min_score") else None,
                "agent_count": int(m["agent_count"]) if m.get("agent_count") else None,
                "episode_json": data,
            }
        )
        raw_total += len(data)
        shard_raw += len(data)
        n_rows += 1
        if len(batch) >= args.row_group_size:
            flush_batch()
        if shard_raw >= shard_raw_bytes:
            flush_batch()
            assert writer is not None
            writer.close()
            writer = None
            shard_idx += 1
            shard_raw = 0

    flush_batch()
    if writer is not None:
        writer.close()

    dt = time.time() - t0
    packed_total = sum(p.stat().st_size for p in shard_paths)
    print(
        f"packed {n_rows} episodes ({raw_total / 1e9:.2f} GB raw) -> "
        f"{len(shard_paths)} shard(s), {packed_total / 1e6:.1f} MB "
        f"({raw_total / max(packed_total, 1):.1f}x) in {dt:.1f}s "
        f"({raw_total / 1e6 / dt:.0f} MB/s raw)"
    )
    for p in shard_paths:
        print(f"  {p.relative_to(Path(args.out_root))}  {p.stat().st_size / 1e6:.1f} MB")
    return 0


def _iter_rows(shard: Path, columns: list[str] | None = None):
    pf = pq.ParquetFile(shard)
    for rb in pf.iter_batches(columns=columns):
        yield from rb.to_pylist()


def cmd_unpack(args: argparse.Namespace) -> int:
    shards = sorted(Path(args.shard_dir).glob("*.parquet"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for shard in shards:
        for row in _iter_rows(shard):
            data = row["episode_json"]
            if hashlib.sha256(data).hexdigest() != row["sha256"]:
                print(f"sha256 MISMATCH for episode {row['episode_id']}", file=sys.stderr)
                return 1
            (out_dir / f"{row['episode_id']}.json").write_bytes(data)
            n += 1
            if args.limit and n >= args.limit:
                print(f"unpacked {n} episodes -> {out_dir}")
                return 0
    print(f"unpacked {n} episodes -> {out_dir}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Every check that must pass before the raw split folder may be deleted."""
    shards = sorted(Path(args.shard_dir).glob("*.parquet"))
    split_dir = Path(args.split_dir).resolve()
    disk_ids = {int(p.stem) for p in split_dir.glob("*.json")}

    rng = random.Random(args.seed)
    packed_ids: set[int] = set()
    n_hash_ok = 0
    sampled: list[tuple[int, bytes]] = []
    for shard in shards:
        for row in _iter_rows(shard):
            data = row["episode_json"]
            if hashlib.sha256(data).hexdigest() != row["sha256"]:
                print(f"FAIL: stored sha256 mismatch, episode {row['episode_id']}")
                return 1
            n_hash_ok += 1
            packed_ids.add(row["episode_id"])
            # reservoir-sample rows for byte-compare against the source files
            if len(sampled) < args.sample:
                sampled.append((row["episode_id"], data))
            elif rng.random() < args.sample / n_hash_ok:
                sampled[rng.randrange(args.sample)] = (row["episode_id"], data)

    if args.limit is None and packed_ids != disk_ids:
        missing = sorted(disk_ids - packed_ids)[:5]
        extra = sorted(packed_ids - disk_ids)[:5]
        print(f"FAIL: id sets differ (missing {missing}... extra {extra}...)")
        return 1

    n_cmp = 0
    for eid, data in sampled:
        src = split_dir / f"{eid}.json"
        if src.exists() and src.read_bytes() != data:
            print(f"FAIL: byte mismatch vs source file, episode {eid}")
            return 1
        n_cmp += src.exists()

    print(
        f"OK: {n_hash_ok} rows hash-verified across {len(shards)} shard(s); "
        f"id set matches {len(disk_ids)} files"
        + ("" if args.limit is None else " (skipped: --limit pack)")
        + f"; {n_cmp} rows byte-identical to source files"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pack", help="split-day folder -> zstd parquet shards")
    p.add_argument("--split-dir", required=True)
    p.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--day", default=None)
    p.add_argument("--split-name", default=None, choices=[None, "train", "eval"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--level", type=int, default=9, help="zstd level (default 9)")
    p.add_argument("--row-group-size", type=int, default=32)
    p.add_argument("--shard-raw-gib", type=float, default=10.0)
    p.set_defaults(fn=cmd_pack)

    p = sub.add_parser("unpack", help="shards -> original {episode_id}.json files")
    p.add_argument("--shard-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(fn=cmd_unpack)

    p = sub.add_parser("verify", help="prove shards faithfully replace a raw folder")
    p.add_argument("--shard-dir", required=True)
    p.add_argument("--split-dir", required=True)
    p.add_argument("--sample", type=int, default=25, help="rows to byte-compare")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="set if pack used --limit")
    p.set_defaults(fn=cmd_verify)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
