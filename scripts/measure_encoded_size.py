"""Phase 2.5 Step 1: encode the train day to parquet and measure bytes/day.

"We never train on JSON, we train on encoded rows" -- this measures the
real number the Phase 2.5 decision hinges on: encoded bytes/day vs the
20 GiB/day raw JSON figure, extrapolated to the full 41+ day corpus.

Writes a scratch parquet file (deleted by default -- pass --keep to retain
it) built from the exact same encode_observation() output the training loop
consumes, not a hand-picked subset of fields.

Usage:
    uv run python scripts/measure_encoded_size.py --split train
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pokemon_tcg import config  # noqa: E402
from pokemon_tcg.il_dataset import iter_decisions, encode_observation, resolve_split_dir  # noqa: E402


def rows_to_table(rows: list[dict]) -> pa.Table:
    columns = {}
    for key in rows[0]:
        vals = [r[key] for r in rows]
        arr = np.stack([v.numpy() for v in vals]) if vals[0].ndim > 0 else np.array([v.item() for v in vals])
        if arr.ndim > 1:
            # flatten fixed-size tensor columns to a list-of-list arrow column
            columns[key] = pa.array(arr.reshape(arr.shape[0], -1).tolist(), type=pa.list_(pa.float32()))
        else:
            columns[key] = pa.array(arr)
    return pa.table(columns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--rows-per-shard", type=int, default=50_000)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    split_dir = resolve_split_dir(args.split)
    n_episodes_available = len(list(split_dir.glob("*.json")))
    n_episodes = args.max_episodes or n_episodes_available
    print(f"split: {split_dir} ({n_episodes}/{n_episodes_available} episodes)")

    out_dir = config.PROJECT_ROOT / "data" / "interim" / "encoded_size_probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_bytes = sum(f.stat().st_size for f in list(split_dir.glob("*.json"))[:n_episodes])

    t0 = time.time()
    rows: list[dict] = []
    n_rows = 0
    shard_idx = 0
    encoded_bytes = 0
    for obs, label, exclude, _meta in iter_decisions(split_dir, args.max_episodes):
        feats = encode_observation(obs, exclude=exclude)
        if feats is None:
            continue
        feats.pop("n_real_options", None)
        feats["label"] = __import__("torch").tensor(label, dtype=__import__("torch").long)
        rows.append(feats)
        n_rows += 1
        if len(rows) >= args.rows_per_shard:
            table = rows_to_table(rows)
            shard_path = out_dir / f"shard_{shard_idx:04d}.parquet"
            pq.write_table(table, shard_path, compression="zstd")
            encoded_bytes += shard_path.stat().st_size
            if not args.keep:
                shard_path.unlink()
            shard_idx += 1
            rows = []
            elapsed = time.time() - t0
            print(f"  {n_rows} rows encoded ({n_rows/elapsed:.0f} rows/sec), "
                  f"{encoded_bytes/2**20:.1f} MiB written so far")

    if rows:
        table = rows_to_table(rows)
        shard_path = out_dir / f"shard_{shard_idx:04d}.parquet"
        pq.write_table(table, shard_path, compression="zstd")
        encoded_bytes += shard_path.stat().st_size
        if not args.keep:
            shard_path.unlink()

    elapsed = time.time() - t0
    print(f"\n{n_rows} rows from {n_episodes} episodes in {elapsed:.1f}s ({n_rows/elapsed:.0f} rows/sec)")
    print(f"raw JSON bytes:     {raw_bytes/2**30:.3f} GiB")
    print(f"encoded parquet:    {encoded_bytes/2**30:.3f} GiB (zstd)")
    print(f"compression ratio:  {raw_bytes/max(encoded_bytes,1):.1f}x")
    print(f"rows/episode:       {n_rows/n_episodes:.1f}")
    bytes_per_day = encoded_bytes * (4554 / n_episodes) if n_episodes else 0
    print(f"\nextrapolated encoded bytes/day (scaled to a full 4554-episode day): "
          f"{bytes_per_day/2**30:.3f} GiB/day")
    print(f"extrapolated to 41 days: {bytes_per_day*41/2**30:.2f} GiB total")
    if not args.keep:
        try:
            out_dir.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    main()
