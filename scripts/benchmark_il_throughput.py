"""Stage-1 throughput benchmark for the batch-size experiment (2026-08-13).

For each batch size this measures two things over the SAME shard stream:

  (a) loader-only rows/sec -- how fast the streaming DataLoader alone can
      produce collated batches. This is the dataloader ceiling.
  (b) train rows/sec -- full pipeline: device transfer, forward, backward,
      grad clip, AdamW step, on the production-size model (192/6/6).

If (b) tracks (a), training is dataloader-bound and batch size is the wrong
knob (the follow-up would be a num_workers/prefetch sweep). If (b) rises with
batch size while staying below (a), the compute step was launch-bound and
batching helps. Peak RSS of the whole process tree is sampled throughout so
the RAM cost of each arm is part of the result, not a surprise.

Experiment card: notes/experiments/2026-08-13-batch-size-throughput.md
Run on an otherwise idle machine -- MPS numbers under contention are void.

Usage:
    uv run python scripts/benchmark_il_throughput.py --out results.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import threading
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pokemon_tcg.device import resolve_device  # noqa: E402
from pokemon_tcg.il_dataset import ShardILDataset, split_meta  # noqa: E402
from pokemon_tcg.il_model import PTCGILConfig, PTCGImitationPolicy  # noqa: E402


class PeakRssSampler:
    """Samples RSS of this process + all children (DataLoader workers live in
    child processes, so self-RSS alone would miss most of the memory)."""

    def __init__(self, interval: float = 0.2) -> None:
        import psutil

        self._proc = psutil.Process()
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_bytes = 0

    def _run(self) -> None:
        import psutil

        while not self._stop.is_set():
            total = 0
            try:
                total = self._proc.memory_info().rss
                for child in self._proc.children(recursive=True):
                    try:
                        total += child.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
            except psutil.Error:
                pass
            self.peak_bytes = max(self.peak_bytes, total)
            self._stop.wait(self._interval)

    def __enter__(self) -> "PeakRssSampler":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()


def make_dataset(args) -> ShardILDataset:
    kind, days, _ = split_meta(args.train_split)
    ds = ShardILDataset(
        kind, days=days, repo_id=args.hub_repo,
        max_episodes=args.max_episodes, seed=args.seed,
    )
    ds.set_epoch(0)  # identical shard order + shuffle rng for every arm
    return ds


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def timed_pass(args, batch_size: int, device: str, mode: str) -> dict:
    """One measured pass. mode='loader' iterates batches only; mode='train'
    additionally runs forward/backward/step on the production-size model."""
    ds = make_dataset(args)
    workers = min(args.num_workers, ds.n_shards)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=workers)

    model = optimizer = None
    autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.precision)
    if mode == "train":
        model = PTCGImitationPolicy(
            PTCGILConfig(hidden_size=args.hidden_size,
                         num_hidden_layers=args.num_layers,
                         num_attention_heads=args.num_heads)
        ).to(device)
        model.train()
        if args.compile:
            model = torch.compile(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    min_rows = max(args.measure_min_rows, args.measure_min_steps * batch_size)
    warmup_left = args.warmup_steps
    rows = steps = 0
    t0 = None
    oom = False
    it = iter(loader)
    with PeakRssSampler() as rss:
        try:
            for batch in it:
                if mode == "train":
                    batch = {k: v.to(device) for k, v in batch.items()}
                    if autocast_dtype is not None:
                        with torch.autocast(device_type=device, dtype=autocast_dtype):
                            out = model(**batch)
                    else:
                        out = model(**batch)
                    loss = out["loss"]
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                if warmup_left > 0:
                    warmup_left -= 1
                    if warmup_left == 0:
                        sync(device)
                        t0 = time.time()
                    continue
                rows += next(iter(batch.values())).shape[0] if mode == "loader" \
                    else batch["label"].shape[0]
                steps += 1
                if rows >= min_rows:
                    break
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                oom = True
            else:
                raise
        sync(device)
        elapsed = (time.time() - t0) if t0 is not None else float("nan")
    del it, loader, ds, model, optimizer
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    return {
        "mode": mode,
        "precision": args.precision,
        "compiled": bool(args.compile),
        "batch_size": batch_size,
        "num_workers": workers,
        "rows_timed": rows,
        "steps_timed": steps,
        "seconds": round(elapsed, 3),
        "rows_per_sec": round(rows / elapsed, 1) if elapsed and rows else 0.0,
        "steps_per_sec": round(steps / elapsed, 3) if elapsed and steps else 0.0,
        "peak_rss_gb": round(rss.peak_bytes / 2**30, 2),
        "oom": oom,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--batch-sizes", default="64,128,256,512,1024")
    ap.add_argument("--modes", default="loader,train",
                    help="comma list of passes per arm; precision/compile arms "
                         "only need 'train' (the loader is unaffected by them)")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--train-split", default="train_combined_v4",
                    help="key into splits.json (production BC corpus)")
    ap.add_argument("--hub-repo", default=None)
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--warmup-steps", type=int, default=30)
    ap.add_argument("--measure-min-rows", type=int, default=12800)
    ap.add_argument("--measure-min-steps", type=int, default=20)
    ap.add_argument("--hidden-size", type=int, default=192)
    ap.add_argument("--num-layers", type=int, default=6)
    ap.add_argument("--num-heads", type=int, default=6)
    ap.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32",
                    help="autocast dtype for the train pass (fp32 = no autocast)")
    ap.add_argument("--compile", action="store_true",
                    help="wrap the model in torch.compile (compile time lands in "
                         "warmup, excluded from timing)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    device = args.device or resolve_device()
    batch_sizes = [int(b) for b in args.batch_sizes.split(",") if b]
    print(f"device={device} split={args.train_split} workers={args.num_workers} "
          f"warmup={args.warmup_steps} steps, timing >= {args.measure_min_rows} "
          f"rows and >= {args.measure_min_steps} steps per arm", flush=True)

    results = []
    modes = [m for m in args.modes.split(",") if m]
    for bs in batch_sizes:
        for mode in modes:
            r = timed_pass(args, bs, device, mode)
            results.append(r)
            print(json.dumps(r), flush=True)

    if "loader" in modes and "train" in modes:
        print(f"\n{'batch':>6} {'loader rows/s':>14} {'train rows/s':>13} "
              f"{'train steps/s':>14} {'peak RSS GB':>12} {'x vs b64':>9}")
        by = {(r["mode"], r["batch_size"]): r for r in results}
        base = by.get(("train", batch_sizes[0]), {}).get("rows_per_sec") or None
        for bs in batch_sizes:
            ld, tr = by[("loader", bs)], by[("train", bs)]
            speedup = f"{tr['rows_per_sec'] / base:.2f}" if base else "-"
            flag = " OOM" if tr["oom"] or ld["oom"] else ""
            print(f"{bs:>6} {ld['rows_per_sec']:>14} {tr['rows_per_sec']:>13} "
                  f"{tr['steps_per_sec']:>14} {tr['peak_rss_gb']:>12} {speedup:>9}{flag}")

    if args.out:
        args.out.write_text(json.dumps(
            {"device": device, "config": {k: str(v) for k, v in vars(args).items()},
             "results": results}, indent=2))
        print(f"\nresults written to {args.out}")


if __name__ == "__main__":
    main()
