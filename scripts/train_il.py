"""Train the Pokemon TCG imitation-learning (behavior cloning) policy.

Reads training decisions directly from the on-disk episode split named in
data/episodes/splits/splits.json (train-2026-07-26) and validates on the
held-out calendar day (eval-2026-07-27) -- no download, no re-splitting,
per the project's held-out-DAY rule.

Usage:
    uv run python scripts/train_il.py --epochs 3
    uv run python scripts/train_il.py --dry-run
        # 50 train / 20 eval episodes, a tiny 2-layer/32-dim model, 1 epoch.
        # Verifies the pipeline end-to-end in ~1-2 minutes on CPU.

Full-split runtime: see notes/phase6_projection.md for a measured
steps/sec -> wall-clock projection; per the project's own stop condition,
do not launch anything projected over 1 hour without checking that note.

Checkpoints are written to models/il_agent/ via
PTCGImitationPolicy.save_pretrained(), which agents/il_agent/agent_core.py
loads straight back with .from_pretrained(). To package a submission, copy
models/il_agent/{config.json,model.safetensors} + a deck.csv +
agents/il_agent/{agent_core.py,main.py} into submissions/il_agent/, mirroring
the layout of submissions/mega_lucario/.

Device: resolved once via pokemon_tcg.device.resolve_device() (MPS if
available, else CPU -- no CUDA branch, see src/pokemon_tcg/device.py).
Override with --device cpu to reproduce/time exactly what the CPU-only
Kaggle evaluator will do.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pokemon_tcg import config  # noqa: E402
from pokemon_tcg.device import log_device_info, resolve_device  # noqa: E402
from pokemon_tcg.il_dataset import ILDataset, resolve_split_dir  # noqa: E402
from pokemon_tcg.il_model import PTCGILConfig, PTCGImitationPolicy  # noqa: E402
from pokemon_tcg.logging_utils import TensorBoardLogger  # noqa: E402

# Measured on the full train day (notes/phase0_discovery_report.md, §0.5):
# global majority-class baseline over every single-choice decision. Logged
# as a flat reference line so every accuracy curve is interpretable next to
# it, not just in isolation.
MAJORITY_BASELINE = 0.381


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=config.PROJECT_ROOT
        ).stdout.strip()
    except Exception:
        return "unknown"


def accuracy(logits: torch.Tensor, label: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == label).float().mean().item()


def cosine_warmup_lr(step: int, warmup_steps: int, total_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(
    model: PTCGImitationPolicy,
    loader: DataLoader,
    device: str,
    max_batches: int | None = None,
) -> dict:
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    per_ctx_correct: dict[int, int] = defaultdict(int)
    per_ctx_total: dict[int, int] = defaultdict(int)
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        total_loss += out["loss"].item()
        total_acc += accuracy(out["logits"], batch["label"])
        n += 1

        pred = out["logits"].argmax(dim=-1)
        correct = (pred == batch["label"]).cpu().numpy()
        ctx = batch["select_context"].cpu().numpy()
        for c, ok in zip(ctx, correct, strict=True):
            per_ctx_total[int(c)] += 1
            per_ctx_correct[int(c)] += int(ok)

    per_ctx_acc = {
        c: per_ctx_correct[c] / per_ctx_total[c] for c in per_ctx_total if per_ctx_total[c] > 0
    }
    model.train()
    return {
        "loss": total_loss / max(n, 1),
        "accuracy": total_acc / max(n, 1),
        "per_context_accuracy": per_ctx_acc,
        "per_context_n": dict(per_ctx_total),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--train-split", default="train", help="key into splits.json")
    ap.add_argument("--eval-split", default="eval", help="key into splits.json")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    ap.add_argument("--max-train-episodes", type=int, default=None)
    ap.add_argument("--max-eval-episodes", type=int, default=None)
    ap.add_argument(
        "--eval-batches", type=int, default=100, help="cap eval batches (streamed too)"
    )
    ap.add_argument("--hidden-size", type=int, default=192)
    ap.add_argument("--num-layers", type=int, default=6)
    ap.add_argument("--num-heads", type=int, default=6)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--out", type=Path, default=config.MODELS_DIR / "il_agent")
    ap.add_argument("--run-dir", type=Path, default=None, help="defaults to runs/<timestamp>")
    ap.add_argument("--device", default=None, help="override resolve_device() (e.g. 'cpu')")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="tiny model, 50 train / 20 eval episodes, 1 epoch -- pipeline smoke test",
    )
    args = ap.parse_args()

    if args.dry_run:
        args.max_train_episodes = args.max_train_episodes or 50
        args.max_eval_episodes = args.max_eval_episodes or 20
        args.epochs = 1
        args.hidden_size, args.num_layers, args.num_heads = 32, 2, 2
        args.eval_batches = 20
        args.log_every = 20
        args.warmup_steps = 10

    set_seed(args.seed)
    device = resolve_device(args.device)

    run_dir = args.run_dir or (config.PROJECT_ROOT / "runs" / time.strftime("%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    log_device_info(run_dir, device)
    (run_dir / "run_config.json").write_text(
        json.dumps({**vars(args), "out": str(args.out), "run_dir": str(run_dir),
                     "git_sha": git_sha(), "majority_baseline": MAJORITY_BASELINE}, indent=2, default=str)
    )
    logger = TensorBoardLogger(run_dir)
    print(f"run dir: {run_dir}  device: {device}  git sha: {git_sha()[:12]}")

    train_dir = resolve_split_dir(args.train_split)
    eval_dir = resolve_split_dir(args.eval_split)
    print(f"train split: {train_dir}")
    print(f"eval split:  {eval_dir}")

    train_ds = ILDataset(train_dir, max_episodes=args.max_train_episodes, seed=args.seed)
    eval_ds = ILDataset(eval_dir, max_episodes=args.max_eval_episodes, shuffle_buffer=1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size)

    model_config = PTCGILConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
    )
    model = PTCGImitationPolicy(model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,} ({n_params/1e6:.2f}M)")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    args.out.mkdir(parents=True, exist_ok=True)
    step = 0
    best_eval_acc = -1.0
    # total_steps for the cosine schedule: ILDataset is a streaming
    # IterableDataset (no __len__), so estimate it rather than block on a
    # full pre-count pass. ROWS_PER_EPISODE is the measured rate from
    # notes/phase6_projection.md (181.3 rows/episode, incl. declines and
    # the multi-select unroll); n_episodes is a cheap glob, not a parse.
    # BUG FIXED HERE: a prior placeholder (`max(warmup_steps*10, 2000)`)
    # hardcoded total_steps=2000 regardless of the real epoch length, so
    # the cosine schedule decayed LR to exactly 0 at step 2000 and stayed
    # there -- a 12,854-step run only actually trained for its first ~2000
    # steps; the remaining ~10,850 were zero-gradient no-ops that still
    # cost wall-clock time. Confirmed from runs/full_epoch1's own log
    # (lr=0.00e+00 from step 2500 through step 12500).
    ROWS_PER_EPISODE = 181.3
    n_episodes = len(list(train_dir.glob("*.json")))
    if args.max_train_episodes:
        n_episodes = min(n_episodes, args.max_train_episodes)
    est_total_steps = max(
        int(n_episodes * ROWS_PER_EPISODE / args.batch_size) * args.epochs,
        args.warmup_steps + 1,
    )
    print(f"estimated total steps for LR schedule: {est_total_steps} "
          f"({n_episodes} episodes x {ROWS_PER_EPISODE} rows/ep / batch {args.batch_size} x {args.epochs} epochs)")
    t0 = time.time()
    for epoch in range(args.epochs):
        running_loss, running_acc, running_n = 0.0, 0.0, 0
        epoch_step0 = step
        epoch_t0 = time.time()
        for batch in train_loader:
            lr = cosine_warmup_lr(step, args.warmup_steps, est_total_steps, args.lr)
            for g in optimizer.param_groups:
                g["lr"] = lr

            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            running_loss += loss.item()
            running_acc += accuracy(out["logits"], batch["label"])
            running_n += 1
            step += 1
            if step % args.log_every == 0:
                elapsed = time.time() - t0
                steps_per_sec = (step - epoch_step0) / max(time.time() - epoch_t0, 1e-9)
                train_loss = running_loss / running_n
                train_acc = running_acc / running_n
                print(
                    f"epoch {epoch} step {step} "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                    f"lr={lr:.2e} grad_norm={grad_norm:.3f} "
                    f"({steps_per_sec:.2f} steps/s, {elapsed:.1f}s)"
                )
                logger.log_scalars(
                    {
                        "train/loss": train_loss,
                        "train/accuracy": train_acc,
                        "train/lr": lr,
                        "train/grad_norm": float(grad_norm),
                        "train/steps_per_sec": steps_per_sec,
                        "reference/majority_baseline": MAJORITY_BASELINE,
                    },
                    step,
                )
                running_loss, running_acc, running_n = 0.0, 0.0, 0

        eval_result = evaluate(model, eval_loader, device, max_batches=args.eval_batches)
        msg = (
            f"epoch {epoch} done -- eval_loss={eval_result['loss']:.4f} "
            f"eval_acc={eval_result['accuracy']:.4f} (majority baseline {MAJORITY_BASELINE:.3f})"
        )
        print(msg)
        logger.log_scalars(
            {
                "eval/loss": eval_result["loss"],
                "eval/accuracy": eval_result["accuracy"],
                "reference/majority_baseline": MAJORITY_BASELINE,
            },
            step,
        )
        for ctx, acc in sorted(eval_result["per_context_accuracy"].items()):
            n_ctx = eval_result["per_context_n"][ctx]
            logger.log_scalar(f"eval_per_context/ctx_{ctx}_accuracy", acc, step)
            logger.log_scalar(f"eval_per_context/ctx_{ctx}_n", n_ctx, step)

        # Checkpoint-on-best-val: only overwrite the shipped checkpoint when
        # eval accuracy improves: val loss/accuracy/Rung-2 win-rate can
        # disagree (Phase 4), but val accuracy is the cheapest signal
        # available inside the training loop itself.
        if eval_result["accuracy"] > best_eval_acc:
            best_eval_acc = eval_result["accuracy"]
            # Device-agnostic checkpoint: state_dict moved to CPU before
            # save, so an MPS-trained checkpoint still loads on a CPU-only
            # evaluator via .from_pretrained() + map_location handling in
            # agent_core.py's _load_model().
            model.to("cpu").save_pretrained(args.out)
            model.to(device)
            (args.out / "train_metadata.json").write_text(
                json.dumps(
                    {
                        "epoch": epoch,
                        "step": step,
                        "eval_loss": eval_result["loss"],
                        "eval_accuracy": eval_result["accuracy"],
                        "majority_baseline": MAJORITY_BASELINE,
                        "git_sha": git_sha(),
                        "device_trained_on": device,
                        "run_dir": str(run_dir),
                        "n_params": n_params,
                        "config": vars(args) | {"out": str(args.out), "run_dir": str(run_dir)},
                    },
                    indent=2,
                    default=str,
                )
            )
            print(f"checkpoint saved to {args.out} (new best eval_acc={best_eval_acc:.4f})")

    logger.close()
    print(f"total training time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
