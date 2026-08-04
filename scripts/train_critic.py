"""Train the S2-E4 state-value critic V(s) on terminal outcomes.

rl_pipeline_v1.md §2.1 row E4: warm-start the trunk from a frozen actor
checkpoint, regress cls_hidden -> outcome in {0, 0.5, 1} (MSE; rows with the
unknown-outcome sentinel -1 are excluded, never fit). The resulting critic is
consumed by scripts/train_il.py --weight-arm adv-exp|adv-binary and is
TRAIN-TIME ONLY — it never ships in a bundle.

Quality gate (printed at the end, next to its baseline per the repo rule that
a number without a control is not a result): eval MSE vs the constant-0.5
predictor's MSE. A critic that cannot beat "every state is a coin flip" makes
E4 collapse into a noisy E1/E2 — do not spend a policy run on it.

Usage:
  uv run python scripts/train_critic.py --init-from models/il_agent \
      --out models/critic_small --total-steps 12900
  uv run python scripts/train_critic.py --dry-run   # CPU-sized smoke
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pokemon_tcg import config  # noqa: E402
from pokemon_tcg.device import resolve_device  # noqa: E402
from pokemon_tcg.il_dataset import ILDataset, resolve_split_dir  # noqa: E402
from pokemon_tcg.il_model import PTCGILConfig  # noqa: E402
from pokemon_tcg.offline_critic import OfflineValueModel, save_critic  # noqa: E402

ROWS_PER_EPISODE = 181.3  # same measured constant train_il.py schedules with


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _split_batch(batch: dict, device: str) -> tuple[dict, torch.Tensor]:
    batch = {k: v.to(device) for k, v in batch.items()}
    outcome = batch.pop("outcome")
    for k in ILDataset.META_KEYS:
        batch.pop(k, None)
    batch.pop("label", None)
    return batch, outcome


def _masked_mse(pred: torch.Tensor, outcome: torch.Tensor) -> torch.Tensor | None:
    known = outcome >= 0.0
    if not bool(known.any()):
        return None
    err = (pred[known] - outcome[known]) ** 2
    return err.mean()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init-from", type=Path, default=config.MODELS_DIR / "il_agent",
                    help="actor checkpoint to warm-start the trunk from")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--eval-split", default="eval")
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    ap.add_argument("--max-train-episodes", type=int, default=None)
    ap.add_argument("--max-eval-episodes", type=int, default=None)
    ap.add_argument("--eval-batches", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--out", type=Path, default=config.MODELS_DIR / "critic_small")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="fresh tiny trunk, 50 train / 20 eval episodes -- smoke test")
    args = ap.parse_args()

    if args.dry_run:
        args.max_train_episodes = args.max_train_episodes or 50
        args.max_eval_episodes = args.max_eval_episodes or 20
        args.epochs = 1
        args.eval_batches = 20
        args.log_every = 20

    set_seed(args.seed)
    device = resolve_device(args.device)

    if args.dry_run:
        critic = OfflineValueModel.from_config(
            PTCGILConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2)
        )
        init_desc = "fresh tiny trunk (dry run)"
    else:
        critic = OfflineValueModel.from_actor_checkpoint(args.init_from)
        init_desc = str(args.init_from)
    critic.to(device).train()
    n_params = sum(p.numel() for p in critic.parameters())
    print(f"critic trunk init: {init_desc}  params: {n_params/1e6:.2f}M  device: {device}")

    train_dir = resolve_split_dir(args.train_split)
    eval_dir = resolve_split_dir(args.eval_split)
    n_episodes = args.max_train_episodes or len(list(train_dir.glob("*.json")))
    total_steps = args.total_steps or max(
        int(n_episodes * ROWS_PER_EPISODE / args.batch_size * args.epochs), 1
    )
    print(f"train split: {train_dir}  ({n_episodes} episodes, schedule {total_steps} steps)")

    optimizer = torch.optim.AdamW(critic.parameters(), lr=args.lr)
    step, t0 = 0, time.time()
    running, running_n = 0.0, 0
    epoch_passes = max(int(np.ceil(args.epochs)), 1)
    for _ in range(epoch_passes):
        if step >= total_steps:
            break
        ds = ILDataset(train_dir, max_episodes=args.max_train_episodes,
                       shuffle_buffer=2000, seed=args.seed + step, with_meta=True)
        for batch in DataLoader(ds, batch_size=args.batch_size):
            if step >= total_steps:
                break
            feats, outcome = _split_batch(batch, device)
            loss = _masked_mse(critic(**feats), outcome)
            if loss is None:
                continue  # batch of only unknown-outcome rows
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), args.grad_clip)
            optimizer.step()
            running += loss.item()
            running_n += 1
            step += 1
            if step % args.log_every == 0:
                print(f"step {step}/{total_steps} train_mse={running/max(running_n,1):.4f} "
                      f"({step/(time.time()-t0):.2f} steps/s)", flush=True)
                running, running_n = 0.0, 0

    # Eval: critic MSE next to the constant-0.5 baseline on the same rows.
    critic.eval()
    se_model, se_const, n_rows = 0.0, 0.0, 0
    eval_ds = ILDataset(eval_dir, max_episodes=args.max_eval_episodes,
                        shuffle_buffer=1, with_meta=True)
    with torch.no_grad():
        for i, batch in enumerate(DataLoader(eval_ds, batch_size=args.batch_size)):
            if i >= args.eval_batches:
                break
            feats, outcome = _split_batch(batch, device)
            known = outcome >= 0.0
            if not bool(known.any()):
                continue
            pred = critic(**feats)[known]
            oc = outcome[known]
            se_model += float(((pred - oc) ** 2).sum())
            se_const += float(((0.5 - oc) ** 2).sum())
            n_rows += int(known.sum())
    eval_mse = se_model / max(n_rows, 1)
    const_mse = se_const / max(n_rows, 1)
    print(f"eval -- critic_mse={eval_mse:.4f} vs constant-0.5 baseline {const_mse:.4f} "
          f"({n_rows} rows) {'PASS' if eval_mse < const_mse else 'FAIL: critic no better than coin-flip prior'}")

    save_critic(critic, args.out)
    (args.out / "train_metadata.json").write_text(json.dumps({
        "step": step,
        "init_from": init_desc,
        "train_split": args.train_split,
        "eval_mse": eval_mse,
        "constant_baseline_mse": const_mse,
        "beats_constant_baseline": eval_mse < const_mse,
        "seed": args.seed,
    }, indent=2))
    print(f"critic saved to {args.out}")


if __name__ == "__main__":
    main()
