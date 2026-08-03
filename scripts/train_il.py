"""Train the Pokemon TCG imitation-learning (behavior cloning) policy.

Training decisions come from one of two interchangeable sources
(--data-source): the on-disk episode splits named in
data/episodes/splits/splits.json, or the packed zstd-Parquet corpus in the
private Hugging Face repo (ADR-001) streamed shard-by-shard with optional
DataLoader workers (--num-workers). Validation is always the held-out
calendar day(s), per the project's held-out-DAY rule -- no re-splitting.

Usage:
    uv run python scripts/train_il.py --epochs 3
    uv run python scripts/train_il.py --epochs 0.5   # sub-epoch: a fraction
        # of one pass, schedule still anneals fully to the requested length.
    uv run python scripts/train_il.py --dry-run
        # 50 train / 20 eval episodes, a tiny 2-layer/32-dim model, 1 epoch.
        # Verifies the pipeline end-to-end in ~1-2 minutes on CPU.

Full-split runtime: see notes/phase6_projection.md for a measured
steps/sec -> wall-clock projection; per the project's own stop condition,
do not launch anything projected over 1 hour without checking that note.

STEP-DRIVEN, NOT EPOCH-DRIVEN. `--epochs` (a float now, not int) only sets
the SCHEDULE LENGTH (est_total_steps = n_episodes * ROWS_PER_EPISODE /
batch_size * epochs) -- the training loop itself runs by step count against
a repeatedly re-iterated streaming loader, not `for epoch in
range(args.epochs)`. This is what makes a genuine 0.5-epoch run possible
(a fractional pass through the data with a schedule that still anneals to
zero exactly at that length) instead of "epochs" only ever meaning whole
passes. A run stopped before its own schedule's total_steps is undertrained
by construction (the LR never annealed) -- always let a run finish, don't
Ctrl-C it expecting a usable partial checkpoint.

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
from pokemon_tcg.il_dataset import ILDataset, ShardILDataset, resolve_split_dir  # noqa: E402
from pokemon_tcg.il_model import PTCGILConfig, PTCGImitationPolicy  # noqa: E402
from pokemon_tcg.logging_utils import TensorBoardLogger  # noqa: E402

# Measured on the full train day (notes/phase0_discovery_report.md, §0.5):
# global majority-class baseline over every single-choice decision. Logged
# as a flat reference line so every accuracy curve is interpretable next to
# it, not just in isolation.
MAJORITY_BASELINE = 0.381

# named top-level submodules to group by for the ||dW||/||W|| report --
# embeddings and the trunk/head converge at different rates (Gate 2), a
# single global norm ratio hides that.
PARAM_GROUPS = [
    "card_emb", "attack_emb", "special_emb", "select_type_emb",
    "select_context_emb", "option_type_emb", "slot_scalar_proj",
    "global_scalar_proj", "opt_scalar_proj", "opt_ref_scalar_proj",
    "cls_param", "slot_pos_emb", "opt_pos_emb", "encoder", "score_head",
]


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


def accuracy(logits: torch.Tensor, label: torch.Tensor, k: int = 1) -> float:
    if k == 1:
        return (logits.argmax(dim=-1) == label).float().mean().item()
    topk = logits.topk(k=min(k, logits.shape[-1]), dim=-1).indices
    return (topk == label.unsqueeze(-1)).any(dim=-1).float().mean().item()


def policy_entropy(logits: torch.Tensor, opt_mask: torch.Tensor) -> float:
    """Mean entropy over legal actions, normalized by log(n_legal).

    Rows with <=1 legal option are trivial (entropy is 0 by construction,
    log(n_legal) would be 0 too) -- excluded from the mean rather than
    counted as a spurious 0/0.
    """
    probs = torch.softmax(logits, dim=-1)
    ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
    n_legal = opt_mask.sum(dim=-1).float()
    valid = n_legal > 1
    if not valid.any():
        return 0.0
    norm_ent = ent[valid] / torch.log(n_legal[valid])
    return norm_ent.mean().item()


def cosine_warmup_lr(step: int, warmup_steps: int, total_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def snapshot_param_norms(model: PTCGImitationPolicy) -> dict[str, tuple[torch.Tensor, float]]:
    """One flat detached copy + its norm per top-level submodule, for the dW/W report."""
    groups: dict[str, list[torch.Tensor]] = defaultdict(list)
    for name, p in model.named_parameters():
        top = name.split(".")[0]
        groups[top].append(p.detach().flatten().clone())
    out = {}
    for top, tensors in groups.items():
        flat = torch.cat(tensors)
        out[top] = (flat, flat.norm().item())
    return out


def weight_delta_ratios(model: PTCGImitationPolicy, initial: dict[str, tuple[torch.Tensor, float]]) -> dict[str, float]:
    current = snapshot_param_norms(model)
    ratios = {}
    for top, (init_flat, init_norm) in initial.items():
        cur_flat, _ = current[top]
        dw = (cur_flat - init_flat).norm().item()
        ratios[top] = dw / max(init_norm, 1e-12)
    return ratios


@torch.no_grad()
def evaluate(
    model: PTCGImitationPolicy,
    loader: DataLoader,
    device: str,
    max_batches: int | None = None,
) -> dict:
    model.eval()
    total_loss, total_acc, total_acc3, total_ent, n = 0.0, 0.0, 0.0, 0.0, 0
    per_ctx_correct: dict[int, int] = defaultdict(int)
    per_ctx_total: dict[int, int] = defaultdict(int)
    # ECE accumulators: 10 equal-width confidence bins over the model's own
    # top-1 softmax probability, aggregated across the whole eval pass (not
    # averaged per-batch -- ECE is not a linear function of its inputs).
    n_bins = 10
    bin_conf_sum = np.zeros(n_bins)
    bin_correct_sum = np.zeros(n_bins)
    bin_count = np.zeros(n_bins)

    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        logits, label = out["logits"], batch["label"]
        total_loss += out["loss"].item()
        total_acc += accuracy(logits, label, k=1)
        total_acc3 += accuracy(logits, label, k=3)
        total_ent += policy_entropy(logits, batch["opt_mask"])
        n += 1

        probs = torch.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)
        correct = (pred == label).float()
        conf_np, correct_np = conf.cpu().numpy(), correct.cpu().numpy()
        bin_idx = np.clip((conf_np * n_bins).astype(int), 0, n_bins - 1)
        for b, c, ok in zip(bin_idx, conf_np, correct_np, strict=True):
            bin_conf_sum[b] += c
            bin_correct_sum[b] += ok
            bin_count[b] += 1

        ctx = batch["select_context"].cpu().numpy()
        correct_ctx = (pred == label).cpu().numpy()
        for c, ok in zip(ctx, correct_ctx, strict=True):
            per_ctx_total[int(c)] += 1
            per_ctx_correct[int(c)] += int(ok)

    ece = 0.0
    n_total = bin_count.sum()
    if n_total > 0:
        for b in range(n_bins):
            if bin_count[b] == 0:
                continue
            avg_conf = bin_conf_sum[b] / bin_count[b]
            avg_acc = bin_correct_sum[b] / bin_count[b]
            ece += (bin_count[b] / n_total) * abs(avg_conf - avg_acc)

    per_ctx_acc = {
        c: per_ctx_correct[c] / per_ctx_total[c] for c in per_ctx_total if per_ctx_total[c] > 0
    }
    model.train()
    return {
        "loss": total_loss / max(n, 1),
        "accuracy": total_acc / max(n, 1),
        "top3_accuracy": total_acc3 / max(n, 1),
        "entropy_normalized": total_ent / max(n, 1),
        "ece": ece,
        "per_context_accuracy": per_ctx_acc,
        "per_context_n": dict(per_ctx_total),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--train-split", default="train", help="key into splits.json")
    ap.add_argument("--eval-split", default="eval", help="key into splits.json")
    ap.add_argument("--epochs", type=float, default=3,
                     help="schedule length in passes over the train split; fractional "
                          "values (e.g. 0.5) run a genuine sub-epoch schedule, not a "
                          "truncated one -- see module docstring")
    ap.add_argument("--total-steps", type=int, default=None,
                     help="override the epochs->steps estimate directly")
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
    ap.add_argument("--data-source", choices=["auto", "local", "hub"], default="auto",
                     help="'local' = raw JSON split folders (original path); 'hub' = "
                          "zstd Parquet shards from the private HF repo (ADR-001; each "
                          "shard is cached after first use, so pass 2+ reads from "
                          "disk); 'auto' = local if the split folder exists, else hub. "
                          "NOTE: hub trains on ALL train days in the repo (the growing "
                          "corpus), not just splits.json's single day -- restrict with "
                          "--hub-days for a like-for-like comparison")
    ap.add_argument("--hub-days", default=None,
                     help="comma-separated YYYY-MM-DD list restricting TRAIN days when "
                          "--data-source hub (eval always uses the repo's eval days)")
    ap.add_argument("--hub-repo", default=None,
                     help=f"dataset repo id (default {config.HF_EPISODES_REPO})")
    ap.add_argument("--num-workers", type=int, default=0,
                     help="DataLoader workers for the train stream (hub source only; "
                          "needs >= that many shards). 0 reproduces the old "
                          "single-process loader, which was 56%% data-wait on MPS")
    ap.add_argument("--eval-every-steps", type=int, default=None,
                     help="eval+checkpoint at this step interval (default: once, at the end)")
    ap.add_argument("--features", default="",
                     help="comma-separated deterministic-future feature groups the model "
                          "consumes (names from il_dataset.GLOBAL_FEATURE_SPECS / "
                          "OPT_FEATURE_SPECS, e.g. 'ko_race,attack_tactical'). The "
                          "encoder always computes every group; this only selects which "
                          "columns the model sees. Empty = baseline architecture.")
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

    source = args.data_source
    if source == "auto":
        try:
            source = "local" if resolve_split_dir(args.train_split).exists() else "hub"
        except (FileNotFoundError, KeyError):
            source = "hub"
    if args.num_workers > 0 and source != "hub":
        sys.exit(
            "--num-workers > 0 requires --data-source hub: ILDataset over raw JSON "
            "has no worker sharding, so every worker would duplicate the data."
        )

    if source == "hub":
        hub_days = args.hub_days.split(",") if args.hub_days else None
        train_ds = ShardILDataset(
            "train", days=hub_days, repo_id=args.hub_repo,
            max_episodes=args.max_train_episodes, seed=args.seed,
        )
        eval_ds = ShardILDataset(
            "eval", repo_id=args.hub_repo,
            max_episodes=args.max_eval_episodes, shuffle_buffer=1,
        )
        n_episodes = train_ds.n_episodes()  # parquet footers only, no downloads
        if args.num_workers > train_ds.n_shards:
            print(f"warning: --num-workers {args.num_workers} > {train_ds.n_shards} "
                  f"train shards; clamping to {train_ds.n_shards}")
            args.num_workers = train_ds.n_shards
        print(f"train source: hub {train_ds.repo_id} ({train_ds.n_shards} shards, "
              f"{n_episodes} episodes{', days ' + args.hub_days if hub_days else ''}, "
              f"{args.num_workers} workers)")
        print(f"eval source:  hub {eval_ds.repo_id} ({eval_ds.n_shards} shards)")
    else:
        train_dir = resolve_split_dir(args.train_split)
        eval_dir = resolve_split_dir(args.eval_split)
        print(f"train split: {train_dir}")
        print(f"eval split:  {eval_dir}")
        train_ds = ILDataset(train_dir, max_episodes=args.max_train_episodes, seed=args.seed)
        eval_ds = ILDataset(eval_dir, max_episodes=args.max_eval_episodes, shuffle_buffer=1)
        n_episodes = len(list(train_dir.glob("*.json")))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers
    )
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size)

    feature_names = [f for f in args.features.split(",") if f]
    from pokemon_tcg.il_dataset import GLOBAL_FEATURE_SPECS, OPT_FEATURE_SPECS
    unknown_features = [
        f for f in feature_names
        if f not in GLOBAL_FEATURE_SPECS and f not in OPT_FEATURE_SPECS
    ]
    if unknown_features:
        sys.exit(f"unknown feature group(s) {unknown_features}; known: "
                 f"{list(GLOBAL_FEATURE_SPECS) + list(OPT_FEATURE_SPECS)}")
    model_config = PTCGILConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        global_features=[f for f in feature_names if f in GLOBAL_FEATURE_SPECS],
        opt_features=[f for f in feature_names if f in OPT_FEATURE_SPECS],
    )
    model = PTCGImitationPolicy(model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,} ({n_params/1e6:.2f}M)")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    initial_param_norms = snapshot_param_norms(model)

    args.out.mkdir(parents=True, exist_ok=True)

    # total_steps for the cosine schedule: both dataset types are streaming
    # IterableDatasets (no __len__), so estimate it rather than block on a
    # full pre-count pass. ROWS_PER_EPISODE is the measured rate from
    # notes/phase6_projection.md (181.3 rows/episode, incl. declines and
    # the multi-select unroll); n_episodes came from a glob (local) or the
    # shard footers (hub) above.
    ROWS_PER_EPISODE = 181.3
    if args.max_train_episodes:
        n_episodes = min(n_episodes, args.max_train_episodes)
    if args.total_steps is not None:
        total_steps = args.total_steps
    else:
        total_steps = max(
            int(n_episodes * ROWS_PER_EPISODE / args.batch_size * args.epochs),
            args.warmup_steps + 1,
        )
    eval_every_steps = args.eval_every_steps or total_steps  # default: once, at the very end
    print(f"total steps for this schedule: {total_steps} "
          f"({n_episodes} episodes x {ROWS_PER_EPISODE} rows/ep / batch {args.batch_size} "
          f"x {args.epochs} epochs-equivalent)")

    step = 0
    best_eval_acc = -1.0
    running_loss, running_acc, running_n = 0.0, 0.0, 0
    t0 = time.time()
    pass_t0 = time.time()

    def save_checkpoint(eval_result: dict, is_final: bool) -> None:
        nonlocal best_eval_acc
        is_best = eval_result["accuracy"] > best_eval_acc
        if not (is_best or is_final):
            return
        best_eval_acc = max(best_eval_acc, eval_result["accuracy"])
        # Device-agnostic checkpoint: state_dict moved to CPU before save, so
        # an MPS-trained checkpoint still loads on a CPU-only evaluator via
        # .from_pretrained() + map_location handling in agent_core.py's
        # _load_model().
        model.to("cpu").save_pretrained(args.out)
        model.to(device)
        dw_ratios = weight_delta_ratios(model, initial_param_norms)
        (args.out / "train_metadata.json").write_text(
            json.dumps(
                {
                    "step": step,
                    "total_steps": total_steps,
                    "epochs_equivalent": args.epochs,
                    "is_final_checkpoint": is_final,
                    "eval_loss": eval_result["loss"],
                    "eval_accuracy": eval_result["accuracy"],
                    "eval_top3_accuracy": eval_result["top3_accuracy"],
                    "eval_entropy_normalized": eval_result["entropy_normalized"],
                    "eval_ece": eval_result["ece"],
                    "majority_baseline": MAJORITY_BASELINE,
                    "weight_delta_ratios": dw_ratios,
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
        tag = "final" if is_final else "best"
        print(f"checkpoint saved to {args.out} ({tag}, eval_acc={eval_result['accuracy']:.4f})")

    def run_eval_and_log(is_final: bool) -> None:
        eval_result = evaluate(model, eval_loader, device, max_batches=args.eval_batches)
        print(
            f"step {step}/{total_steps} eval -- loss={eval_result['loss']:.4f} "
            f"acc={eval_result['accuracy']:.4f} top3={eval_result['top3_accuracy']:.4f} "
            f"entropy_norm={eval_result['entropy_normalized']:.4f} ece={eval_result['ece']:.4f} "
            f"(majority baseline {MAJORITY_BASELINE:.3f})"
        )
        logger.log_scalars(
            {
                "eval/loss": eval_result["loss"],
                "eval/accuracy": eval_result["accuracy"],
                "eval/top3_accuracy": eval_result["top3_accuracy"],
                "eval/entropy_normalized": eval_result["entropy_normalized"],
                "eval/ece": eval_result["ece"],
                "reference/majority_baseline": MAJORITY_BASELINE,
            },
            step,
        )
        for ctx, acc in sorted(eval_result["per_context_accuracy"].items()):
            n_ctx = eval_result["per_context_n"][ctx]
            logger.log_scalar(f"eval_per_context/ctx_{ctx}_accuracy", acc, step)
            logger.log_scalar(f"eval_per_context/ctx_{ctx}_n", n_ctx, step)
        dw_ratios = weight_delta_ratios(model, initial_param_norms)
        for group, ratio in dw_ratios.items():
            logger.log_scalar(f"weight_delta_ratio/{group}", ratio, step)
        save_checkpoint(eval_result, is_final=is_final)

    # Step-driven training loop: re-iterate the streaming loader as many
    # times as needed to reach total_steps, instead of `for epoch in
    # range(args.epochs)` -- this is what makes a fractional-epoch schedule
    # (a genuine sub-epoch pass, not a truncated multi-epoch one) possible.
    pass_idx = 0
    while step < total_steps:
        # Reshuffle the shard order + shuffle rng each pass. No-op for
        # ILDataset, which has no set_epoch and keeps its original
        # fixed-seed behavior.
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(pass_idx)
        pass_idx += 1
        for batch in train_loader:
            if step >= total_steps:
                break
            lr = cosine_warmup_lr(step, args.warmup_steps, total_steps, args.lr)
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
                steps_per_sec = args.log_every / max(time.time() - pass_t0, 1e-9)
                pass_t0 = time.time()
                train_loss = running_loss / running_n
                train_acc = running_acc / running_n
                print(
                    f"step {step}/{total_steps} "
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

            if step % eval_every_steps == 0 or step >= total_steps:
                run_eval_and_log(is_final=(step >= total_steps))

    logger.close()
    print(f"total training time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
