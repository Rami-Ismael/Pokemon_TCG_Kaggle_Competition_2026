"""Rung 1 (offline) evaluation: top-1 / top-3 action-match on the held-out
day, reported per SelectContext, always printed next to the majority-class
baseline computed on the SAME rows (same label scheme -- including the
decline slot and multi-select unroll added in this pass, so the baseline is
apples-to-apples with what the model is actually scored against, not the
raw Phase 0 numbers which predate those label changes).

Usage:
    uv run python scripts/eval_rung1.py --model-dir models/il_agent --max-episodes 500
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pokemon_tcg import config  # noqa: E402
from pokemon_tcg.device import resolve_device  # noqa: E402
from pokemon_tcg.il_dataset import (  # noqa: E402
    encode_observation,
    iter_decisions,
    resolve_split_dir,
)
from pokemon_tcg.il_model import PTCGImitationPolicy  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=config.MODELS_DIR / "il_agent")
    ap.add_argument("--eval-split", default="eval")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    model = PTCGImitationPolicy.from_pretrained(args.model_dir).to(device).eval()
    eval_dir = resolve_split_dir(args.eval_split)
    print(f"eval split: {eval_dir}")
    print(f"model: {args.model_dir}  device: {device}")

    ctx_n = Counter()
    ctx_top1 = Counter()
    ctx_top3 = Counter()
    ctx_majority_label = defaultdict(Counter)  # first pass: find each context's majority label

    n = 0
    with torch.no_grad():
        for obs, label, exclude in iter_decisions(eval_dir, args.max_episodes):
            feats = encode_observation(obs, exclude=exclude)
            if feats is None:
                continue
            ctx = int(feats["select_context"].item())
            feats.pop("n_real_options", None)
            batch = {k: v.unsqueeze(0).to(device) for k, v in feats.items()}
            logits = model(**batch)["logits"][0]
            top3 = logits.topk(min(3, logits.numel())).indices.tolist()

            ctx_n[ctx] += 1
            ctx_majority_label[ctx][label] += 1
            if top3[0] == label:
                ctx_top1[ctx] += 1
            if label in top3:
                ctx_top3[ctx] += 1
            n += 1
            if n % 5000 == 0:
                print(f"  ...{n} rows scored")

    print(f"\ntotal rows scored: {n}\n")
    print(f"{'ctx':>4} {'n':>8} {'majority%':>10} {'top1%':>8} {'top3%':>8}  gap-vs-majority")
    total_top1 = total_top3 = total_maj = 0
    for ctx in sorted(ctx_n, key=lambda c: -ctx_n[c]):
        cnt = ctx_n[ctx]
        maj = ctx_majority_label[ctx].most_common(1)[0][1] / cnt
        top1 = ctx_top1[ctx] / cnt
        top3 = ctx_top3[ctx] / cnt
        total_top1 += ctx_top1[ctx]
        total_top3 += ctx_top3[ctx]
        total_maj += ctx_majority_label[ctx].most_common(1)[0][1]
        gap = top1 - maj
        flag = "  <-- BELOW majority baseline" if gap < 0 else ""
        print(f"{ctx:>4} {cnt:>8} {100*maj:>9.1f}% {100*top1:>7.1f}% {100*top3:>7.1f}%  {gap:+.1%}{flag}")

    print(f"\n{'GLOBAL':>4} {n:>8} {100*total_maj/n:>9.1f}% {100*total_top1/n:>7.1f}% {100*total_top3/n:>7.1f}%")


if __name__ == "__main__":
    main()
