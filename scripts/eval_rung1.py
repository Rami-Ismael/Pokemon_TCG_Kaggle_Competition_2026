"""Rung 1 (offline) evaluation: top-1 / top-3 action-match on the held-out
day, reported per SelectContext, always printed next to the majority-class
baseline computed on the SAME rows (same label scheme -- including the
decline slot and multi-select unroll added in this pass, so the baseline is
apples-to-apples with what the model is actually scored against, not the
raw Phase 0 numbers which predate those label changes).

Data source: like train_il.py, 'auto' uses the local split folder ONLY if
it holds (approximately) the full registered episode count -- after ADR-001
raw days are pruned post-Hub-verify, so a local folder can exist as a
near-empty stub while the corpus lives on the Hub. 'hub' streams the eval
day's zstd Parquet shards (deterministic order: sorted shards, pack order).

Usage:
    uv run python scripts/eval_rung1.py --model-dir models/il_agent --max-episodes 500
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pokemon_tcg import config  # noqa: E402
from pokemon_tcg.device import resolve_device  # noqa: E402
from pokemon_tcg.il_dataset import (  # noqa: E402
    _resolve_hub_shards,
    encode_observation,
    iter_decisions,
    iter_episode_decisions,
    resolve_split_dir,
)
from pokemon_tcg.il_model import PTCGImitationPolicy  # noqa: E402


def iter_hub_decisions(
    split: str,
    max_episodes: int | None,
    repo_id: str | None,
    min_score: float | None = None,
):
    """Yield (obs, label, exclude) from the split's Hub shards, deterministically.

    `min_score` keeps only episodes whose in-shard `min_score` (the LOWER of
    the two players' ratings) meets the threshold -- the skill-filtered gate
    metric. Episodes with a null score are skipped when filtering.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    repo = repo_id or config.HF_EPISODES_REPO
    cols = ["episode_json"] if min_score is None else ["min_score", "episode_json"]
    n = 0
    for rel in _resolve_hub_shards(repo, split, days=None):
        pf = pq.ParquetFile(hf_hub_download(repo, rel, repo_type="dataset"))
        for rb in pf.iter_batches(batch_size=8, columns=cols):
            batch = rb.to_pydict()
            for i, raw in enumerate(batch["episode_json"]):
                if min_score is not None:
                    ms = batch["min_score"][i]
                    if ms is None or ms < min_score:
                        continue
                if max_episodes is not None and n >= max_episodes:
                    return
                n += 1
                yield from iter_episode_decisions(json.loads(raw))


def _local_split_complete(split: str) -> bool:
    """Same pruned-stub guard as train_il.py: exists AND ~full per splits.json."""
    try:
        split_dir = resolve_split_dir(split)
    except (FileNotFoundError, KeyError):
        return False
    if not split_dir.exists():
        return False
    meta = json.loads(
        (config.EPISODES_DIR / "splits" / "splits.json").read_text()
    ).get(split, {})
    registered = meta.get("episodes")
    n_local = sum(1 for _ in split_dir.glob("*.json"))
    if registered and n_local < 0.9 * registered:
        print(f"warning: local split {split!r} has {n_local} of {registered} "
              f"registered episodes (pruned stub) -- not using it")
        return False
    return n_local > 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=config.MODELS_DIR / "il_agent")
    ap.add_argument("--eval-split", default="eval")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--data-source", choices=["auto", "local", "hub"], default="auto")
    ap.add_argument("--hub-repo", default=None,
                     help=f"dataset repo id (default {config.HF_EPISODES_REPO})")
    ap.add_argument("--min-score", type=float, default=None,
                     help="hub source only: keep only eval episodes whose min_score "
                          "(lower of the two players' ratings) is >= this")
    args = ap.parse_args()

    device = resolve_device(args.device)
    model = PTCGImitationPolicy.from_pretrained(args.model_dir).to(device).eval()

    source = args.data_source
    if source == "auto":
        source = "local" if _local_split_complete(args.eval_split) else "hub"
    elif source == "local" and not _local_split_complete(args.eval_split):
        sys.exit(f"--data-source local, but local {args.eval_split!r} split is "
                 "missing or a pruned stub. Use --data-source hub.")
    if source == "hub":
        decisions = iter_hub_decisions(
            args.eval_split, args.max_episodes, args.hub_repo, args.min_score
        )
        print(f"eval source: hub ({args.hub_repo or config.HF_EPISODES_REPO}, "
              f"split {args.eval_split}"
              + (f", min_score>={args.min_score}" if args.min_score else "") + ")")
    else:
        if args.min_score is not None:
            sys.exit("--min-score requires --data-source hub (scores live in shards)")
        eval_dir = resolve_split_dir(args.eval_split)
        decisions = iter_decisions(eval_dir, args.max_episodes)
        print(f"eval split: {eval_dir}")
    print(f"model: {args.model_dir}  device: {device}")

    ctx_n = Counter()
    ctx_top1 = Counter()
    ctx_top3 = Counter()
    ctx_majority_label = defaultdict(Counter)  # first pass: find each context's majority label

    # Forced rows -- where opt_mask leaves <=1 selectable slot -- are free
    # correct answers for the model AND for the majority baseline: action
    # masking already guarantees them at inference, so scoring them measures
    # legality, not preference. Partition on the same mask the model is
    # scored against so the split cannot drift from what the head sees.
    split_n = {"forced": Counter(), "contested": Counter()}
    split_top1 = {"forced": Counter(), "contested": Counter()}
    split_top3 = {"forced": Counter(), "contested": Counter()}
    split_majority_label = {
        "forced": defaultdict(Counter), "contested": defaultdict(Counter)
    }
    # Mask-aware floor. The majority-class baseline picks a per-context label
    # INDEX and never sees opt_mask, while the model's head is -inf-masked
    # (il_model.py:227) -- so on a row with one scoreable slot the model is
    # correct by construction and the baseline can still miss. random-legal
    # expected top-1 = mean(1/n_choices) is the honest floor: same information
    # the model has, no learning. This is the offline twin of the
    # `random_legal` benchmark agent.
    split_randleg = {"forced": 0.0, "contested": 0.0}

    n = 0
    with torch.no_grad():
        for obs, label, exclude, _meta in decisions:
            feats = encode_observation(obs, exclude=exclude)
            if feats is None:
                continue
            ctx = int(feats["select_context"].item())
            n_choices = int(feats["opt_mask"].sum().item())
            bucket = "forced" if n_choices <= 1 else "contested"
            feats.pop("n_real_options", None)
            batch = {k: v.unsqueeze(0).to(device) for k, v in feats.items()}
            logits = model(**batch)["logits"][0]
            top3 = logits.topk(min(3, logits.numel())).indices.tolist()

            ctx_n[ctx] += 1
            ctx_majority_label[ctx][label] += 1
            split_n[bucket][ctx] += 1
            split_majority_label[bucket][ctx][label] += 1
            split_randleg[bucket] += 1.0 / max(n_choices, 1)
            if top3[0] == label:
                ctx_top1[ctx] += 1
                split_top1[bucket][ctx] += 1
            if label in top3:
                ctx_top3[ctx] += 1
                split_top3[bucket][ctx] += 1
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

    # Forced-vs-contested split. The contested row is the honest headline: it
    # drops the rows the action mask already decides. Majority is recomputed
    # per-context WITHIN each bucket, so both columns stay apples-to-apples.
    print(f"\n{'bucket':>10} {'n':>8} {'share':>7} {'rand-legal%':>12} {'majority%':>10} "
          f"{'top1%':>8} {'top3%':>8}  gap-vs-rand")
    for bucket in ("forced", "contested"):
        bn = sum(split_n[bucket].values())
        if not bn:
            print(f"{bucket:>10} {0:>8}       -            -          -        -        -     -")
            continue
        b_top1 = sum(split_top1[bucket].values())
        b_top3 = sum(split_top3[bucket].values())
        b_maj = sum(
            c.most_common(1)[0][1] for c in split_majority_label[bucket].values()
        )
        b_rl = split_randleg[bucket]
        print(f"{bucket:>10} {bn:>8} {100*bn/n:>6.2f}% {100*b_rl/bn:>11.1f}% "
              f"{100*b_maj/bn:>9.1f}% {100*b_top1/bn:>7.1f}% {100*b_top3/bn:>7.1f}%  "
              f"{(b_top1-b_rl)/bn:+.1%}")
    print("  rand-legal = mean(1/n_choices), the mask-aware floor. On forced rows the\n"
          "  model scores 100% BY CONSTRUCTION (one scoreable slot, logits -inf-masked),\n"
          "  so its 'gap' there is an artifact -- read the contested row only.")
    cn = sum(split_n["contested"].values())
    if cn:
        c_top1 = sum(split_top1["contested"].values())
        print(f"\nheadline top-1 {100*total_top1/n:.1f}% (all rows) vs "
              f"{100*c_top1/cn:.1f}% (contested only) -- "
              f"delta {100*(total_top1/n - c_top1/cn):+.1f} pt of inflation from "
              f"{100*(n-cn)/n:.2f}% forced rows")


if __name__ == "__main__":
    main()
