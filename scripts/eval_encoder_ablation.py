"""Paired held-out-day evaluation for the encoder ablation.

Stage 1 metric of notes/experiments/2026-08-05-modernbert-vs-bert-encoder.md.

Why a dedicated script instead of running eval_rung1.py twelve times: the
comparison has to be PAIRED on identical rows, and eval_rung1 re-streams and
re-encodes the eval day per checkpoint (batch size 1) -- twelve times the
encode cost, and no per-row record to pair on. Here the eval rows are encoded
ONCE into a batched cache, every checkpoint is scored on that same cache, and
we keep the per-row correct/incorrect vector so the arm-vs-baseline delta can
be tested paired (McNemar) instead of as two independent proportions.

Usage:
    uv run python scripts/eval_encoder_ablation.py --max-episodes 400
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pokemon_tcg import config  # noqa: E402
from pokemon_tcg.device import resolve_device  # noqa: E402
from pokemon_tcg.il_dataset import encode_observation, iter_episode_decisions  # noqa: E402
from pokemon_tcg.il_model import PTCGImitationPolicy  # noqa: E402

ARMS = ["a_bert_base", "b_mbert_matched", "c_mbert_xxs", "d_bert_xxsgeom"]
BASELINE_ARM = "a_bert_base"
SEEDS = [42, 43, 44]


def build_eval_cache(split: str, max_episodes: int, repo_id: str | None, batch_size: int):
    """Encode a deterministic prefix of the held-out day into stacked batches.

    Deterministic because _resolve_hub_shards returns sorted shards and we read
    them in pack order -- the same rows, in the same order, for every arm. It is
    a PREFIX of the day, so treat the absolute accuracy as a prefix estimate
    (see the max-episodes-is-a-biased-prefix finding); the paired DELTAS between
    arms are unaffected, which is what this script exists to measure.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from pokemon_tcg.il_dataset import _resolve_hub_shards

    repo = repo_id or config.HF_EPISODES_REPO
    rows, labels, contexts = [], [], []
    n_eps = 0
    for rel in _resolve_hub_shards(repo, split, days=None):
        pf = pq.ParquetFile(hf_hub_download(repo, rel, repo_type="dataset"))
        for rb in pf.iter_batches(batch_size=8, columns=["episode_json"]):
            for raw in rb.to_pydict()["episode_json"]:
                if n_eps >= max_episodes:
                    return _stack(rows, labels, contexts, batch_size)
                n_eps += 1
                for obs, label, exclude, _meta in iter_episode_decisions(json.loads(raw)):
                    feats = encode_observation(obs, exclude=exclude)
                    if feats is None:
                        continue
                    contexts.append(int(feats["select_context"].item()))
                    feats.pop("n_real_options", None)
                    rows.append(feats)
                    labels.append(label)
                if n_eps % 50 == 0 and n_eps:
                    print(f"  ...{n_eps} episodes encoded, {len(rows)} rows", flush=True)
    return _stack(rows, labels, contexts, batch_size)


def _stack(rows, labels, contexts, batch_size):
    if not rows:
        sys.exit("no eval rows encoded -- check the split/repo")
    keys = list(rows[0].keys())
    batches = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        batches.append({k: torch.stack([r[k] for r in chunk]) for k in keys})
    return batches, torch.tensor(labels), torch.tensor(contexts)


@torch.no_grad()
def score(model_dir: Path, batches, device) -> torch.Tensor:
    """Per-row top-1 correctness (bool), in cache order."""
    model = PTCGImitationPolicy.from_pretrained(model_dir).to(device).eval()
    preds = []
    for b in batches:
        logits = model(**{k: v.to(device) for k, v in b.items()})["logits"]
        preds.append(logits.argmax(dim=-1).cpu())
    del model
    return torch.cat(preds)


def mcnemar(a: torch.Tensor, b: torch.Tensor) -> tuple[int, int, float]:
    """Paired test on two correctness vectors. Returns (b_only, a_only, p).

    Normal approximation to the exact binomial on the discordant pairs; with
    tens of thousands of rows the discordant count is far into the regime where
    that is fine, and it degrades to p=1.0 when nothing disagrees.
    """
    b_only = int((b & ~a).sum())  # arm right, baseline wrong
    a_only = int((a & ~b).sum())  # baseline right, arm wrong
    n = b_only + a_only
    if n == 0:
        return b_only, a_only, 1.0
    z = (abs(b_only - a_only) - 1) / math.sqrt(n)
    p = math.erfc(max(z, 0.0) / math.sqrt(2))
    return b_only, a_only, p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", type=Path, default=config.MODELS_DIR / "encoder_ablation")
    ap.add_argument("--eval-split", default="eval")
    ap.add_argument("--max-episodes", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--hub-repo", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=config.REPORTS_DIR / "encoder_ablation.json")
    ap.add_argument("--models", nargs="+", default=None, metavar="LABEL=PATH",
                    help="compare these checkpoints instead of the ARMS x SEEDS "
                         "grid, e.g. 'bert_full=models/il_agent_full_0804' "
                         "'mbert_full=models/il_agent_full_mbert_0805'. The FIRST "
                         "one is the baseline everything else is paired against. "
                         "Used for the Stage 2 production-recipe comparison, "
                         "where each arm is a single run, not a seed family.")
    args = ap.parse_args()

    global ARMS, SEEDS, BASELINE_ARM
    explicit: dict[str, Path] = {}
    if args.models:
        for spec in args.models:
            if "=" not in spec:
                sys.exit(f"--models entry {spec!r} must be LABEL=PATH")
            label, path = spec.split("=", 1)
            explicit[label] = Path(path)
        ARMS = list(explicit)
        SEEDS = [0]  # single run per arm; the key is just f"{label}_s0"
        BASELINE_ARM = ARMS[0]

    device = resolve_device(args.device)
    print(f"device: {device}  eval: {args.eval_split} prefix of {args.max_episodes} episodes")
    batches, labels, contexts = build_eval_cache(
        args.eval_split, args.max_episodes, args.hub_repo, args.batch_size
    )
    n = labels.numel()
    print(f"eval cache: {n} rows in {len(batches)} batches\n")

    correct: dict[str, torch.Tensor] = {}
    meta: dict[str, dict] = {}
    for arm in ARMS:
        for seed in SEEDS:
            key = f"{arm}_s{seed}"
            d = explicit[arm] if explicit else args.models_dir / key
            if not (d / "config.json").exists():
                print(f"MISSING {key} -- skipped")
                continue
            correct[key] = score(d, batches, device) == labels
            cfg = json.loads((d / "config.json").read_text())
            meta[key] = {
                "encoder_type": cfg.get("encoder_type", "bert"),
                "hidden_size": cfg["hidden_size"],
                "num_hidden_layers": cfg["num_hidden_layers"],
                "num_attention_heads": cfg["num_attention_heads"],
                "intermediate_size": cfg["intermediate_size"],
            }
            print(f"{key:<24} top1={correct[key].float().mean():.4f}")

    def arm_seeds(arm):
        return [f"{arm}_s{s}" for s in SEEDS if f"{arm}_s{s}" in correct]

    base_keys = arm_seeds(BASELINE_ARM)
    if not base_keys:
        sys.exit(f"baseline arm {BASELINE_ARM} has no trained seeds -- cannot pair")
    base_accs = [correct[k].float().mean().item() for k in base_keys]
    base_mean = sum(base_accs) / len(base_accs)
    base_sd = (
        math.sqrt(sum((a - base_mean) ** 2 for a in base_accs) / (len(base_accs) - 1))
        if len(base_accs) > 1
        else float("nan")
    )

    print(
        f"\n{'arm':<18} {'seeds':>5} {'top1 mean':>10} {'sd':>8} "
        f"{'Δ vs A':>9} {'seeds>A':>8} {'McNemar p (per seed)':>26}"
    )
    results = {}
    for arm in ARMS:
        keys = arm_seeds(arm)
        if not keys:
            continue
        accs = [correct[k].float().mean().item() for k in keys]
        mean = sum(accs) / len(accs)
        sd = (
            math.sqrt(sum((a - mean) ** 2 for a in accs) / (len(accs) - 1))
            if len(accs) > 1
            else float("nan")
        )
        # Pair seed-to-seed against the baseline arm: same seed, same rows.
        paired, ps, wins = [], [], 0
        for s in SEEDS:
            bk, ak = f"{BASELINE_ARM}_s{s}", f"{arm}_s{s}"
            if bk in correct and ak in correct:
                d = correct[ak].float().mean().item() - correct[bk].float().mean().item()
                paired.append(d)
                wins += d > 0
                ps.append(mcnemar(correct[bk], correct[ak])[2])
        delta = sum(paired) / len(paired) if paired else float("nan")
        results[arm] = {
            "seeds": keys,
            "per_seed_top1": accs,
            "mean_top1": mean,
            "sd_top1": sd,
            "paired_delta_vs_baseline": delta,
            "per_seed_paired_delta": paired,
            "seeds_beating_baseline": wins,
            "mcnemar_p_per_seed": ps,
            "config": meta[keys[0]],
        }
        pstr = ", ".join(f"{p:.3g}" for p in ps)
        print(
            f"{arm:<18} {len(keys):>5} {mean:>10.4f} {sd:>8.4f} "
            f"{delta:>+9.4f} {wins:>4}/{len(paired):<3} {pstr:>26}"
        )

    # Pre-registered decision rule from the experiment card.
    if math.isnan(base_sd):
        print(
            "\nSingle run per arm: no across-seed sd, so the Stage 1 bar cannot be "
            "evaluated here. Read the paired Δ and McNemar p only -- the decision "
            "metric at Stage 2 is the ladder, not this number."
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"eval_rows": n, "arms": results}, indent=2))
        print(f"\nwrote {args.out}")
        return
    print(
        f"\nPre-registered bar: adopt an arm iff mean paired Δ > baseline across-seed "
        f"sd ({base_sd:.4f}) AND ≥2/3 seeds positive."
    )
    for arm, r in results.items():
        if arm == BASELINE_ARM:
            continue
        clears = (
            r["paired_delta_vs_baseline"] > base_sd
            and r["seeds_beating_baseline"] >= 2
        )
        print(f"  {arm:<18} {'CLEARS -> Stage 2' if clears else 'does not clear -> null'}")
        r["clears_preregistered_bar"] = bool(clears)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "eval_split": args.eval_split,
                "eval_episodes_prefix": args.max_episodes,
                "eval_rows": n,
                "baseline_arm": BASELINE_ARM,
                "baseline_across_seed_sd": base_sd,
                "arms": results,
            },
            indent=2,
        )
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
