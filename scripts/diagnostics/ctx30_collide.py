"""Hypothesis (b) test: for ctx-30 (DISCARD_ENERGY), are distinct legal options
encoded into IDENTICAL feature rows, and does the model therefore emit
identical logits for them?

Encodes every ctx-30 decision on the eval day, then for each row compares the
per-option feature slices bitwise and the model's logits.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pokemon_tcg import config  # noqa: E402
from pokemon_tcg.device import resolve_device  # noqa: E402
from pokemon_tcg.il_dataset import (  # noqa: E402
    encode_observation, iter_episode_decisions, _resolve_hub_shards,
)
from pokemon_tcg.il_model import PTCGImitationPolicy  # noqa: E402

OPT_KEYS = ["opt_type", "opt_attack", "opt_special", "opt_scalar",
            "opt_ref_card_id", "opt_ref_scalar", "extra_opt"]


def hub_decisions(split, max_episodes):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    repo = config.HF_EPISODES_REPO
    n = 0
    for rel in _resolve_hub_shards(repo, split, days=None):
        pf = pq.ParquetFile(hf_hub_download(repo, rel, repo_type="dataset"))
        for rb in pf.iter_batches(batch_size=8, columns=["episode_json"]):
            for raw in rb.to_pydict()["episode_json"]:
                if max_episodes is not None and n >= max_episodes:
                    print(f"episodes consumed: {n}")
                    return
                n += 1
                for t in iter_episode_decisions(json.loads(raw)):
                    yield (t[0], t[1], t[2], t[3]._replace(episode_id=n))
    print(f"episodes consumed: {n}")


def opt_signature(feats, i):
    """Bitwise signature of option slot i across every option feature tensor."""
    parts = []
    for k in OPT_KEYS:
        v = feats[k]
        parts.append(tuple(v[i].flatten().tolist()) if v.dim() > 1 else (v[i].item(),))
    return tuple(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=config.MODELS_DIR / "il_agent")
    ap.add_argument("--eval-split", default="eval")
    ap.add_argument("--max-episodes", type=int, default=100)
    ap.add_argument("--ctx", type=int, default=30)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = resolve_device(None)
    model = PTCGImitationPolicy.from_pretrained(args.model_dir).to(device).eval()

    recs = []
    with torch.no_grad():
        for obs, label, exclude, meta in hub_decisions(args.eval_split, args.max_episodes):
            feats = encode_observation(obs, exclude=exclude)
            if feats is None or int(feats["select_context"].item()) != args.ctx:
                continue
            n_real = int(feats.pop("n_real_options"))
            batch = {k: v.unsqueeze(0).to(device) for k, v in feats.items()}
            logits = model(**batch)["logits"][0]

            sigs = [opt_signature(feats, i) for i in range(n_real)]
            groups = {}
            for i, s in enumerate(sigs):
                groups.setdefault(s, []).append(i)
            # logit spread inside each identical-feature group
            max_spread = 0.0
            for idxs in groups.values():
                if len(idxs) > 1:
                    lv = [float(logits[i]) for i in idxs]
                    max_spread = max(max_spread, max(lv) - min(lv))
            label_group = groups[sigs[label]] if label < n_real else [label]
            recs.append({
                "episode": meta.episode_id, "turn": meta.turn, "seat": meta.seat,
                "n_real": n_real, "label": label,
                "top1": int(logits.argmax()),
                "n_distinct_sigs": len(groups),
                "largest_group": max(len(v) for v in groups.values()),
                "fully_degenerate": len(groups) == 1 and n_real > 1,
                "max_logit_spread_within_group": max_spread,
                "label_group": label_group,
                "top1_in_label_group": int(logits.argmax()) in label_group,
                "raw_opts": obs["select"]["option"][:n_real],
            })

    n = len(recs)
    multi = [r for r in recs if r["n_real"] > 1]
    deg = [r for r in multi if r["fully_degenerate"]]
    part = [r for r in multi if not r["fully_degenerate"] and r["largest_group"] > 1]
    print(f"\nctx {args.ctx} rows: {n}   forced(n=1): {n-len(multi)}   multi-option: {len(multi)}")
    print(f"  fully degenerate (ALL options encode identically): {len(deg)}/{len(multi)}")
    print(f"  partially degenerate: {len(part)}/{len(multi)}")
    print(f"  no collision at all: {len(multi)-len(deg)-len(part)}/{len(multi)}")
    spreads = [r["max_logit_spread_within_group"] for r in multi if r["largest_group"] > 1]
    if spreads:
        print(f"  max logit spread within an identical-feature group: {max(spreads):.3e}")
    print(f"\n  top-1 correct overall: {sum(r['top1']==r['label'] for r in recs)}/{n}")
    print(f"  top-1 correct, multi-option: {sum(r['top1']==r['label'] for r in multi)}/{len(multi)}")
    print(f"  top-1 lands in the label's identical-feature group (multi): "
          f"{sum(r['top1_in_label_group'] for r in multi)}/{len(multi)}")
    print(f"  top-1 correct, fully degenerate: {sum(r['top1']==r['label'] for r in deg)}/{len(deg)}")
    print(f"  label dist | fully degenerate: {Counter(r['label'] for r in deg).most_common()}")
    args.out.write_text(json.dumps(recs, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
