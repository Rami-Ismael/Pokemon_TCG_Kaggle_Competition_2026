"""Is the dropped `energyIndex` decision-relevant, or a genuine tie?

For each fully-degenerate ctx-30 row, resolve each option's (area,index,
playerIndex,energyIndex) to the actual attached energy card and compare types.
If every colliding option points at the SAME energy type, the choice is
game-theoretically arbitrary (hypothesis a). If types differ, the encoder is
dropping real information (hypothesis b).
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
    encode_observation, iter_episode_decisions, _resolve_hub_shards, _get_card,
)
from pokemon_tcg.il_model import PTCGImitationPolicy  # noqa: E402

sys.path.insert(0, str(config.PROJECT_ROOT / "data" / "external" / "cg-lib"))
from cg.api import State, SelectData, EnergyType  # noqa: E402
from cg.utils import to_dataclass  # noqa: E402


def hub_decisions(split, max_episodes):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    n = 0
    for rel in _resolve_hub_shards(config.HF_EPISODES_REPO, split, days=None):
        pf = pq.ParquetFile(hf_hub_download(config.HF_EPISODES_REPO, rel, repo_type="dataset"))
        for rb in pf.iter_batches(batch_size=8, columns=["episode_json"]):
            for raw in rb.to_pydict()["episode_json"]:
                if max_episodes is not None and n >= max_episodes:
                    print(f"episodes consumed: {n}")
                    return
                n += 1
                for t in iter_episode_decisions(json.loads(raw)):
                    yield (t[0], t[1], t[2], t[3]._replace(episode_id=n))
    print(f"episodes consumed: {n}")


def energies_of(state, select, o, my_index):
    """The energy-type list attached to the Pokemon this option targets."""
    pi = o.playerIndex if o.playerIndex is not None else my_index
    card = _get_card(state, select, o.area, o.index, pi)
    return list(getattr(card, "energies", None) or []), card


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=config.MODELS_DIR / "il_agent")
    ap.add_argument("--eval-split", default="eval")
    ap.add_argument("--max-episodes", type=int, default=100)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = resolve_device(None)
    model = PTCGImitationPolicy.from_pretrained(args.model_dir).to(device).eval()

    recs, same_type, diff_type, unresolved = [], 0, 0, 0
    own, opp = 0, 0
    with torch.no_grad():
        for obs, label, exclude, meta in hub_decisions(args.eval_split, args.max_episodes):
            feats = encode_observation(obs, exclude=exclude)
            if feats is None or int(feats["select_context"].item()) != 30:
                continue
            n_real = int(feats.pop("n_real_options"))
            if n_real < 2:
                continue
            batch = {k: v.unsqueeze(0).to(device) for k, v in feats.items()}
            top1 = int(model(**batch)["logits"][0].argmax())

            state = to_dataclass(obs["current"], State)
            select = to_dataclass(obs["select"], SelectData)
            my_index = state.yourIndex
            per_opt = []
            for o in select.option[:n_real]:
                ens, card = energies_of(state, select, o, my_index)
                ei = o.energyIndex
                et = ens[ei] if (ei is not None and ei < len(ens)) else None
                per_opt.append({
                    "energyIndex": ei, "energy_type": et,
                    "attached": ens,
                    "target_card_id": getattr(card, "id", None),
                    "playerIndex": o.playerIndex, "area": o.area, "index": o.index,
                })
            pis = {p["playerIndex"] for p in per_opt}
            if pis == {my_index}:
                own += 1
            elif pis == {1 - my_index}:
                opp += 1

            types = [p["energy_type"] for p in per_opt]
            if any(t is None for t in types):
                unresolved += 1
                kind = "unresolved"
            elif len(set(types)) == 1:
                same_type += 1
                kind = "all same energy type (genuine tie)"
            else:
                diff_type += 1
                kind = "energy types DIFFER (dropped info matters)"
            recs.append({"episode": meta.episode_id, "turn": meta.turn,
                         "label": label, "top1": top1, "n_real": n_real,
                         "kind": kind, "types": types,
                         "label_type": types[label] if label < len(types) else None,
                         "top1_type": types[top1] if top1 < len(types) else None,
                         "per_opt": per_opt})

    n = len(recs)
    print(f"\nmulti-option ctx-30 rows: {n}")
    print(f"  all colliding options point at the SAME energy type : {same_type}  ({same_type/n:.1%})")
    print(f"  energy types DIFFER across options                  : {diff_type}  ({diff_type/n:.1%})")
    print(f"  unresolved (could not read attached energies)        : {unresolved}")
    print(f"\n  target is own Pokemon: {own}   opponent's Pokemon: {opp}   mixed/other: {n-own-opp}")

    dt = [r for r in recs if r["kind"].startswith("energy types DIFFER")]
    if dt:
        acc = sum(r["top1"] == r["label"] for r in dt)
        print(f"\n  on the {len(dt)} rows where type actually differs: model top1 correct {acc}/{len(dt)}")
        print(f"    label energy type dist: {Counter(r['label_type'] for r in dt).most_common()}")
        print(f"    model  energy type dist: {Counter(r['top1_type'] for r in dt).most_common()}")
    st = [r for r in recs if r["kind"].startswith("all same")]
    if st:
        acc = sum(r["top1"] == r["label"] for r in st)
        print(f"\n  on the {len(st)} genuine-tie rows: model top1 'correct' {acc}/{len(st)} = {acc/len(st):.1%}")
        print(f"    (any policy is equally good in the game; only index-matching differs)")
    args.out.write_text(json.dumps(recs, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
