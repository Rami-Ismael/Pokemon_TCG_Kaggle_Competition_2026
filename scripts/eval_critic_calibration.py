"""Calibration audit for an offline critic V(s) — experiment
notes/experiments/2026-08-05-critic-calibration-sparse-reward.md.

Measures, on the held-out eval day (real outcomes, never the relabeled
control stream):

- MSE vs BOTH constant baselines (0.5 and the eval base rate). A critic that
  cannot beat "predict the average" has learned nothing per-state.
- Win/loss AUC (rank-based). Ties are excluded; measured tie mass is ~0
  (0 ties in 656 benchmark games + 60 raw episodes, 2026-08-05).
- 10-bin reliability table + ECE on predictions clipped to [0, 1]. The head
  is a raw linear output; clipping is monotone so AUC is unaffected, and the
  clip fraction is reported so silent saturation is visible.
- Phase profile: rows bucketed by game turn quartile. A critic that reads
  the board sharpens toward the end of the game (AUC and sharpness rise
  Q1 -> Q4); a flat profile means it only learned the base rate.

Pre-registered decisions (from the experiment card):
  useful      = beats base-rate MSE by >=5% relative AND AUC >= 0.60
  calibrated  = ECE <= 0.05
  drop        = AUC < 0.55

Usage:
  uv run python scripts/eval_critic_calibration.py --critic-dir models/critic_alldays \
      --out reports/critic_calibration_alldays.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pokemon_tcg import config  # noqa: E402
from pokemon_tcg.device import resolve_device  # noqa: E402
from pokemon_tcg.il_dataset import (  # noqa: E402
    ILDataset,
    ShardILDataset,
    resolve_split_dir,
    split_meta,
)
from pokemon_tcg.offline_critic import load_critic  # noqa: E402


def _auc(pred: np.ndarray, y: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney), ties in pred handled by midranks."""
    pos, neg = pred[y == 1.0], pred[y == 0.0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allp = np.concatenate([pos, neg])
    order = allp.argsort(kind="mergesort")
    ranks = np.empty(len(allp))
    ranks[order] = np.arange(1, len(allp) + 1)
    # midranks for exact pred ties
    for v in np.unique(allp):
        m = allp == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def _ece_table(pred: np.ndarray, y: np.ndarray, n_bins: int = 10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows, ece = [], 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (pred >= lo) & (pred < hi if i < n_bins - 1 else pred <= hi)
        if m.sum() == 0:
            rows.append({"bin": [float(lo), float(hi)], "n": 0})
            continue
        mp, fw = float(pred[m].mean()), float(y[m].mean())
        rows.append({"bin": [float(lo), float(hi)], "n": int(m.sum()),
                     "mean_pred": mp, "frac_win": fw})
        ece += (m.sum() / len(pred)) * abs(mp - fw)
    return rows, float(ece)


def _bucket_metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    _, ece = _ece_table(pred, y)
    return {
        "n": int(len(y)),
        "auc": _auc(pred, y),
        "sharpness_mean_abs_dev_from_half": float(np.abs(pred - 0.5).mean()),
        "ece": ece,
        "base_rate": float(y.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--critic-dir", type=Path, required=True)
    ap.add_argument("--eval-split", default="eval")
    ap.add_argument("--data-source", choices=["auto", "local", "hub"], default="auto")
    ap.add_argument("--hub-repo", default=None)
    ap.add_argument("--max-eval-episodes", type=int, default=None)
    ap.add_argument("--eval-batches", type=int, default=4000,
                    help="cap on eval batches (default 4000 x 64 = 256k rows)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=None,
                    help="default reports/critic_calibration_<critic-dir-name>.json")
    args = ap.parse_args()

    device = resolve_device(args.device)
    critic = load_critic(args.critic_dir, device=device)
    print(f"critic: {args.critic_dir}  device: {device}")

    source = args.data_source
    if source == "auto":
        try:
            _, _, expected = split_meta(args.eval_split)
            d = resolve_split_dir(args.eval_split)
            n = sum(1 for p in d.glob("*.json") if p.exists()) if d.is_dir() else 0
            source = "local" if n == expected else "hub"
        except (FileNotFoundError, KeyError):
            source = "hub"
    if source == "hub":
        kind, days, _ = split_meta(args.eval_split)
        ds = ShardILDataset(kind, days=days, repo_id=args.hub_repo,
                            max_episodes=args.max_eval_episodes,
                            shuffle_buffer=1, with_meta=True)
        print(f"eval source: hub days {','.join(days)}")
    else:
        ds = ILDataset(resolve_split_dir(args.eval_split),
                       max_episodes=args.max_eval_episodes,
                       shuffle_buffer=1, with_meta=True)
        print(f"eval source: local {resolve_split_dir(args.eval_split)}")

    preds, outs, turns = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(DataLoader(ds, batch_size=args.batch_size)):
            if i >= args.eval_batches:
                break
            feats = {k: v.to(device) for k, v in batch.items()
                     if k not in ILDataset.META_KEYS and k != "label"}
            v = critic(**feats)
            preds.append(v.float().cpu().numpy())
            outs.append(batch["outcome"].numpy())
            turns.append(batch["turn"].numpy())
            if (i + 1) % 500 == 0:
                print(f"  {(i + 1) * args.batch_size} rows...", flush=True)

    raw = np.concatenate(preds)
    y = np.concatenate(outs)
    turn = np.concatenate(turns)
    known = y >= 0.0
    raw, y, turn = raw[known], y[known], turn[known]
    pred = np.clip(raw, 0.0, 1.0)
    clip_frac = float(((raw < 0) | (raw > 1)).mean())
    base_rate = float(y.mean())

    mse_model = float(((pred - y) ** 2).mean())
    mse_raw = float(((raw - y) ** 2).mean())
    mse_half = float(((0.5 - y) ** 2).mean())
    mse_base = float(((base_rate - y) ** 2).mean())

    wl = (y == 0.0) | (y == 1.0)  # ties excluded from AUC/ECE
    auc = _auc(pred[wl], y[wl])
    bins, ece = _ece_table(pred[wl], y[wl])

    q = np.quantile(turn[turn >= 0], [0.25, 0.5, 0.75])
    phase = {}
    labels = [f"turn<= {q[0]:.0f}", f"turn {q[0]:.0f}-{q[1]:.0f}",
              f"turn {q[1]:.0f}-{q[2]:.0f}", f"turn > {q[2]:.0f}"]
    masks = [turn <= q[0], (turn > q[0]) & (turn <= q[1]),
             (turn > q[1]) & (turn <= q[2]), turn > q[2]]
    for lab, m in zip(labels, masks):
        mm = m & wl
        if mm.sum():
            phase[lab] = _bucket_metrics(pred[mm], y[mm])

    rel_gain_vs_base = (mse_base - mse_model) / mse_base if mse_base else 0.0
    verdict = {
        "useful": bool(rel_gain_vs_base >= 0.05 and auc >= 0.60),
        "calibrated": bool(ece <= 0.05),
        "drop": bool(auc < 0.55),
    }

    result = {
        "critic_dir": str(args.critic_dir),
        "critic_metadata": json.loads(
            (args.critic_dir / "train_metadata.json").read_text()
        ) if (args.critic_dir / "train_metadata.json").exists() else None,
        "eval_split": args.eval_split,
        "data_source": source,
        "n_rows": int(len(y)),
        "n_tie_rows": int((y == 0.5).sum()),
        "base_rate": base_rate,
        "clip_fraction": clip_frac,
        "mse": {"model_clipped": mse_model, "model_raw": mse_raw,
                "const_0.5": mse_half, "const_base_rate": mse_base,
                "rel_gain_vs_base_rate": float(rel_gain_vs_base)},
        "auc_win_loss": auc,
        "ece_10bin": ece,
        "reliability_bins": bins,
        "phase_by_turn_quartile": phase,
        "verdict": verdict,
    }
    out = args.out or (config.REPORTS_DIR
                       / f"critic_calibration_{args.critic_dir.name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print(f"\nrows {len(y)}  (ties {result['n_tie_rows']})  base_rate {base_rate:.3f}  "
          f"clip_frac {clip_frac:.3f}")
    print(f"MSE  model {mse_model:.4f}  vs const-0.5 {mse_half:.4f}  "
          f"vs base-rate {mse_base:.4f}  (rel gain {rel_gain_vs_base:+.1%})")
    print(f"AUC {auc:.4f}   ECE {ece:.4f}")
    for lab in phase:
        p = phase[lab]
        print(f"  {lab:>16}: auc {p['auc']:.3f}  sharp {p['sharpness_mean_abs_dev_from_half']:.3f}  "
              f"ece {p['ece']:.3f}  (n {p['n']})")
    print(f"verdict: {verdict}")
    print(f"written -> {out}")


if __name__ == "__main__":
    main()
