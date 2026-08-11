"""Fit the leaf-value transform for search: Platt scaling + centering.

Stage 3 requires the MCTS leaf value to be **centered**. The reason is
mechanical, not cosmetic: `search_prior_mcts._create_node` flips the leaf's
sign at opponent-to-move nodes (`v = value if yourIndex == your_index else
-value`, the same `v = -v` kiyotah's reference does). A constant component `b`
in the leaf therefore enters the tree as `+b` on our plies and `-b` on theirs,
so it does not cancel — it becomes a systematic preference for lines that end
on one turn parity. Already measured once here: a constant +0.084 leaf moved
**10.5%** of decisions (memory `search-leaf-value-must-be-centered`).

Two transforms are fitted, in this order:

1. **Platt scaling** `p = sigmoid(a * logit(clip(v_raw)) + b)` — the critic's
   head is a raw linear output with ECE 0.062 and the standing verdict
   "useful: true, calibrated: false". Platt is the minimal monotone fix: it
   cannot change AUC (rank order is preserved), only the probabilities.
2. **Centering** `v_leaf = clamp(2 * (p - center), -1, 1)` where `center` is
   the mean calibrated prediction over the decision distribution. With
   `center = 0.5` this is exactly the current shipped mapping, so the default
   is backwards compatible and the change is auditable.

**Known limitation (E2b manipulation check, 2026-08-06): this fits `center`
on root decision states, but the leaf value is consumed at internal tree
nodes — a deeper, perspective-alternating population.** A root-fitted centre
once left an in-play leaf mean of +0.2382 while looking unbiased on roots.
Before quoting any critic-in-the-loop arm, record `leaf_value_in_play` from
the actual search and require |mean| <= 0.01 (memory
`search-leaf-value-must-be-centered`).

**Fit/report split.** Parameters are fitted on episodes with even
`episode_id` and every reported number comes from the odd half. Fitting and
reporting on the same rows would make the calibration look better than it is,
which is the whole failure mode this file exists to avoid.

The shuffled-label control critic is run through the identical pipeline; if
centering or Platt "improves" the control, the measurement is broken.

Usage (critic_trainday is LOST — dangling worktree symlinks, no HF copy; the
retrained audited critic is critic_outcome_day_2026-07-26_seed42, backed up
on Rami/ptcg-s2v2-arms):
  uv run python scripts/fit_critic_calibration.py \
      --critic-dir models/critic_outcome_day_2026-07-26_seed42
  uv run python scripts/fit_critic_calibration.py --critic-dir <shuffled-control-dir> \
      --out reports/critic_leafcal_shuffled.json --no-write-calibration
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

CALIBRATION_FILE = "calibration.json"
_EPS = 1e-6


def _auc(pred: np.ndarray, y: np.ndarray) -> float:
    pos, neg = pred[y == 1.0], pred[y == 0.0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allp = np.concatenate([pos, neg])
    order = allp.argsort(kind="mergesort")
    ranks = np.empty(len(allp))
    ranks[order] = np.arange(1, len(allp) + 1)
    for v in np.unique(allp):
        m = allp == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def _ece(pred: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (pred >= lo) & (pred < hi if i < n_bins - 1 else pred <= hi)
        if m.sum():
            total += (m.sum() / len(pred)) * abs(pred[m].mean() - y[m].mean())
    return float(total)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def fit_platt(raw: np.ndarray, y: np.ndarray, iters: int = 400) -> tuple[float, float]:
    """Fit sigmoid(a*z + b) by NLL, z = logit(clip(raw, 0, 1)). LBFGS, 2 params."""
    z = torch.tensor(_logit(np.clip(raw, 0.0, 1.0)), dtype=torch.float64)
    t = torch.tensor(y, dtype=torch.float64)
    a = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([a, b], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(a * z + b, t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(a.detach()), float(b.detach())


def apply_platt(raw: np.ndarray, a: float, b: float) -> np.ndarray:
    z = _logit(np.clip(raw, 0.0, 1.0))
    return 1.0 / (1.0 + np.exp(-(a * z + b)))


def _metrics(pred: np.ndarray, y: np.ndarray, base_rate: float) -> dict:
    wl = (y == 0.0) | (y == 1.0)
    mse = float(((pred - y) ** 2).mean())
    mse_base = float(((base_rate - y) ** 2).mean())
    return {
        "mse": mse,
        "mse_const_base_rate": mse_base,
        "mse_const_half": float(((0.5 - y) ** 2).mean()),
        "rel_gain_vs_base_rate": float((mse_base - mse) / mse_base) if mse_base else 0.0,
        "auc": _auc(pred[wl], y[wl]),
        "ece_10bin": _ece(pred[wl], y[wl]),
        "mean_pred": float(pred.mean()),
        # The quantity search actually consumes, BEFORE centering:
        "mean_leaf_uncentered": float(np.clip(2.0 * (pred - 0.5), -1, 1).mean()),
    }


def stream_rows(critic, args, device) -> tuple[np.ndarray, ...]:
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

    preds, outs, eids = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(DataLoader(ds, batch_size=args.batch_size)):
            if i >= args.eval_batches:
                break
            feats = {k: v.to(device) for k, v in batch.items()
                     if k not in ILDataset.META_KEYS and k != "label"}
            preds.append(critic(**feats).float().cpu().numpy())
            outs.append(batch["outcome"].numpy())
            eids.append(batch["episode_id"].numpy())
            if (i + 1) % 500 == 0:
                print(f"  {(i + 1) * args.batch_size} rows...", flush=True)
    raw = np.concatenate(preds)
    y = np.concatenate(outs)
    eid = np.concatenate(eids)
    known = y >= 0.0
    return raw[known], y[known], eid[known], source


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--critic-dir", type=Path, required=True)
    ap.add_argument("--eval-split", default="eval")
    ap.add_argument("--data-source", choices=["auto", "local", "hub"], default="auto")
    ap.add_argument("--hub-repo", default=None)
    ap.add_argument("--max-eval-episodes", type=int, default=None)
    ap.add_argument("--eval-batches", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-write-calibration", action="store_true",
                    help="report only; do not write calibration.json into the critic dir")
    args = ap.parse_args()

    device = resolve_device(args.device)
    critic = load_critic(args.critic_dir, device=device)
    print(f"critic: {args.critic_dir}  device: {device}")

    raw, y, eid, source = stream_rows(critic, args, device)
    fit_m = (eid % 2) == 0          # fit half
    rep_m = ~fit_m                  # report half
    print(f"\n{len(y)} known-outcome rows: {fit_m.sum()} fit / {rep_m.sum()} report "
          f"(split by episode_id parity)")
    if fit_m.sum() < 1000 or rep_m.sum() < 1000:
        raise SystemExit("too few rows in one half to fit/report honestly")

    base_rate = float(y[rep_m].mean())

    # --- arm 0: raw, mapped the way search currently does it -----------------
    m_raw = _metrics(np.clip(raw[rep_m], 0.0, 1.0), y[rep_m], base_rate)

    # --- arm 1: Platt-scaled (monotone; AUC must be unchanged) ---------------
    a, b = fit_platt(raw[fit_m], y[fit_m])
    cal_rep = apply_platt(raw[rep_m], a, b)
    m_platt = _metrics(cal_rep, y[rep_m], base_rate)

    # --- centering: mean calibrated prediction on the FIT half ---------------
    center = float(apply_platt(raw[fit_m], a, b).mean())
    leaf_centered = np.clip(2.0 * (cal_rep - center), -1.0, 1.0)
    leaf_uncentered = np.clip(2.0 * (cal_rep - 0.5), -1.0, 1.0)

    result = {
        "critic_dir": str(args.critic_dir),
        "critic_metadata": json.loads((args.critic_dir / "train_metadata.json").read_text())
        if (args.critic_dir / "train_metadata.json").exists() else None,
        "eval_split": args.eval_split,
        "data_source": source,
        "n_rows_total": int(len(y)),
        "n_rows_fit": int(fit_m.sum()),
        "n_rows_report": int(rep_m.sum()),
        "base_rate_report_half": base_rate,
        "platt": {"a": a, "b": b},
        "center": center,
        "raw": m_raw,
        "platt_scaled": m_platt,
        "leaf": {
            "mean_uncentered": float(leaf_uncentered.mean()),
            "mean_centered": float(leaf_centered.mean()),
            "std_centered": float(leaf_centered.std()),
            "abs_bias_removed": abs(float(leaf_uncentered.mean())),
        },
        "verdict": {
            # Same pre-registered thresholds as eval_critic_calibration.py.
            "useful": bool(m_platt["rel_gain_vs_base_rate"] >= 0.05 and m_platt["auc"] >= 0.60),
            "calibrated_after_platt": bool(m_platt["ece_10bin"] <= 0.05),
            "drop": bool(m_platt["auc"] < 0.55),
        },
    }

    out = args.out or (config.REPORTS_DIR / f"critic_leafcal_{args.critic_dir.name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    if not args.no_write_calibration:
        (args.critic_dir / CALIBRATION_FILE).write_text(json.dumps({
            "platt_a": a, "platt_b": b, "center": center,
            "fitted_on": f"{args.eval_split} even-episode_id half, n={int(fit_m.sum())}",
            "source_report": str(out),
        }, indent=2))
        print(f"wrote {args.critic_dir / CALIBRATION_FILE}")

    print(f"\nbase rate (report half) {base_rate:.4f}")
    print(f"{'':<14}{'MSE':>9}{'rel gain':>10}{'AUC':>8}{'ECE':>8}{'mean p':>9}")
    for name, m in (("raw", m_raw), ("platt", m_platt)):
        print(f"{name:<14}{m['mse']:>9.4f}{m['rel_gain_vs_base_rate']:>9.1%}"
              f"{m['auc']:>8.4f}{m['ece_10bin']:>8.4f}{m['mean_pred']:>9.4f}")
    print(f"const base-rate{m_raw['mse_const_base_rate']:>9.4f}{0.0:>9.1%}"
          f"{0.5:>8.4f}{'-':>8}{base_rate:>9.4f}")
    print(f"\nPlatt a={a:.4f} b={b:.4f}   center={center:.4f}")
    print(f"leaf value mean: uncentered {leaf_uncentered.mean():+.4f} -> "
          f"centered {leaf_centered.mean():+.4f}  (std {leaf_centered.std():.4f})")
    print(f"verdict: {result['verdict']}")
    print(f"written -> {out}")


if __name__ == "__main__":
    main()
