"""Figures for rl_pipeline_v4 work items 2 (logging half) and 7.

Two charts, each with its control drawn beside the claim:

  kl_dual_reference.png  -- kl_to_prior vs kl_to_il_prior across a smoke run
      that forces a retarget. The CONTROL is the pre-retarget region, where the
      two references are the same policy and the series must coincide exactly;
      if they diverge there, the instrument is wrong. The claim is the
      post-retarget gap.

  pool_prereg_calibration.png -- the frozen local-Glicko -> ladder map with its
      residual band, every anchor it was fit on, and the pre-registered
      candidates plotted at their predicted score with no actual yet. The
      CONTROL is the anchor scatter: the pre-registered points must sit on the
      same line, drawn before any of them reads back.

Usage:
  uv run python scripts/plot_v4_workitems_2_7.py --metrics <smoke>/train_metrics.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pokemon_tcg import config  # noqa: E402
from pool_predictiveness import LADDER_ANCHORS  # noqa: E402
from prereg_pool_prediction import CALIB, PREREG, _fold, _read  # noqa: E402


def plot_kl(metrics: Path, out: Path) -> None:
    rows = [json.loads(ln) for ln in metrics.read_text().splitlines() if ln.strip()]
    ep = [r["epoch"] for r in rows]
    kl = [r.get("kl_to_prior") for r in rows]
    kl_il = [r.get("kl_to_il_prior") for r in rows]
    distinct = [bool(r.get("kl_refs_distinct")) for r in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(ep, kl, "o-", label="kl_to_prior (anchor — retargeted on promotion)")
    ax.plot(ep, kl_il, "s--", label="kl_to_il_prior (original IL prior — never retargeted)")

    # Shade the control region: references identical, series must coincide.
    first_distinct = next((e for e, d in zip(ep, distinct) if d), None)
    if first_distinct is not None:
        ax.axvspan(min(ep) - 0.2, first_distinct - 0.5, alpha=0.10, color="tab:green")
        ax.axvline(first_distinct - 0.5, color="tab:red", ls=":", lw=1.5)
        ax.text(first_distinct - 0.45, max(filter(None, kl_il)) * 0.92,
                " retarget", color="tab:red", fontsize=9, va="top")
        ax.text(min(ep) - 0.15, max(filter(None, kl_il)) * 0.92,
                " CONTROL: same reference,\n series must coincide",
                fontsize=8, va="top", color="darkgreen")

    ax.set_yscale("log")
    ax.set_xlabel("update (epoch)")
    ax.set_ylabel("mean nats per decision (log scale)")
    ax.set_title("Two KL references: after a retarget the anchor reads ~0\n"
                 "while drift from the human prior is the whole story")
    ax.set_xticks(ep)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3, which="both")
    fig.text(0.5, 0.005,
             "3-update CPU smoke run (--smoke, forced gate) — an instrument "
             "check, NOT a training result.",
             ha="center", fontsize=7.5, style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def plot_prereg(source: Path, out: Path) -> None:
    g = json.loads(source.read_text())
    pts = [(n, g[n]["rating"], LADDER_ANCHORS[n][0]) for n in g if n in LADDER_ANCHORS]
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    a, b, sd = CALIB["intercept"], CALIB["slope"], CALIB["residual_sd"]

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.scatter(xs, ys, s=34, color="tab:blue", zorder=3,
               label=f"anchors the map was fit on (n={len(pts)})")
    lo, hi = min(xs) - 40, max(xs) + 80
    line = [lo, hi]
    ax.plot(line, [a + b * v for v in line], color="tab:blue", lw=1.4,
            label=CALIB["fit"])
    ax.fill_between(line, [a + b * v - 2 * sd for v in line],
                    [a + b * v + 2 * sd for v in line],
                    alpha=0.12, color="tab:blue",
                    label=f"±2 residual SD (±{2 * sd:.0f})")

    folded = _fold(_read())
    for r in sorted(folded.values(), key=lambda r: -r["local_glicko"]):
        ax.scatter([r["local_glicko"]], [r["predicted_ladder"]], marker="*",
                   s=210, color="tab:red", zorder=4)
        ax.annotate(r["candidate"], (r["local_glicko"], r["predicted_ladder"]),
                    textcoords="offset points", xytext=(6, -11), fontsize=8,
                    color="tab:red")
    if folded:
        ax.scatter([], [], marker="*", s=140, color="tab:red",
                   label="PRE-REGISTERED, no ladder read yet")

    ax.set_xlabel("local pool Glicko (compounded scale, frozen snapshot)")
    ax.set_ylabel("ladder score")
    ax.set_title("Pre-registered predictions, locked before any read-back\n"
                 f"in-sample ρ = {CALIB['in_sample_rho']:+.3f} "
                 f"(n={CALIB['n_anchors']}, p={CALIB['in_sample_p']:.4f}) — "
                 "out-of-sample is what these stars will test")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.text(0.5, 0.005,
             "Most anchors are self-reported, not ledger-verified; the ordinal "
             "prediction is the registered claim, not the point estimate.",
             ha="center", fontsize=7.5, style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--source", type=Path,
                    default=config.REPORTS_DIR / "pool_prereg_source_glicko.json")
    args = ap.parse_args()
    figs = config.FIGURES_DIR
    figs.mkdir(parents=True, exist_ok=True)
    plot_kl(args.metrics, figs / "kl_dual_reference.png")
    plot_prereg(args.source, figs / "pool_prereg_calibration.png")
    print(f"pre-registration file: {PREREG}")


if __name__ == "__main__":
    main()
