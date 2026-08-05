"""Stage-2 (REWEIGHT) plumbing acceptance check — rl_pipeline_v2.md §2.B1.

Streams decisions with the DecisionMeta fields and reports what the
weighting arms will actually consume:

* row- and episode-level outcome distribution (should be ~50/50 win/loss by
  construction — both seats of each match are yielded),
* manifest join coverage (fraction of episodes with an avg_score), and the
  avg_score quartiles, including the Q75 threshold S2-E3 starts from,
* the figure reports/figures/s2_plumbing_outcomes_avgscore.png.

Sources: the raw-JSON split folders (--data-source local) or the ADR-001
Hub shards (--data-source hub) — the restored corpus lives on the Hub, so
the v2 restart audit runs with `--data-source hub --hub-days
2026-07-01,2026-07-26`. 'auto' picks local iff the split folder resolves
AND actually contains episode files (the 08-03 cleanup left dangling
symlink farms behind — counting resolvable files is the rule).

Usage:
    uv run python scripts/check_weight_plumbing.py --max-episodes 500
    uv run python scripts/check_weight_plumbing.py --data-source hub \
        --hub-days 2026-07-01,2026-07-26          # full restored corpus
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pokemon_tcg import config  # noqa: E402
from pokemon_tcg.il_dataset import (  # noqa: E402
    _resolve_hub_shards,
    iter_decisions,
    iter_episode_decisions,
    load_manifest_scores,
    resolve_split_dir,
)


def iter_hub_decisions(split: str, days: list[str] | None, max_episodes: int | None):
    """Yield (obs, label, exclude, meta) from the Hub shards — no encoding,
    so the scan is json.loads-bound, not model-bound."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    files = _resolve_hub_shards(config.HF_EPISODES_REPO, split, days)
    if not files:
        sys.exit(f"no Hub shards for split {split!r}, days {days}")
    n = 0
    for rel in files:
        pf = pq.ParquetFile(hf_hub_download(config.HF_EPISODES_REPO, rel, repo_type="dataset"))
        for rb in pf.iter_batches(batch_size=8, columns=["episode_id", "episode_json"]):
            for eid, raw in zip(
                rb.column(0).to_pylist(), rb.column(1).to_pylist(), strict=True
            ):
                if max_episodes is not None and n >= max_episodes:
                    return
                n += 1
                yield from iter_episode_decisions(json.loads(raw), episode_id=int(eid))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--data-source", choices=["auto", "local", "hub"], default="auto")
    ap.add_argument("--hub-days", default=None,
                    help="comma-separated YYYY-MM-DD list (hub source only)")
    args = ap.parse_args()

    source = args.data_source
    if source == "auto":
        try:
            d = resolve_split_dir(args.split)
            # resolvable files, not directory entries: dangling symlinks from
            # the 08-03 cleanup make an empty dir look populated
            source = "local" if any(
                p.exists() for p in d.glob("*.json")
            ) else "hub"
        except (FileNotFoundError, KeyError):
            source = "hub"

    scores = load_manifest_scores()
    print(f"manifest entries: {len(scores)}")
    if source == "hub":
        days = args.hub_days.split(",") if args.hub_days else None
        print(f"source: hub {config.HF_EPISODES_REPO} split={args.split} days={days or 'all'}")
        decisions = iter_hub_decisions(args.split, days, args.max_episodes)
    else:
        split_dir = resolve_split_dir(args.split)
        print(f"source: local {split_dir}")
        decisions = iter_decisions(split_dir, args.max_episodes)

    row_outcomes: Counter[float] = Counter()
    seat_outcome: dict[tuple[int, int], float] = {}
    n_rows = 0
    for _obs, _label, _exclude, meta in decisions:
        row_outcomes[meta.outcome] += 1
        seat_outcome[(meta.episode_id, meta.seat)] = meta.outcome
        n_rows += 1
        if n_rows % 100_000 == 0:
            print(f"  ...{n_rows} rows", flush=True)

    episodes = sorted({ep for ep, _ in seat_outcome})
    n_eps = len(episodes)
    in_manifest = [ep for ep in episodes if ep in scores]
    ep_scores = np.array([scores[ep] for ep in in_manifest])

    print(f"\nrows: {n_rows}   episodes seen: {n_eps}")
    print("\nrow-level outcome distribution (winner seats yield more rows only if"
          " winners take more decisions per game):")
    name = {1.0: "win", 0.5: "draw", 0.0: "loss", -1.0: "UNKNOWN (no rewards pair)"}
    for oc, cnt in sorted(row_outcomes.items(), reverse=True):
        print(f"  {name.get(oc, oc):>28}: {cnt:>8}  ({100 * cnt / n_rows:.1f}%)")
    unknown = row_outcomes.get(-1.0, 0)
    if unknown:
        print(f"  ⚠️ {unknown} rows have unknown outcome -- weighting arms must exclude them")

    seat_counts = Counter(seat_outcome.values())
    print("\nepisode-seat outcome counts (should be equal win/loss by construction):")
    for oc, cnt in sorted(seat_counts.items(), reverse=True):
        print(f"  {name.get(oc, oc):>28}: {cnt}")

    print(f"\nmanifest join: {len(in_manifest)}/{n_eps} episodes matched "
          f"({100 * len(in_manifest) / max(n_eps, 1):.1f}%)")
    if len(ep_scores):
        q25, q50, q75 = np.percentile(ep_scores, [25, 50, 75])
        print(f"avg_score quartiles: Q25={q25:.1f}  median={q50:.1f}  Q75={q75:.1f}")
        print(f"  -> S2-E3's starting skill filter keeps episodes with avg_score >= {q75:.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = [name.get(oc, str(oc)) for oc in sorted(row_outcomes, reverse=True)]
    counts = [row_outcomes[oc] for oc in sorted(row_outcomes, reverse=True)]
    axes[0].bar(labels, counts, color=["#2a9d8f", "#e9c46a", "#e76f51", "#999999"][: len(labels)])
    axes[0].set_title(f"Row outcomes ({args.split}, {n_eps} episodes)")
    axes[0].set_ylabel("decision rows")
    if len(ep_scores):
        axes[1].hist(ep_scores, bins=40, color="#457b9d")
        axes[1].axvline(q75, color="#e76f51", linestyle="--", label=f"Q75 = {q75:.0f}")
        axes[1].set_title("Episode avg_score (manifest player rating)")
        axes[1].set_xlabel("avg_score")
        axes[1].legend()
    fig.tight_layout()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = config.FIGURES_DIR / "s2_plumbing_outcomes_avgscore.png"
    fig.savefig(out, dpi=150)
    print(f"\nfigure: {out}")


if __name__ == "__main__":
    main()
