"""Backfill 2026-07-26 manifest avg_score via the team-name proxy — v2 §2.B0.

Why this exists: the train day 2026-07-26 has ZERO manifest rows (suspected-
error source #1 of the Stage-2 restart), and BOTH direct recovery paths are
dead ends, verified 2026-08-04:

* `ingest_episodes.py backfill-manifest` walks leaderboard → team-submissions
  → episodes, but the CLI returns ONE page of most-recent episodes per
  submission — a week-old day is beyond page 1 (measured: 44 episodes listed
  for 07-26 across the current top teams, 0 of 4,554 overlapping the corpus).
* The replay JSON itself carries no rating fields (only `info.TeamNames`).

The proxy (registered during corpus discovery as option (b)):
150+ of the ~172 train-day team names also appear on days whose manifest DOES
carry real historical ratings (2026-07-01 train, 2026-07-27 eval). A team's
skill proxy is the MEDIAN of `avg_score` over its rated-day episodes — under
Glicko matchmaking locality (opponents are rating-adjacent) episode avg_score
≈ each participant's own rating, so the median over many episodes is a stable
per-team estimate. A 07-26 episode's approximated avg_score is then the mean
of its two teams' proxies, written to manifest.csv only when BOTH teams are
known — blank means unknown, never a guess (same rule as ingest_episodes).

Known limitations, stated: assumes day-to-day skill stability across ~1-25
days (07-01 ↔ 07-26); proxy scores are on the rated days' scale, not 07-26's
true at-the-time scale. E3's report must chart per-day threshold sensitivity
so a scale artifact can't masquerade as a skill effect (rl_pipeline_v2 §2.B2).

Usage:
    uv run python scripts/backfill_manifest_names.py            # dry report
    uv run python scripts/backfill_manifest_names.py --write    # merge rows
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pokemon_tcg import config  # noqa: E402
from pokemon_tcg.il_dataset import load_manifest_scores  # noqa: E402

# Rated source days -> (split, day); target day is the manifest hole.
SOURCE_DAYS = [("train", "2026-07-01"), ("eval", "2026-07-27")]
TARGET = ("train", "2026-07-26")


def _team_names(raw: bytes) -> list[str] | None:
    """info.TeamNames from a replay JSON, cheap path first.

    The `info` block serializes near the head of the file (keys are ordered:
    configuration, description, id, info, ...), so a bounded substring find
    beats a full ~4MB json.loads by ~100x. Any oddity (a name containing
    ']', missing marker) falls back to the full parse.
    """
    i = raw.find(b'"TeamNames"', 0, 300_000)
    if i >= 0:
        try:
            j = raw.index(b"[", i)
            k = raw.index(b"]", j) + 1
            names = json.loads(raw[j:k])
            if isinstance(names, list) and len(names) == 2:
                return names
        except (ValueError, json.JSONDecodeError):
            pass
    try:
        d = json.loads(raw)
        names = (d.get("info") or {}).get("TeamNames")
        return names if isinstance(names, list) and len(names) == 2 else None
    except (json.JSONDecodeError, AttributeError):
        return None


def _iter_day_rows(split: str, day: str):
    """(episode_id, create_time, raw_bytes) for every episode of a Hub day."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    prefix = f"{split}/day={day}/"
    shards = sorted(
        f for f in HfApi().list_repo_files(config.HF_EPISODES_REPO, repo_type="dataset")
        if f.startswith(prefix) and f.endswith(".parquet")
    )
    if not shards:
        sys.exit(f"no Hub shards under {prefix}")
    for rel in shards:
        pf = pq.ParquetFile(hf_hub_download(config.HF_EPISODES_REPO, rel, repo_type="dataset"))
        for rb in pf.iter_batches(
            batch_size=8, columns=["episode_id", "create_time", "episode_json"]
        ):
            yield from zip(
                rb.column(0).to_pylist(),
                rb.column(1).to_pylist(),
                rb.column(2).to_pylist(),
                strict=True,
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help="merge the proxy rows into manifest.csv (default: report only)")
    ap.add_argument("--min-episodes-per-team", type=int, default=3,
                    help="a team needs at least this many rated episodes for a proxy")
    args = ap.parse_args()

    scores = load_manifest_scores()
    print(f"manifest entries with avg_score: {len(scores)}")

    team_scores: dict[str, list[float]] = defaultdict(list)
    for split, day in SOURCE_DAYS:
        n, rated = 0, 0
        for eid, _ct, raw in _iter_day_rows(split, day):
            n += 1
            sc = scores.get(int(eid))
            if sc is None:
                continue
            names = _team_names(raw)
            if not names:
                continue
            rated += 1
            for name in names:
                team_scores[name].append(sc)
        print(f"{split}/{day}: {n} episodes scanned, {rated} rated with names")

    proxy = {
        name: statistics.median(v)
        for name, v in team_scores.items()
        if len(v) >= args.min_episodes_per_team
    }
    thin = len(team_scores) - len(proxy)
    print(f"team proxies: {len(proxy)} teams (median over rated episodes; "
          f"{thin} teams dropped with <{args.min_episodes_per_team} episodes)")

    split, day = TARGET
    new_rows: dict[int, dict] = {}
    n, both, one, none = 0, 0, 0, 0
    approx: list[float] = []
    for eid, ct, raw in _iter_day_rows(split, day):
        n += 1
        names = _team_names(raw)
        known = [proxy[t] for t in (names or []) if t in proxy]
        if names and len(known) == 2:
            both += 1
            a = sum(known) / 2
            approx.append(a)
            new_rows[int(eid)] = {
                "episode_id": int(eid),
                "create_time": ct,
                "avg_score": f"{a:.6f}",
                "min_score": f"{min(known):.6f}",
                "sum_score": f"{sum(known):.6f}",
                "agent_count": 2,
                "size_bytes": "",
            }
        elif known:
            one += 1
        else:
            none += 1

    print(f"{split}/{day}: {n} episodes; proxy coverage BOTH teams {both} "
          f"({100 * both / max(n, 1):.1f}%), one team {one}, neither {none}")
    if approx:
        qs = statistics.quantiles(approx, n=4)
        print(f"approximated avg_score quartiles: Q25={qs[0]:.1f} "
              f"median={qs[1]:.1f} Q75={qs[2]:.1f}")

    if args.write and new_rows:
        from ingest_episodes import _merge_manifest
        _merge_manifest(new_rows)
    elif new_rows:
        print("(dry run -- re-run with --write to merge into manifest.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
