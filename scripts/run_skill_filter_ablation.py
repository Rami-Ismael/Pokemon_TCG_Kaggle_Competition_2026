"""Skill-filtered demonstrations ablation (pivot from the feature ablation).

Question: does behavior-cloning on episodes played by higher-rated players
(min_score = the LOWER of the two players' ratings, so BOTH POVs cloned
from a decision are above threshold) beat cloning everything, at equal
training steps?

Motivated by the feature-ablation negative result
(reports/feature_ablation/report.md): BC accuracy rewards imitating the
logged population, so making the logged population BETTER is the lever
that feature engineering wasn't.

Arms (equal --total-steps, seeds shared with the feature sweep). Filtering
is WITHIN-DAY: a pooled threshold turned out to be a day filter in
disguise (top50 came out 91% from 2026-07-01 because that day's ratings
run ~60 points hotter -- measured 2026-08-04, one arm burned before the
confound was caught), so every arm now holds the day mix fixed:
- unfiltered_scored: every scored episode on every scored day
- top50_wd / top25_wd: within each day, episodes at or above that day's
  own median / 75th percentile of min_score. Fewer episodes, repeated
  more -- that trade IS the treatment (equal steps, standing rule #4).

Scores come from the shard columns where present, with a manifest.csv
fallback join on episode_id (the 2026-07-26 shards were packed before
score-joining). Days with no scores anywhere (2026-07-28..2026-08-01,
2026-08-03) are excluded from EVERY arm, including the baseline.

Gate metric (both reported, per arm): eval_rung1 on the held-out day,
(a) unfiltered eval episodes -- comparable to every previous number, and
(b) the top-25%-min_score subset of the eval day -- "does it imitate
strong players better?", which is the outcome the filter is aimed at.
Accept-for-full-retrain iff the paired mean improvement on metric (b)
exceeds the across-seed spread AND metric (a) does not collapse (> -1pp
mean). The ship decision after a full-budget retrain belongs to
scripts/benchmark_agents.py, never to accuracy.

Usage:
    uv run python scripts/run_skill_filter_ablation.py
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from pokemon_tcg import config  # noqa: E402

SEEDS = [42, 43, 44]
TOTAL_STEPS = 4000
EVAL_EPISODES = 150

OUT_ROOT = REPO / "reports" / "skill_filter"
RESULTS_DIR = OUT_ROOT / "results"
IDS_DIR = OUT_ROOT / "allowlists"
MODELS_DIR = config.MODELS_DIR / "skill_filter"
RUNS_DIR = REPO / "runs" / "skill_filter"

_GLOBAL_RE = re.compile(r"GLOBAL\s+(\d+)\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%")


def collect_scores() -> dict[int, tuple[str, float]]:
    """episode_id -> (day, min_score) for every scored train episode on the Hub."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from pokemon_tcg.il_dataset import _resolve_hub_shards

    manifest: dict[int, float] = {}
    with (config.EPISODES_DIR / "manifest.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            try:
                manifest[int(r["episode_id"])] = float(r["min_score"])
            except (ValueError, KeyError):
                pass

    scores: dict[int, tuple[str, float]] = {}
    n_total = 0
    for rel in _resolve_hub_shards(config.HF_EPISODES_REPO, "train", days=None):
        day = rel.split("/")[1].removeprefix("day=")
        pf = pq.ParquetFile(
            hf_hub_download(config.HF_EPISODES_REPO, rel, repo_type="dataset")
        )
        t = pf.read(columns=["episode_id", "min_score"]).to_pydict()
        for eid, ms in zip(t["episode_id"], t["min_score"], strict=True):
            n_total += 1
            if ms is None:
                ms = manifest.get(eid)
            if ms is not None:
                scores[eid] = (day, float(ms))
    print(f"train episodes on Hub: {n_total}; with a min_score: {len(scores)} "
          f"({100 * len(scores) / max(n_total, 1):.0f}%)")
    return scores


def run_one(arm: str, ids_file: Path | None, seed: int, total_steps: int,
            eval_min_score: float) -> dict:
    tag = f"{arm}-s{seed}"
    result_path = RESULTS_DIR / f"{tag}.json"
    if result_path.exists():
        print(f"[skip] {tag} (result exists)")
        return json.loads(result_path.read_text())

    out_dir = MODELS_DIR / tag
    train_cmd = [
        "uv", "run", "python", str(REPO / "scripts" / "train_il.py"),
        "--seed", str(seed),
        "--total-steps", str(total_steps),
        "--data-source", "hub",
        "--num-workers", "2",
        "--out", str(out_dir),
        "--run-dir", str(RUNS_DIR / tag),
    ]
    if ids_file is not None:
        train_cmd += ["--episode-ids-file", str(ids_file)]
    t0 = time.time()
    print(f"[train] {tag}", flush=True)
    train = subprocess.run(train_cmd, cwd=REPO, capture_output=True, text=True)
    if train.returncode != 0:
        print(train.stdout[-3000:], file=sys.stderr)
        print(train.stderr[-3000:], file=sys.stderr)
        raise RuntimeError(f"training failed for {tag}")
    train_secs = time.time() - t0

    def run_eval(min_score: float | None) -> dict:
        cmd = [
            "uv", "run", "python", str(REPO / "scripts" / "eval_rung1.py"),
            "--model-dir", str(out_dir),
            "--data-source", "hub",
            "--max-episodes", str(EVAL_EPISODES),
        ]
        if min_score is not None:
            cmd += ["--min-score", str(min_score)]
        ev = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        m = _GLOBAL_RE.search(ev.stdout)
        if ev.returncode != 0 or not m:
            print(ev.stdout[-3000:], file=sys.stderr)
            print(ev.stderr[-3000:], file=sys.stderr)
            raise RuntimeError(f"eval failed for {tag} (min_score={min_score})")
        return {
            "rows": int(m.group(1)),
            "majority_pct": float(m.group(2)),
            "top1_pct": float(m.group(3)),
            "top3_pct": float(m.group(4)),
        }

    result = {
        "arm": arm,
        "seed": seed,
        "total_steps": total_steps,
        "train_secs": round(train_secs, 1),
        "eval_all": run_eval(None),
        "eval_highskill": run_eval(eval_min_score),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    print(f"[done] {tag}: all={result['eval_all']['top1_pct']:.1f}% "
          f"highskill={result['eval_highskill']['top1_pct']:.1f}% "
          f"({train_secs:.0f}s train)", flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--total-steps", type=int, default=TOTAL_STEPS)
    args = ap.parse_args()

    scores = collect_scores()
    by_day: dict[str, list[tuple[int, float]]] = {}
    for eid, (day, ms) in scores.items():
        by_day.setdefault(day, []).append((eid, ms))

    # Within-day thresholds: each day contributes its own top half/quarter,
    # so every arm has the SAME day mix and the treatment is skill, not day.
    day_thr: dict[str, tuple[float, float]] = {}
    for day, pairs in sorted(by_day.items()):
        vals = sorted(ms for _, ms in pairs)
        med = statistics.median(vals)
        q75 = statistics.quantiles(vals, n=4)[2] if len(vals) >= 4 else med
        day_thr[day] = (med, q75)
        print(f"day {day}: n={len(vals)} median={med:.0f} q75={q75:.0f}")

    IDS_DIR.mkdir(parents=True, exist_ok=True)
    arm_ids = {
        "unfiltered_scored": [eid for eid, _ in
                              (p for pairs in by_day.values() for p in pairs)],
        "top50_wd": [eid for day, pairs in by_day.items()
                     for eid, ms in pairs if ms >= day_thr[day][0]],
        "top25_wd": [eid for day, pairs in by_day.items()
                     for eid, ms in pairs if ms >= day_thr[day][1]],
    }
    arms: dict[str, Path | None] = {}
    for name, ids in arm_ids.items():
        p = IDS_DIR / f"{name}.txt"
        p.write_text("\n".join(str(i) for i in sorted(ids)))
        arms[name] = p
        id_set = set(ids)
        comp = {day: sum(1 for eid, _ in pairs if eid in id_set)
                for day, pairs in sorted(by_day.items())}
        print(f"arm {name}: {len(ids)} episodes, per-day {comp}")

    # High-skill eval threshold: 75th percentile of the EVAL day's own
    # min_score (fully scored in-shard), computed by eval_rung1 via
    # --min-score. Use the q75 measured earlier from eval shards.
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from pokemon_tcg.il_dataset import _resolve_hub_shards
    ev_scores: list[float] = []
    for rel in _resolve_hub_shards(config.HF_EPISODES_REPO, "eval", days=None):
        pf = pq.ParquetFile(
            hf_hub_download(config.HF_EPISODES_REPO, rel, repo_type="dataset")
        )
        ev_scores += [s for s in pf.read(columns=["min_score"]).to_pydict()["min_score"]
                      if s is not None]
    eval_thr = statistics.quantiles(ev_scores, n=4)[2]
    print(f"eval high-skill threshold (q75 of eval day min_score): {eval_thr:.0f}")

    results: dict[str, dict[int, dict]] = {}
    for arm, ids_file in arms.items():
        for seed in SEEDS:
            results.setdefault(arm, {})[seed] = run_one(
                arm, ids_file, seed, args.total_steps, eval_thr
            )

    base = results["unfiltered_scored"]
    lines = ["| arm | high-skill top1 by seed (%) | mean | paired d (pp) | mean d | spread | all-eval mean d | verdict |",
             "|---|---|---|---|---|---|---|---|"]
    accepted: list[str] = []
    for arm, by in results.items():
        hs = [by[s]["eval_highskill"]["top1_pct"] for s in SEEDS]
        if arm == "unfiltered_scored":
            lines.append(f"| unfiltered_scored | {', '.join(f'{x:.1f}' for x in hs)} | "
                         f"{sum(hs)/3:.2f} | — | — | — | — | reference |")
            continue
        d = [by[s]["eval_highskill"]["top1_pct"] - base[s]["eval_highskill"]["top1_pct"]
             for s in SEEDS]
        d_all = [by[s]["eval_all"]["top1_pct"] - base[s]["eval_all"]["top1_pct"]
                 for s in SEEDS]
        mean_d, spread = sum(d) / 3, max(d) - min(d)
        mean_d_all = sum(d_all) / 3
        ok = mean_d > spread and mean_d > 0 and mean_d_all > -1.0
        if ok:
            accepted.append(arm)
        lines.append(f"| {arm} | {', '.join(f'{x:.1f}' for x in hs)} | {sum(hs)/3:.2f} | "
                     f"{', '.join(f'{x:+.2f}' for x in d)} | {mean_d:+.2f} | "
                     f"{spread:.2f} | {mean_d_all:+.2f} | {'ACCEPT' if ok else 'drop'} |")

    table = "\n".join(lines)
    print("\n" + table)
    print(f"\naccepted: {accepted or 'NONE'}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "summary.json").write_text(json.dumps(
        {"protocol": {"seeds": SEEDS, "total_steps": args.total_steps,
                      "eval_episodes": EVAL_EPISODES,
                      "eval_highskill_threshold": eval_thr,
                      "accept_rule": "mean paired high-skill top-1 d > spread(d), > 0, "
                                     "and all-eval mean d > -1pp"},
         "accepted": accepted,
         "results": {a: {str(s): r for s, r in by.items()} for a, by in results.items()}},
        indent=2))
    (OUT_ROOT / "summary.md").write_text(
        "# Skill-filtered demonstrations (accuracy gate)\n\n"
        f"Equal steps {args.total_steps}; seeds {SEEDS}; eval {EVAL_EPISODES} episodes "
        f"(high-skill subset: eval-day min_score >= {eval_thr:.0f}).\n\n"
        + table + f"\n\nAccepted: {accepted or 'NONE'}\n\n"
        "Accuracy only gates. A full-budget retrain + benchmark_agents decides "
        "shipping (see the driver docstring).\n")
    print(f"summary written to {OUT_ROOT / 'summary.md'}")


if __name__ == "__main__":
    main()
