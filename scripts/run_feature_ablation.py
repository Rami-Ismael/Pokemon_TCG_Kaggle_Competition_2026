"""Feature-ablation driver for the deterministic-future features.

Protocol (notes/feature_ablation_candidates.md, Step 3):
- one feature GROUP per arm; each arm = baseline encoder + that group only
- >= 3 seeds per arm, identical data, and a fixed number of PASSES over that
  data (--epochs). No arm is pinned to another arm's step count.
- metric: eval_rung1 top-1/top-3 on the held-out day, vs the majority-class
  baseline computed on the same rows
- accept a group iff the mean paired (same-seed) top-1 improvement over the
  baseline arm exceeds the across-seed spread (max - min) of those paired
  differences; flat or negative -> drop
- finish with one combined arm (all accepted groups together)

Accuracy only GATES features. Whether anything ships is decided by
scripts/benchmark_agents.py on the combined model (Step 4), never by
accuracy alone.

Runs are resumable: a completed (arm, seed) leaves
reports/feature_ablation/results/{arm}-s{seed}.json and is skipped on
re-run. Training subprocesses inherit the repo's device policy
(resolve_device inside train_il.py -- never hardcoded here).

Usage:
    uv run python scripts/run_feature_ablation.py            # arms + combined
    uv run python scripts/run_feature_ablation.py --dry-run  # 2 tiny arms
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from pokemon_tcg import config  # noqa: E402

ARMS: dict[str, str] = {
    "baseline": "",
    "ko_race": "ko_race",
    "prize_race": "prize_race",
    "energy_deficit": "energy_deficit",
    "status_conditions": "status_conditions",
    "attack_tactical": "attack_tactical",
    "attach_enable": "attach_enable",
    "retreat_switch": "retreat_switch",
}
SEEDS = [42, 43, 44]
# Budget as data passes, not as a pinned step count. 0.3 epochs over the
# 4,554-episode train day is ~3,900 steps at batch 64 (4,554 x 181.3 rows/ep
# / 64), i.e. about what the first sweep spent -- but derived from the arm's
# own corpus rather than copied from another arm.
EPOCHS = 0.3
EVAL_EPISODES = 150

RESULTS_DIR = REPO / "reports" / "feature_ablation" / "results"
MODELS_DIR = config.MODELS_DIR / "ablation"
RUNS_DIR = REPO / "runs" / "ablation"

# eval_rung1's final line: "GLOBAL <n> <maj>% <top1>% <top3>%"
_GLOBAL_RE = re.compile(
    r"GLOBAL\s+(\d+)\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%"
)


def run_one(arm: str, features: str, seed: int, epochs: float, dry: bool) -> dict:
    tag = f"{arm}-s{seed}"
    result_path = RESULTS_DIR / f"{tag}.json"
    if result_path.exists():
        print(f"[skip] {tag} (result exists)")
        return json.loads(result_path.read_text())

    out_dir = MODELS_DIR / tag
    t0 = time.time()
    # Hub source explicitly: the local split folders are pruned stubs after
    # ADR-001 (24 of 4,554 train files on 2026-08-03) -- 'local' would
    # silently memorize a handful of episodes. --hub-days pins the original
    # train day so every arm sees the exact same corpus.
    train_cmd = [
        "uv", "run", "python", str(REPO / "scripts" / "train_il.py"),
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--features", features,
        "--data-source", "hub",
        "--hub-days", "2026-07-26",
        "--num-workers", "2",
        "--out", str(out_dir),
        "--run-dir", str(RUNS_DIR / tag),
    ]
    if dry:
        train_cmd += ["--max-train-episodes", "50", "--max-eval-episodes", "20",
                      "--hidden-size", "32", "--num-layers", "2", "--num-heads", "2",
                      "--eval-batches", "20", "--warmup-steps", "10"]
    print(f"[train] {tag}: {' '.join(train_cmd)}", flush=True)
    train = subprocess.run(train_cmd, cwd=REPO, capture_output=True, text=True)
    if train.returncode != 0:
        print(train.stdout[-3000:], file=sys.stderr)
        print(train.stderr[-3000:], file=sys.stderr)
        raise RuntimeError(f"training failed for {tag}")
    train_secs = time.time() - t0

    eval_cmd = [
        "uv", "run", "python", str(REPO / "scripts" / "eval_rung1.py"),
        "--model-dir", str(out_dir),
        "--data-source", "hub",
        "--max-episodes", str(10 if dry else EVAL_EPISODES),
    ]
    t1 = time.time()
    ev = subprocess.run(eval_cmd, cwd=REPO, capture_output=True, text=True)
    if ev.returncode != 0:
        print(ev.stdout[-3000:], file=sys.stderr)
        print(ev.stderr[-3000:], file=sys.stderr)
        raise RuntimeError(f"eval failed for {tag}")
    m = _GLOBAL_RE.search(ev.stdout)
    if not m:
        print(ev.stdout[-3000:], file=sys.stderr)
        raise RuntimeError(f"could not parse eval_rung1 GLOBAL line for {tag}")

    result = {
        "arm": arm,
        "features": features,
        "seed": seed,
        "epochs": epochs,
        "eval_rows": int(m.group(1)),
        "majority_pct": float(m.group(2)),
        "top1_pct": float(m.group(3)),
        "top3_pct": float(m.group(4)),
        "train_secs": round(train_secs, 1),
        "eval_secs": round(time.time() - t1, 1),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    print(f"[done] {tag}: top1={result['top1_pct']:.1f}% top3={result['top3_pct']:.1f}% "
          f"(majority {result['majority_pct']:.1f}%, {train_secs:.0f}s train)", flush=True)
    return result


def aggregate(results: dict[str, dict[int, dict]]) -> tuple[list[str], str]:
    """Accept/reject per the paired-difference rule; returns (accepted, report_md)."""
    lines = ["| arm | top1 by seed (%) | mean top1 | paired d vs baseline (pp) | mean d | spread(d) | verdict |",
             "|---|---|---|---|---|---|---|"]
    base = results["baseline"]
    accepted: list[str] = []
    for arm, by_seed in results.items():
        t1s = [by_seed[s]["top1_pct"] for s in SEEDS]
        mean_t1 = sum(t1s) / len(t1s)
        if arm == "baseline":
            lines.append(f"| baseline | {', '.join(f'{x:.1f}' for x in t1s)} | "
                         f"{mean_t1:.2f} | — | — | — | reference |")
            continue
        d = [by_seed[s]["top1_pct"] - base[s]["top1_pct"] for s in SEEDS]
        mean_d = sum(d) / len(d)
        spread = max(d) - min(d)
        verdict = "ACCEPT" if mean_d > spread and mean_d > 0 else "drop"
        if verdict == "ACCEPT":
            accepted.append(arm)
        lines.append(f"| {arm} | {', '.join(f'{x:.1f}' for x in t1s)} | {mean_t1:.2f} | "
                     f"{', '.join(f'{x:+.2f}' for x in d)} | {mean_d:+.2f} | "
                     f"{spread:.2f} | {verdict} |")
    return accepted, "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="2 arms, tiny models, smoke only")
    ap.add_argument("--epochs", type=float, default=EPOCHS,
                    help="passes over each arm's own corpus (fractional allowed)")
    ap.add_argument("--skip-combined", action="store_true")
    args = ap.parse_args()

    arms = dict(list(ARMS.items())[:2]) if args.dry_run else ARMS
    epochs = 0.01 if args.dry_run else args.epochs

    results: dict[str, dict[int, dict]] = {}
    for arm, features in arms.items():
        for seed in SEEDS:
            results.setdefault(arm, {})[seed] = run_one(
                arm, features, seed, epochs, args.dry_run
            )

    accepted, table = aggregate(results)
    print("\n" + table)
    print(f"\naccepted groups: {accepted or 'NONE'}")

    combined_results = None
    if accepted and not args.skip_combined:
        feats = ",".join(accepted)
        combined_results = {}
        for seed in SEEDS:
            combined_results[seed] = run_one(
                "combined", feats, seed, epochs, args.dry_run
            )
        base = results["baseline"]
        d = [combined_results[s]["top1_pct"] - base[s]["top1_pct"] for s in SEEDS]
        mean_d = sum(d) / len(d)
        spread = max(d) - min(d)
        joint_ok = mean_d > spread and mean_d > 0
        table += (f"\n| combined ({feats}) | "
                  f"{', '.join(f'{combined_results[s]['top1_pct']:.1f}' for s in SEEDS)} | "
                  f"{sum(combined_results[s]['top1_pct'] for s in SEEDS)/len(SEEDS):.2f} | "
                  f"{', '.join(f'{x:+.2f}' for x in d)} | {mean_d:+.2f} | {spread:.2f} | "
                  f"{'JOINTLY CONFIRMED' if joint_ok else 'NOT confirmed jointly'} |")
        print(table.splitlines()[-1])

    summary = {
        "protocol": {"seeds": SEEDS, "epochs": epochs,
                     "eval_episodes": EVAL_EPISODES,
                     "accept_rule": "mean paired top-1 diff vs baseline > across-seed spread (max-min) of the diffs, and > 0"},
        "accepted": accepted,
        "results": {a: {str(s): r for s, r in by.items()} for a, by in results.items()},
        "combined": {str(s): r for s, r in (combined_results or {}).items()},
    }
    out = REPO / "reports" / "feature_ablation" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    (REPO / "reports" / "feature_ablation" / "summary.md").write_text(
        "# Feature ablation (accuracy gate)\n\n"
        f"{epochs} passes over each arm's own corpus (arms here share one "
        f"corpus, so the realised step counts coincide -- no arm is pinned to "
        f"another's); seeds {SEEDS}; eval on first "
        f"{EVAL_EPISODES} held-out-day episodes via eval_rung1.\n\n" + table + "\n\n"
        f"Accepted: {accepted or 'NONE'}\n\n"
        "Accuracy only gates features -- the ship decision belongs to "
        "scripts/benchmark_agents.py (see Step 4).\n"
    )
    print(f"\nsummary written to {out}")


if __name__ == "__main__":
    main()
