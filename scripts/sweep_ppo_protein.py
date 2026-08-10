"""Protein hyperparameter sweep over Stage-3 PPO (train_ppo_puffer.py).

Runs in the SAME side venv as the trainer (PufferLib 3.0 + pyro):

    .venv-ppo/bin/python scripts/sweep_ppo_protein.py --smoke      # plumbing
    .venv-ppo/bin/python scripts/sweep_ppo_protein.py              # real sweep
    .venv-ppo/bin/python scripts/sweep_ppo_protein.py --validate   # decision

Design + pre-registered decision rule: notes/experiments/
2026-08-04-protein-ppo-sweep.md. One Protein suggestion per run; each run is
a fresh train_ppo_puffer.py subprocess (cg engine is a per-process singleton,
and MPS state must die with the run) on a FIXED WALL-CLOCK budget
(--train-seconds), promotion gate off. Score = decisive-game win rate of the
run's final snapshot vs the frozen anchor (promotion.evaluate_gate, 50
mirrored pairs). Protein 3.0 seeds with the search center, so run #1 is
exactly the incumbent config — the baseline arm.

The budget is time, not steps. The first sweep pinned every run to 60,416
steps and let wall-clock float from 666s to 1,589s — a 2.4x compute spread
handed to whichever configs happened to be slow. Now every config gets the
same wall-clock and reaches whatever step count its own throughput allows;
a config that trains slowly is charged for it in progress made, and
`global_step` is recorded per run as the throughput diagnostic. Runs are
NOT discarded for reaching fewer steps than their neighbours — that
rejection was the step-matching rule wearing a failure-handler's coat.

--anneal-horizon keeps the LR/entropy schedule SHAPE identical across runs
(it is the schedule's denominator, a fixed property of the search space,
not a target any run must reach). A slow config therefore stops earlier on
that schedule, at a higher LR and entropy than a fast one — that is a real
consequence of being slow at a fixed compute budget, and is intended.

Failure handling: PufferLib 3.0's Protein.observe() takes is_failure but its
body appends every call to success_observations, so failed runs (crashes)
are NOT observed at all — only logged. A run whose training log shows a
preflight refusal is retried after a wait, not counted: preflight failure
means the machine was busy, not that the config failed. The launcher also
refuses to START a training run while another repo training/benchmark
process is alive (MPS single-tenancy).

Every run appends one JSON line to <out>/sweep_log.jsonl; restarting the
sweep replays that log into Protein (re-observe) and continues where it left
off, so the sweep is resumable across sessions.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from pokemon_tcg import config  # noqa: E402

import pufferlib  # noqa: E402

assert str(getattr(pufferlib, "__version__", "")).startswith("3"), (
    f"pufferlib {getattr(pufferlib, '__version__', '?')} found -- this driver "
    "needs the 3.0 sweep API (Space(mean=...), seed_with_search_center). "
    "Run it with .venv-ppo/bin/python, not uv run.")

import pufferlib.sweep as psweep  # noqa: E402

ANCHOR_DEFAULT = config.MODELS_DIR / "s2" / "e1_seed43"

# Search space centered on the incumbent operating point (see experiment
# card). kl_coef=0 is excluded on purpose: it switches trainer class (stock
# PuffeRL, no anchor) -- a different code path, not a point on this surface.
# minibatch max 1024 == batch_size (8 envs x 128 horizon), PuffeRL's hard
# minibatch<=batch constraint.
SWEEP_CONFIG = {
    "method": "Protein",
    "metric": "win_rate_vs_anchor",
    "goal": "maximize",
    "downsample": 0,
    "learning_rate": dict(distribution="log_normal", min=3e-6, mean=3e-5, max=3e-4, scale="auto"),
    "ent_coef": dict(distribution="log_normal", min=1e-5, mean=1e-3, max=3e-2, scale="auto"),
    "kl_coef": dict(distribution="log_normal", min=5e-3, mean=5e-2, max=5e-1, scale="auto"),
    "update_epochs": dict(distribution="int_uniform", min=1, mean=3, max=4, scale="auto"),
    "minibatch_size": dict(distribution="uniform_pow2", min=128, mean=512, max=1024, scale="auto"),
    "gae_lambda": dict(distribution="logit_normal", min=0.8, mean=0.95, max=0.99, scale="auto"),
}

# Anything matching this, outside our own process tree, blocks a launch: two
# MPS jobs (or a training run + a benchmark) invalidate each other's numbers.
# \w* after train_ppo matters: train_ppo_puffer.py is the main offender, and
# the original strict form silently missed it (caught 2026-08-05 when another
# session's run was only stopped by the generic preflight CPU check).
COMPETING_RE = re.compile(
    r"(train_il|train_ppo\w*|train_critic|benchmark_agents|ingest_episodes|"
    r"exploiter_curve_sweep|eval_exploiter)\.py")

MIN_DECISIVE_SCORE = 20  # below this many decisive eval games, don't score


def competing_processes(min_total_pcpu: float = 25.0) -> list[str]:
    """Competing training/benchmark processes that are actually RUNNING.

    The pcpu floor exists because a hung run (a known failure mode: 0% CPU
    forever, e.g. the stranded-worker deadlock) would otherwise block a
    launch indefinitely; idle matches are reported as a note, not a blocker.
    """
    out = subprocess.run(["ps", "-axo", "pid=,pcpu=,command="],
                         capture_output=True, text=True).stdout
    hits, total = [], 0.0
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid, pcpu, cmd = parts
        if COMPETING_RE.search(cmd) and "sweep_ppo_protein" not in cmd:
            total += float(pcpu)
            hits.append(f"{pid} ({pcpu}% cpu) {cmd[:100]}")
    if hits and total < min_total_pcpu:
        print(f"[sweep] note: matching but IDLE process(es) (sum "
              f"{total:.0f}% cpu < {min_total_pcpu:.0f}%), likely hung -- "
              "not treating as competition:")
        for h in hits:
            print(f"    {h}")
        return []
    return hits


def wait_for_quiet(poll_s: float = 120.0) -> None:
    waited = 0.0
    while True:
        hits = competing_processes()
        if not hits:
            if waited:
                print(f"[sweep] machine quiet after {waited/60:.0f} min wait")
            return
        if waited == 0:
            print("[sweep] waiting for competing run(s) to finish "
                  "(MPS is single-tenant):")
            for h in hits:
                print(f"    {h}")
        time.sleep(poll_s)
        waited += poll_s


def sanitize(params: dict) -> dict:
    # Protein's unnormalize hands back numpy scalars; cast so the params dict
    # is json-serializable and argv-safe.
    p = {k: float(v) for k, v in params.items()}
    p["update_epochs"] = int(round(p["update_epochs"]))
    p["minibatch_size"] = int(round(p["minibatch_size"]))
    return p


def train_once(idx: int, params: dict, run_dir: Path, args) -> dict:
    """One training subprocess; retries while preflight refuses the machine."""
    run_dir.mkdir(parents=True, exist_ok=True)
    tag = f"sweep{args.sweep_tag}_r{idx:02d}"
    cmd = [
        "nice", "-n", "5", sys.executable,
        str(REPO / "scripts" / "train_ppo_puffer.py"),
        "--promote-every", "0",
        "--snapshot-every-s", "999999",
        "--max-seconds", str(args.train_seconds),
        "--out", str(run_dir),
        "--run-tag", tag,
        "--lr", f"{params['learning_rate']:.8g}",
        "--ent-coef", f"{params['ent_coef']:.8g}",
        "--kl-coef", f"{params['kl_coef']:.8g}",
        "--update-epochs", str(params["update_epochs"]),
        "--minibatch-size", str(params["minibatch_size"]),
        "--gae-lambda", f"{params['gae_lambda']:.6g}",
    ]
    if args.smoke:
        # cpu-only plumbing test: allowed to coexist with a live MPS run
        # (the doctor's own guidance), so no quiet-wait and no preflight.
        cmd += ["--smoke", "--skip-preflight"]
    else:
        # Schedule horizon only -- --max-seconds above is what ends the run.
        cmd += ["--total-timesteps", str(args.anneal_horizon)]

    log_path = run_dir / "train_log.txt"
    for attempt in range(1, 13):
        if not args.smoke:
            wait_for_quiet()
        t0 = time.time()
        with log_path.open("a") as fh:
            fh.write(f"\n===== attempt {attempt} cmd: {' '.join(cmd)}\n")
            fh.flush()
            try:
                proc = subprocess.run(
                    cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=REPO,
                    timeout=args.train_seconds + 900)
            except subprocess.TimeoutExpired:
                return {"ok": False, "seconds": time.time() - t0, "tag": tag,
                        "error": "hard timeout (train ignored --max-seconds)"}
        seconds = time.time() - t0
        tail = log_path.read_text()[-3000:]
        if proc.returncode == 0:
            return {"ok": True, "seconds": seconds, "tag": tag}
        if "preflight" in tail:
            print(f"[sweep] run {idx} attempt {attempt}: preflight refused "
                  f"(machine busy) -- waiting 5 min and retrying")
            time.sleep(300)
            continue
        return {"ok": False, "seconds": seconds, "tag": tag,
                "error": f"exit {proc.returncode}; log tail: {tail[-400:]}"}
    return {"ok": False, "seconds": 0.0, "tag": tag,
            "error": "preflight refused 12 times"}


def final_snapshot(run_dir: Path) -> Path | None:
    # Legacy u<step>_final names and the readable <run_tag>_step<step>_final ones.
    snaps = sorted(d for d in run_dir.glob("*_final")
                   if (d / "config.json").exists())
    return snaps[-1] if snaps else None


def evaluate_snapshot(snap: Path, anchor: Path, pairs: int, workers: int) -> dict:
    from pokemon_tcg.promotion import evaluate_gate
    return evaluate_gate(snap, anchor, pairs=pairs, workers=workers,
                         threshold=0.5)


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


def replay_log(protein, log_path: Path) -> int:
    """Re-observe prior scored runs so a restarted sweep continues, not restarts."""
    last_idx = 0
    if not log_path.exists():
        return last_idx
    for line in log_path.read_text().splitlines():
        row = json.loads(line)
        last_idx = max(last_idx, row["idx"])
        if not row.get("failure"):
            protein.observe(row["params"], row["score"], row["cost_s"])
    return last_idx


def run_sweep(args) -> None:
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "sweep_log.jsonl"
    anchor = args.anchor
    assert anchor.exists(), f"anchor checkpoint missing: {anchor}"

    protein = psweep.Protein(deepcopy(SWEEP_CONFIG))
    start_idx = replay_log(protein, log_path)
    if start_idx:
        print(f"[sweep] resumed: {start_idx} prior run(s) replayed from {log_path}")

    for idx in range(start_idx + 1, args.runs + 1):
        params, info = protein.suggest(None)
        params = sanitize(params)
        print(f"\n[sweep] run {idx}/{args.runs}: {params}"
              + (f"  (gp info: {info})" if info else ""))
        run_dir = out / f"run{idx:02d}"
        result = train_once(idx, params, run_dir, args)

        row = {"idx": idx, "ts": round(time.time(), 1), "params": params,
               "run_tag": result["tag"], "cost_s": round(result["seconds"], 1),
               "failure": False, "score": None}
        snap = final_snapshot(run_dir)
        if not result["ok"] or snap is None:
            row.update(failure=True,
                       error=result.get("error", "no final snapshot"))
        else:
            meta = json.loads((snap / "ppo_metadata.json").read_text())
            row["global_step"] = meta.get("global_step")
            row["snap"] = str(snap)
            # Every run that produced a snapshot is scored, however far it got
            # in its wall-clock budget: reaching fewer steps is a RESULT of the
            # config (slow), not grounds for discarding it.
            verdict = evaluate_snapshot(snap, anchor, args.pairs,
                                        args.eval_workers)
            decisive = verdict["wins"] + verdict["losses"]
            row["gate"] = {k: verdict[k] for k in
                           ("wins", "losses", "draws", "games", "win_rate",
                            "seconds")}
            min_dec = 1 if args.smoke else MIN_DECISIVE_SCORE
            if decisive < min_dec:
                row.update(failure=True,
                           error=f"only {decisive} decisive eval games")
            else:
                row["score"] = verdict["win_rate"]

        if not row["failure"]:
            protein.observe(row["params"], row["score"], row["cost_s"])
        with log_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        status = (f"score={row['score']:.3f}" if row["score"] is not None
                  else f"FAILED ({row.get('error', '?')[:80]})")
        print(f"[sweep] run {idx} done in {row['cost_s']:.0f}s -- {status}")

    summarize(log_path)


def summarize(log_path: Path) -> None:
    rows = [json.loads(l) for l in log_path.read_text().splitlines()]
    scored = sorted((r for r in rows if r.get("score") is not None),
                    key=lambda r: r["score"], reverse=True)
    print(f"\n[sweep] {len(rows)} runs, {len(scored)} scored. Top 5:")
    for r in scored[:5]:
        g = r.get("gate", {})
        lo, hi = wilson_ci(g.get("wins", 0), g.get("wins", 0) + g.get("losses", 0))
        print(f"  run{r['idx']:02d} score={r['score']:.3f} "
              f"[{lo:.3f},{hi:.3f}] {r['params']}")
    inc = next((r for r in rows if r["idx"] == 1), None)
    if inc and inc.get("score") is not None:
        print(f"  incumbent (run01) score={inc['score']:.3f}")
    if scored:
        best = scored[0]
        (log_path.parent / "best.json").write_text(json.dumps(best, indent=2))
        print(f"[sweep] best -> {log_path.parent / 'best.json'}")


def validate(args) -> None:
    """Pre-registered decision: best vs run01 (incumbent), 150 mirrored pairs."""
    log_path = args.out / "sweep_log.jsonl"
    rows = [json.loads(l) for l in log_path.read_text().splitlines()]
    scored = sorted((r for r in rows if r.get("score") is not None),
                    key=lambda r: r["score"], reverse=True)
    best = scored[0]
    inc = next(r for r in rows if r["idx"] == 1 and not r.get("failure"))
    if best["idx"] == 1:
        print("[validate] best config IS the incumbent -- decision: DROP "
              "(operating point already best at this scale); no games needed")
        return
    print(f"[validate] best run{best['idx']:02d} {best['params']}\n"
          f"      vs incumbent run01 {inc['params']}\n"
          f"      {args.validate_pairs} mirrored pairs...")
    wait_for_quiet()  # 16 policy processes; don't pile onto a live run
    verdict = evaluate_snapshot(Path(best["snap"]), Path(inc["snap"]),
                                args.validate_pairs, args.eval_workers)
    decisive = verdict["wins"] + verdict["losses"]
    lo, hi = wilson_ci(verdict["wins"], decisive)
    adopt = lo > 0.5
    print(f"[validate] best vs incumbent: {verdict['wins']}W/"
          f"{verdict['losses']}L/{verdict['draws']}D  "
          f"win_rate={verdict['win_rate']:.3f}  95% Wilson [{lo:.3f},{hi:.3f}]"
          f"  -> {'ADOPT swept config' if adopt else 'DROP (CI includes 0.5)'}")
    (args.out / "validation.json").write_text(json.dumps(
        {"best": best, "incumbent": inc, "verdict": verdict,
         "wilson_ci": [lo, hi], "adopt": adopt}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--train-seconds", type=float, default=1200.0,
                    help="wall-clock budget every run gets -- THE budget. "
                         "1200s sits mid-range of the first sweep's 666-1589s "
                         "spread at 60k steps.")
    ap.add_argument("--anneal-horizon", type=int, default=60_000,
                    help="LR/entropy schedule denominator passed through as "
                         "--total-timesteps; NOT a step target, and runs are "
                         "never rejected for finishing below it.")
    ap.add_argument("--pairs", type=int, default=50)
    ap.add_argument("--eval-workers", type=int, default=8)
    ap.add_argument("--anchor", type=Path, default=ANCHOR_DEFAULT)
    ap.add_argument("--out", type=Path, default=config.MODELS_DIR / "ppo_sweep")
    ap.add_argument("--sweep-tag", default=time.strftime("%m%d"))
    ap.add_argument("--validate", action="store_true",
                    help="run the pre-registered best-vs-incumbent decision "
                         "match instead of sweeping")
    ap.add_argument("--validate-pairs", type=int, default=150)
    ap.add_argument("--smoke", action="store_true",
                    help="2 tiny cpu runs + 3-pair evals: plumbing test only")
    args = ap.parse_args()

    if args.smoke:
        args.runs = min(args.runs, 2)
        args.pairs = 3
        args.eval_workers = 2

    if args.validate:
        validate(args)
    else:
        run_sweep(args)


if __name__ == "__main__":
    main()
