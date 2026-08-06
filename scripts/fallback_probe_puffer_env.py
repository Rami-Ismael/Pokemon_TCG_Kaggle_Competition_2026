"""Fallback diagnostic for a frozen module opponent UNDER the PufferLib
training harness (PTCGGym), not the benchmark harness.

scripts/fallback_diagnostic.py measures an agent driven by env.run() in the
benchmark harness. The exploiter experiment instead drives the opponent
through PTCGGym._advance_until_learner() (callback inversion, per-select
calls, env-level exception guard), so the validity check must run in THAT
code path: if il_agent silently answers from _safe_choice here, an exploiter
trained in this env learns to beat the fallback, not the model.

Run in the PPO side venv (gymnasium lives there):

    .venv-ppo/bin/python scripts/fallback_probe_puffer_env.py --episodes 6

The learner plays uniformly-random legal actions (a fresh PPO explorer's
behavior distribution) so the opponent sees maximally off-distribution
states — the harsh case for encode/fallback paths.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["PTCG_FALLBACK_TRACK"] = "1"  # must precede the agent module import
os.environ.setdefault("PTCG_DEVICE", "cpu")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from pokemon_tcg import config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", default="il_agent")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    ap.add_argument("--out", type=Path,
                    default=config.REPORTS_DIR / "fallback_probe_puffer_env.json")
    args = ap.parse_args()

    torch.set_num_threads(1)
    from pokemon_tcg.puffer_env import OBS_LAYOUT, _OFFSETS, _SIZES, PTCGGym

    mask_i = [k for k, _, _ in OBS_LAYOUT].index("opt_mask")
    off, size = int(_OFFSETS[mask_i]), int(_SIZES[mask_i])

    env = PTCGGym(league=[("module", args.opponent)], mix=(0.0, 1.0),
                  seed=args.seed)
    rng = np.random.default_rng(args.seed)

    outcomes, steps = [], 0
    t0 = time.time()
    for _ in range(args.episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            legal = np.flatnonzero(obs[off:off + size] > 0.5)
            action = int(rng.choice(legal)) if len(legal) else 0
            obs, reward, done, _, info = env.step(action)
            steps += 1
        outcomes.append(reward)

    import benchmark_agents as bench
    mod = bench._LOADED_MODULES.get(args.opponent)
    if mod is None or not hasattr(mod, "diag_snapshot"):
        sys.exit(f"{args.opponent} not tracked (no diag_snapshot) — probe invalid")
    snap = mod.diag_snapshot()

    decisions = snap.get("decisions", 0)
    fallbacks = int(snap.get("fallbacks", 0))
    rate = fallbacks / decisions if decisions else float("nan")
    report = {
        "harness": "PTCGGym (puffer_env)", "opponent": args.opponent,
        "episodes": args.episodes, "learner": "uniform-random-legal",
        "learner_outcomes": outcomes, "learner_steps": steps,
        "seconds": round(time.time() - t0, 1),
        "opponent_decisions": decisions, "opponent_fallbacks": fallbacks,
        "opponent_fallback_rate": rate, "diag_snapshot": snap,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(f"opponent={args.opponent} episodes={args.episodes} "
          f"decisions={decisions} fallbacks={fallbacks} "
          f"rate={100 * rate:.2f}%")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
