"""Stage-3 first measurement (rl_pipeline_v1.md §3.1): vectorized self-play
throughput with the actual sampling policy on both seats. Every Stage-3
budget number (updates/hour, games per candidate) hangs off this.

Usage:
    uv run python scripts/probe_selfplay_throughput.py --workers 8 --games 32
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pokemon_tcg import config  # noqa: E402
from pokemon_tcg.selfplay import (  # noqa: E402
    play_one_game, sample_lineage_opponent, worker_init)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(config.MODELS_DIR / "s2" / "e1_seed43"),
                    help="learner checkpoint (default: the Stage-3 init)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--games", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--league-mix", action="store_true",
                    help="draw opponents per the lineage mix (live mirror / "
                         "our frozen checkpoints) instead of pure mirror; uses "
                         "il_agent as the one past checkpoint. Externals are "
                         "never opponents -- see selfplay.sample_lineage_opponent")
    args = ap.parse_args()

    rng = random.Random(config.RANDOM_SEED)
    league = [str(config.MODELS_DIR / "il_agent")] if args.league_mix else []
    tasks = []
    for i in range(args.games):
        spec = (sample_lineage_opponent(rng, league) if args.league_mix
                else ("ckpt", "__mirror__"))
        tasks.append((i, spec, i % 2))

    ctx = mp.get_context("spawn")
    t0 = time.time()
    with ctx.Pool(args.workers, initializer=worker_init,
                  initargs=(args.ckpt, args.temperature)) as pool_:
        results = pool_.map(play_one_game, tasks)
    wall = time.time() - t0

    n_rows = sum(len(r["rows"]) for r in results)
    n_dec = sum(r["decisions"] for r in results)
    n_fb = sum(r["fallbacks"] for r in results)
    outcomes = [r["outcome"] for r in results]
    game_secs = [r["seconds"] for r in results]

    print(f"\nckpt: {args.ckpt}")
    print(f"workers={args.workers} games={args.games} wall={wall:.1f}s "
          f"(incl. ~{max(game_secs) and (wall - sum(game_secs)/args.workers):.0f}s worker startup)")
    print(f"games/sec:        {args.games / wall:.2f}")
    print(f"rollout rows:     {n_rows} ({n_rows / wall:.0f} rows/sec)")
    print(f"decisions:        {n_dec}  fallbacks: {n_fb} ({100 * n_fb / max(n_dec + n_fb, 1):.2f}%)")
    print(f"mean game length: {n_rows / max(args.games, 1):.0f} rows, "
          f"mean {sum(game_secs)/len(game_secs):.2f}s/game in-worker")
    print(f"learner win rate: {sum(outcomes) / len(outcomes):.2f} "
          f"(mirror ~0.5 expected)")
    print(f"\nprojection: {3600 * n_rows / wall:.0f} rows/hour "
          f"-> a 4096-row PPO update every {4096 * wall / max(n_rows, 1):.0f}s")


if __name__ == "__main__":
    main()
