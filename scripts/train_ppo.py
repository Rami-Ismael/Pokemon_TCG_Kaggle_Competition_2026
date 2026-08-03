"""Stage-3 SELFPLAY: on-policy PPO fine-tuning over live cabt games.

rl_pipeline_v1.md §3. Nothing persistent is written except checkpoints and
TensorBoard logs (disk budget rule). Rollouts are collected by a spawn-pool
of workers playing full games (pokemon_tcg/selfplay.py), consumed by one PPO
update, and freed.

Usage:
    uv run python scripts/train_ppo.py --dry-run          # 2 tiny updates, smoke test
    uv run python scripts/train_ppo.py --updates 50       # real run (check the
                                                          # projected wall clock it
                                                          # prints; >1h needs approval)

Stop signals (watch the logs, per the plan): KL-to-prior blowup or per-context
entropy collapse without Rung-2 gain.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pokemon_tcg import config  # noqa: E402
from pokemon_tcg.device import log_device_info, resolve_device  # noqa: E402
from pokemon_tcg.il_model import PTCGImitationPolicy  # noqa: E402
from pokemon_tcg.logging_utils import TensorBoardLogger  # noqa: E402
from pokemon_tcg.ppo import ValueModel, ppo_losses, prior_logprobs, stack_rows  # noqa: E402
from pokemon_tcg.selfplay import play_one_game, sample_league, worker_init  # noqa: E402

STRONG_POOL = ["kiyotah_dragapult", "mechi22_alakazam", "plamen06_steel"]


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=config.PROJECT_ROOT).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-from", type=Path, default=config.MODELS_DIR / "s2" / "e1_seed43",
                    help="actor init + critic trunk init + the frozen KL prior")
    ap.add_argument("--updates", type=int, default=50)
    ap.add_argument("--games-per-update", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ppo-epochs", type=int, default=3)
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--kl-coef", type=float, default=0.05)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--lr-actor", type=float, default=3e-5)
    ap.add_argument("--lr-critic", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="rollout sampling temperature; PPO ratios assume 1.0")
    ap.add_argument("--league", default=str(config.MODELS_DIR / "il_agent"),
                    help="comma-separated past-checkpoint dirs for the league bucket")
    ap.add_argument("--pool", default=",".join(STRONG_POOL),
                    help="comma-separated public-pool agent names (benchmark registry)")
    ap.add_argument("--mix", default="0.5,0.3,0.2", help="mirror,league,pool draw probabilities")
    ap.add_argument("--snapshot-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    ap.add_argument("--out", type=Path, default=config.MODELS_DIR / "ppo")
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="2 updates x 8 games x 2 workers, snapshot at the end")
    args = ap.parse_args()

    if args.dry_run:
        args.updates, args.games_per_update, args.workers = 2, 8, 2
        args.snapshot_every = 2

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    rng = random.Random(args.seed)
    mix = tuple(float(x) for x in args.mix.split(","))
    league = [x for x in args.league.split(",") if x]
    pool = [x for x in args.pool.split(",") if x]

    run_dir = args.run_dir or (config.PROJECT_ROOT / "runs" / f"ppo_{time.strftime('%Y%m%d-%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_device_info(run_dir, device)
    (run_dir / "run_config.json").write_text(
        json.dumps({**vars(args), "git_sha": git_sha()}, indent=2, default=str))
    logger = TensorBoardLogger(run_dir)

    actor = PTCGImitationPolicy.from_pretrained(args.init_from).to(device)
    critic = ValueModel(str(args.init_from)).to(device)
    prior = PTCGImitationPolicy.from_pretrained(args.init_from).to(device).eval()
    for p in prior.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.AdamW([
        {"params": actor.parameters(), "lr": args.lr_actor},
        {"params": critic.parameters(), "lr": args.lr_critic},
    ])
    print(f"actor/prior/critic init: {args.init_from}  device: {device}")
    print(f"league: {league}  pool: {pool}  mix: {mix}")

    # Workers reload the CURRENT actor from this dir at every pool (re)spawn.
    current_ckpt = run_dir / "current_ckpt"
    ctx = mp.get_context("spawn")
    t_start = time.time()

    for update in range(1, args.updates + 1):
        # --- collect (on-policy: fresh pool sees fresh weights) ---
        t0 = time.time()
        actor.to("cpu").save_pretrained(current_ckpt)
        actor.to(device)
        tasks = [(i, sample_league(rng, league, pool, mix), i % 2)
                 for i in range(args.games_per_update)]
        with ctx.Pool(args.workers, initializer=worker_init,
                      initargs=(str(current_ckpt), args.temperature)) as p_:
            results = p_.map(play_one_game, tasks)
        t_collect = time.time() - t0

        rows, outcomes, wins = [], [], {"mirror": [0, 0], "league": [0, 0], "pool": [0, 0]}
        n_fallbacks = 0
        for r in results:
            rows.extend(r["rows"])
            outcomes.extend([r["outcome"]] * len(r["rows"]))
            n_fallbacks += r["fallbacks"]
            bucket = ("mirror" if r["opponent"][1] == "__mirror__"
                      else "league" if r["opponent"][0] == "ckpt" else "pool")
            wins[bucket][0] += r["outcome"] == 1.0
            wins[bucket][1] += 1
        if not rows:
            print(f"update {update}: no rollout rows collected -- stopping")
            break

        # --- PPO update ---
        t1 = time.time()
        batch = stack_rows(rows, outcomes, device)
        prior_logp = prior_logprobs(prior, batch)
        n = batch["action"].shape[0]
        metrics_acc: dict[str, float] = {}
        n_mb = 0
        for _ in range(args.ppo_epochs):
            perm = torch.randperm(n, device=device)
            for s in range(0, n, args.minibatch):
                idx = perm[s:s + args.minibatch]
                mb = {k: v[idx] for k, v in batch.items()}
                loss, metrics = ppo_losses(actor, critic, mb, prior_logp[idx],
                                           clip=args.clip, kl_coef=args.kl_coef,
                                           value_coef=args.value_coef)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()), args.grad_clip)
                optimizer.step()
                for k, v in metrics.items():
                    metrics_acc[k] = metrics_acc.get(k, 0.0) + v
                n_mb += 1
        t_update = time.time() - t1

        avg = {k: v / n_mb for k, v in metrics_acc.items()}
        wr = {b: (f"{w}/{g}" if g else "-") for b, (w, g) in wins.items()}
        print(f"update {update}/{args.updates}  rows={n}  "
              f"collect={t_collect:.1f}s update={t_update:.1f}s  "
              f"win mirror={wr['mirror']} league={wr['league']} pool={wr['pool']}  "
              f"kl={avg['kl_to_prior']:.4f} ent={avg['entropy_normalized']:.3f} "
              f"vloss={avg['loss/value']:.4f} clip%={100*avg['ratio/clipped_frac']:.1f}")
        logger.log_scalars({**{f"ppo/{k}": v for k, v in avg.items()},
                            "rollout/rows": n,
                            "rollout/fallbacks": n_fallbacks,
                            "rollout/collect_seconds": t_collect,
                            "rollout/update_seconds": t_update,
                            **{f"winrate/{b}": (w / g if g else 0.0)
                               for b, (w, g) in wins.items()}}, update)

        if update % args.snapshot_every == 0 or update == args.updates:
            snap = args.out / f"u{update:04d}"
            actor.to("cpu").save_pretrained(snap)
            actor.to(device)
            (snap / "ppo_metadata.json").write_text(json.dumps(
                {"update": update, "init_from": str(args.init_from),
                 "config": vars(args), "git_sha": git_sha(),
                 "kl_to_prior": avg["kl_to_prior"],
                 "entropy_normalized": avg["entropy_normalized"]},
                indent=2, default=str))
            print(f"  snapshot -> {snap}")

    logger.close()
    print(f"total: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
