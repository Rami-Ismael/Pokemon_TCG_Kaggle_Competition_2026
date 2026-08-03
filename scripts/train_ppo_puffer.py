"""Stage-3 SELFPLAY on PufferLib 3.0's PPO (PuffeRL) — runs in the side venv:

    .venv-ppo/bin/python scripts/train_ppo_puffer.py --smoke
    .venv-ppo/bin/python scripts/train_ppo_puffer.py --total-timesteps 500000

Uses PufferLib's own trainer (rollout buffers, puff-advantage, minibatch
epochs, LR anneal) over vectorized PTCGGym envs — replacing the custom loop
in scripts/train_ppo.py per the user's 2026-08-03 direction. The thin driver
below exists only to (a) neutralize PuffeRL's hardcoded CUDA autocast context
on MPS, and (b) periodically save the actor in save_pretrained format so env
workers' mirror opponents can hot-reload it (LatestCheckpointOpponent) and so
snapshots drop straight into the benchmark harness.

Known deliberate differences from the plan noted in rl_pipeline_v1.md §3.2:
stock PuffeRL loss has an entropy bonus, not a KL-to-prior anchor. v1 runs
stock (small ent_coef); if eval shows prior-forgetting, the KL term gets
patched into a PuffeRL subclass next.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pokemon_tcg import config  # noqa: E402
from pokemon_tcg.puffer_env import make_puffer_env  # noqa: E402
from pokemon_tcg.puffer_policy import PTCGPufferPolicy  # noqa: E402

import pufferlib.pufferl as pufferl  # noqa: E402
import pufferlib.vector as pvector  # noqa: E402

MIN_DISK_GB = 2.0   # snapshots + logs headroom (laptop disk budget is tight)
MIN_RAM_GB = 3.0    # 8 workers x (torch + engine + 2 policies) + MPS learner


def preflight(num_workers: int) -> None:
    """Refuse to launch into a machine that can't host the run.

    Checks, in order: free disk (snapshots/logs), available RAM, and heavy
    COMPETING processes (another training run, a benchmark, a parallel
    session's job). Contention doesn't just slow this run -- it invalidates
    its throughput-based budget arithmetic and can starve env workers into
    timeout losses that poison the rollout data. Override: --skip-preflight.
    """
    import shutil as _sh
    import subprocess as _sp

    free_gb = _sh.disk_usage(str(config.PROJECT_ROOT)).free / 1e9
    assert free_gb >= MIN_DISK_GB, (
        f"preflight: only {free_gb:.1f} GB disk free (< {MIN_DISK_GB}) -- clean up first")

    try:
        import psutil
        avail_gb = psutil.virtual_memory().available / 1e9
    except ImportError:
        out = _sp.run(["vm_stat"], capture_output=True, text=True).stdout
        pages = {ln.split(":")[0].strip(): int(ln.split(":")[1].strip(" ."))
                 for ln in out.splitlines()
                 if ":" in ln and ln.split(":")[1].strip(" .").isdigit()}
        avail_gb = (pages.get("Pages free", 0) + pages.get("Pages inactive", 0)) * 16384 / 1e9
    assert avail_gb >= MIN_RAM_GB, (
        f"preflight: only {avail_gb:.1f} GB RAM available (< {MIN_RAM_GB})")

    me = os.getpid()
    ps = _sp.run(["ps", "-axo", "pid=,ppid=,pcpu=,comm="], capture_output=True, text=True).stdout
    hogs = []
    for ln in ps.splitlines():
        parts = ln.split(None, 3)
        if len(parts) < 4:
            continue
        pid, ppid, pcpu, comm = int(parts[0]), int(parts[1]), float(parts[2]), parts[3]
        if pid == me or ppid == me:
            continue
        if pcpu > 50.0:
            hogs.append((pid, pcpu, comm.strip()[:60]))
    total = sum(h[1] for h in hogs)
    if hogs:
        print(f"preflight WARNING: {len(hogs)} competing heavy process(es), "
              f"{total:.0f}% CPU total:")
        for pid, pcpu, comm in hogs[:6]:
            print(f"  pid {pid}  {pcpu:.0f}%  {comm}")
    assert total <= 300.0, (
        "preflight: heavy contention (>3 cores busy elsewhere) -- another run is "
        "active; wait for it or pass --skip-preflight to launch anyway")
    print(f"preflight OK: disk {free_gb:.1f} GB, RAM {avail_gb:.1f} GB, "
          f"competing CPU {total:.0f}%, workers requested {num_workers}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-from", type=Path, default=config.MODELS_DIR / "s2" / "e1_seed43")
    ap.add_argument("--total-timesteps", type=int, default=500_000)
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--bptt-horizon", type=int, default=128)
    ap.add_argument("--minibatch-size", type=int, default=512)
    ap.add_argument("--update-epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--ent-coef", type=float, default=0.001)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--kl-coef", type=float, default=0.0,
                    help="KL-to-prior anchor coefficient; 0 = stock pufferl loss. "
                         ">0 uses PuffeRLPriorKL (pokemon_tcg/pufferl_kl.py)")
    ap.add_argument("--kl-prior", type=Path, default=config.MODELS_DIR / "s2" / "e1_seed43",
                    help="frozen reference policy for the anchor (the BC lineage prior)")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="internal wall-clock budget: stop cleanly (with a final "
                         "snapshot) once exceeded -- replaces external kill timers")
    ap.add_argument("--league", default=str(config.MODELS_DIR / "il_agent"),
                    help="comma-separated frozen checkpoint dirs for the env's "
                         "league bucket (the 30%% draw)")
    ap.add_argument("--pool-weights", default=None,
                    help="comma-separated draw weights for the public-pool trio "
                         "(kiyotah,mechi22,plamen06), e.g. '0.6,0.25,0.15' for "
                         "PFSP-style frontier weighting; default uniform")
    ap.add_argument("--init-policy-full", type=Path, default=None,
                    help="policy_full.pt from a prior run: restores actor AND "
                         "warmed value head (overrides --init-from for weights; "
                         "--init-from still sets the KL prior/architecture)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--snapshot-every-s", type=float, default=300.0,
                    help="actor save_pretrained cadence (mirror hot-reload + benchmark)")
    ap.add_argument("--out", type=Path, default=config.MODELS_DIR / "ppo_puffer")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="launch even if disk/RAM/contention checks fail")
    ap.add_argument("--smoke", action="store_true",
                    help="4 envs / 2 workers / ~3 tiny epochs on cpu")
    args = ap.parse_args()

    if args.smoke:
        args.num_envs, args.num_workers = 2, 2
        # PuffeRL requires minibatch_size <= batch_size (= num_envs * horizon)
        args.bptt_horizon, args.minibatch_size = 32, 64
        args.total_timesteps = 3 * args.num_envs * args.bptt_horizon
        args.device = "cpu"

    # HARD CONSTRAINT (verified 2026-08-03): the cg/cabt native engine keeps ONE
    # battle state per process; two live envs in one process corrupt each other
    # and the dylib exits silently. Exactly one env per worker, always.
    assert args.num_envs == args.num_workers, (
        f"cg engine is a per-process singleton: need num_envs == num_workers, "
        f"got {args.num_envs} != {args.num_workers}")

    if not args.skip_preflight:
        preflight(args.num_workers)

    # load_config() runs its own argparse over sys.argv (and errors on our
    # flags) -- hand it an empty argv while it parses.
    argv, sys.argv = sys.argv, [sys.argv[0]]
    try:
        cfg = pufferl.load_config("default")
    finally:
        sys.argv = argv
    cfg["train"].update({
        "device": args.device,
        "optimizer": "adam",  # NOT muon: fine-tuning a BC prior, see plan §8.1
        "learning_rate": args.lr,
        "anneal_lr": True,
        "gamma": args.gamma,
        "gae_lambda": 0.95,
        "update_epochs": args.update_epochs,
        "ent_coef": args.ent_coef,
        "vf_coef": 0.5,
        "max_grad_norm": 1.0,
        "total_timesteps": args.total_timesteps,
        "batch_size": "auto",
        "bptt_horizon": args.bptt_horizon,
        "minibatch_size": args.minibatch_size,
        "compile": False,
        "checkpoint_interval": 10_000_000,  # we snapshot ourselves (see below)
        "data_dir": str(config.PROJECT_ROOT / "runs" / "ppo_puffer"),
        "seed": config.RANDOM_SEED,
    })

    mirror_root = str(args.out)  # env workers hot-reload newest snapshot from here
    args.out.mkdir(parents=True, exist_ok=True)

    league = [("ckpt", p) for p in args.league.split(",") if p]
    pool_weights = ([float(x) for x in args.pool_weights.split(",")]
                    if args.pool_weights else None)
    vecenv = pvector.make(
        make_puffer_env,
        env_kwargs={"mirror_root": mirror_root, "anchor_ckpt": str(args.init_from),
                    "league": league, "pool_weights": pool_weights},
        backend=pvector.Multiprocessing,  # never Serial >1 env: cg singleton
        num_envs=args.num_envs,
        num_workers=args.num_workers,
        seed=config.RANDOM_SEED,
    )
    policy = PTCGPufferPolicy(str(args.init_from)).to(args.device)
    if args.init_policy_full is not None:
        # Full-policy restore: actor AND warmed value head from a prior run --
        # the critic-cold-start tax (explained_var -0.9 for the first chunk of
        # every generation) only stops once generations inherit both nets.
        state = torch.load(args.init_policy_full, map_location=args.device)
        policy.load_state_dict(state)
        print(f"restored full policy (actor + critic) from {args.init_policy_full}")
    print(f"policy params: {sum(p.numel() for p in policy.parameters()):,}")

    # Everything below runs under try/finally with vecenv.close(): any raise
    # (including PuffeRL's own config validation) would otherwise strand the
    # spawned env workers and deadlock the interpreter at exit -- that exact
    # failure presented as "the smoke test hangs forever" on 2026-08-03.
    try:
        train_config = dict(**cfg["train"], env="ptcg_selfplay")
        if args.kl_coef > 0:
            from pokemon_tcg.pufferl_kl import PuffeRLPriorKL

            prior = PTCGPufferPolicy(str(args.kl_prior)).to(args.device)
            trainer = PuffeRLPriorKL(train_config, vecenv, policy, None,
                                     kl_prior_policy=prior, kl_coef=args.kl_coef)
            print(f"KL anchor ON: coef={args.kl_coef} prior={args.kl_prior}")
        else:
            trainer = pufferl.PuffeRL(train_config, vecenv, policy, None)
        # PuffeRL hardcodes torch.amp.autocast(device_type='cuda'); on mps/cpu
        # with fp32 that context is at best a no-op, at worst an error.
        trainer.amp_context = contextlib.nullcontext()

        # Seed snapshot so mirror opponents have something to load from step 0.
        policy.actor.to("cpu").save_pretrained(args.out / "u0")
        policy.actor.to(args.device)

        last_snap = time.time()
        t_run = time.time()
        epoch = 0
        while trainer.global_step < train_config["total_timesteps"]:
            if args.max_seconds and time.time() - t_run > args.max_seconds:
                print(f"wall-clock budget {args.max_seconds:.0f}s reached at "
                      f"step {trainer.global_step} -- stopping cleanly")
                break
            trainer.evaluate()
            logs = trainer.train()
            epoch += 1
            if time.time() - last_snap >= args.snapshot_every_s:
                snap = args.out / f"u{trainer.global_step}"
                policy.actor.to("cpu").save_pretrained(snap)
                policy.actor.to(args.device)
                (snap / "ppo_metadata.json").write_text(json.dumps(
                    {"global_step": trainer.global_step, "epoch": epoch,
                     "init_from": str(args.init_from), "trainer": "pufferl-3.0"},
                    indent=2))
                last_snap = time.time()
                print(f"  snapshot -> {snap}")

        snap = args.out / f"u{trainer.global_step}_final"
        policy.actor.to("cpu").save_pretrained(snap)
        # Full policy (actor + warmed critic) for the next generation's
        # --init-policy-full; ~26 MB, well inside the disk budget.
        policy.to("cpu")
        torch.save(policy.state_dict(), args.out / "policy_full.pt")
        print(f"final snapshot -> {snap}")
        trainer.print_dashboard()
        trainer.close()  # closes the vecenv too; the finally below is then a no-op
    finally:
        # Both cleanups matter independently: unclosed env workers deadlock
        # interpreter exit in atexit, and PuffeRL's non-daemon Utilization
        # thread deadlocks it in wait_for_thread_shutdown.
        try:
            vecenv.close()
        except Exception:
            pass
        try:
            trainer.utilization.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
