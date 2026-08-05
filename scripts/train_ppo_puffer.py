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

KL-regularized self-play stabilization (Orbit Wars 1st place / Lux AI S1),
both pieces ON by default:

1. Frozen-reference anchor -- continual, on for the whole run: the loss
   carries kl_coef * KL(pi_theta || pi_ref) over LEGAL actions only
   (PuffeRLPriorKL, pokemon_tcg/pufferl_kl.py), with pi_ref frozen and
   forward-only. --kl-coef 0 falls back to stock PuffeRL (that arm is the
   throughput/ablation baseline; it also disables the gate below, which
   needs the reference policy).
2. Checkpoint promotion (the ratchet): every --promote-every updates, the
   live actor plays mirrored-pair eval matches against pi_ref
   (pokemon_tcg/promotion.py). Win >70% of decisive games -> pi_ref becomes
   a frozen copy of the live policy (saved under <out>/refs/), else the old
   reference stands. Every decision lands in <out>/promotion_log.jsonl; the
   KL pull stays on throughout -- continual anchoring, not a warm start.

Per-update losses (incl. kl_to_prior) append to <out>/train_metrics.jsonl.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pokemon_tcg import config  # noqa: E402
from pokemon_tcg.device import resolve_device  # noqa: E402
from pokemon_tcg.promotion import evaluate_gate  # noqa: E402
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
    # One env per worker, always (cg per-process singleton, asserted below);
    # 8x8 is the measured real-run topology (rl_pipeline_v1.md ~61 steps/s).
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--bptt-horizon", type=int, default=128)
    ap.add_argument("--minibatch-size", type=int, default=512)
    ap.add_argument("--update-epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--ent-coef", type=float, default=0.001)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--kl-coef", type=float, default=0.05,
                    help="beta for the frozen-reference anchor "
                         "beta*KL(pi_theta||pi_ref); sweepable. 0 = stock "
                         "pufferl loss (no anchor forward pass, promotion "
                         "gate disabled) -- the ablation/throughput baseline. "
                         ">0 uses PuffeRLPriorKL (pokemon_tcg/pufferl_kl.py)")
    ap.add_argument("--kl-prior", type=Path, default=None,
                    help="initial frozen reference policy pi_ref for the anchor "
                         "(default: --init-from, i.e. the IL-lineage checkpoint "
                         "training starts from, so the anchor and the init "
                         "always match unless deliberately overridden)")
    ap.add_argument("--promote-every", type=int, default=20,
                    help="run the promotion gate every N updates (0 = never). "
                         "Needs --kl-coef > 0: the gate plays live vs the "
                         "anchor reference and ratchets it forward on a win")
    ap.add_argument("--promote-pairs", type=int, default=50,
                    help="mirrored pairs per gate (2x this many games); keep "
                         ">=50 for real runs, lower only for smoke tests")
    ap.add_argument("--promote-threshold", type=float, default=0.70,
                    help="promote when live wins STRICTLY more than this "
                         "fraction of decisive (non-draw) gate games")
    ap.add_argument("--promote-workers", type=int, default=8,
                    help="spawned eval-game processes per gate (cg engine is "
                         "a per-process singleton, so parallelism = processes)")
    ap.add_argument("--run-tag", default=time.strftime("ppo_%Y%m%d_%H%M%S"),
                    help="tag stamped into snapshots, promoted-ref metadata, "
                         "and every metrics/promotion log row")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="internal wall-clock budget: stop cleanly (with a final "
                         "snapshot) once exceeded -- replaces external kill timers")
    ap.add_argument("--opponent-module", default=None,
                    help="EXPLOITER MODE: train 100%% of episodes against this "
                         "single frozen benchmark agent (an AGENT_FILES name, "
                         "e.g. il_agent — loaded via load_agent, real main.py "
                         "bundle + its own deck). Overrides --league/--pool-"
                         "weights, disables the mirror bucket, and turns on "
                         "strict per-episode seat alternation. The opponent is "
                         "never updated; fallback tracking is enabled in the "
                         "env workers so the frozen target's decisions stay "
                         "auditable (see scripts/fallback_probe_puffer_env.py)")
    ap.add_argument("--league", default=str(config.MODELS_DIR / "il_agent"),
                    help="comma-separated frozen checkpoint dirs for the env's "
                         "league bucket (the 30%% draw)")
    ap.add_argument("--pool-weights", default=None,
                    help="comma-separated draw weights for the public-pool trio "
                         "(kiyotah,mechi22,plamen06), e.g. '0.6,0.25,0.15' for "
                         "PFSP-style frontier weighting; default uniform")
    ap.add_argument("--mix", default="0.625,0.375,0",
                    help="mirror,league,public-pool draw shares (must sum to 1). "
                         "Default drops the public pool entirely — pure "
                         "self-play + fictitious-self-play league, per the "
                         "2026-08-04 decision (external decks now unseen in "
                         "training; see notes/experiments/2026-08-04-league-"
                         "pool-composition.md). Old behavior: '0.5,0.3,0.2'")
    ap.add_argument("--init-policy-full", type=Path, default=None,
                    help="policy_full.pt from a prior run: restores actor AND "
                         "warmed value head (overrides --init-from for weights; "
                         "--init-from still sets the KL prior/architecture)")
    ap.add_argument("--device", default=None,
                    help="mps|cpu; default resolves via pokemon_tcg.device."
                         "resolve_device (PTCG_DEVICE env var, then auto)")
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
    args.device = resolve_device(args.device)
    if args.kl_prior is None:
        args.kl_prior = args.init_from
    if args.promote_every > 0 and args.kl_coef <= 0:
        print("NOTE: --kl-coef 0 runs stock PuffeRL (no reference policy), "
              "so the promotion gate is disabled for this run")

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
    mix = tuple(float(x) for x in args.mix.split(","))
    if len(mix) != 3 or abs(sum(mix) - 1.0) > 1e-6:
        ap.error(f"--mix needs 3 shares summing to 1, got {args.mix}")
    if mix[2] == 0 and not league:
        # _draw_opponent falls through to the public pool when the league
        # bucket is empty; with a zero pool share that would silently
        # reintroduce external opponents.
        ap.error("--mix with zero public-pool share requires a non-empty --league")
    env_kwargs = {"mirror_root": mirror_root, "anchor_ckpt": str(args.init_from),
                  "league": league, "pool_weights": pool_weights, "mix": mix}
    if args.opponent_module:
        # Exploiter mode: every draw lands in the league bucket, whose single
        # entry is the frozen module agent. No mirror (the opponent must never
        # track the learner), strict seat alternation, opponent decisions
        # audited by the fallback tracker (flag inherited by spawned workers;
        # read at agent-module import, so it must be set before vecenv).
        os.environ.setdefault("PTCG_FALLBACK_TRACK", "1")
        env_kwargs = {"mirror_root": None, "anchor_ckpt": str(args.init_from),
                      "league": [("module", args.opponent_module)],
                      "mix": (0.0, 1.0, 0.0), "pool_weights": None,
                      "alternate_seats": True}
        print(f"EXPLOITER MODE: frozen opponent = module '{args.opponent_module}' "
              f"(100% of episodes, seats strictly alternating)")
    vecenv = pvector.make(
        make_puffer_env,
        env_kwargs=env_kwargs,
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

        epoch = 0

        def save_actor(snap: Path, **extra) -> Path:
            """save_pretrained snapshot + run/step-tagged metadata.

            The cpu round-trip is the established pattern here: Module.to()
            swaps param data in place, so the optimizer's device-resident
            state survives (prior generations g2/g3 trained through it).
            """
            policy.actor.to("cpu").save_pretrained(snap)
            policy.actor.to(args.device)
            (snap / "ppo_metadata.json").write_text(json.dumps(
                {"global_step": trainer.global_step, "epoch": epoch,
                 "run_tag": args.run_tag, "init_from": str(args.init_from),
                 "kl_coef": args.kl_coef, "trainer": "pufferl-3.0", **extra},
                indent=2))
            return snap

        def append_jsonl(path: Path, row: dict) -> None:
            with path.open("a") as fh:
                fh.write(json.dumps(row) + "\n")

        metrics_path = args.out / "train_metrics.jsonl"
        promo_path = args.out / "promotion_log.jsonl"
        # Current frozen reference ON DISK: starts as the anchor prior, moves
        # forward only when the gate promotes. Eval workers load from disk,
        # never from the MPS learner.
        ref_dir = str(args.kl_prior)
        gate_on = args.kl_coef > 0 and args.promote_every > 0

        # Seed snapshot so mirror opponents have something to load from step 0.
        save_actor(args.out / "u0")

        last_snap = time.time()
        t_run = time.time()
        while trainer.global_step < train_config["total_timesteps"]:
            if args.max_seconds and time.time() - t_run > args.max_seconds:
                print(f"wall-clock budget {args.max_seconds:.0f}s reached at "
                      f"step {trainer.global_step} -- stopping cleanly")
                break
            trainer.evaluate()
            trainer.train()
            epoch += 1

            # Per-update metrics (kl_to_prior included when the anchor is on).
            sps = trainer.global_step / max(time.time() - t_run, 1e-9)
            row = {"ts": round(time.time(), 3), "run_tag": args.run_tag,
                   "global_step": int(trainer.global_step), "epoch": epoch,
                   "sps": round(sps, 2), "ref": ref_dir}
            row.update({k: (float(v) if math.isfinite(v) else None)
                        for k, v in dict(getattr(trainer, "losses", {})).items()
                        if isinstance(v, (int, float))})
            append_jsonl(metrics_path, row)

            if gate_on and epoch % args.promote_every == 0:
                # Promotion gate (the ratchet): mirrored pairs of live vs the
                # frozen reference; training is paused for its duration.
                live_snap = save_actor(args.out / f"u{trainer.global_step}")
                print(f"promotion gate at update {epoch} (step "
                      f"{trainer.global_step}): live vs {ref_dir}, "
                      f"{args.promote_pairs} mirrored pairs...")
                verdict = evaluate_gate(
                    live_snap, ref_dir, pairs=args.promote_pairs,
                    workers=args.promote_workers,
                    threshold=args.promote_threshold)
                if verdict["promote"]:
                    ref_snap = save_actor(
                        args.out / "refs" / f"u{trainer.global_step}",
                        promoted_from=str(live_snap), gate=verdict)
                    trainer.retarget_prior(policy)
                    ref_dir = str(ref_snap)
                verdict.update({"global_step": int(trainer.global_step),
                                "epoch": epoch, "run_tag": args.run_tag,
                                "new_ref": ref_dir})
                append_jsonl(promo_path, verdict)
                outcome = (f"PROMOTED -> {ref_dir}" if verdict["promote"]
                           else f"kept {ref_dir}")
                print(f"  gate: {verdict['wins']}W/{verdict['losses']}L/"
                      f"{verdict['draws']}D of {verdict['games']} "
                      f"win_rate={verdict['win_rate']:.3f} vs "
                      f">{args.promote_threshold:.2f} -- {outcome} "
                      f"[{verdict['seconds']:.0f}s]")
                # The gate's live snapshot doubles as the periodic one.
                last_snap = time.time()
            elif time.time() - last_snap >= args.snapshot_every_s:
                snap = save_actor(args.out / f"u{trainer.global_step}")
                last_snap = time.time()
                print(f"  snapshot -> {snap}")

        elapsed = time.time() - t_run
        print(f"throughput: {trainer.global_step} steps in {elapsed:.0f}s = "
              f"{trainer.global_step / max(elapsed, 1e-9):.1f} steps/s "
              f"(kl_coef={args.kl_coef}, includes gate pauses)")

        snap = args.out / f"u{trainer.global_step}_final"
        policy.actor.to("cpu").save_pretrained(snap)
        (snap / "ppo_metadata.json").write_text(json.dumps(
            {"global_step": trainer.global_step, "epoch": epoch,
             "run_tag": args.run_tag, "init_from": str(args.init_from),
             "kl_coef": args.kl_coef, "final_ref": ref_dir,
             "trainer": "pufferl-3.0"}, indent=2))
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
