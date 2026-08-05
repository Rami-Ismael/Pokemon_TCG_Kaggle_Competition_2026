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
from pokemon_tcg.deck_pool import DeckPool  # noqa: E402
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
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="v2 default 1e-4 (was 3e-5): gen-2 measured clipfrac "
                         "~0.5%%/approx_kl ~0.002 at 3e-5 -- updates far inside "
                         "the trust region, little net movement per budget")
    ap.add_argument("--ent-coef-init", "--ent-coef", dest="ent_coef_init",
                    type=float, default=0.01,
                    help="entropy bonus at step 0 (rl_pipeline_v2 §3.2: the "
                         "schedule is its own knob; Orbit Wars 3rd place "
                         "called entropy annealing the most important one)")
    ap.add_argument("--ent-coef-final", type=float, default=0.001,
                    help="entropy bonus after the anneal (equal to init = "
                         "constant, the old behavior)")
    ap.add_argument("--ent-anneal-frac", type=float, default=0.5,
                    help="fraction of total timesteps over which the entropy "
                         "coef anneals linearly init->final, then holds")
    ap.add_argument("--gamma", type=float, default=0.997,
                    help="v2 default 0.997 (gen-2's recorded call: time "
                         "preference against game-dragging; ~1 stays legal "
                         "for terminal-only episodic reward)")
    ap.add_argument("--gae-lambda", type=float, default=0.95)
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
                    help="comma-separated INITIAL draw weights for the public-"
                         "pool trio (kiyotah,mechi22,plamen06); default uniform. "
                         "With PFSP refresh on, these only seed the first "
                         "refresh interval")
    ap.add_argument("--opp-hold", type=int, default=4,
                    help="PFSP-lite: each env keeps its drawn opponent for "
                         "this many consecutive episodes (stable win-rate "
                         "runs harvested from rollouts, §3.3)")
    ap.add_argument("--pfsp-refresh-every", type=int, default=10,
                    help="rewrite pool_weights.json from rollout-harvested "
                         "win-rate EMAs every N updates (0 = static weights; "
                         "also moot when --mix zeroes the public-pool share). "
                         "Weights follow w = max(wr*(1-wr), 0.05): peak "
                         "learning signal near 50%% matchups, floored so no "
                         "opponent fully disappears")
    ap.add_argument("--pfsp-ema", type=float, default=0.15,
                    help="EMA step for harvested per-opponent win rates")
    ap.add_argument("--deck-pool", default=None,
                    help="deck-diversity sweep: pool spec sampled once per "
                         "episode. Forms: comma-separated deck refs (a path, a "
                         "configs/deck_lists stem, or an agents/<name>), "
                         "'@manifest.txt' (one ref per line), 'all:decklists', "
                         "or 'all:agents' (content-deduped). Default None keeps "
                         "the single hardcoded deck — pre-sweep behaviour")
    ap.add_argument("--deck-pool-k", type=int, default=None,
                    help="take a K-sized subset of --deck-pool (the swept axis)")
    ap.add_argument("--deck-pool-seed", type=int, default=None,
                    help="which K-subset to take (the sweep's nesting seed); "
                         "default None = first K by name. Subsets are nested: "
                         "for a fixed seed, P_4 subset-of P_16 subset-of P_33")
    ap.add_argument("--deck-pool-pin", default=None,
                    help="comma-separated refs forced to the front of the "
                         "subset order, matched by deck CONTENT. The sweep "
                         "pins il_agent so K=1 is exactly the v1 baseline deck")
    ap.add_argument("--mirror-deck", action="store_true",
                    help="both seats play the episode's pooled deck. Required "
                         "for the mirror control: without it a module opponent "
                         "keeps serving its own bundled deck and a K>1 pool "
                         "silently becomes a cross-deck matchup")
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
        "gae_lambda": args.gae_lambda,
        "update_epochs": args.update_epochs,
        "ent_coef": args.ent_coef_init,
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
    POOL_NAMES = ["kiyotah_dragapult", "mechi22_alakazam", "plamen06_steel"]
    weights_path = None
    if args.pfsp_refresh_every > 0:
        # Seed the weights file BEFORE envs spawn so no worker reads a stale
        # file left by a previous run into the same out dir. (Inert when the
        # --mix public-pool share is zero -- no pool draws, no harvest.)
        weights_path = args.out / "pool_weights.json"
        init_w = pool_weights or [1.0] * len(POOL_NAMES)
        weights_path.write_text(json.dumps(dict(zip(POOL_NAMES, init_w))))
    mix = tuple(float(x) for x in args.mix.split(","))
    if len(mix) != 3 or abs(sum(mix) - 1.0) > 1e-6:
        ap.error(f"--mix needs 3 shares summing to 1, got {args.mix}")
    if mix[2] == 0 and not league:
        # _draw_opponent falls through to the public pool when the league
        # bucket is empty; with a zero pool share that would silently
        # reintroduce external opponents.
        ap.error("--mix with zero public-pool share requires a non-empty --league")
    env_kwargs = {"mirror_root": mirror_root, "anchor_ckpt": str(args.init_from),
                  "league": league, "pool_weights": pool_weights, "mix": mix,
                  "pool_weights_path": str(weights_path) if weights_path else None,
                  "opp_hold": args.opp_hold}
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

    if args.deck_pool:
        pool = DeckPool.from_spec(
            args.deck_pool, limit=args.deck_pool_k, seed=args.deck_pool_seed,
            pin=args.deck_pool_pin.split(",") if args.deck_pool_pin else None)
        env_kwargs["deck_pool"] = pool
        env_kwargs["mirror_deck"] = args.mirror_deck
        print(f"DECK POOL: K={len(pool)} mirror={args.mirror_deck} :: "
              f"{', '.join(pool.names)}")
        if len(pool) > 1 and not args.mirror_deck:
            print("WARNING: K>1 without --mirror-deck. The opponent will keep "
                  "playing its own bundled deck while the learner varies, so "
                  "this measures deck MATCHUP, not policy exploitability.")
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

        # PFSP-lite harvest state (§3.3): per-opponent win-rate EMAs from the
        # wr_<slug> terminal infos, consumed-length bookkeeping because
        # pufferl's stats lists accumulate until its throttled log clears them.
        from collections import defaultdict as _dd
        wr_ema: dict[str, float] = {}
        wr_seen: dict[str, int] = _dd(int)
        wr_games: dict[str, int] = _dd(int)

        def harvest(stats) -> None:
            for k, v in list(stats.items()):
                if not k.startswith("wr_") or not isinstance(v, list):
                    continue
                if len(v) < wr_seen[k]:
                    wr_seen[k] = 0  # upstream cleared its stats on a log tick
                fresh, wr_seen[k] = v[wr_seen[k]:], len(v)
                slug = k[3:]
                for x in fresh:
                    x = float(x)
                    wr_ema[slug] = (x if slug not in wr_ema
                                    else (1 - args.pfsp_ema) * wr_ema[slug]
                                    + args.pfsp_ema * x)
                    wr_games[slug] += 1

        total_ts = train_config["total_timesteps"]

        def ent_coef_at(step: int) -> float:
            span = max(int(total_ts * args.ent_anneal_frac), 1)
            frac = min(step / span, 1.0)
            return args.ent_coef_init + (args.ent_coef_final - args.ent_coef_init) * frac

        last_snap = time.time()
        t_run = time.time()
        while trainer.global_step < train_config["total_timesteps"]:
            if args.max_seconds and time.time() - t_run > args.max_seconds:
                print(f"wall-clock budget {args.max_seconds:.0f}s reached at "
                      f"step {trainer.global_step} -- stopping cleanly")
                break
            # Entropy schedule (§3.2): PuffeRL reads config['ent_coef'] per
            # train() call, and self.config is the dict we passed by reference.
            ent_coef = ent_coef_at(trainer.global_step)
            trainer.config["ent_coef"] = ent_coef
            harvest(trainer.evaluate())
            trainer.train()
            epoch += 1

            if weights_path is not None and epoch % args.pfsp_refresh_every == 0:
                # w = wr(1-wr) peaks at 50% matchups (max learning signal
                # under terminal-only reward), floored so nobody vanishes;
                # unseen opponents sit at the 0.25 uniform-ish default.
                w = {n: max(round(wr_ema.get(n, 0.5) * (1 - wr_ema.get(n, 0.5)), 4),
                            0.05) for n in POOL_NAMES}
                tmp = weights_path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(w))
                os.replace(tmp, weights_path)

            # Per-update metrics (kl_to_prior included when the anchor is on).
            sps = trainer.global_step / max(time.time() - t_run, 1e-9)
            row = {"ts": round(time.time(), 3), "run_tag": args.run_tag,
                   "global_step": int(trainer.global_step), "epoch": epoch,
                   "sps": round(sps, 2), "ref": ref_dir,
                   "ent_coef": round(ent_coef, 6)}
            row.update({k: (float(v) if math.isfinite(v) else None)
                        for k, v in dict(getattr(trainer, "losses", {})).items()
                        if isinstance(v, (int, float))})
            if wr_ema:
                row["pool_wr"] = {s: round(v, 3) for s, v in sorted(wr_ema.items())}
                row["pool_games"] = dict(sorted(wr_games.items()))
            pc = getattr(trainer, "per_context", None)
            if pc:
                row["per_context"] = pc
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
