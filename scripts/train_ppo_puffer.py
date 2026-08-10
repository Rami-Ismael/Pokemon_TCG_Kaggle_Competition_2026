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
import hashlib
import json
import math
import os
import re
import subprocess
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

def _deck_fingerprint(env_kwargs: dict) -> dict:
    """What deck(s) this run actually trained on.

    The single most load-bearing missing field: with no --deck-pool the
    learner AND its checkpoint opponents play `selfplay.DECK_PATH`, which is
    hardcoded to agents/il_agent/deck.csv (Mega Lucario ex). Every self-play
    generation through g3 trained only on that deck, but nothing in the
    checkpoint said so -- so selfplay_g1 was submitted piloting Grimmsnarl, a
    deck it had never self-played on, and its 254.9 was read as a generation
    result rather than an off-distribution one.
    """
    pool = env_kwargs.get("deck_pool")
    if pool is not None:
        return {"deck_source": "deck_pool", "deck_names": list(pool.names),
                "deck_k": len(pool), "mirror_deck": bool(env_kwargs.get("mirror_deck"))}
    from pokemon_tcg import selfplay as _sp
    path = _sp.DECK_PATH
    try:
        raw = path.read_bytes()
        cards = [int(x) for x in raw.decode().split() if x.strip()][:60]
        return {"deck_source": "hardcoded", "deck_path": str(path),
                "deck_sha256": hashlib.sha256(raw).hexdigest(),
                "deck_n_cards": len(cards)}
    except OSError:
        return {"deck_source": "hardcoded", "deck_path": str(path),
                "deck_sha256": None, "deck_n_cards": None}


def _init_chain(start: Path | str | None, _depth: int = 0) -> list[dict]:
    """Walk init_from back through prior generations.

    A generation's own `global_step` is per-run, so g3's 1,229,824 is NOT its
    total training -- cumulative across the chain is. Returns newest-first.
    """
    out: list[dict] = []
    cur = Path(start) if start else None
    seen: set[str] = set()
    while cur is not None and _depth < 32:
        key = str(cur)
        if key in seen:
            break
        seen.add(key)
        try:
            m = json.loads((cur / "ppo_metadata.json").read_text())
        except (OSError, json.JSONDecodeError):
            out.append({"dir": key, "global_step": None, "note": "no ppo_metadata (BC init or missing)"})
            break
        out.append({"dir": key, "global_step": m.get("global_step"),
                    "run_tag": m.get("run_tag")})
        nxt = m.get("init_from")
        cur = Path(nxt) if nxt and nxt != "None" else None
        _depth += 1
    return out


def _git_sha() -> str | None:
    """Commit only. git_state() is the single implementation -- a run launched
    from a dirty tree is not the commit it names, and only that function says
    so, but the snapshot provenance block wants the bare sha."""
    return git_state()["commit"]


def _slug(text: str) -> str:
    """Filesystem-safe name component: anything odd collapses to '-'."""
    return re.sub(r"[^A-Za-z0-9.-]+", "-", str(text)).strip("-")


def _checkpoint_slug(path: Path) -> str:
    """Cite a checkpoint by its path under models/ with slashes folded, so
    models/selfplay_g1/refs/u430080 reads selfplay_g1-refs-u430080 instead of
    the bare u430080."""
    p = Path(path)
    try:
        parts = p.resolve().relative_to(config.MODELS_DIR.resolve()).parts
    except ValueError:
        parts = (p.name,)
    return _slug("-".join(parts))


def build_run_tag(label: str | None, mix: tuple[float, ...],
                  opponent_module: str | None, deck_pool,
                  mirror_deck: bool, init_from: Path, seed: int) -> str:
    """The run's name, with every axis that has burned us spelled out.

    ppo_puffer/u120832 trained with 20% external opponents and selfplay_g1
    with 0%; the names said neither, so the 275.1-vs-254.9 ladder gap was
    unattributable without exhuming shell scripts and git history. Same story
    for the deck (see _deck_fingerprint). So the tag always carries the
    opponent mix, the deck, the init checkpoint, and the seed. A --run-tag
    label is kept as a prefix, never as a replacement.
    """
    def pct(x: float) -> str:
        return format(round(x * 100, 1), "g")

    if opponent_module:
        opponents = f"exploit100-{_slug(opponent_module)}"
    else:
        # `mix` is (mirror, league) after validation -- the externals share is
        # pinned to 0 and its bucket is deleted from the env. Still spelled out
        # in the tag: the point of this name is that no run can hide its
        # opponent mix, and "external0" states it rather than leaving a reader
        # to infer it from an absence.
        external = mix[2] if len(mix) > 2 else 0.0
        opponents = (f"mirror{pct(mix[0])}-league{pct(mix[1])}"
                     f"-external{pct(external)}")
    if deck_pool is not None:
        deck = f"deckpool{len(deck_pool)}" + ("-mirror" if mirror_deck else "")
    else:
        from pokemon_tcg import selfplay as _sp
        deck = f"deck-{_slug(_sp.DECK_PATH.parent.name)}"
    # args.seed, NOT config.RANDOM_SEED: --seed is what actually moves the
    # vecenv (and so every opponent/deck draw). Stamping the module constant
    # would name three differently-seeded arms of one comparison identically.
    return "_".join([_slug(label) if label else "ppo", opponents, deck,
                     f"from-{_checkpoint_slug(init_from)}",
                     f"seed{seed}"])


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


def git_state() -> dict:
    """Commit + dirtiness of the tree this run launched from.

    Recorded because argparse DEFAULTS MOVE. `--mix` did not exist before
    e8298fc (2026-08-04 20:06); runs before it drew the public pool 20% of the
    time from PTCGGym's own default, and runs after it drew it 0%. Reading
    today's `--mix` help text onto an 08-03 run gets the opponent distribution
    exactly backwards, which is a wrong answer about what the run even was.
    A commit makes the run's real configuration recoverable.
    """
    def _git(*a: str) -> str | None:
        try:
            return subprocess.run(("git", *a), cwd=config.PROJECT_ROOT,
                                  capture_output=True, text=True,
                                  timeout=10).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None

    return {"commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(_git("status", "--porcelain"))}


def opponent_composition(args, env_kwargs: dict) -> dict:
    """The resolved answer to 'what did this run actually play against?'

    Derived here, at launch, rather than inferred later from flags: the league
    bucket holds a MIX of our own checkpoints ("ckpt") and external benchmark
    agents ("module"), so the share of external opponents is not any single
    flag -- it is mix[1] weighted by how much of the league is external. That
    composite is the number every past post-mortem got wrong.

    In normal self-play this is now provably 0.0: --mix pins the externals
    share to zero and --league entries are validated to be our own saved
    checkpoints. It is still computed rather than hardcoded, so if either
    guard is ever weakened the number moves and says so.
    """
    mix = tuple(env_kwargs.get("mix", (0.625, 0.375)))
    league = env_kwargs.get("league") or []
    n_ext = sum(1 for kind, _ in league if kind == "module")
    league_ext_frac = (n_ext / len(league)) if league else 0.0
    # The public-pool bucket is DELETED from the env, not zeroed, and --mix
    # pins its share to 0, so there is no third term to add and no roster to
    # name. The league can still be all-external in exploiter mode
    # (--opponent-module), which is the one path this must keep reporting.
    pool_share = mix[2] if len(mix) > 2 else 0.0
    return {
        "mix_mirror_league": list(mix),
        "league": [list(x) for x in league],
        "league_external_fraction": round(league_ext_frac, 4),
        # P(opponent is not one of our own policies) for a single draw.
        "external_opponent_share": round(mix[1] * league_ext_frac + pool_share, 4),
        "opp_hold_episodes": env_kwargs.get("opp_hold"),
        "alternate_seats": bool(env_kwargs.get("alternate_seats", False)),
    }


def write_run_config(args, env_kwargs: dict) -> Path:
    """Freeze the fully resolved run configuration to <out>/run_config.json.

    ppo_metadata.json records where a snapshot came from; this records what the
    run WAS -- resolved args, git state, and the opponent/deck distribution --
    so no future reader has to reconstruct it from argparse defaults that have
    since moved. Written before the first env spawns, so it exists even if the
    run dies early.
    """
    resolved = {str(k): (str(v) if isinstance(v, Path) else v)
                for k, v in sorted(vars(args).items())}
    cfg = {
        "run_tag": args.run_tag,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "started_unix": round(time.time(), 3),
        "git": git_state(),
        "opponents": opponent_composition(args, env_kwargs),
        # _deck_fingerprint is the shared implementation (it also hashes the
        # deck file, which catches an edited deck.csv that keeps its name).
        "decks": {"deck_pool_spec": args.deck_pool, **_deck_fingerprint(env_kwargs)},
        "args": resolved,
    }
    out = args.out / "run_config.json"
    out.write_text(json.dumps(cfg, indent=2) + "\n")
    o, d = cfg["opponents"], cfg["decks"]
    print(f"run config -> {out}")
    print(f"  git {cfg['git']['commit'][:9] if cfg['git']['commit'] else '?'}"
          f"{' DIRTY' if cfg['git']['dirty'] else ''} on {cfg['git']['branch']}")
    print(f"  opponents: mix={o['mix_mirror_league_pool']} "
          f"external share={o['external_opponent_share']:.0%}  "
          f"decks: {d.get('deck_source')} "
          f"K={d.get('deck_k', 1)} mirror={d.get('mirror_deck', False)}")
    return out


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
    ap.add_argument("--il-prior", type=Path, default=None,
                    help="ORIGINAL IL checkpoint, used as a second frozen "
                         "reference that is NEVER retargeted, so kl_to_il_prior "
                         "answers 'has it forgotten the human prior' after the "
                         "anchor has moved (rl_pipeline_v4 §3.3). Diagnostic "
                         "only -- never enters the loss. Default: --kl-prior, "
                         "which is right for generation 1 and WRONG from gen 2 "
                         "on, where --init-from is a promoted self-play "
                         "checkpoint: pass the real IL checkpoint explicitly. "
                         "'none' disables the second reference")
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
    ap.add_argument("--run-tag", default=None,
                    help="optional human label, kept as a PREFIX: the final "
                         "tag always appends the derived descriptor (opponent "
                         "mix, deck, init checkpoint, seed — see "
                         "build_run_tag) so no run can hide its opponent mix "
                         "behind a nickname again. The full tag is stamped "
                         "into snapshot dir names, promoted-ref metadata, and "
                         "every metrics/promotion log row")
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
                         "league bucket (our own lineage only)")
    ap.add_argument("--opp-hold", type=int, default=4,
                    help="each env keeps its drawn opponent for this many "
                         "consecutive episodes, so per-opponent win-rate "
                         "estimates harvested from rollouts are runs rather "
                         "than single samples")
    ap.add_argument("--deck-pool", default=None,
                    help="deck pool. BOTH seats draw from it, independently, "
                         "once per episode. Forms: comma-separated deck refs (a "
                         "path, a configs/deck_lists stem, or an agents/<name>), "
                         "'@manifest.txt' (one ref per line), 'all:decklists', "
                         "'all:agents', or 'all:decks' (the union, content-"
                         "deduped). Default None keeps the single hardcoded "
                         "deck on both seats — the pre-2026-08-05 behaviour. "
                         "Gate any pool through scripts/probe_deck_legality.py "
                         "first: an illegal deck is a free loss every episode "
                         "it is drawn")
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
                    help="SAME-DECK CONTROL: one draw per episode, both seats. "
                         "Default (off) draws the two seats' decks "
                         "independently. Use this arm when a measurement must "
                         "not be contaminated by deck MATCHUP — e.g. the "
                         "exploitability sweep")
    ap.add_argument("--field-pool", default="",
                    help="MUST BE EMPTY. Accepted only so that a script or a "
                         "merge carrying a field pool of external agents fails "
                         "LOUDLY instead of silently training against them. "
                         "Self-play means the agent plays its own lineage: the "
                         "live mirror and our own frozen checkpoints. Rule "
                         "baselines and other competitors' submissions are "
                         "never training opponents — that is what keeps them "
                         "unseen, and the local evaluation pool a real "
                         "out-of-sample filter. For a deliberate single frozen "
                         "external target use --opponent-module (exploiter "
                         "mode), which is labelled as the diagnostic it is")
    ap.add_argument("--deck-report-every", type=int, default=10,
                    help="append per-deck and per-matchup win-rate tables to "
                         "<out>/deck_metrics.jsonl every N updates")
    ap.add_argument("--mix", default="0.625,0.375,0",
                    help="mirror,league[,externals] draw shares (must sum to 1). "
                         "THE LAST NUMBER IS ALWAYS ZERO and a nonzero value is "
                         "a hard error, not an option: opponents are "
                         "LINEAGE-ONLY (the live mirror and our own frozen "
                         "checkpoints). Externals stay unseen in training so "
                         "the local evaluation pool remains a real "
                         "out-of-sample filter. The bucket itself is DELETED "
                         "from the env — the third share exists only as a "
                         "pinned zero, so there is no code path to fall into "
                         "(extends notes/experiments/2026-08-04-league-pool-"
                         "composition.md)")
    ap.add_argument("--init-policy-full", type=Path, default=None,
                    help="policy_full.pt from a prior run: restores actor AND "
                         "warmed value head (overrides --init-from for weights; "
                         "--init-from still sets the KL prior/architecture)")
    ap.add_argument("--device", default=None,
                    help="mps|cpu; default resolves via pokemon_tcg.device."
                         "resolve_device (PTCG_DEVICE env var, then auto)")
    ap.add_argument("--snapshot-every-s", type=float, default=300.0,
                    help="actor save_pretrained cadence (mirror hot-reload + benchmark)")
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED,
                    help="seeds the PPO trainer AND the vecenv (which is what "
                         "moves the per-env opponent/deck draws). Arms of a "
                         "controlled comparison must be run at the SAME seeds, "
                         "since seed-to-seed spread here is large enough to "
                         "swamp the effect being measured")
    ap.add_argument("--out", type=Path, default=None,
                    help="snapshot dir; default models/<full run tag>, so "
                         "runs stop piling into one anonymous models/ppo_puffer")
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
        if args.run_tag is None:
            args.run_tag = "smoke"  # a smoke run must never look like a real one
    args.device = resolve_device(args.device)
    if args.kl_prior is None:
        args.kl_prior = args.init_from
    if args.il_prior is None:
        args.il_prior = args.kl_prior
    elif str(args.il_prior).lower() == "none":
        args.il_prior = None
    if args.promote_every > 0 and args.kl_coef <= 0:
        print("NOTE: --kl-coef 0 runs stock PuffeRL (no reference policy), "
              "so the promotion gate is disabled for this run")

    # HARD CONSTRAINT (verified 2026-08-03): the cg/cabt native engine keeps ONE
    # battle state per process; two live envs in one process corrupt each other
    # and the dylib exits silently. Exactly one env per worker, always.
    assert args.num_envs == args.num_workers, (
        f"cg engine is a per-process singleton: need num_envs == num_workers, "
        f"got {args.num_envs} != {args.num_workers}")

    # Run identity BEFORE anything is created on disk: parse the opponent/deck
    # config, derive the full run tag from it, and only then resolve --out
    # (default models/<tag>), so every directory this run writes is named
    # after what the run actually is.
    if args.field_pool.strip():
        ap.error(
            f"--field-pool {args.field_pool!r}: must be empty. Self-play trains "
            f"the agent against its LINEAGE (the live mirror and our own frozen "
            f"checkpoints) and nothing else. Externals as training opponents "
            f"would stop them being unseen, which is the only thing making the "
            f"local evaluation pool an out-of-sample read. Use "
            f"--opponent-module for a deliberate single frozen external "
            f"target (exploiter mode).")
    league = [("ckpt", p) for p in args.league.split(",") if p]
    # The league is LINEAGE: our own saved checkpoints, nothing else. Without
    # this check an entry is taken as a checkpoint path on faith, so an agent
    # NAME lands here as ("ckpt", "wmh_grimmsnarl") and fails inside a spawned
    # worker -- which presents as that seat crashing every episode, i.e. free
    # wins for the learner, not as an error anyone sees.
    not_ckpt = [p for _, p in league if not (Path(p) / "config.json").exists()]
    if not_ckpt:
        ap.error(
            f"--league entries are not saved checkpoints: {not_ckpt}. The "
            f"league holds OUR OWN frozen checkpoints (a dir with a "
            f"config.json) — self-play trains against its own lineage. A "
            f"benchmark agent name here is an external opponent wearing a "
            f"checkpoint's clothes; use --opponent-module if you deliberately "
            f"want one frozen external target (exploiter mode).")
    mix = tuple(float(x) for x in args.mix.split(","))
    if len(mix) not in (2, 3) or abs(sum(mix) - 1.0) > 1e-6:
        ap.error(f"--mix needs 2 shares (mirror,league) or 3 with a ZERO "
                 f"externals share, all summing to 1; got {args.mix}")
    if len(mix) == 3:
        # The third share is accepted only as a pinned zero. Scripts and notes
        # across the repo still write three numbers, and silently dropping a
        # nonzero one would be worse than refusing it -- that is precisely how
        # externals would slip back into training and quietly destroy the local
        # evaluation pool's out-of-sample status.
        if mix[2] != 0.0:
            ap.error(
                f"--mix {args.mix}: the externals share must be 0. Opponents "
                f"are lineage-only by design (the live mirror and our own "
                f"frozen checkpoints); the externals bucket is deleted from "
                f"the env, so a nonzero share has nothing to draw from. Train "
                f"against externals only via --opponent-module, which is "
                f"exploiter mode and is labelled as such.")
        mix = mix[:2]
    if mix[1] > 0 and not league:
        ap.error("--mix gives the league bucket a nonzero share but --league "
                 "is empty")
    deck_pool = None
    if args.deck_pool:
        deck_pool = DeckPool.from_spec(
            args.deck_pool, limit=args.deck_pool_k, seed=args.deck_pool_seed,
            pin=args.deck_pool_pin.split(",") if args.deck_pool_pin else None)
    args.run_tag = build_run_tag(args.run_tag, mix, args.opponent_module,
                                 deck_pool, args.mirror_deck, args.init_from,
                                 args.seed)
    if args.out is None:
        args.out = config.MODELS_DIR / args.run_tag
    print(f"run tag: {args.run_tag}")
    print(f"out dir: {args.out}")

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
        "seed": args.seed,
    })

    mirror_root = str(args.out)  # env workers hot-reload newest snapshot from here
    args.out.mkdir(parents=True, exist_ok=True)

    env_kwargs = {"mirror_root": mirror_root, "anchor_ckpt": str(args.init_from),
                  "league": league, "mix": mix, "opp_hold": args.opp_hold}
    if args.opponent_module:
        # Exploiter mode: every draw lands in the league bucket, whose single
        # entry is the frozen module agent. No mirror (the opponent must never
        # track the learner), strict seat alternation, opponent decisions
        # audited by the fallback tracker (flag inherited by spawned workers;
        # read at agent-module import, so it must be set before vecenv).
        os.environ.setdefault("PTCG_FALLBACK_TRACK", "1")
        env_kwargs = {"mirror_root": None, "anchor_ckpt": str(args.init_from),
                      "league": [("module", args.opponent_module)],
                      "mix": (0.0, 1.0), "alternate_seats": True}
        print(f"EXPLOITER MODE: frozen opponent = module '{args.opponent_module}' "
              f"(100% of episodes, seats strictly alternating)")

    if deck_pool is not None:
        env_kwargs["deck_pool"] = deck_pool
        env_kwargs["mirror_deck"] = args.mirror_deck
        mode = ("MIRROR (one draw, both seats)" if args.mirror_deck
                else "INDEPENDENT (two draws per episode)")
        print(f"DECK POOL: K={len(deck_pool)} mode={mode} :: "
              f"{', '.join(deck_pool.names)}")
        if len(deck_pool) == 1:
            print("NOTE: K=1 -- both seats always play the same single deck, "
                  "so this is the single-deck control regardless of "
                  "--mirror-deck.")
    else:
        print("DECK POOL: none -- both seats play the single hardcoded deck "
              "(selfplay.load_deck / agents/il_agent/deck.csv). This is the "
              "single-deck CONTROL arm.")

    write_run_config(args, env_kwargs)

    vecenv = pvector.make(
        make_puffer_env,
        env_kwargs=env_kwargs,
        backend=pvector.Multiprocessing,  # never Serial >1 env: cg singleton
        num_envs=args.num_envs,
        num_workers=args.num_workers,
        seed=args.seed,
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
            # Separate module instance on purpose: retarget_prior() mutates
            # `prior` in place, and sharing the object would drag the
            # never-retargeted IL reference along with it (§3.3).
            il_prior = None
            if args.il_prior is not None:
                il_prior = PTCGPufferPolicy(str(args.il_prior)).to(args.device)
            trainer = PuffeRLPriorKL(train_config, vecenv, policy, None,
                                     kl_prior_policy=prior, kl_coef=args.kl_coef,
                                     il_prior_policy=il_prior)
            print(f"KL anchor ON: coef={args.kl_coef} prior={args.kl_prior}")
            if il_prior is not None:
                same = str(args.il_prior) == str(args.kl_prior)
                print(f"  second reference (diagnostic, never retargeted): "
                      f"{args.il_prior}"
                      + ("  [identical to the anchor until the first retarget: "
                         "kl_to_il_prior is a copy of kl_to_prior until then]"
                         if same else ""))
            else:
                print("  second reference DISABLED: kl_to_il_prior will not be "
                      "logged, so forgetting the human prior is unmeasurable "
                      "after the first retarget")
        else:
            trainer = pufferl.PuffeRL(train_config, vecenv, policy, None)
        # PuffeRL hardcodes torch.amp.autocast(device_type='cuda'); on mps/cpu
        # with fp32 that context is at best a no-op, at worst an error.
        trainer.amp_context = contextlib.nullcontext()

        epoch = 0

        # Everything a later reader needs to attribute a generation's result
        # WITHOUT re-reading the shell script that launched it. g1->g2 moved
        # five axes at once (init weights, warm critic, KL anchor, league
        # size, wall budget) and none of them were recorded in the
        # checkpoint, so the step was unattributable after the fact.
        chain = _init_chain(args.init_from)
        provenance = {
            "trainer": "pufferl-3.0",
            "git_sha": _git_sha(),
            "init_from": str(args.init_from),
            "init_policy_full": str(args.init_policy_full) if args.init_policy_full else None,
            "kl_prior": str(args.kl_prior) if args.kl_prior else None,
            "kl_coef": args.kl_coef,
            "league": [s for s in args.league.split(",") if s],
            "opponent_module": args.opponent_module,
            "mix_mirror_league": args.mix,
            "opponents": "lineage-only (live mirror + our frozen checkpoints)",
            "field_pool": args.field_pool or None,
            # DERIVED, not read off a flag -- reading it off a flag is what made
            # the 275.1-vs-254.9 gap look like a generation effect. On this
            # branch the guards make it provably 0.0 (externals share pinned to
            # zero, league entries validated as our own checkpoints), and it is
            # recorded anyway so a future change that breaks that shows up here.
            "external_opponent_share":
                opponent_composition(args, env_kwargs)["external_opponent_share"],
            "opp_hold": args.opp_hold,
            "seed": args.seed,
            "promote_every": args.promote_every,
            "promote_pairs": args.promote_pairs,
            "promote_threshold": args.promote_threshold,
            "hyperparams": {
                "lr": args.lr, "gamma": args.gamma, "gae_lambda": args.gae_lambda,
                "ent_coef_init": args.ent_coef_init, "ent_coef_final": args.ent_coef_final,
                "ent_anneal_frac": args.ent_anneal_frac,
                "minibatch_size": args.minibatch_size, "bptt_horizon": args.bptt_horizon,
                "update_epochs": args.update_epochs, "num_envs": args.num_envs,
                "num_workers": args.num_workers, "device": str(args.device),
            },
            "budget": {"total_timesteps": args.total_timesteps,
                       "max_seconds": args.max_seconds},
            "init_chain": chain,
            "cumulative_steps_prior": sum(
                c["global_step"] or 0 for c in chain),
            **_deck_fingerprint(env_kwargs),
        }
        # Mutable because save_actor closes over it but t_run/gate counts are
        # only known once the loop is running.
        run_state = {"t_run": None, "gates": 0, "promotions": 0}

        def snap_name(step: int, final: bool = False) -> str:
            """Snapshot dir names carry the full run tag: a bare u<step> dir
            copied into agents/ becomes an agent named u120832 that says
            nothing about how it trained. Steps are zero-padded so lexical
            sort equals step order."""
            return f"{args.run_tag}_step{step:08d}" + ("_final" if final else "")

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
                 "run_tag": args.run_tag,
                 "cumulative_steps": provenance["cumulative_steps_prior"]
                                     + int(trainer.global_step),
                 "wall_seconds": (round(time.time() - run_state["t_run"], 1)
                                  if run_state["t_run"] else None),
                 "gates_run": run_state["gates"],
                 "promotions": run_state["promotions"],
                 **provenance, **extra},
                indent=2))
            return snap

        def append_jsonl(path: Path, row: dict) -> None:
            with path.open("a") as fh:
                fh.write(json.dumps(row) + "\n")

        metrics_path = args.out / "train_metrics.jsonl"
        deck_metrics_path = args.out / "deck_metrics.jsonl"
        promo_path = args.out / "promotion_log.jsonl"
        # Current frozen reference ON DISK: starts as the anchor prior, moves
        # forward only when the gate promotes. Eval workers load from disk,
        # never from the MPS learner.
        ref_dir = str(args.kl_prior)
        gate_on = args.kl_coef > 0 and args.promote_every > 0

        # Seed snapshot so mirror opponents have something to load from step 0.
        save_actor(args.out / snap_name(0))

        # Outcome harvest from the env's terminal infos. THREE separate cuts
        # (opponent type / learner deck / full matchup), because a pooled win
        # rate is precisely the statistic that hides a policy dominating one
        # deck and losing on the rest. Running SUM and COUNT, not an EMA: the
        # reportable number is a cumulative win rate with its own n attached,
        # and a cell's n is what says whether to read it at all.
        # Consumed-position bookkeeping, keyed on LIST IDENTITY.
        #
        # pufferl.train() does `self.stats = defaultdict(list)` every update, so
        # each evaluate() hands back brand-new list objects. Tracking only a
        # consumed LENGTH across that boundary silently drops samples: a sparse
        # key reappears in the fresh dict with len(v)==1 while the stale count
        # is also 1, so `v[1:]` is empty and that episode is lost. The loss
        # scales with key sparsity -- measured on selfplay_deckpool29_seed42, 3 opponent
        # keys captured ~100%, 29 deck keys 59%, 841 matchup keys 20%.
        #
        # Holding a REFERENCE to the previous list makes `is` a sound identity
        # test (id() alone could be recycled by the allocator): same object ->
        # take the delta, new object -> take everything.
        from collections import defaultdict as _dd
        PREFIXES = {"wr_": "opp", "wrd_": "deck", "wrm_": "matchup"}
        wr_sum: dict[str, dict[str, float]] = {v: _dd(float)
                                               for v in PREFIXES.values()}
        wr_n: dict[str, dict[str, int]] = {v: _dd(int)
                                           for v in PREFIXES.values()}
        prev_list: dict[str, list] = {}
        consumed: dict[str, int] = _dd(int)

        def harvest(stats) -> None:
            for k, v in list(stats.items()):
                # longest prefix first: 'wrd_'/'wrm_' also start with 'wr'
                pre = next((p_ for p_ in ("wrd_", "wrm_", "wr_")
                            if k.startswith(p_)), None)
                if pre is None or not isinstance(v, list):
                    continue
                fresh = v[consumed[k]:] if prev_list.get(k) is v else v[:]
                prev_list[k] = v
                consumed[k] = len(v)
                fam, slug = PREFIXES[pre], k[len(pre):]
                for x in fresh:
                    wr_sum[fam][slug] += float(x)
                    wr_n[fam][slug] += 1

        def table(fam: str) -> dict:
            """{slug: [win_rate, n]} sorted by win rate, worst first."""
            return {s: [round(wr_sum[fam][s] / n, 4), n]
                    for s, n in sorted(wr_n[fam].items(),
                                       key=lambda kv: wr_sum[fam][kv[0]] / kv[1])}

        total_ts = train_config["total_timesteps"]

        def ent_coef_at(step: int) -> float:
            span = max(int(total_ts * args.ent_anneal_frac), 1)
            frac = min(step / span, 1.0)
            return args.ent_coef_init + (args.ent_coef_final - args.ent_coef_init) * frac

        last_snap = time.time()
        t_run = time.time()
        run_state["t_run"] = t_run
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

            # Per-update metrics (kl_to_prior included when the anchor is on).
            sps = trainer.global_step / max(time.time() - t_run, 1e-9)
            row = {"ts": round(time.time(), 3), "run_tag": args.run_tag,
                   "global_step": int(trainer.global_step), "epoch": epoch,
                   "sps": round(sps, 2), "ref": ref_dir,
                   "ent_coef": round(ent_coef, 6)}
            row.update({k: (float(v) if math.isfinite(v) else None)
                        for k, v in dict(getattr(trainer, "losses", {})).items()
                        if isinstance(v, (int, float))})
            if wr_n["opp"]:
                row["opp_wr"] = table("opp")
            pc = getattr(trainer, "per_context", None)
            if pc:
                row["per_context"] = pc
            # Name BOTH references in every row. `ref` moves on promotion;
            # `il_ref` never does. kl_refs_distinct=False means the two are
            # still the same policy, so kl_to_il_prior duplicates kl_to_prior
            # and the pair carries one measurement, not two (§3.3).
            if args.il_prior is not None:
                row["il_ref"] = str(args.il_prior)
                row["kl_refs_distinct"] = bool(
                    getattr(trainer, "kl_refs_distinct", False))
            append_jsonl(metrics_path, row)

            # Deck/matchup tables live in their own file: with K decks the
            # matchup table is K^2 cells and would swamp every metrics row.
            if wr_n["deck"] and epoch % args.deck_report_every == 0:
                append_jsonl(deck_metrics_path, {
                    "global_step": int(trainer.global_step), "epoch": epoch,
                    "run_tag": args.run_tag,
                    "per_deck": table("deck"), "per_matchup": table("matchup")})

            if gate_on and epoch % args.promote_every == 0:
                # Promotion gate (the ratchet): mirrored pairs of live vs the
                # frozen reference; training is paused for its duration.
                live_snap = save_actor(args.out / snap_name(trainer.global_step))
                print(f"promotion gate at update {epoch} (step "
                      f"{trainer.global_step}): live vs {ref_dir}, "
                      f"{args.promote_pairs} mirrored pairs...")
                verdict = evaluate_gate(
                    live_snap, ref_dir, pairs=args.promote_pairs,
                    workers=args.promote_workers,
                    threshold=args.promote_threshold,
                    deck_pool=deck_pool, mirror_deck=args.mirror_deck,
                    seed=args.seed + epoch)
                run_state["gates"] += 1
                if verdict["promote"]:
                    run_state["promotions"] += 1
                    ref_snap = save_actor(
                        args.out / "refs" / snap_name(trainer.global_step),
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
                snap = save_actor(args.out / snap_name(trainer.global_step))
                last_snap = time.time()
                print(f"  snapshot -> {snap}")

        elapsed = time.time() - t_run
        # Per-deck / per-matchup summary. The WORST cell is printed with its n
        # because the headline win rate is the one number that can be high
        # while some deck is unplayable -- and a matchup cell with n in the
        # single digits is noise, not a weakness.
        if wr_n["deck"]:
            deck_tbl, mu_tbl = table("deck"), table("matchup")
            pooled_n = sum(wr_n["deck"].values())
            pooled = sum(wr_sum["deck"].values()) / max(pooled_n, 1)
            summary = {
                "run_tag": args.run_tag, "global_step": int(trainer.global_step),
                "deck_pool": args.deck_pool, "deck_pool_k": len(deck_pool) if deck_pool else 1,
                "mirror_deck": args.mirror_deck, "mix": list(mix),
                "pooled_win_rate": round(pooled, 4), "episodes": pooled_n,
                "per_deck": deck_tbl, "per_matchup": mu_tbl,
                "worst_deck": next(iter(deck_tbl.items()), None),
                "worst_matchup": next(iter(mu_tbl.items()), None),
            }
            (args.out / "deck_summary.json").write_text(json.dumps(summary, indent=2))
            print(f"pooled win rate {pooled:.3f} over {pooled_n} episodes "
                  f"-- READ THE CELLS BELOW, NOT THIS NUMBER")
            print("per-deck (learner side), worst first:")
            for name, (w, n) in deck_tbl.items():
                print(f"  {w:.3f}  n={n:<6} {name}")
            worst = summary["worst_matchup"]
            if worst:
                print(f"worst matchup: {worst[0]}  wr={worst[1][0]:.3f} "
                      f"n={worst[1][1]}")
            print(f"deck summary -> {args.out / 'deck_summary.json'}")
        print(f"throughput: {trainer.global_step} steps in {elapsed:.0f}s = "
              f"{trainer.global_step / max(elapsed, 1e-9):.1f} steps/s "
              f"(kl_coef={args.kl_coef}, includes gate pauses)")

        snap = args.out / snap_name(trainer.global_step, final=True)
        policy.actor.to("cpu").save_pretrained(snap)
        (snap / "ppo_metadata.json").write_text(json.dumps(
            {"global_step": trainer.global_step, "epoch": epoch,
             "run_tag": args.run_tag, "final_ref": ref_dir,
             "cumulative_steps": provenance["cumulative_steps_prior"]
                                 + int(trainer.global_step),
             "wall_seconds": round(elapsed, 1),
             "gates_run": run_state["gates"],
             "promotions": run_state["promotions"],
             "stopped_on": ("max_seconds" if args.max_seconds
                            and elapsed > args.max_seconds else "total_timesteps"),
             **provenance}, indent=2))
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
