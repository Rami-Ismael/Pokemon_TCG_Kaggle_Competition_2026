"""v6 step `lineage_selfplay` driver (rl_pipeline_v6.md §1.2).

Per generation:
  1. Play --games direct-engine games: the current policy vs an OSFP-sampled
     opponent (probability --p-opt the mirror, else uniform over the saved
     lineage). Both seats' decks are sampled from the deck pool every game.
     Every game is recorded as a corpus-format episode and written straight
     into zstd parquet shards (pack_episodes schema) -- no raw JSON on disk
     unless --keep-raw.
  2. Resume training: scripts/train_il.py --init-from <current> with the same
     binary-advantage loss over human episodes (hub stream) UNION all
     self-play shards produced so far (--selfplay-shards).
  3. Mint decision (--mint-rule):
       cadence  every generation's checkpoint joins the lineage.
       tryout   the checkpoint joins only if it wins strictly more than
                --tryout-threshold of decisive games against the NEWEST
                lineage member (promotion.evaluate_gate, mirrored pairs,
                same deck pool as training).
     Either way the new checkpoint becomes the current policy; the rules
     differ only in lineage membership. The two rules are the two arms of
     notes/experiments/2026-08-13-lineage-minting-rule.md.

Zero external agents anywhere (v6 §1.3): lineage entries must be saved
checkpoints of our own model, and the episode writer lives only here.

Usage (single generation, tryout arm):
    uv run python scripts/run_lineage_selfplay.py \
        --current models/binaryadv_alldays_jun16-aug07_seed42 \
        --lineage models/bc_alldays52_jun16_aug07_seed42,models/binaryadv_alldays_jun16-aug07_seed42 \
        --critic-dir models/critic_outcome_bcalldays52trunk_2026-08-01_to_2026-08-07_seed42 \
        --mint-rule tryout --games 1000 --out models/lineage_selfplay_tryout_seed42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import random
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pokemon_tcg import config  # noqa: E402
from pokemon_tcg import lineage_selfplay  # noqa: E402
from pokemon_tcg.deck_pool import DeckPool  # noqa: E402
from pokemon_tcg.promotion import evaluate_gate  # noqa: E402

import pack_episodes  # noqa: E402  (same dir; SCHEMA reuse)

MIN_DISK_GB = 5.0
EPISODE_ID_BASE = 9_000_000_000  # far above Kaggle's ~55M id range


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=config.PROJECT_ROOT,
        ).stdout.strip()
    except OSError:
        return "unknown"


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def preflight(out: Path) -> None:
    free_gb = shutil.disk_usage(out).free / 2**30
    if free_gb < MIN_DISK_GB:
        sys.exit(f"preflight: only {free_gb:.1f} GB free (< {MIN_DISK_GB}); "
                 f"clear disk before generating episodes")
    print(f"preflight: {free_gb:.1f} GB free")


def validate_ckpt(path: str) -> str:
    p = Path(path)
    if not (p / "config.json").exists():
        sys.exit(f"not a saved checkpoint (no config.json): {p} -- lineage "
                 f"and --current must be OUR OWN checkpoint dirs (v6 §1.3)")
    return str(p.resolve())


class ShardWriter:
    """Streams finished episodes into pack_episodes-schema parquet shards."""

    def __init__(self, out_dir: Path, day: str, per_shard: int, level: int = 9):
        import pyarrow.parquet as pq  # noqa: F401  (validated import early)

        self.out_dir = out_dir
        self.day = day
        self.per_shard = per_shard
        self.level = level
        self.rows: list[dict] = []
        self.shard_idx = 0
        self.n_written = 0
        self.files: list[Path] = []
        out_dir.mkdir(parents=True, exist_ok=True)

    def add(self, episode_id: int, raw: bytes) -> None:
        self.rows.append({
            "episode_id": episode_id,
            "day": self.day,
            "split_folder": "lineage_selfplay",
            "n_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "create_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "avg_score": None,
            "min_score": None,
            "agent_count": 2,
            "episode_json": raw,
        })
        self.n_written += 1
        if len(self.rows) >= self.per_shard:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = self.out_dir / f"shard-{self.shard_idx:03d}.parquet"
        with pq.ParquetWriter(
            path, pack_episodes.SCHEMA, compression="zstd",
            compression_level=self.level,
        ) as w:
            w.write_table(
                pa.Table.from_pylist(self.rows, schema=pack_episodes.SCHEMA))
        self.files.append(path)
        self.shard_idx += 1
        self.rows = []


def play_generation(gen: int, current: str, lineage: list[str], pool: DeckPool,
                    args, out: Path) -> dict:
    """Generate one generation's episodes; returns summary stats."""
    rng = random.Random(f"{args.seed}:gen{gen}")
    name_by_ckpt = {c: Path(c).name for c in lineage}
    shard_dir = out / "shards" / f"gen{gen:03d}"
    writer = ShardWriter(shard_dir, day=f"selfplay-gen{gen:03d}",
                         per_shard=args.episodes_per_shard)
    raw_dir = out / "episodes_raw" / f"gen{gen:03d}" if args.keep_raw else None
    if raw_dir:
        raw_dir.mkdir(parents=True, exist_ok=True)
    log_path = out / "selfplay_log.jsonl"

    dropped_decks: set[str] = set()

    def draw_deck(r: random.Random) -> tuple[str, list[int]]:
        for _ in range(50):
            name, deck = pool.sample(r)
            if name not in dropped_decks:
                return name, deck
        sys.exit("deck pool exhausted: every deck was rejected by the engine")

    tasks = []
    for i in range(args.games):
        mirror = rng.random() < args.p_opt
        opp = None if mirror else rng.choice(lineage)
        d0_name, d0 = draw_deck(rng)
        d1_name, d1 = draw_deck(rng)
        tasks.append({
            "game_idx": i,
            "opponent_ckpt": opp,
            "current_seat": i % 2,
            "decks": [d0, d1],
            "deck_names": [d0_name, d1_name],
            "info": {
                "source": "lineage_selfplay",
                "engine": "direct",
                "gen": gen,
                "game_idx": i,
                "current_ckpt": Path(current).name,
                "opponent": "mirror" if opp is None else name_by_ckpt.get(
                    opp, Path(opp).name),
                "current_seat": i % 2,
                "decks": [d0_name, d1_name],
            },
        })

    t0 = time.time()
    outcomes: list[float] = []
    by_opp: dict[str, list[float]] = defaultdict(list)
    opp_hist: Counter = Counter()
    n_fallbacks = n_decisions = n_rejects = n_deck_rejects = 0

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(args.workers, len(tasks)),
                  initializer=lineage_selfplay.worker_init,
                  initargs=(current, args.temperature)) as p:
        for res in p.imap_unordered(
                lineage_selfplay.play_one_recorded_game, tasks,
                chunksize=1):
            task = tasks[res["game_idx"]]
            opp_name = task["info"]["opponent"]
            if res["episode_json"] is None:
                seat = res["deck_rejected_seat"]
                bad = task["deck_names"][seat]
                dropped_decks.add(bad)
                n_deck_rejects += 1
                print(f"  DECK REJECTED by engine: {bad} (seat {seat}) -- "
                      f"dropped from this run's pool")
                append_jsonl(log_path, {"gen": gen, **task["info"],
                                        "deck_rejected": bad})
                continue
            eid = EPISODE_ID_BASE + gen * 1_000_000 + res["game_idx"]
            writer.add(eid, res["episode_json"])
            if raw_dir:
                (raw_dir / f"{eid}.json").write_bytes(res["episode_json"])
            outcomes.append(res["outcome"])
            opp_hist[opp_name] += 1
            by_opp[opp_name].append(res["outcome"])
            n_fallbacks += sum(res["fallbacks"])
            n_decisions += sum(res["decisions"])
            n_rejects += res["engine_rejects"]
            append_jsonl(log_path, {
                "gen": gen, **task["info"], "episode_id": eid,
                "outcome": res["outcome"], "result": res["result"],
                "forfeiter": res["forfeiter"],
                "decisions": res["decisions"], "fallbacks": res["fallbacks"],
                "engine_rejects": res["engine_rejects"],
                "seconds": round(res["seconds"], 3),
            })
    writer.flush()

    elapsed = time.time() - t0
    wins = sum(1 for o in outcomes if o == 1.0)
    draws = sum(1 for o in outcomes if o == 0.5)
    losses = sum(1 for o in outcomes if o == 0.0)
    summary = {
        "gen": gen,
        "games_played": len(outcomes),
        "episodes_written": writer.n_written,
        "shards": [str(f) for f in writer.files],
        "current_record_wdl": [wins, draws, losses],
        "opponent_histogram": dict(opp_hist),
        "win_rate_by_opponent": {
            k: round(sum(1 for o in v if o == 1.0)
                     / max(1, sum(1 for o in v if o != 0.5)), 3)
            for k, v in by_opp.items()
        },
        "fallback_rate": round(n_fallbacks / max(1, n_decisions), 5),
        "engine_rejects": n_rejects,
        "deck_rejects": n_deck_rejects,
        "dropped_decks": sorted(dropped_decks),
        "seconds": round(elapsed, 1),
        "games_per_s": round(len(outcomes) / max(elapsed, 1e-9), 3),
    }
    print(f"  gen {gen}: {len(outcomes)} games in {elapsed:.0f}s "
          f"({summary['games_per_s']:.2f} games/s), current W/D/L "
          f"{wins}/{draws}/{losses}, fallback_rate "
          f"{summary['fallback_rate']:.4f}")
    for k, v in sorted(summary["win_rate_by_opponent"].items()):
        print(f"    vs {k}: {v:.3f} decisive win rate "
              f"(n={opp_hist[k]})")
    return summary


def run_resume_training(gen: int, current: str, out: Path, args) -> str:
    """train_il.py resume over human ∪ all self-play shards so far."""
    ckpt_name = f"lineage_selfplay_{args.mint_rule}_gen{gen}_seed{args.seed}"
    ckpt_out = out / ckpt_name
    cmd = [
        sys.executable, str(Path(__file__).parent / "train_il.py"),
        "--init-from", current,
        "--weight-arm", "adv-binary",
        "--critic-dir", str(args.critic_dir),
        "--data-source", "hub",
        "--train-split", args.train_split,
        "--selfplay-shards", str(out / "shards"),
        "--num-workers", str(args.train_workers),
        "--epochs", str(args.epochs),
        "--seed", str(args.seed),
        "--out", str(ckpt_out),
        "--run-dir", str(out / "train_runs" / f"gen{gen:03d}"),
    ] + args.train_arg
    print(f"  resume training -> {ckpt_out}")
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return validate_ckpt(str(ckpt_out))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--current", required=True,
                    help="starting policy checkpoint dir (v6: the "
                         "binary_advantage checkpoint)")
    ap.add_argument("--lineage", required=True,
                    help="comma-separated initial lineage checkpoint dirs "
                         "(v6: imitation ancestor + binary_advantage)")
    ap.add_argument("--mint-rule", required=True, choices=["tryout", "cadence"],
                    help="when a generation checkpoint joins the lineage: "
                         "tryout = only after winning >--tryout-threshold of "
                         "decisive games vs the newest lineage member; "
                         "cadence = every generation")
    ap.add_argument("--games", type=int, default=1000,
                    help="self-play games per generation")
    ap.add_argument("--generations", type=int, default=1)
    ap.add_argument("--p-opt", type=float, default=0.5,
                    help="probability the opponent is the current policy "
                         "(the OSFP optimism); else uniform over the lineage")
    ap.add_argument("--deck-pool", default="all:decks",
                    help="DeckPool spec; BOTH seats draw independently every "
                         "game (v6: deck diversity is the defense against "
                         "self-play collapse)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    ap.add_argument("--out", type=Path, required=True,
                    help="run dir: shards/, checkpoints, logs live here")
    ap.add_argument("--episodes-per-shard", type=int, default=256)
    ap.add_argument("--keep-raw", action="store_true",
                    help="also write each episode as raw JSON (debugging; "
                         "disk-expensive)")
    ap.add_argument("--tryout-pairs", type=int, default=50)
    ap.add_argument("--tryout-threshold", type=float, default=0.70)
    ap.add_argument("--tryout-workers", type=int, default=8)
    ap.add_argument("--skip-train", action="store_true",
                    help="generate episodes only (no resume step, no mint "
                         "decision) -- smoke/dataset-only mode")
    ap.add_argument("--critic-dir", type=Path, default=None,
                    help="calibrated critic for adv-binary (required unless "
                         "--skip-train); calibration.json is mandatory")
    ap.add_argument("--train-split", default="train_combined_v4")
    ap.add_argument("--train-workers", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=1,
                    help="resume epochs per generation")
    ap.add_argument("--train-arg", action="append", default=[],
                    help="extra raw args appended to the train_il.py resume "
                         "command (repeatable)")
    args = ap.parse_args()

    if not args.skip_train:
        if args.critic_dir is None:
            ap.error("--critic-dir is required unless --skip-train")
        if not (args.critic_dir / "calibration.json").exists():
            ap.error(f"{args.critic_dir}/calibration.json missing -- the "
                     f"uncalibrated critic is worse than a constant; fit "
                     f"calibration first (scripts/fit_critic_calibration.py)")

    current = validate_ckpt(args.current)
    lineage = [validate_ckpt(p) for p in args.lineage.split(",") if p]
    pool = DeckPool.from_spec(args.deck_pool)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    preflight(out)

    run_config = {
        **{k: str(v) if isinstance(v, Path) else v
           for k, v in vars(args).items()},
        "current_resolved": current,
        "lineage_resolved": list(lineage),
        "deck_pool_k": len(pool),
        "deck_pool_names": pool.names,
        "engine": "direct",
        "git_sha": git_sha(),
        "episode_id_base": EPISODE_ID_BASE,
    }
    (out / "run_config.json").write_text(json.dumps(run_config, indent=2))
    print(f"lineage self-play: mint_rule={args.mint_rule} "
          f"games/gen={args.games} generations={args.generations} "
          f"p_opt={args.p_opt} deck_pool K={len(pool)} workers={args.workers}")
    print(f"  current: {current}")
    for c in lineage:
        print(f"  lineage: {c}")

    mint_log = out / "mint_log.jsonl"
    for gen in range(1, args.generations + 1):
        print(f"generation {gen}/{args.generations}: playing {args.games} "
              f"games (current = {Path(current).name})")
        summary = play_generation(gen, current, lineage, pool, args, out)
        append_jsonl(out / "generation_log.jsonl", summary)

        if args.skip_train:
            print("  --skip-train: no resume step, no mint decision")
            continue

        candidate = run_resume_training(gen, current, out, args)

        if args.mint_rule == "cadence":
            verdict = {"rule": "cadence", "promote": True}
        else:
            newest = lineage[-1]
            print(f"  tryout: {Path(candidate).name} vs {Path(newest).name}, "
                  f"{args.tryout_pairs} mirrored pairs...")
            verdict = evaluate_gate(
                candidate, newest, pairs=args.tryout_pairs,
                workers=args.tryout_workers,
                threshold=args.tryout_threshold,
                deck_pool=pool, mirror_deck=False,
                seed=args.seed + gen, engine="direct")
            verdict["rule"] = "tryout"
            print(f"  tryout: {verdict['wins']}W/{verdict['losses']}L/"
                  f"{verdict['draws']}D win_rate={verdict['win_rate']:.3f} "
                  f"vs >{args.tryout_threshold:.2f} -> "
                  f"{'MINTED' if verdict['promote'] else 'not minted'}")
        if verdict["promote"]:
            lineage.append(candidate)
        current = candidate
        append_jsonl(mint_log, {
            "gen": gen, "candidate": candidate, "minted": verdict["promote"],
            **{k: v for k, v in verdict.items() if k != "promote"},
            "lineage_size": len(lineage),
        })
        (out / "lineage.json").write_text(json.dumps({
            "current": current, "lineage": lineage,
            "mint_rule": args.mint_rule, "generations_done": gen,
        }, indent=2))

    print("done. final state:")
    print(f"  current: {current}")
    print(f"  lineage ({len(lineage)}): "
          f"{', '.join(Path(c).name for c in lineage)}")
    print(f"  logs: {out}/generation_log.jsonl, {out}/mint_log.jsonl, "
          f"{out}/selfplay_log.jsonl")


if __name__ == "__main__":
    main()
