"""Per-deck and per-matchup win rates for a checkpoint, at a readable n.

WHY THIS EXISTS SEPARATELY FROM TRAINING METRICS. The env emits wrd_/wrm_ stats
for free, but they are unusable for two reasons: they average over a policy that
was changing all run and a mirror opponent that changed with it, and the matchup
grid is K^2 cells while episodes grow only linearly with compute (at K=29 and a
2.5h arm, ~5.6 episodes per matchup cell). This script fixes both: a FROZEN
checkpoint, a FIXED opponent, and games allocated per cell rather than sprayed
over a grid.

READ THE CELLS, NOT THE POOLED RATE. A pooled win rate is exactly the statistic
that hides a policy dominating one deck and losing the rest, which is the
failure mode deck-varied training is meant to remove. Output is sorted
worst-first, every cell carries n and a binomial SE, and the headline is the
WORST cell, not the mean.

CHOOSING THE OPPONENT. Default is a checkpoint (our own lineage) because the
deck axis must be the only thing moving. Beware pointing --opponent at an
external module agent: several are hand-written for one specific deck
(kojimar's is a Mega Lucario ex pilot), so forcing it onto a grid deck measures
that agent falling apart off its own list, not our policy's deck robustness.
`rule_baseline` is the safe external choice -- it is deck-agnostic by
construction.

    uv run python scripts/eval_deck_grid.py --agent models/g4_treatment_s42/u..._final \\
        --decks all:decklists --opponent models/il_agent --games-per-cell 12
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokemon_tcg import config  # noqa: E402
from pokemon_tcg.deck_pool import DeckPool  # noqa: E402

_W: dict = {}


def _worker_init(agent_dir: str, opp_ref: str, opp_kind: str) -> None:
    import torch

    torch.set_num_threads(1)
    from kaggle_environments import make as kaggle_make

    from pokemon_tcg.deck_pool import deck_override_agent
    from pokemon_tcg.selfplay import SamplingPolicy, as_env_agent

    _W["make"] = kaggle_make
    _W["decks"] = {0: None, 1: None}
    _W["seat"] = 0

    ours = as_env_agent(
        SamplingPolicy(agent_dir, temperature=1.0, record=False).agent)
    if opp_kind == "ckpt":
        opp = as_env_agent(
            SamplingPolicy(opp_ref, temperature=1.0, record=False).agent)
    else:
        sys.path.insert(0, str(config.PROJECT_ROOT / "scripts"))
        import benchmark_agents
        opp = benchmark_agents.load_agent(opp_ref)

    # Deck belongs to the SEAT, not to the policy: mirrored pairs swap which
    # policy sits in which seat, and the grid cell is defined by (our deck,
    # their deck), so the getters must follow the seat.
    _W["ours"] = deck_override_agent(ours, lambda: _W["decks"][_W["seat"]])
    _W["opp"] = deck_override_agent(opp, lambda: _W["decks"][1 - _W["seat"]])


def _play(task) -> tuple:
    """task = (our_deck_name, our_deck, opp_deck_name, opp_deck, our_seat)."""
    our_name, our_deck, opp_name, opp_deck, seat = task
    _W["seat"] = seat
    _W["decks"] = {seat: list(our_deck), 1 - seat: list(opp_deck)}
    agents = [None, None]
    agents[seat] = _W["ours"]
    agents[1 - seat] = _W["opp"]
    env = _W["make"]("cabt")
    final = env.run(agents)[-1]
    r = [final[i]["reward"] for i in range(2)]
    scored = [float("-inf") if x is None else x for x in r]
    mine, theirs = scored[seat], scored[1 - seat]
    return our_name, opp_name, (0.5 if mine == theirs
                                else (1.0 if mine > theirs else 0.0))


def se(p: float, n: int) -> float:
    return math.sqrt(max(p * (1 - p), 0.0) / n) if n else float("nan")


def summarize(sums, counts, label, min_n, limit=20):
    tbl = {k: (sums[k] / counts[k], counts[k]) for k in counts}
    order = sorted(tbl.items(), key=lambda kv: kv[1][0])
    readable = [(k, v) for k, v in order if v[1] >= min_n]
    print(f"\n{label}: {len(tbl)} cells, {sum(counts.values())} games, "
          f"{len(readable)} at n>={min_n}")
    for k, (wr, n) in order[:limit]:
        mark = "" if n >= min_n else "   <- UNREADABLE (n too small)"
        print(f"  {wr:6.3f} +-{se(wr, n):5.3f}  n={n:<5} {k}{mark}")
    if len(order) > limit:
        print(f"  ... {len(order) - limit} more (sorted worst-first)")
    return (readable[0] if readable else None), tbl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, help="checkpoint dir to evaluate")
    ap.add_argument("--decks", default="all:decklists",
                    help="deck pool spec for OUR side (and the opponent's, "
                         "unless --opp-decks). Keep this small: the grid is "
                         "|decks| x |opp-decks| cells and each needs games")
    ap.add_argument("--opp-decks", default=None,
                    help="deck spec for the opponent side (default: same as "
                         "--decks, i.e. the full square grid)")
    ap.add_argument("--opponent", default=str(config.MODELS_DIR / "il_agent"),
                    help="a checkpoint DIR (lineage, the default and the "
                         "clean choice) or a benchmark agent name. See the "
                         "module docstring before pointing this at an "
                         "external module agent")
    ap.add_argument("--games-per-cell", type=int, default=12,
                    help="games per (our deck, their deck) cell; rounded up to "
                         "an even number so seats are mirrored")
    ap.add_argument("--workers", type=int, default=8,
                    help="spawned processes (cg engine is a per-process "
                         "singleton, so parallelism = processes)")
    ap.add_argument("--min-n", type=int, default=20,
                    help="cells below this are printed but marked UNREADABLE "
                         "and excluded from the worst-cell verdict")
    ap.add_argument("--out", type=Path,
                    default=config.REPORTS_DIR / "deck_grid_eval.json")
    args = ap.parse_args()

    ours = DeckPool.from_spec(args.decks)
    theirs = DeckPool.from_spec(args.opp_decks) if args.opp_decks else ours
    opp_kind = "ckpt" if (Path(args.opponent) / "config.json").exists() else "module"

    per_pair = args.games_per_cell + (args.games_per_cell % 2)  # even: mirrored
    tasks = []
    for (an, ad), (bn, bd) in itertools.product(ours.entries, theirs.entries):
        for i in range(per_pair):
            tasks.append((an, ad, bn, bd, i % 2))

    n_cells = len(ours) * len(theirs)
    print(f"agent      {args.agent}")
    print(f"opponent   {args.opponent}  ({opp_kind})")
    print(f"grid       {len(ours)} x {len(theirs)} = {n_cells} cells "
          f"x {per_pair} games = {len(tasks)} games")
    print(f"per-deck   {per_pair * len(theirs)} games per deck (our side)")
    est = len(tasks) * 0.9 / max(args.workers, 1)
    print(f"estimate   ~{est / 60:.0f} min at ~0.9 s/game on {args.workers} workers\n")

    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(args.workers, len(tasks)),
                  initializer=_worker_init,
                  initargs=(args.agent, args.opponent, opp_kind)) as pool:
        results = pool.map(_play, tasks)

    deck_s, deck_n = defaultdict(float), defaultdict(int)
    mu_s, mu_n = defaultdict(float), defaultdict(int)
    for our_name, opp_name, outcome in results:
        deck_s[our_name] += outcome
        deck_n[our_name] += 1
        key = f"{our_name}~{opp_name}"
        mu_s[key] += outcome
        mu_n[key] += 1

    pooled = sum(deck_s.values()) / max(sum(deck_n.values()), 1)
    print(f"pooled win rate {pooled:.3f} over {len(results)} games "
          f"-- READ THE CELLS, NOT THIS")
    worst_deck, deck_tbl = summarize(deck_s, deck_n, "per deck (our side)",
                                     args.min_n)
    worst_mu, mu_tbl = summarize(mu_s, mu_n, "per matchup (ours ~ theirs)",
                                 args.min_n)

    if worst_deck:
        print(f"\nWORST DECK:    {worst_deck[0]}  wr={worst_deck[1][0]:.3f} "
              f"n={worst_deck[1][1]}  (SE {se(*worst_deck[1]):.3f})")
    if worst_mu:
        print(f"WORST MATCHUP: {worst_mu[0]}  wr={worst_mu[1][0]:.3f} "
              f"n={worst_mu[1][1]}  (SE {se(*worst_mu[1]):.3f})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "agent": args.agent, "opponent": args.opponent, "opp_kind": opp_kind,
        "decks": args.decks, "opp_decks": args.opp_decks or args.decks,
        "games": len(results), "games_per_cell": per_pair,
        "seconds": round(time.time() - t0, 1),
        "pooled_win_rate": round(pooled, 4),
        "per_deck": {k: [round(v[0], 4), v[1]] for k, v in deck_tbl.items()},
        "per_matchup": {k: [round(v[0], 4), v[1]] for k, v in mu_tbl.items()},
        "worst_deck": worst_deck, "worst_matchup": worst_mu,
        "min_n": args.min_n,
    }, indent=2))
    print(f"\nreport: {args.out}  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
