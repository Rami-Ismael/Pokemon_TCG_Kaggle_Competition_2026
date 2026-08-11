"""Strength measurement: il_agent (search-free IL policy) vs mcts_il_agent (same
policy as MCTS prior, offline critic as leaf value).

Three measurements, all mirrored-pair games (seat swap cancels first-player
advantage; see benchmark_agents.play_match):

  1. Head-to-head: il_agent vs mcts_il_agent directly.
  2. il_agent vs each public-field opponent (the `rung2` group).
  3. mcts_il_agent vs the same opponents.

(2) and (3) are the strength numbers -- win rate against a common external
field. (1) tells you whether search changes outcomes against the exact policy
it wraps. Every win rate is reported with a binomial sigma; a point estimate
alone is not a result here.

Also reported, per our agent:
  - fallback diagnostics (diag_snapshot): if mcts_il_agent fell back to
    _safe_choice on most decisions, its number measures the fallback, not MCTS.
  - seconds/game: local think time is unbudgeted, the ladder gives a 600 s
    per-match bank -- a strength edge that blows the bank is not deployable.

Isolated from the standing Glicko file on purpose (no rating persistence);
results go to reports/il_vs_mcts_strength.json and stdout.

USAGE
-----
    uv run python scripts/eval_il_vs_mcts.py                     # full field
    uv run python scripts/eval_il_vs_mcts.py --pairs 2 --opponents kiyotah_dragapult  # smoke
    uv run python scripts/eval_il_vs_mcts.py --search-count 30   # MCTS budget
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark_agents as ba  # noqa: E402

IL_AGENT = "il_agent"
MCTS_AGENT = "mcts_il_agent"


def binom_sigma(wins: int, games: int) -> float:
    """Std. error of a win-rate estimate, sqrt(p(1-p)/n)."""
    if games == 0:
        return float("nan")
    p = wins / games
    return math.sqrt(max(p * (1 - p), 1e-12) / games)


def fmt_rate(wins: int, games: int) -> str:
    if games == 0:
        return "   n/a"
    p = wins / games
    return f"{100 * p:5.1f}% ± {100 * binom_sigma(wins, games):4.1f} ({wins}/{games})"


def play_pairing(fns: dict, name_a: str, name_b: str, pairs: int, env_factory) -> dict:
    a_wins, b_wins, draws, secs = ba.play_match(
        fns[name_a], fns[name_b], env_factory, pairs=pairs,
        name_a=name_a, name_b=name_b,
    )
    games = 2 * pairs
    decisive = a_wins + b_wins
    return {
        "a": name_a, "b": name_b,
        "a_wins": a_wins, "b_wins": b_wins, "draws": draws,
        "games": games,
        "a_winrate": a_wins / games,
        "a_sigma": binom_sigma(a_wins, games),
        "seconds": secs,
        "sec_per_game": secs / games,
        "decisive": decisive,
    }


def diag_for(name: str) -> dict | None:
    mod = ba._LOADED_MODULES.get(name)
    if mod is not None and hasattr(mod, "diag_snapshot"):
        return mod.diag_snapshot()
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pairs", type=int, default=5,
                    help="mirrored pairs per (our agent, opponent) pairing; 2 games each (default 5 = 10 games)")
    ap.add_argument("--h2h-pairs", type=int, default=15,
                    help="mirrored pairs for the direct il_agent vs mcts_il_agent match (default 15 = 30 games)")
    ap.add_argument("--opponents", default="rung2",
                    help="comma list of opponent agents or group names (default: rung2, the public field)")
    ap.add_argument("--search-count", type=int, default=None,
                    help="MCTS_IL_SEARCH_COUNT for mcts_il_agent (overrides the env var when "
                         "passed; unset flag leaves the env/agent default 30)")
    ap.add_argument("--out", type=Path, default=REPO / "reports" / "il_vs_mcts_strength.json")
    args = ap.parse_args()

    if args.search_count is not None:
        # Must land in the env BEFORE load_agent execs the module (read at import).
        os.environ["MCTS_IL_SEARCH_COUNT"] = str(args.search_count)
    os.environ.setdefault("PTCG_FALLBACK_TRACK", "1")

    opponents = ba.resolve_agent_selection(args.opponents.split(","))
    opponents = [o for o in opponents if o not in (IL_AGENT, MCTS_AGENT)]
    ours = [IL_AGENT, MCTS_AGENT]

    from kaggle_environments import make
    env_factory = lambda: make("cabt")  # noqa: E731

    print(f"Loading agents: {ours + opponents}")
    ba._LOADED_MODULES.clear()
    fns = {name: ba.load_agent(name) for name in ours + opponents}
    for mod in ba._LOADED_MODULES.values():
        if hasattr(mod, "diag_reset"):
            mod.diag_reset()

    t0 = time.time()
    results = {"h2h": None, "field": {IL_AGENT: [], MCTS_AGENT: []}}

    # 1) Head-to-head
    if args.h2h_pairs > 0:
        print(f"\n[1/3] head-to-head: {IL_AGENT} vs {MCTS_AGENT} "
              f"({args.h2h_pairs} mirrored pairs = {2 * args.h2h_pairs} games)")
        r = play_pairing(fns, IL_AGENT, MCTS_AGENT, args.h2h_pairs, env_factory)
        results["h2h"] = r
        print(f"      {IL_AGENT}: {fmt_rate(r['a_wins'], r['games'])}   "
              f"draws {r['draws']}   {r['sec_per_game']:.1f} s/game")

    # 2) + 3) each of ours vs the common field
    for i, mine in enumerate(ours, start=2):
        print(f"\n[{i}/3] {mine} vs field ({args.pairs} mirrored pairs per opponent)")
        for opp in opponents:
            r = play_pairing(fns, mine, opp, args.pairs, env_factory)
            results["field"][mine].append(r)
            print(f"      vs {opp:28s} {fmt_rate(r['a_wins'], r['games'])}   "
                  f"draws {r['draws']}   {r['sec_per_game']:.1f} s/game")

    # Pooled field strength
    print("\n=== field strength (pooled over the common opponent set) ===")
    pooled = {}
    for mine in ours:
        w = sum(r["a_wins"] for r in results["field"][mine])
        g = sum(r["games"] for r in results["field"][mine])
        spg = (sum(r["seconds"] for r in results["field"][mine]) / g) if g else float("nan")
        pooled[mine] = {"wins": w, "games": g,
                        "winrate": (w / g) if g else None,
                        "sigma": binom_sigma(w, g),
                        "sec_per_game": spg}
        print(f"  {mine:14s} {fmt_rate(w, g)}   {spg:.1f} s/game")
    if all(pooled[m]["games"] for m in ours):
        d = pooled[MCTS_AGENT]["winrate"] - pooled[IL_AGENT]["winrate"]
        sd = math.hypot(pooled[MCTS_AGENT]["sigma"], pooled[IL_AGENT]["sigma"])
        print(f"  delta (mcts - il): {100 * d:+.1f} pp ± {100 * sd:.1f}")
        pooled["delta_mcts_minus_il"] = {"pp": d, "sigma": sd}

    # Fallback diagnostics: was the search path actually used?
    print("\n=== fallback diagnostics ===")
    diags = {}
    for mine in ours:
        d = diag_for(mine)
        diags[mine] = d
        if d is None:
            print(f"  {mine}: no diag interface exposed")
        elif not d.get("enabled", False):
            print(f"  {mine}: tracking disabled")
        else:
            print(f"  {mine}: {d['fallbacks']}/{d['decisions']} decisions fell back "
                  f"({100 * d['fallback_rate']:.1f}%)  "
                  + " ".join(f"{k}={v}" for k, v in d.items()
                             if k not in ("enabled", "decisions", "fallbacks", "fallback_rate")))
            if d["fallback_rate"] > 0.05:
                print(f"  [WARN] {mine} fallback rate > 5% -- its win rate partly "
                      f"measures _safe_choice, not the intended policy/search.")

    out = {
        "config": {
            "pairs_per_opponent": args.pairs,
            "h2h_pairs": args.h2h_pairs,
            "opponents": opponents,
            "search_count": os.environ.get("MCTS_IL_SEARCH_COUNT", "30 (agent default)"),
            "note": ("local think time is unbudgeted; the ladder budget is a 600 s/match "
                     "bank -- read sec_per_game against that before any deploy claim"),
        },
        "h2h": results["h2h"],
        "field": results["field"],
        "pooled": pooled,
        "fallback_diagnostics": diags,
        "wall_clock_seconds": time.time() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}  (total {out['wall_clock_seconds']:.0f} s)")


if __name__ == "__main__":
    main()
