"""Agent performance comparison benchmark for the Pokémon TCG AI Battle Challenge.

Runs a round-robin tournament between the agent implementations we have and
reports head-to-head win rates, a win matrix, and Glicko-1 ratings.

Win-rate alone is a noisy, history-blind estimate: it weighs every game
equally and forgets everything once you rerun the script. Glicko-1 (see
scripts/glicko1.py) instead tracks a rating + rating deviation (RD) per
agent, persisted in reports/glicko_ratings.json across runs, so ratings
compound over the agent's *full* battle history and carry a confidence
interval that narrows as more evidence comes in.

WHY THIS EXISTS
---------------
We now have several "our" algorithms in the repo:
  * rule_baseline        -- the sample rule-based Mega Lucario ex agent
                            (agents/mega_lucario/agent_core.py)
  * improved_prob_main   -- the earlier cleaned reimplementation
                            (agents/improved_probabilistic/main.py)
  * agent_core_improved  -- the faithful, engine-search-verified reimplementation
                            (agents/mega_lucario/agent_core_improved.py)
  * proto                -- the prototype search agent (scripts/_proto_agent.py)

This script pits them against each other (and against themselves) so we can
see whether the search re-ranker actually buys win rate over the pure rule
baseline, and whether the two "improved probabilistic" variants differ.

USAGE
-----
    uv run python scripts/benchmark_agents.py            # default 8 games/pair
    uv run python scripts/benchmark_agents.py --games 12
    uv run python scripts/benchmark_agents.py --agents rule_baseline,agent_core_improved

Output: a win-rate table (rows = player A, cols = player B) and per-agent
overall win rate, printed to stdout and saved next to the script.

Note: each match is 2 games (seat A=p0,p1 then A=p1,p0) so first-player bias
is cancelled. With --games N you get N such mirrored pairs per ordered pair.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import glicko1  # noqa: E402

GLICKO_PATH = REPO / "reports" / "glicko_ratings.json"

# Make `cg` importable (packaged engine) before importing any agent module.
for _cand in (REPO / "data" / "external" / "cg-lib", REPO / "agents" / "mega_lucario"):
    if (_cand / "cg" / "api.py").exists():
        sys.path.insert(0, str(_cand))
        break


AGENT_FILES = {
    "rule_baseline": REPO / "agents" / "mega_lucario" / "agent_core.py",
    "improved_prob_main": REPO / "agents" / "improved_probabilistic" / "main.py",
    "agent_core_improved": REPO / "agents" / "mega_lucario" / "agent_core_improved.py",
    "proto": REPO / "scripts" / "_proto_agent.py",
    "il_agent": REPO / "agents" / "il_agent" / "agent_core.py",
    # Floor test, not a competitor: uniform-random legal moves on the same
    # frozen deck. If a trained policy doesn't beat this decisively, its
    # offline accuracy isn't credible evidence it learned anything.
    "random_legal": REPO / "agents" / "random_legal" / "agent_core.py",
}

# Where each agent's real competition entry point (main.py) lives, if any.
# Loading via main.py guarantees the deck is wired up exactly as the harness
# would at submission time.
AGENT_MAIN = {
    "rule_baseline": REPO / "submissions" / "mega_lucario" / "main.py",
    "agent_core_improved": REPO / "submissions" / "mega_lucario_improved" / "main.py",
}


def load_agent(name: str):
    """Load an agent's `agent` callable.

    Prefers the bundle `main.py` (real entry point, deck wired up). Falls back
    to the bare module — for bare modules that need `my_deck` injected (e.g.
    the sample rule baseline), we read it from the matching deck.csv.
    """
    main_py = AGENT_MAIN.get(name)
    if main_py and main_py.exists():
        spec = importlib.util.spec_from_file_location(f"_bench_{name}", main_py)
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(main_py.parent))
        spec.loader.exec_module(mod)
        fn = getattr(mod, "agent", None)
        if not callable(fn):
            raise AttributeError(f"{name} bundle has no callable `agent`")
        return fn

    path = AGENT_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"no agent file for {name}: {path}")
    spec = importlib.util.spec_from_file_location(f"_bench_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(mod)
    # Bare modules that reference a module-level `my_deck` (e.g. the sample
    # rule baseline) have it injected by their main.py; wire it here from deck.csv.
    if not hasattr(mod, "my_deck") and (path.parent / "deck.csv").exists():
        deck = [int(x) for x in (path.parent / "deck.csv").read_text().splitlines() if x.strip()][:60]
        mod.my_deck = deck
    fn = getattr(mod, "agent", None)
    if not callable(fn):
        raise AttributeError(f"{name} has no callable `agent`")
    return fn


def load_glicko_ratings(path: Path = GLICKO_PATH) -> dict[str, glicko1.Rating]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {name: glicko1.Rating(rating=v["rating"], rd=v["rd"]) for name, v in raw.items()}


def save_glicko_ratings(ratings: dict[str, glicko1.Rating], path: Path = GLICKO_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {name: {"rating": r.rating, "rd": r.rd} for name, r in ratings.items()},
        indent=2,
    ))


def play_match(agent_a, agent_b, env_factory, pairs: int = 1) -> tuple[int, int, int, float]:
    """Play `pairs` mirrored game pairs (both seat orders each). Returns (a_wins, b_wins, draws, seconds).

    Each mirrored pair cancels first-player advantage: game 1 has A in seat 0,
    game 2 has B in seat 0. A win for "player 0" maps to whichever agent sits
    in player 0's seat.
    """
    t0 = time.time()
    a_wins = b_wins = draws = 0
    for _ in range(pairs):
        for seat0 in (agent_a, agent_b):
            seat1 = agent_b if seat0 is agent_a else agent_a
            env = env_factory()
            trace = env.run([seat0, seat1])
            final = trace[-1]
            r = [final[i]["reward"] for i in range(2)]
            if r[0] == r[1]:
                draws += 1
            elif r[0] > r[1]:
                a_wins += 1 if seat0 is agent_a else 0
                b_wins += 1 if seat0 is agent_b else 0
            else:
                a_wins += 1 if seat0 is agent_b else 0
                b_wins += 1 if seat0 is agent_a else 0
    return a_wins, b_wins, draws, time.time() - t0


def run_benchmark(agents: list[str], games_per_pair: int = 8):
    from kaggle_environments import make

    print(f"Loading {len(agents)} agents: {', '.join(agents)}")
    fns = {name: load_agent(name) for name in agents}

    n = len(agents)
    wins = {a: {b: 0 for b in agents} for a in agents}  # wins[a][b] = a beat b
    games = {a: {b: 0 for b in agents} for a in agents}
    totals = {a: {"w": 0, "g": 0} for a in agents}
    # (player_a, player_b, score_a) triples for the Glicko-1 rating period.
    # Self-play is excluded: an agent playing itself carries no information
    # about its rating relative to the field.
    glicko_games: list[tuple[str, str, float]] = []

    total_pairs = n * (n - 1) // 2 + n  # unordered incl. self-play
    done = 0
    for i in range(n):
        for j in range(i, n):
            a, b = agents[i], agents[j]
            pairs = games_per_pair if a != b else max(1, games_per_pair // 2)
            aw, bw, dr, secs = play_match(fns[a], fns[b], lambda: make("cabt"), pairs)
            wins[a][b] += aw
            wins[b][a] += bw
            games[a][b] += aw + bw + dr
            games[b][a] += aw + bw + dr
            totals[a]["w"] += aw
            totals[b]["w"] += bw
            totals[a]["g"] += aw + bw + dr
            totals[b]["g"] += aw + bw + dr
            if a != b:
                glicko_games += [(a, b, 1.0)] * aw
                glicko_games += [(a, b, 0.0)] * bw
                glicko_games += [(a, b, 0.5)] * dr
            done += 1
            print(
                f"[{done}/{total_pairs}] {a} vs {b}: "
                f"{a} {aw}-{bw}{('-' + str(dr) + 'd') if dr else ''}  "
                f"({aw+bw+dr} games, {secs:.1f}s)"
            )

    # ---- report ----
    print("\n=== Head-to-head win matrix (rows beat cols) ===")
    header = "agent".ljust(22) + "".join(b[:6].ljust(8) for b in agents) + "win%"
    print(header)
    overall = {}
    for a in agents:
        row = a.ljust(22)
        w = totals[a]["w"]
        g = totals[a]["g"]
        overall[a] = (100.0 * w / g) if g else 0.0
        cells = ""
        for b in agents:
            if a == b:
                cells += "-".ljust(8)
            else:
                cells += f"{wins[a][b]}".ljust(8)
        row += cells + f"{overall[a]:5.1f}"
        print(row)

    print("\n=== Overall win rate (all games) ===")
    for a in sorted(agents, key=lambda x: -overall[x]):
        print(f"  {a:22s} {overall[a]:5.1f}%  ({totals[a]['w']}/{totals[a]['g']})")

    # ---- Glicko-1 ratings ----
    # Win-rate is a single-run snapshot; Glicko carries a rating + confidence
    # (RD) forward across every benchmark run, weighted by opponent strength,
    # so it's a better estimate of true skill than this run's win% alone.
    prior_ratings = load_glicko_ratings()
    glicko = glicko1.run_rating_period(prior_ratings, glicko_games)
    save_glicko_ratings(glicko)

    print("\n=== Glicko-1 ratings (persisted across runs, higher = stronger) ===")
    ranked = sorted(agents, key=lambda x: -glicko[x].rating)
    for a in ranked:
        r = glicko[a]
        print(f"  {a:22s} {r.rating:7.1f}  (RD {r.rd:5.1f}, 95% CI +/-{2*r.rd:5.1f})")

    result = {
        "agents": agents,
        "games_per_pair": games_per_pair,
        "wins": wins,
        "games": games,
        "overall_win_pct": overall,
        "glicko": {a: {"rating": glicko[a].rating, "rd": glicko[a].rd} for a in agents},
    }
    out_path = REPO / "reports" / "agent_benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nsaved: {out_path}")
    print(f"saved: {GLICKO_PATH}")
    return result


def main():
    ap = argparse.ArgumentParser(description="Benchmark agent vs agent performance.")
    ap.add_argument("--agents", default=",".join(AGENT_FILES.keys()),
                    help="comma-separated subset of: " + ", ".join(AGENT_FILES))
    ap.add_argument("--games", type=int, default=8, dest="games_per_pair",
                    help="mirrored game pairs per ordered agent pair")
    args = ap.parse_args()
    agents = [a.strip() for a in args.agents.split(",") if a.strip()]
    unknown = [a for a in agents if a not in AGENT_FILES]
    if unknown:
        sys.exit(f"unknown agent(s): {unknown}. Available: {list(AGENT_FILES)}")
    run_benchmark(agents, args.games_per_pair)


if __name__ == "__main__":
    main()
