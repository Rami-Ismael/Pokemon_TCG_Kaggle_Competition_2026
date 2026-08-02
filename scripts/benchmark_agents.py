"""Agent performance comparison benchmark for the Pokémon TCG AI Battle Challenge.

Runs a round-robin tournament between the agent implementations we have and
reports head-to-head win rates, a win matrix, and Glicko-1 ratings.

Win-rate alone is a noisy, history-blind estimate: it weighs every game
equally and forgets everything once you rerun the script. Glicko-1 (see
scripts/glicko1.py) instead tracks a rating + rating deviation (RD) per
agent, persisted in reports/glicko_ratings.json across runs, so ratings
compound over the agent's *full* battle history and carry a confidence
interval that narrows as more evidence comes in. GXE (also from
scripts/glicko1.py) folds rating + RD into a single win-probability-%
against a fixed average opponent -- the same pair of stats Pokemon
Showdown itself reports, and what the Metamon paper cites for its own
agent evaluations.

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
  * grunt                -- deck-agnostic greedy one-ply agent: always takes
                            the attack with the highest type-chart-adjusted
                            damage, and switches to the best type matchup
                            when forced (agents/grunt/agent_core.py)

This script pits them against each other (and against themselves) so we can
see whether the search re-ranker actually buys win rate over the pure rule
baseline, and whether the two "improved probabilistic" variants differ.

USAGE
-----
    uv run python scripts/benchmark_agents.py            # default 8 games/pair
    uv run python scripts/benchmark_agents.py --games 12
    uv run python scripts/benchmark_agents.py --agents rule_baseline,agent_core_improved

    # Deck-arm sweep: same il_agent policy/weights, different submitted deck
    # (configs/deck_lists/<tag>.csv), isolated from the standing Glicko file:
    uv run python scripts/benchmark_agents.py \
        --agents il_agent@dragapult_ex,il_agent@alakazam,kiyotah_dragapult \
        --no-glicko-persist --out reports/deck_selection_benchmark.json

Output: a win-rate table (rows = player A, cols = player B) and per-agent
overall win rate, printed to stdout and saved next to the script.

Note: each match is 2 games (seat A=p0,p1 then A=p1,p0) so first-player bias
is cancelled. With --games N you get N such mirrored pairs per ordered pair.

NO COMMON RANDOM SEEDS. Checked against the actual engine: `cabt.json`'s
configuration schema exposes no seed field, `kaggle_environments.make()`'s
generic `configuration` dict has no seed convention this env reads, and the
Python-side `cg.game.battle_start(deck0, deck1)` call takes no seed
parameter -- deck shuffling and opening hands are decided inside the
compiled native engine (`libcg-arm64.so` / `cg.dll` / `libcg.dylib`), which
exposes no RNG control from Python. Different runs of the same pairing see
different shuffles/hands; there is no way to hold that constant with the
harness as given. Recorded here so this isn't silently assumed later.
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
# Agents import `from cg.api import ...`, so we need a cg package that ships
# api.py (the agent-facing wrapper) *and* a native lib for this platform. The
# canonical copy is data/external/cg-lib (its libcg.dylib is the arm64 build).
# When this script runs from a git worktree, that dir isn't checked out, so we
# also walk up parent directories to find the primary checkout's cg-lib. The
# other in-repo cg copies (sample-agent-output, submissions/*) ship only a
# Linux libcg.so and dlopen-crash on macOS, so they're intentionally not used.
_cg_cands = [REPO / "data" / "external" / "cg-lib"]
_cg_cands += [p / "data" / "external" / "cg-lib" for p in REPO.parents]
_cg_cands.append(REPO / "agents" / "mega_lucario")
for _cand in _cg_cands:
    if (_cand / "cg" / "api.py").exists():
        sys.path.insert(0, str(_cand))
        break


AGENT_FILES = {
    "rule_baseline": REPO / "agents" / "mega_lucario" / "agent_core.py",
    "improved_prob_main": REPO / "agents" / "improved_probabilistic" / "main.py",
    "agent_core_improved": REPO / "agents" / "mega_lucario" / "agent_core_improved.py",
    "proto": REPO / "scripts" / "_proto_agent.py",
    "il_agent": REPO / "agents" / "il_agent" / "agent_core.py",
    # kojimar's "Simple Baseline + Matchup Tests" Mega Lucario ex, ported as a
    # bare module (literal DECK -> my_deck). Distinct 60-card list from
    # rule_baseline/mega_lucario; see agents/kojimar_lucario/agent_core.py.
    "kojimar_lucario": REPO / "agents" / "kojimar_lucario" / "agent_core.py",
    "grunt": REPO / "agents" / "grunt" / "agent_core.py",
    # Floor test, not a competitor: uniform-random legal moves on the same
    # frozen deck. If a trained policy doesn't beat this decisively, its
    # offline accuracy isn't credible evidence it learned anything.
    "random_legal": REPO / "agents" / "random_legal" / "agent_core.py",
    # Public opponent-pool vetted batch (Q25 / notes/phase0_discovery_report.md
    # gap: benchmarking only against our own agents was never evidence about
    # the ladder). Pulled from Kaggle's competition Code tab, individually
    # reviewed for safety before wiring in -- see notebooks/reference/INDEX.md.
    "kiyotah_dragapult": REPO / "agents" / "kiyotah_dragapult" / "agent_core.py",
    "kiyotah_iono": REPO / "agents" / "kiyotah_iono" / "agent_core.py",
    "kiyotah_abomasnow": REPO / "agents" / "kiyotah_abomasnow" / "agent_core.py",
    "dedquoc_rule_engine": REPO / "agents" / "dedquoc_rule_engine" / "agent_core.py",
    "ryotasueyoshi_alakazam": REPO / "agents" / "ryotasueyoshi_alakazam" / "agent_core.py",
    "makthanithin_improved_prob": REPO / "agents" / "makthanithin_improved_prob" / "agent_core.py",
    # jek1wantaufik's rule-based Mega Lucario ex agent, recovered from the
    # Kaggle model `jek1wantaufik/buddy` (main.py + deck.pkl) that its
    # "Strategy Analysis" notebook only statically analyses -- the notebook
    # itself ships no runnable agent(). Independent implementation of the
    # Lucario archetype (its own scoring constants), reviewed for safety
    # before wiring in -- see notebooks/reference/jek1wantaufik-strategy-analysis/NOTES.md.
    "jek1wantaufik_lucario": REPO / "agents" / "jek1wantaufik_lucario" / "agent_core.py",
    # mechi22's notebook ships main.py as a base64 blob (SHA256-verified, not
    # encrypted) to deter forking -- decoded to plain source for this repo's
    # review; see notebooks/reference/mechi22-alakazam/main_decoded.py.
    "mechi22_alakazam": REPO / "agents" / "mechi22_alakazam" / "agent_core.py",
    # Phase 1 archetype-coverage recruitment (notes/phase1_gate1_report.md):
    # the pool was 9/14 agents on the exact frozen Mega Lucario ex deck before
    # this. Archaludon ex / Cinderace metal-tempo -- a genuinely different
    # archetype (Metal-type, not Fighting/Psychic/Dragon/Electric/Grass-Ice
    # like everything else in the pool).
    "plamen06_steel": REPO / "agents" / "plamen06_steel" / "agent_core.py",
    # Second wave of public-pool opponents, wired to widen the pool's *strength*
    # range so it predicts the ladder, not just rank our own agents. Each was
    # individually source-reviewed; see the "Benchmark-wiring wave" section of
    # notebooks/reference/INDEX.md.
    #   romanrozen_strong_start -- Kaggle LB ~950 Probabilistic Expectimax with a
    #     UCB1/MCTS re-ranker over *real* cg engine search rollouts. Strongest
    #     public opponent in the pool. Byte-identical to aristophanivan's
    #     improved-probabilistic-agent (same lineage), so only one copy is wired.
    #     NB: also a near-duplicate policy of kojimar_lucario is NOT re-added --
    #     kojimar's "Simple Baseline" is already wired above as kojimar_lucario.
    "romanrozen_strong_start": REPO / "agents" / "romanrozen_strong_start" / "agent_core.py",
    #   avikdas567_heuristic -- weak: scores options by substring-matching
    #     str(option). Fills the tier between random_legal and the real policies.
    "avikdas567_heuristic": REPO / "agents" / "avikdas567_heuristic" / "agent_core.py",
    #   makimakiai_rl -- RL-tuned linear-weights policy over option types. The
    #     upstream notebook only ships the *training harness*; its LEARNED_WEIGHTS
    #     bake is nondeterministic and wasn't reproduced, so this runs on the
    #     notebook's DEFAULT (untrained) weights -- a legal, distinct-method
    #     opponent, but NOT the tuned agent. Labeled untrained in INDEX.md.
    "makimakiai_rl": REPO / "agents" / "makimakiai_rl" / "agent_core.py",
}

# Where each agent's real competition entry point (main.py) lives, if any.
# Loading via main.py guarantees the deck is wired up exactly as the harness
# would at submission time.
AGENT_MAIN = {
    "rule_baseline": REPO / "submissions" / "mega_lucario" / "main.py",
    "agent_core_improved": REPO / "submissions" / "mega_lucario_improved" / "main.py",
}


# Populated by load_agent as a side effect, keyed by the same `name` passed
# in. Lets callers that only kept the `agent` callable (e.g. eval_rung3_sanity.py)
# stay on the unchanged load_agent(name) -> fn signature, while run_benchmark
# can still reach each module's diag_snapshot()/diag_reset() (see "Fallback
# diagnostics" below) if the agent exposes the fallback-tracking pattern.
_LOADED_MODULES: dict[str, object] = {}


DECK_LISTS_DIR = REPO / "configs" / "deck_lists"


def load_agent(name: str):
    """Load an agent's `agent` callable.

    Prefers the bundle `main.py` (real entry point, deck wired up). Falls back
    to the bare module — for bare modules that need `my_deck` injected (e.g.
    the sample rule baseline), we read it from the matching deck.csv.

    `name` may carry a deck override as `<agent>@<deck-tag>` (e.g.
    `il_agent@dragapult_ex`), where `<deck-tag>.csv` must exist under
    configs/deck_lists/. The base agent loads and runs completely unmodified;
    only its module-level `my_deck` is overwritten after loading, so the SAME
    policy/weights can be benchmarked piloting a different 60-card deck. Each
    call builds a fresh module object (no caching), so distinct deck-tagged
    labels for the same base agent never share state.
    """
    base_name, _, deck_tag = name.partition("@")
    main_py = AGENT_MAIN.get(base_name)
    if main_py and main_py.exists():
        spec = importlib.util.spec_from_file_location(f"_bench_{name}", main_py)
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(main_py.parent))
        spec.loader.exec_module(mod)
        fn = getattr(mod, "agent", None)
        if not callable(fn):
            raise AttributeError(f"{name} bundle has no callable `agent`")
    else:
        path = AGENT_FILES[base_name]
        if not path.exists():
            raise FileNotFoundError(f"no agent file for {base_name}: {path}")
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
            raise AttributeError(f"{base_name} has no callable `agent`")

    if deck_tag:
        if not hasattr(mod, "my_deck"):
            raise AttributeError(f"{base_name} has no `my_deck` to override (not deck-injectable)")
        deck_csv = DECK_LISTS_DIR / f"{deck_tag}.csv"
        if not deck_csv.exists():
            raise FileNotFoundError(f"deck override '{deck_tag}' not found: {deck_csv}")
        mod.my_deck = [int(x) for x in deck_csv.read_text().splitlines() if x.strip()][:60]

    _LOADED_MODULES[name] = mod
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


def play_match(agent_a, agent_b, env_factory, pairs: int = 1,
               name_a: str = "agent_a", name_b: str = "agent_b") -> tuple[int, int, int, float]:
    """Play `pairs` mirrored game pairs (both seat orders each). Returns (a_wins, b_wins, draws, seconds).

    Each mirrored pair cancels first-player advantage: game 1 has A in seat 0,
    game 2 has B in seat 0. A win for "player 0" maps to whichever agent sits
    in player 0's seat.
    """
    t0 = time.time()
    a_wins = b_wins = draws = 0
    for _ in range(pairs):
        # a_in_seat0 tracks which *role* (a or b) sits in seat 0, independent
        # of object identity -- true self-play passes the same callable as
        # both agent_a and agent_b, so `is` checks can't distinguish the
        # roles (both "seat0 is agent_a" and "seat0 is agent_b" are true on
        # every game, double-counting every decisive result).
        for a_in_seat0 in (True, False):
            seat0 = agent_a if a_in_seat0 else agent_b
            seat1 = agent_b if a_in_seat0 else agent_a
            env = env_factory()
            trace = env.run([seat0, seat1])
            final = trace[-1]
            r = [final[i]["reward"] for i in range(2)]
            # kaggle_environments gives a crashed agent reward=None instead of
            # a comparable score (its exception ends the episode early) --
            # treat that seat as an outright loss instead of letting the
            # `None > int` comparison below take the whole tournament down.
            seat_role = ("a", "b") if a_in_seat0 else ("b", "a")
            for seat_idx, role in enumerate(seat_role):
                if r[seat_idx] is None:
                    who = name_a if role == "a" else name_b
                    print(f"  [WARN] {who} crashed (status={final[seat_idx].get('status')}); scoring as a loss")
                    r[seat_idx] = -float("inf")
            if r[0] == r[1]:
                draws += 1
            else:
                winner_role = seat_role[0] if r[0] > r[1] else seat_role[1]
                if winner_role == "a":
                    a_wins += 1
                else:
                    b_wins += 1
    return a_wins, b_wins, draws, time.time() - t0


def run_benchmark(agents: list[str], games_per_pair: int = 8,
                   glicko_path: Path = GLICKO_PATH, out_path: Path | None = None,
                   persist_glicko: bool = True):
    from kaggle_environments import make

    print(f"Loading {len(agents)} agents: {', '.join(agents)}")
    _LOADED_MODULES.clear()  # drop any modules from a prior run_benchmark() call in this process
    fns = {name: load_agent(name) for name in agents}
    for mod in _LOADED_MODULES.values():
        if hasattr(mod, "diag_reset"):
            mod.diag_reset()

    n = len(agents)
    wins = {a: {b: 0 for b in agents} for a in agents}  # wins[a][b] = a beat b
    games = {a: {b: 0 for b in agents} for a in agents}
    wall_clock = {a: {b: 0.0 for b in agents} for a in agents}  # seconds, symmetric per pairing
    totals = {a: {"w": 0, "g": 0} for a in agents}
    # (player_a, player_b, score_a) triples for the Glicko-1 rating period.
    # Self-play is excluded: an agent playing itself carries no information
    # about its rating relative to the field.
    glicko_games: list[tuple[str, str, float]] = []

    total_pairs = n * (n - 1) // 2 + n  # unordered incl. self-play
    done = 0
    run_t0 = time.time()
    for i in range(n):
        for j in range(i, n):
            a, b = agents[i], agents[j]
            pairs = games_per_pair if a != b else max(1, games_per_pair // 2)
            aw, bw, dr, secs = play_match(fns[a], fns[b], lambda: make("cabt"), pairs, name_a=a, name_b=b)
            wins[a][b] += aw
            wins[b][a] += bw
            games[a][b] += aw + bw + dr
            games[b][a] += aw + bw + dr
            wall_clock[a][b] += secs
            wall_clock[b][a] += secs
            totals[a]["w"] += aw
            totals[b]["w"] += bw
            totals[a]["g"] += aw + bw + dr
            totals[b]["g"] += aw + bw + dr
            if a != b:
                glicko_games += [(a, b, 1.0)] * aw
                glicko_games += [(a, b, 0.0)] * bw
                glicko_games += [(a, b, 0.5)] * dr
            done += 1
            n_games_pair = aw + bw + dr
            print(
                f"[{done}/{total_pairs}] {a} vs {b}: "
                f"{a} {aw}-{bw}{('-' + str(dr) + 'd') if dr else ''}  "
                f"({n_games_pair} games, {secs:.1f}s, {secs/n_games_pair:.2f}s/game)"
            )
    total_wall_clock = time.time() - run_t0

    # ---- report ----
    print("\n=== Head-to-head win matrix (rows beat cols; cell = row's wins [95% CI]) ===")
    header = "agent".ljust(22) + "".join(b[:6].ljust(8) for b in agents) + "win% [95% CI]"
    print(header)
    overall = {}
    overall_ci = {}
    for a in agents:
        row = a.ljust(22)
        w = totals[a]["w"]
        g = totals[a]["g"]
        overall[a] = (100.0 * w / g) if g else 0.0
        overall_ci[a] = glicko1.wilson_ci(w, g)
        cells = ""
        for b in agents:
            if a == b:
                cells += "-".ljust(8)
            else:
                cells += f"{wins[a][b]}".ljust(8)
        lo, hi = overall_ci[a][1] * 100, overall_ci[a][2] * 100
        row += cells + f"{overall[a]:5.1f}  [{lo:4.1f},{hi:4.1f}]"
        print(row)

    print("\n=== Per-pairing win rate with Wilson 95% CI (row vs col) ===")
    for a in agents:
        for b in agents:
            if a >= b:
                continue
            n_ab = games[a][b]
            if n_ab == 0:
                continue
            p, lo, hi = glicko1.wilson_ci(wins[a][b], n_ab)
            print(f"  {a} vs {b}: {wins[a][b]}/{n_ab} = {p*100:5.1f}% [{lo*100:4.1f},{hi*100:4.1f}]  "
                  f"({wall_clock[a][b]:.1f}s, {wall_clock[a][b]/n_ab:.2f}s/game)")

    print("\n=== Overall win rate (all games), Wilson 95% CI ===")
    for a in sorted(agents, key=lambda x: -overall[x]):
        lo, hi = overall_ci[a][1] * 100, overall_ci[a][2] * 100
        print(f"  {a:22s} {overall[a]:5.1f}%  [{lo:4.1f},{hi:4.1f}]  ({totals[a]['w']}/{totals[a]['g']})")

    # ---- Glicko-1 ratings ----
    # Win-rate is a single-run snapshot; Glicko carries a rating + confidence
    # (RD) forward across every benchmark run, weighted by opponent strength,
    # so it's a better estimate of true skill than this run's win% alone.
    prior_ratings = load_glicko_ratings(glicko_path) if persist_glicko else {}
    glicko = glicko1.run_rating_period(prior_ratings, glicko_games)
    if persist_glicko:
        save_glicko_ratings(glicko, glicko_path)

    print("\n=== Glicko-1 ratings (persisted across runs, higher = stronger) ===")
    ranked = sorted(agents, key=lambda x: -glicko[x].rating)
    for a in ranked:
        r = glicko[a]
        print(
            f"  {a:22s} {r.rating:7.1f}  (RD {r.rd:5.1f}, 95% CI +/-{2*r.rd:5.1f})  "
            f"GXE {glicko1.gxe(r):5.1f}%"
        )

    # ---- Fallback diagnostics ----
    # Agents that expose the _DIAG/diag_snapshot pattern (see
    # agents/mega_lucario/agent_core_improved.py, agents/improved_probabilistic/main.py,
    # agents/mechi22_alakazam/agent_core.py) count how often their never-crash
    # fallback layers actually fire across every game just played. A nonzero
    # fallback_rate here means search or the heuristic is silently failing on
    # real inputs -- worth investigating even though the agent never crashed.
    fallback_diag = {}
    for a in agents:
        mod = _LOADED_MODULES.get(a)
        if mod is not None and hasattr(mod, "diag_snapshot"):
            fallback_diag[a] = mod.diag_snapshot()
    if fallback_diag:
        print("\n=== Fallback diagnostics (fraction of decisions that hit a fallback layer) ===")
        for a, snap in fallback_diag.items():
            print(f"  {a:22s} fallback_rate={snap.get('fallback_rate', 0.0):6.2%}  "
                  f"decisions={snap.get('decisions', 0)}  raw={dict(snap)}")

    result = {
        "agents": agents,
        "games_per_pair": games_per_pair,
        "seeded": False,
        "seed_note": "cabt exposes no RNG seed from Python (native engine, no seed param on "
                      "battle_start or in cabt.json's configuration schema) -- deck shuffles "
                      "and opening hands are NOT held constant across runs or pairings.",
        "wins": wins,
        "games": games,
        "wall_clock_sec": wall_clock,
        "total_wall_clock_sec": total_wall_clock,
        "overall_win_pct": overall,
        "overall_win_pct_wilson_ci": {
            a: {"p": overall_ci[a][0] * 100, "lo": overall_ci[a][1] * 100, "hi": overall_ci[a][2] * 100}
            for a in agents
        },
        "glicko": {
            a: {"rating": glicko[a].rating, "rd": glicko[a].rd, "gxe": glicko1.gxe(glicko[a])}
            for a in agents
        },
        "fallback_diag": fallback_diag,
    }
    out_path = out_path or (REPO / "reports" / "agent_benchmark.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\ntotal wall clock: {total_wall_clock:.1f}s")
    print(f"saved: {out_path}")
    print(f"saved: {glicko_path}" if persist_glicko else "glicko: not persisted (isolated run)")
    return result


def main():
    ap = argparse.ArgumentParser(description="Benchmark agent vs agent performance.")
    ap.add_argument("--agents", default=",".join(AGENT_FILES.keys()),
                    help="comma-separated subset of: " + ", ".join(AGENT_FILES))
    ap.add_argument("--games", type=int, default=8, dest="games_per_pair",
                    help="mirrored game pairs per ordered agent pair")
    ap.add_argument("--glicko-path", type=Path, default=GLICKO_PATH,
                    help="where to load/persist Glicko ratings (default: reports/glicko_ratings.json)")
    ap.add_argument("--out", type=Path, default=None,
                    help="where to save the result JSON (default: reports/agent_benchmark.json)")
    ap.add_argument("--no-glicko-persist", action="store_true",
                    help="score Glicko for this run's printout but don't read/write --glicko-path "
                         "(use for isolated runs, e.g. deck-arm sweeps, that shouldn't pollute "
                         "the standing ratings)")
    args = ap.parse_args()
    agents = [a.strip() for a in args.agents.split(",") if a.strip()]
    unknown = [a for a in agents if a.partition("@")[0] not in AGENT_FILES]
    if unknown:
        sys.exit(f"unknown agent(s): {unknown}. Available: {list(AGENT_FILES)}")
    run_benchmark(agents, args.games_per_pair, glicko_path=args.glicko_path,
                  out_path=args.out, persist_glicko=not args.no_glicko_persist)


if __name__ == "__main__":
    main()
