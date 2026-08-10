"""Ablation A0 — does real search + a working BC prior beat pure AdvancedPolicy?

Redesigned version of the A0 ablation. The original plan (USE_BC_PRIOR=0 vs 1
on the AS-SUBMITTED 827.8 code) was abandoned after tracing the code: at real
MAIN decisions maxCount==1 always, so SEARCH_ALGO's len(candidates)==1 branch
returns before bc_prior.candidate_probs() is ever reached in that exact
source. USE_BC_PRIOR was architecturally inert in submission 55162376 (proven
both statically and dynamically in the reproducibility audit).

This script instead measures HEAD's agent_core_improved.py (which has the
rank_all() fix, so SEARCH_ALGO gets real multi-candidate lists) in three
arms, one env-var configuration per process invocation (env vars are read at
module import time):

    control        USE_SEARCH=0                    -> current default; behaviorally
                                                        equals 827.8's real behavior
    search_noprior USE_SEARCH=1 USE_BC_PRIOR=0      -> the previously-benchmarked
                                                        losing config (approximately —
                                                        prior state during that run
                                                        wasn't recorded)
    search_prior   USE_SEARCH=1 USE_BC_PRIOR=1      -> never actually measured before
                   (BC_PRIOR_C=0.75, the hardcoded default)

Each arm plays `--pairs` mirrored pairs (2*pairs games) against a single
fixed opponent (default: makthanithin_improved_prob, for continuity with the
one existing historical data point). Appends results to
reports/ablation_a0.json; run scripts/ablation_a0_summary.py after all three
arms to see Wilson 95% CIs.

Usage (one invocation per arm — env vars are read at import time):
    USE_SEARCH=0                 uv run python scripts/ablation_a0.py --arm control
    USE_SEARCH=1 USE_BC_PRIOR=0  uv run python scripts/ablation_a0.py --arm search_noprior
    USE_SEARCH=1 USE_BC_PRIOR=1  uv run python scripts/ablation_a0.py --arm search_prior
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS_PATH = REPO / "reports" / "ablation_a0.json"


def load_bare_agent(path: Path):
    spec = importlib.util.spec_from_file_location(f"_a0_{path.stem}_{id(path)}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(mod)
    if not hasattr(mod, "my_deck") and (path.parent / "deck.csv").exists():
        deck = [int(x) for x in (path.parent / "deck.csv").read_text().splitlines() if x.strip()]
        mod.my_deck = deck[:60]
    return mod


def play_match(agent_a, agent_b, env_factory, pairs: int):
    """Mirrored pairs; a_wins/b_wins/draws counted from the fixed-arm agent's POV as 'a'."""
    from kaggle_environments import make  # noqa: F401 (env_factory already bound)

    t0 = time.time()
    a_wins = b_wins = draws = 0
    per_game = []
    for _ in range(pairs):
        for seat0 in (agent_a, agent_b):
            seat1 = agent_b if seat0 is agent_a else agent_a
            env = env_factory()
            trace = env.run([seat0, seat1])
            final = trace[-1]
            r = [final[i]["reward"] for i in range(2)]
            steps = len(trace)
            if r[0] == r[1]:
                draws += 1
                outcome = "draw"
            elif r[0] > r[1]:
                a_wins += 1 if seat0 is agent_a else 0
                b_wins += 1 if seat0 is agent_b else 0
                outcome = "a" if seat0 is agent_a else "b"
            else:
                a_wins += 1 if seat0 is agent_b else 0
                b_wins += 1 if seat0 is agent_a else 0
                outcome = "b" if seat0 is agent_a else "a"
            per_game.append({"steps": steps, "winner": outcome, "a_seat": 0 if seat0 is agent_a else 1})
    return a_wins, b_wins, draws, time.time() - t0, per_game


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["control", "search_noprior", "search_prior"])
    ap.add_argument("--opponent", default="makthanithin_improved_prob")
    ap.add_argument("--pairs", type=int, default=5, help="mirrored pairs (2x games)")
    ap.add_argument("--my-deck-csv", default=None,
                     help="override agent_core_improved's my_deck with this deck.csv "
                          "(default: its own hardcoded Mega Lucario ex DECK). Use this to "
                          "test the search+BC-prior engine on a deck whose cards actually "
                          "have trained embeddings (e.g. agents/ryotasueyoshi_alakazam/deck.csv).")
    ap.add_argument("--results-path", default=str(RESULTS_PATH),
                     help="where to append results (default: reports/ablation_a0.json)")
    args = ap.parse_args()
    results_path = Path(args.results_path)

    agent_path = REPO / "agents" / "mega_lucario" / "agent_core_improved.py"
    opp_path = REPO / "agents" / args.opponent / "agent_core.py"
    if not opp_path.exists():
        sys.exit(f"no such opponent agent file: {opp_path}")

    a0_mod = load_bare_agent(agent_path)
    opp_mod = load_bare_agent(opp_path)

    deck_tag = "mega_lucario_ex (default)"
    if args.my_deck_csv:
        deck_path = Path(args.my_deck_csv)
        deck = [int(x) for x in deck_path.read_text().splitlines() if x.strip()]
        assert len(deck) == 60, f"{deck_path} must be a 60-card deck, got {len(deck)}"
        a0_mod.my_deck = deck
        deck_tag = str(deck_path)

    use_search = getattr(a0_mod, "USE_SEARCH", None)
    use_bc_prior = getattr(a0_mod, "USE_BC_PRIOR", None)
    bc_prior_c = getattr(a0_mod, "BC_PRIOR_C", None)
    bc_prior_ok = getattr(a0_mod, "_BC_PRIOR_OK", None)
    print(f"arm={args.arm} deck={deck_tag} USE_SEARCH={use_search} USE_BC_PRIOR={use_bc_prior} "
          f"BC_PRIOR_C={bc_prior_c} bc_prior_available={bc_prior_ok}")

    from kaggle_environments import make

    def env_factory():
        return make("cabt")

    a_wins, b_wins, draws, elapsed, per_game = play_match(a0_mod.agent, opp_mod.agent, env_factory, args.pairs)
    n_games = 2 * args.pairs
    print(f"arm={args.arm} vs {args.opponent}: {n_games} games, "
          f"a0_wins={a_wins} opp_wins={b_wins} draws={draws}, {elapsed:.1f}s total, "
          f"{elapsed / n_games:.1f}s/game")

    results_path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if results_path.exists():
        existing = json.loads(results_path.read_text())
    existing.append({
        "arm": args.arm,
        "deck": deck_tag,
        "opponent": args.opponent,
        "use_search": use_search,
        "use_bc_prior": use_bc_prior,
        "bc_prior_c": bc_prior_c,
        "bc_prior_available": bc_prior_ok,
        "pairs": args.pairs,
        "n_games": n_games,
        "a0_wins": a_wins,
        "opp_wins": b_wins,
        "draws": draws,
        "elapsed_sec": elapsed,
        "per_game": per_game,
    })
    results_path.write_text(json.dumps(existing, indent=2))
    print(f"appended to {results_path}")


if __name__ == "__main__":
    main()
