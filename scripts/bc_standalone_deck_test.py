"""Does the standalone BC-cloned policy (agents/il_agent, no search, no heuristic)
play better on a deck whose cards it actually saw in training?

Direct test of the deck-embedding hypothesis in notes/phase1_decisions.md: the
frozen Mega Lucario ex deck's defining cards (Riolu/Mega Lucario ex) appear in
0/800 training-corpus perspectives, so the IL policy's card embeddings for
them are untrained. Alakazam's defining cards (Abra/Kadabra/Alakazam) appear
in 16.1% (129/800) -- see notes/phase0_gate0_report.md's "A0''" section.

This is PURE BC cloning, not the search+prior hybrid tested in ablation_a0.py:
agents/il_agent/agent_core.py's agent() is the trained transformer scoring
options directly, no cg.api search, no hand-written heuristic anywhere in the
decision path.

Two arms, same architecture, only the deck differs:
    il_agent_lucario   -- agents/il_agent's own deck.csv (Mega Lucario ex,
                           0% training prevalence)
    il_agent_alakazam  -- my_deck overridden to agents/ryotasueyoshi_alakazam's
                           deck.csv (16.1% training prevalence)

Each arm plays `--pairs` mirrored pairs against each of `--opponents`.
Results appended to reports/bc_standalone_deck_test.json.

Usage:
    uv run python scripts/bc_standalone_deck_test.py --deck lucario --pairs 8
    uv run python scripts/bc_standalone_deck_test.py --deck alakazam --pairs 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablation_a0 import load_bare_agent, play_match  # noqa: E402
from benchmark_agents import AGENT_FILES, load_agent  # noqa: E402

RESULTS_PATH = REPO / "reports" / "bc_standalone_deck_test.json"

DECKS = {
    "lucario": None,  # None = use il_agent's own sibling deck.csv (default load path)
    "alakazam": REPO / "agents" / "ryotasueyoshi_alakazam" / "deck.csv",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", required=True, choices=list(DECKS))
    ap.add_argument("--opponents", default="makthanithin_improved_prob,rule_baseline")
    ap.add_argument("--pairs", type=int, default=8, help="mirrored pairs (2x games) per opponent")
    args = ap.parse_args()
    opponents = [o.strip() for o in args.opponents.split(",") if o.strip()]

    agent_path = REPO / "agents" / "il_agent" / "agent_core.py"
    il_mod = load_bare_agent(agent_path)  # loads its own deck.csv (Lucario) by default

    deck_tag = "agents/il_agent/deck.csv (Mega Lucario ex, default)"
    if DECKS[args.deck] is not None:
        deck_path = DECKS[args.deck]
        deck = [int(x) for x in deck_path.read_text().splitlines() if x.strip()]
        assert len(deck) == 60, f"{deck_path} must be 60 cards, got {len(deck)}"
        il_mod.my_deck = deck
        deck_tag = str(deck_path.relative_to(REPO))

    print(f"deck={args.deck} ({deck_tag})")

    from kaggle_environments import make

    def env_factory():
        return make("cabt")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else []

    for opp_name in opponents:
        if opp_name not in AGENT_FILES:
            print(f"skipping unknown opponent: {opp_name}")
            continue
        opp_agent_fn = load_agent(opp_name)  # benchmark_agents' loader: prefers AGENT_MAIN's main.py

        wins, losses, draws, elapsed, per_game = play_match(il_mod.agent, opp_agent_fn, env_factory, args.pairs)
        n_games = 2 * args.pairs
        print(f"  vs {opp_name}: {n_games} games, il_agent_wins={wins} opp_wins={losses} draws={draws}, "
              f"{elapsed:.1f}s total, {elapsed / n_games:.2f}s/game")

        existing.append({
            "deck": args.deck,
            "deck_tag": deck_tag,
            "opponent": opp_name,
            "pairs": args.pairs,
            "n_games": n_games,
            "il_agent_wins": wins,
            "opp_wins": losses,
            "draws": draws,
            "elapsed_sec": elapsed,
            "per_game": per_game,
        })
        RESULTS_PATH.write_text(json.dumps(existing, indent=2))

    print(f"appended to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
