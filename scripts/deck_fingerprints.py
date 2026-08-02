"""Group the benchmark agent pool by the 60-card deck each agent pilots.

WHY THIS EXISTS
---------------
scripts/benchmark_agents.py has no deck axis: a deck reaches an agent through
its module-level `my_deck`, which `load_agent` fills from the agent's own
sibling deck.csv. Deck is therefore welded to agent identity, and every cell of
the round-robin win matrix is a *policy + deck* result. Reading "agent X beats
agent Y" as "policy X beats policy Y" is only sound when X and Y happen to
pilot the same 60 cards.

This script says which pairs those are. It hashes each agent's deck.csv and
groups by the hash, so a comparison can be labelled up front as either

    CLEAN      -- same deck, different policy: a real policy comparison
    CONFOUNDED -- different decks: policy and archetype move together, and the
                  result attributes nothing to either on its own

Use it before quoting any head-to-head number as evidence about a policy --
in particular before claiming a trained model beats the field, where the field
mostly pilots decks the model never trained to pilot.

Agents with no sibling deck.csv (their decklist is a module literal, or is
injected by a bundle main.py) are reported separately rather than guessed at.

USAGE
-----
    uv run python scripts/deck_fingerprints.py
    uv run python scripts/deck_fingerprints.py --agent il_agent   # just its clean peers
    uv run python scripts/deck_fingerprints.py --json reports/deck_fingerprints.json
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import benchmark_agents as bench  # noqa: E402

NO_DECK = "(no deck.csv)"


def deck_fingerprint(agent: str) -> tuple[str, list[str]]:
    """Return (fingerprint, cards) for `agent`'s sibling deck.csv.

    Fingerprint is a short hash of the ordered 60-card list, matching exactly
    what `benchmark_agents.load_agent` injects as `my_deck` (same slice, same
    blank-line filtering) -- so two agents share a fingerprint iff the harness
    hands them the identical decklist.
    """
    deck_csv = bench.AGENT_FILES[agent].parent / "deck.csv"
    if not deck_csv.exists():
        return NO_DECK, []
    cards = [x.strip() for x in deck_csv.read_text().splitlines() if x.strip()][:60]
    return hashlib.sha256(",".join(cards).encode()).hexdigest()[:8], cards


def group_by_deck(agents: list[str] | None = None) -> dict[str, list[str]]:
    """Map deck fingerprint -> agents piloting it, for `agents` (default: all)."""
    agents = agents if agents is not None else list(bench.AGENT_FILES)
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for name in agents:
        fp, _ = deck_fingerprint(name)
        groups[fp].append(name)
    return dict(groups)


def clean_peers(agent: str, agents: list[str] | None = None) -> list[str]:
    """Agents that pilot the identical deck to `agent` -- its unconfounded opponents.

    Empty when `agent` has no deck.csv, or is the only one on its deck: then
    every head-to-head it has is policy-and-deck at once.
    """
    fp, _ = deck_fingerprint(agent)
    if fp == NO_DECK:
        return []
    return [a for a in group_by_deck(agents).get(fp, []) if a != agent]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agent", help="report this agent's clean (same-deck) peers only")
    ap.add_argument("--json", type=Path, help="also write the grouping to this path")
    args = ap.parse_args()

    groups = group_by_deck()
    n_agents = len(bench.AGENT_FILES)

    if args.agent:
        if args.agent not in bench.AGENT_FILES:
            sys.exit(f"unknown agent: {args.agent}. Available: {list(bench.AGENT_FILES)}")
        fp, cards = deck_fingerprint(args.agent)
        peers = clean_peers(args.agent)
        print(f"{args.agent}: deck {fp} ({len(cards)} cards)\n")
        if fp == NO_DECK:
            print("  no sibling deck.csv -- decklist is a module literal or bundle-injected;")
            print("  cannot be matched against the pool here.")
        elif peers:
            print(f"  CLEAN (same deck, {len(peers)} agents) -- policy comparison is valid:")
            for p in sorted(peers):
                print(f"    {p}")
            confounded = n_agents - len(peers) - 1
            print(f"\n  CONFOUNDED: the other {confounded} agents pilot a different deck.")
        else:
            print("  sole agent on this deck -- EVERY head-to-head it has is confounded")
            print("  by archetype. No clean policy comparison is available in this pool.")
    else:
        real = {k: v for k, v in groups.items() if k != NO_DECK}
        shared = {k: v for k, v in real.items() if len(v) > 1}
        print(f"{n_agents} registered agents, {len(real)} distinct decks "
              f"({len(shared)} of them shared by 2+ agents)\n")
        for fp, names in sorted(real.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            tag = "CLEAN group" if len(names) > 1 else "sole pilot "
            print(f"  {fp}  {tag}  ({len(names)})  {', '.join(sorted(names))}")
        if NO_DECK in groups:
            print(f"\n  {NO_DECK}  (deck is a module literal or bundle-injected): "
                  f"{', '.join(sorted(groups[NO_DECK]))}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "note": "deck fingerprint -> agents piloting that exact 60-card list. "
                    "Head-to-heads within a group are policy comparisons; across "
                    "groups they are policy+deck and attribute to neither alone.",
            "groups": {k: sorted(v) for k, v in sorted(groups.items())},
        }, indent=2))
        print(f"\nsaved: {args.json}")


if __name__ == "__main__":
    main()
