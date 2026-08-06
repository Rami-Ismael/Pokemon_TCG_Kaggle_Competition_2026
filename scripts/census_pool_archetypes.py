"""Stage 0 census: which archetype does every pool agent ACTUALLY pilot?

Agent directory names are not trustworthy -- `prvsiyan_grimbelief_alakazam` and
`prvsiyan_templates_alakazam` both ship a deck.csv byte-identical to
configs/deck_lists/marnies_grimmsnarl_ex.csv. Before harvesting new public
agents to repair the pool's Grimmsnarl coverage, count what is already wired.

Deck identity key is the one reports/deck_selection.md fixed in §1: the ace /
carry Pokemon = the Pokemon card maximizing (evolution_stage + ex/megaEx bonus,
copy_count). Same key, so the counts here are comparable to that report's
corpus census.

Usage:
    uv run python scripts/census_pool_archetypes.py
    uv run python scripts/census_pool_archetypes.py --archetype grimmsnarl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parent.parent

for _cand in (REPO / "data" / "external" / "cg-lib", REPO / "agents" / "mega_lucario"):
    if (_cand / "cg" / "api.py").exists():
        sys.path.insert(0, str(_cand))
        break

from cg.api import all_card_data  # noqa: E402

CARD = {c.cardId: c for c in all_card_data()}
POKEMON_TYPE = 0  # cardType for Pokemon; energy is 5, trainers differ


def _is_pokemon(card) -> bool:
    """A card is a Pokemon iff it declares a stage flag or has HP."""
    return bool(card.basic or card.stage1 or card.stage2) or card.hp > 0


def _ace_rank(card, copies: int) -> tuple[int, int]:
    """(evolution stage + ex/megaEx bonus, copy count) -- the deck_selection key."""
    stage = 2 if card.stage2 else 1 if card.stage1 else 0
    bonus = 2 if card.megaEx else 1 if card.ex else 0
    return (stage + bonus, copies)


def ace_pokemon(deck_ids: list[int]) -> str:
    counts = Counter(deck_ids)
    best_name, best_key = "(none)", (-1, -1)
    for cid, n in counts.items():
        card = CARD.get(cid)
        if card is None or not _is_pokemon(card):
            continue
        key = _ace_rank(card, n)
        if key > best_key:
            best_name, best_key = card.name, key
    return best_name


def parse_deck(path: Path) -> list[int]:
    return [int(x) for x in path.read_text().split() if x.strip()][:60]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archetype", default=None,
                    help="case-insensitive substring filter on the ace name")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    glicko_path = REPO / "reports" / "glicko_ratings.json"
    glicko = {}
    if glicko_path.exists():
        raw = json.loads(glicko_path.read_text())
        raw = raw.get("ratings", raw)
        for k, v in raw.items():
            glicko[k] = v["rating"] if isinstance(v, dict) else v

    rows = []
    for deck_path in sorted((REPO / "agents").glob("*/deck.csv")):
        agent = deck_path.parent.name
        ids = parse_deck(deck_path)
        rows.append({
            "agent": agent,
            "ace": ace_pokemon(ids),
            "n_cards": len(ids),
            "glicko": glicko.get(agent),
        })

    if args.archetype:
        needle = args.archetype.lower()
        rows = [r for r in rows if needle in r["ace"].lower()]

    by_ace: dict[str, list[dict]] = {}
    for r in rows:
        by_ace.setdefault(r["ace"], []).append(r)

    print(f"{len(rows)} agent decks, {len(by_ace)} distinct archetypes\n")
    for ace, members in sorted(by_ace.items(), key=lambda kv: -len(kv[1])):
        rated = [m for m in members if m["glicko"] is not None]
        top = max((m["glicko"] for m in rated), default=None)
        top_s = f"{top:.1f}" if top is not None else "unrated"
        print(f"=== {ace}  ({len(members)} pilots, {len(rated)} rated, top {top_s})")
        for m in sorted(members, key=lambda m: -(m["glicko"] or -1)):
            g = f"{m['glicko']:8.1f}" if m["glicko"] is not None else "  UNRATED"
            flag = "" if m["n_cards"] == 60 else f"  !! {m['n_cards']} cards"
            print(f"    {g}  {m['agent']}{flag}")
        print()

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
