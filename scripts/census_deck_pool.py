"""What is a deck pool actually uniform OVER?

`--deck-pool` samples uniformly over pool MEMBERS, but members are 60-card
lists, not archetypes. If 12 of 35 lists are Mega Lucario ex variants, "uniform
over 35 decks" is really "34% Lucario" — and a policy trained on it is not
uniform over the strategy space in the way the flag's name suggests. Deck names
do not settle this either: the pool's names come from agent directories and
those have already been shown to mislead (2 Grimmsnarl pilots filed under
"*_alakazam").

So this reports three cuts of a pool, all derived from deck CONTENT:

 1. ARCHETYPE — decks grouped by their headline Pokemon (the highest-stage
    non-basic evolution present, resolved through cg's own card DB). Two lists
    that win the same way land in the same bucket even under different names.
 2. NEAR-DUPLICATES — pairs above --jaccard on card multiset overlap. Content
    dedup already removed EXACT duplicates; these are the 58/60 variants that
    survive it and still double an archetype's sampling weight.
 3. The resulting effective sampling distribution over archetypes.

    uv run python scripts/census_deck_pool.py --deck-pool '@configs/deck_pools/legal_decks.txt'
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokemon_tcg import config  # noqa: E402
from pokemon_tcg.deck_pool import _expand_spec, parse_deck  # noqa: E402
from pokemon_tcg.selfplay import _safe_choice  # noqa: E402  (sets the cg path)

assert _safe_choice  # imported for its cg sys.path side effect


def card_index() -> dict:
    """{cardId: CardData} from cg's public DB; empty if the lib is unavailable."""
    try:
        from cg.api import all_card_data
        return {c.cardId: c for c in all_card_data()}
    except Exception as exc:  # pragma: no cover - environment-dependent
        print(f"WARNING: cg card DB unavailable ({exc}); archetypes will be "
              f"reported as 'unknown'")
        return {}


def _attr(cd, *names, default=None):
    """cg's CardData field names have moved before; try several, don't guess."""
    for n in names:
        v = getattr(cd, n, None)
        if v not in (None, ""):
            return v
    return default


def headline(deck: list[int], db: dict) -> str:
    """The deck's identity card: the rarest-in-deck highest-stage Pokemon.

    Evolution stage first (a Mega/Stage-2 line names the deck), then fewest
    copies as the tiebreak — the 1-2 copy finisher identifies an archetype far
    better than the 4-copy basic every deck runs.
    """
    counts = Counter(deck)
    best, best_key = None, None
    for cid, n in counts.items():
        cd = db.get(cid)
        if cd is None:
            continue
        kind = str(_attr(cd, "cardType", "type", "category", default="")).upper()
        if "POKEMON" not in kind and kind not in ("", "0"):
            continue
        stage = _attr(cd, "stage", "evolutionStage", "subType", default=0)
        try:
            stage_n = int(stage)
        except (TypeError, ValueError):
            s = str(stage).upper()
            stage_n = 3 if "MEGA" in s else 2 if "TWO" in s or "2" in s else \
                1 if "ONE" in s or "1" in s else 0
        hp = int(_attr(cd, "hp", "HP", default=0) or 0)
        key = (stage_n, hp, -n)
        if best_key is None or key > best_key:
            best, best_key = cid, key
    if best is None:
        return "unknown"
    name = _attr(db[best], "name", "cardName", default=None)
    return f"{name}" if name else f"card{best}"


def jaccard(a: list[int], b: list[int]) -> float:
    ca, cb = Counter(a), Counter(b)
    inter = sum((ca & cb).values())
    union = sum((ca | cb).values())
    return inter / union if union else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck-pool", default="@configs/deck_pools/legal_decks.txt")
    ap.add_argument("--jaccard", type=float, default=0.85,
                    help="report deck pairs at or above this card-overlap")
    ap.add_argument("--out", type=Path,
                    default=config.REPORTS_DIR / "deck_pool_census.json")
    args = ap.parse_args()

    db = card_index()
    pairs = _expand_spec(args.deck_pool)
    decks = [(name, parse_deck(path)) for name, path in pairs]
    print(f"pool spec {args.deck_pool!r}: K={len(decks)}\n")

    arch: dict[str, list[str]] = defaultdict(list)
    for name, deck in decks:
        arch[headline(deck, db)].append(name)

    print(f"ARCHETYPES ({len(arch)} distinct headline Pokemon over {len(decks)} lists)")
    rows = sorted(arch.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    for h, names in rows:
        share = len(names) / len(decks)
        print(f"  {len(names):>2} lists  {share:5.1%}  {h}")
        print(f"            {', '.join(sorted(names))}")

    near = []
    for i in range(len(decks)):
        for j in range(i + 1, len(decks)):
            s = jaccard(decks[i][1], decks[j][1])
            if s >= args.jaccard:
                near.append((round(s, 3), decks[i][0], decks[j][0]))
    near.sort(reverse=True)
    print(f"\nNEAR-DUPLICATES at jaccard >= {args.jaccard}: {len(near)} pair(s)")
    for s, a, b in near[:40]:
        print(f"  {s:.3f}  {a}  ~  {b}")
    if len(near) > 40:
        print(f"  ... and {len(near) - 40} more")

    report = {
        "spec": args.deck_pool, "k": len(decks),
        "archetypes": {h: sorted(n) for h, n in rows},
        "archetype_share": {h: round(len(n) / len(decks), 4) for h, n in rows},
        "near_duplicates": [{"jaccard": s, "a": a, "b": b} for s, a, b in near],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nreport: {args.out}")
    top = rows[0]
    print(f"HEAVIEST ARCHETYPE: {top[0]} at {len(top[1]) / len(decks):.1%} of draws. "
          f"Uniform over LISTS is not uniform over strategies -- say so when "
          f"reporting results from this pool.")


if __name__ == "__main__":
    main()
