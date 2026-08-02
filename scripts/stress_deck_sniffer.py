"""Where does the deck sniffer break?

identify_opponent_deck.py showed near-perfect ARCHETYPE ID against the current pool.
That pool is easy: every archetype's Pokemon lines are disjoint, so a single sighting
is decisive and there are no teched/hybrid boards to confuse it. This script stresses
the sniffer along the two axes where it should degrade, and measures how far it degrades.

Axis 1 - Archetype robustness under teching (offline, precise).
    Feed the classifier hand-built "visible board" bags: minimal single-basic reveals,
    clean archetypes, cross-archetype tech splashes, and 50/50 hybrids. Report the
    prediction AND the margin (top signature-score minus runner-up). Margin 0 = a tie the
    argmax breaks arbitrarily = the real failure mode. Shows the current pool never ties
    only because nobody splashes a rival archetype's line.

Axis 2 - Exact-deck fingerprinting from public tells (in-game).
    Archetype != deck. Within an archetype, can we name WHICH of the sibling pool decks
    the opponent is on, from public zones alone? For each archetype with >1 pool member we
    derive each deck's "tell cards" = cards present in it but in no sibling (data-driven,
    presence-based). During real kojimar_lucario games we collect the opponent's public
    board+discard and pin the deck iff exactly one sibling's tells were seen. This exposes
    a sharp asymmetry: the Alakazam pair carries distinct tells and IS fingerprintable; the
    Lucario pair shares every card type (differs only in counts) and is NOT.

Usage:
    uv run python scripts/stress_deck_sniffer.py                 # both axes
    uv run python scripts/stress_deck_sniffer.py --games 30
    uv run python scripts/stress_deck_sniffer.py --skip-ingame   # axis 1 only (fast)
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

# Reuse the sniffer, signatures, engine bootstrap, and loaders from the main experiment.
from identify_opponent_deck import (  # noqa: E402
    CARD,
    OPPONENTS,
    SIGNATURES,
    DeckSniffer,
    battle_finish,
    battle_select,
    battle_start,
    load_agent_module,
    to_observation_class,
    opponent_visible_ids,
    wilson_or_sigma,
)


def name(cid: int) -> str:
    c = CARD.get(cid)
    return c.name if c is not None else f"#{cid}"


def margin(sniffer: DeckSniffer) -> tuple[str | None, int]:
    """Prediction and its margin = (top score - runner-up score). 0 => a tie."""
    s = sniffer.scores()
    if not s:
        return None, 0
    vals = sorted(s.values(), reverse=True)
    top = vals[0]
    runner = vals[1] if len(vals) > 1 else 0
    return sniffer.predict(), top - runner


# --------------------------------------------------------------------------- #
# Axis 1 - adversarial visible-board bags.
# --------------------------------------------------------------------------- #
def bag(*card_ids: int) -> list[int]:
    return list(card_ids)


AXIS1_CASES: list[tuple[str, list[int]]] = [
    ("minimal: lone Riolu (basic)",            bag(677)),
    ("minimal: lone Snover (basic)",           bag(722)),
    ("clean lucario board",                    bag(677, 678, 673, 676)),
    ("clean dragapult board",                  bag(119, 120, 121)),
    ("clean alakazam board",                   bag(741, 742, 743)),
    # A Lucario deck splashing a single Dragapult-line tech (Latias ex).
    ("lucario + 1 dragapult tech (Latias)",    bag(677, 678, 676, 184)),
    # A deck showing one Snover next to a full Lucario line (water tech splash).
    ("lucario + 1 water tech (Snover)",        bag(677, 678, 673, 722)),
    # Balanced hybrid: equal signature Pokemon from two archetypes -> genuine tie.
    ("50/50 hybrid lucario|dragapult",         bag(677, 678, 119, 121)),
    ("50/50 hybrid alakazam|abomasnow",        bag(741, 743, 722, 723)),
    # Only a shared splash tech visible (Fezandipiti ex 140 is in 3 decks) -> no signal.
    ("only shared tech visible (Fezandipiti)", bag(140)),
]


def run_axis1() -> None:
    print("=" * 78)
    print("AXIS 1 - archetype robustness under teching / hybrids (offline)")
    print("=" * 78)
    print(f"{'visible board':<42}{'prediction':<12}{'margin':<8}scores")
    print("-" * 78)
    for label, ids in AXIS1_CASES:
        sn = DeckSniffer()
        sn.observe(ids)
        pred, m = margin(sn)
        scores = sn.scores()
        score_str = ", ".join(f"{a}:{v}" for a, v in sorted(scores.items(), key=lambda kv: -kv[1]))
        flag = ""
        if pred is None:
            flag = "  (no signal)"
        elif m == 0:
            flag = "  <-- TIE (argmax arbitrary)"
        print(f"{label:<42}{str(pred):<12}{m:<8}{score_str}{flag}")
    print("\n  Reading: margin 0 = the classifier is guessing between tied archetypes. The"
          "\n  live pool never hits this because no pool deck splashes a rival's Pokemon line;"
          "\n  a genuinely teched opponent (rows above) collapses the margin and can flip the call.\n")


# --------------------------------------------------------------------------- #
# Axis 2 - exact-deck fingerprinting from public tells, in real games.
# --------------------------------------------------------------------------- #
def deck_list(agent_name: str) -> list[int]:
    p = REPO / "agents" / agent_name / "deck.csv"
    return [int(x) for x in p.read_text().splitlines() if x.strip()][:60]


def derive_tells() -> dict[str, dict[str, set[int]]]:
    """For each archetype with >1 pool member, tell[deck] = card ids present in that deck
    but in NO sibling (presence-based). Split into board-visible (Pokemon) vs discard-only."""
    from cg.api import CardType

    members: dict[str, list[str]] = defaultdict(list)
    for a, arch in OPPONENTS.items():
        members[arch].append(a)

    tells: dict[str, dict[str, set[int]]] = {}
    for arch, decks in members.items():
        if len(decks) < 2:
            continue
        presence = {d: set(deck_list(d)) for d in decks}
        for d in decks:
            others = set().union(*(presence[o] for o in decks if o != d))
            excl = presence[d] - others
            pk = {c for c in excl if CARD[c].cardType == CardType.POKEMON}
            tr = {c for c in excl if CARD[c].cardType != CardType.POKEMON}
            tells.setdefault(arch, {})[d] = excl
            tells[arch].setdefault("_pokemon", set()).update(pk)  # for reporting only
    return tells


def play_collect_public(my_agent, my_deck, opp_agent, opp_deck, my_seat, max_steps=2000):
    """Play one game; return the multiset of ALL opponent public card ids seen
    (board + discard, across every one of my decision points)."""
    deck0, deck1 = (my_deck, opp_deck) if my_seat == 0 else (opp_deck, my_deck)
    seen: Counter = Counter()
    obs, start = battle_start(deck0, deck1)
    if obs is None:
        return None, seen, f"start_fail:{start.errorPlayer}:{start.errorType}"
    try:
        for _ in range(max_steps):
            obc = to_observation_class(obs)
            if obc.current.result >= 0:
                return obc.current.result, seen, ""
            yi = obc.current.yourIndex
            if yi == my_seat:
                for cid in opponent_visible_ids(obc.current, 1 - my_seat):
                    seen[cid] += 1
                act = my_agent(obs)
            else:
                act = opp_agent(obs)
            obs = battle_select(act)
        return None, seen, "max_steps"
    except Exception as exc:
        return None, seen, f"{type(exc).__name__}:{exc}"
    finally:
        battle_finish()


def run_axis2(games: int) -> None:
    print("=" * 78)
    print("AXIS 2 - exact-deck fingerprinting from public tells (in-game)")
    print("=" * 78)

    tells = derive_tells()
    members: dict[str, list[str]] = defaultdict(list)
    for a, arch in OPPONENTS.items():
        members[arch].append(a)
    multi = {arch: decks for arch, decks in members.items() if len(decks) > 1}

    # Report the derived tells so the asymmetry is explicit.
    from cg.api import CardType
    for arch, decks in multi.items():
        print(f"\n  {arch} pair -- derived public tell cards (present in one deck, no sibling):")
        for d in decks:
            excl = tells[arch][d]
            pk = sorted(c for c in excl if CARD[c].cardType == CardType.POKEMON)
            tr = sorted(c for c in excl if CARD[c].cardType != CardType.POKEMON)
            print(f"     {d:<28} board-visible: {[name(c) for c in pk] or 'none'}")
            print(f"     {'':<28} discard-only:  {[name(c) for c in tr] or 'none'}")

    me = load_agent_module("kojimar_lucario")
    my_deck = list(me.my_deck)

    print(f"\n  Piloting kojimar_lucario, {games} games vs each sibling deck. A game is PINNED to"
          "\n  a deck iff exactly one sibling's tells appeared in the opponent's public zones.\n")
    header = f"{'opponent':<28}{'archetype':<11}{'pinned-correct':<16}{'unresolved':<13}{'mispinned':<11}errors"
    print(header)
    print("-" * len(header))

    for arch, decks in multi.items():
        for target in decks:
            opp = load_agent_module(target)
            opp_deck = deck_list(target)
            correct = unresolved = mispin = errors = valid = 0
            for g in range(games):
                my_seat = g % 2
                result, seen, err = play_collect_public(me.agent, my_deck, opp.agent, opp_deck, my_seat)
                if err:
                    errors += 1
                    continue
                valid += 1
                # which siblings' tells were seen?
                hit = [d for d in decks if tells[arch][d] & set(seen)]
                if len(hit) == 1 and hit[0] == target:
                    correct += 1
                elif len(hit) == 0:
                    unresolved += 1
                else:
                    mispin += 1  # saw the wrong sibling's tell, or both
            rate, sigma = wilson_or_sigma(correct, valid)
            print(f"{target:<28}{arch:<11}{f'{rate:.2f}±{sigma:.2f}':<16}"
                  f"{f'{unresolved}/{valid}':<13}{f'{mispin}/{valid}':<11}{errors}")

    print("\n  Reading: 'pinned-correct' = fraction of games we named the exact deck from public"
          "\n  tells. High for Alakazam (distinct tell cards surface); near-zero + all-unresolved"
          "\n  for Lucario (siblings share every card type, so no tell can ever fire).\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=20, help="in-game samples per sibling deck (axis 2)")
    ap.add_argument("--skip-ingame", action="store_true", help="run axis 1 only (no games)")
    args = ap.parse_args()

    run_axis1()
    if not args.skip_ingame:
        run_axis2(args.games)


if __name__ == "__main__":
    main()
