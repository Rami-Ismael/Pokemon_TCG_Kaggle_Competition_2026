"""Can you tell which deck the opponent is playing, just from a match?

This experiment pilots the ported kojimar Mega Lucario ex agent
(agents/kojimar_lucario) against the public opponent pool and, on every one of the
agent's own decision points, runs a *deck sniffer* over ONLY the opponent's public
zones (active + bench + discard + stadium). It never touches the opponent's hand,
deck, or prizes -- those are masked by the engine (hand is None, deck/prize are
counts only), and using them would violate the repo's no-opponent-private-info rule.

The kojimar policy already does two hard-coded versions of this
(`_opponent_is_water_deck`, `_opponent_is_crustle_wall`); this generalizes it into a
full archetype classifier and *measures* how recoverable the opponent deck actually is.

Two questions, two honest answers this script quantifies:

  1. Archetype ("which deck") -- recoverable, and usually within the first turn or two,
     because each archetype's Pokemon lines are disjoint across the pool.
  2. Exact 60-card list -- NOT recoverable within an archetype. Four pool decks share the
     Lucario Pokemon lines (673-678) and two share the Alakazam line (741-743), so their
     visible boards are identical; the hidden zones hold the trainers that differ.

Usage:
    uv run python scripts/identify_opponent_deck.py                # all opponents, 20 games each
    uv run python scripts/identify_opponent_deck.py --games 40
    uv run python scripts/identify_opponent_deck.py --opponents kiyotah_dragapult,kiyotah_iono
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Make `cg` importable (packaged engine). Mirrors scripts/benchmark_agents.py.
for _cand in (REPO / "data" / "external" / "cg-lib", REPO / "agents" / "mega_lucario"):
    if (_cand / "cg" / "api.py").exists():
        sys.path.insert(0, str(_cand))
        break

from cg.api import CardType, all_card_data, to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

CARD = {c.cardId: c for c in all_card_data()}


# --------------------------------------------------------------------------- #
# Archetype signatures.
#
# Each archetype is keyed on the Pokemon it is FORCED to reveal on board (basics and
# their evolutions), restricted to IDs that are exclusive to that archetype within the
# pool -- so a single sighting is decisive. Splash techs shared across archetypes
# (e.g. Fezandipiti ex 140, in both Dragapult and Alakazam) are deliberately excluded;
# keying on them would confuse the classifier. Signatures were derived from the pool's
# own deck.csv Pokemon lists, not guessed.
# --------------------------------------------------------------------------- #
SIGNATURES: dict[str, set[int]] = {
    "lucario":   {673, 674, 675, 676, 677, 678},              # Makuhita..Mega Lucario ex
    "dragapult": {119, 120, 121, 184, 235, 1071},             # Dreepy..Dragapult ex, Latias, Budew, Meowth
    "iono":      {265, 268, 269, 270, 271},                   # Iono's Voltorb/Tadbulb/Bellibolt/Wattrel/Kilowattrel
    "abomasnow": {721, 722, 723},                             # Kyogre, Snover, Mega Abomasnow ex  (water)
    "alakazam":  {65, 66, 142, 305, 343, 741, 742, 743, 858}, # Dunsparce lines, Genesect, Shaymin, Abra..Alakazam, Psyduck
    "crustle":   {344, 345},                                  # Dwebble, Crustle (not in current pool; kojimar keys on it)
}

# Ground-truth archetype for each pool opponent, and its deck.csv path.
OPPONENTS: dict[str, str] = {
    "kiyotah_dragapult": "dragapult",
    "kiyotah_iono": "iono",
    "kiyotah_abomasnow": "abomasnow",
    "dedquoc_rule_engine": "lucario",
    "ryotasueyoshi_alakazam": "alakazam",
    "makthanithin_improved_prob": "lucario",
    "mechi22_alakazam": "alakazam",
}


class DeckSniffer:
    """Accumulates the opponent's publicly-visible Pokemon across a match and
    classifies the archetype. Argmax over (distinct signature Pokemon seen)."""

    def __init__(self) -> None:
        self.seen: set[int] = set()

    def observe(self, visible_card_ids: list[int]) -> None:
        for cid in visible_card_ids:
            if cid in _SIG_LOOKUP:
                self.seen.add(cid)

    def scores(self) -> dict[str, int]:
        s: dict[str, int] = defaultdict(int)
        for cid in self.seen:
            s[_SIG_LOOKUP[cid]] += 1
        return dict(s)

    def predict(self) -> str | None:
        s = self.scores()
        if not s:
            return None
        # argmax; ties broken deterministically by archetype name for reproducibility.
        best = max(s.values())
        return sorted(a for a, v in s.items() if v == best)[0]


# card id -> archetype (signatures are disjoint by construction, so this is 1:1).
_SIG_LOOKUP: dict[int, str] = {}
for _arch, _ids in SIGNATURES.items():
    for _cid in _ids:
        assert _cid not in _SIG_LOOKUP, f"signature collision on {_cid}"
        _SIG_LOOKUP[_cid] = _arch


def load_agent_module(name: str):
    path = REPO / "agents" / name / "agent_core.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(f"_agent_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    deck_csv = path.parent / "deck.csv"
    if not hasattr(mod, "my_deck") and deck_csv.exists():
        mod.my_deck = [int(x) for x in deck_csv.read_text().splitlines() if x.strip()][:60]
    return mod


def opponent_visible_ids(state, opp_index: int) -> list[int]:
    """Only public zones: opponent active + bench + discard, plus the stadium.
    Hand is None (masked); deck/prizes are counts only -- never touched."""
    opp = state.players[opp_index]
    ids: list[int] = []
    for zone in (opp.active, opp.bench, opp.discard):
        if zone:
            ids += [getattr(c, "id", None) for c in zone if c is not None]
    if state.stadium:
        ids += [c.id for c in state.stadium]
    return [i for i in ids if i is not None]


def play_and_sniff(my_agent, my_deck, opp_agent, opp_deck, my_seat: int, max_steps: int = 2000):
    """Play one game. On each of MY decision points, snapshot the opponent's public
    board and update the sniffer. Returns (result, prediction_timeline, n_my_turns).

    prediction_timeline: list of (turn, prediction) at each of my decision points."""
    deck0, deck1 = (my_deck, opp_deck) if my_seat == 0 else (opp_deck, my_deck)
    sniffer = DeckSniffer()
    timeline: list[tuple[int, str | None]] = []
    obs, start = battle_start(deck0, deck1)
    if obs is None:
        return None, timeline, 0, f"start_fail:{start.errorPlayer}:{start.errorType}"
    try:
        for _ in range(max_steps):
            obc = to_observation_class(obs)
            if obc.current.result >= 0:
                return obc.current.result, timeline, len(timeline), ""
            yi = obc.current.yourIndex
            if yi == my_seat:
                sniffer.observe(opponent_visible_ids(obc.current, 1 - my_seat))
                timeline.append((obc.current.turn, sniffer.predict()))
                act = my_agent(obs)
            else:
                act = opp_agent(obs)
            obs = battle_select(act)
        return None, timeline, len(timeline), "max_steps"
    except Exception as exc:  # keep the harness alive; report as an errored game
        return None, timeline, len(timeline), f"{type(exc).__name__}:{exc}"
    finally:
        battle_finish()


def wilson_or_sigma(k: int, n: int) -> tuple[float, float]:
    """Return (rate, +/- normal sigma). Sigma is sqrt(p(1-p)/n)."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    return p, math.sqrt(p * (1 - p) / n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=20, help="games per opponent (split evenly across both seats)")
    ap.add_argument("--opponents", default=",".join(OPPONENTS),
                    help="comma-separated subset of: " + ", ".join(OPPONENTS))
    ap.add_argument("--max-steps", type=int, default=2000)
    args = ap.parse_args()

    opponents = [o.strip() for o in args.opponents.split(",") if o.strip()]
    unknown = [o for o in opponents if o not in OPPONENTS]
    if unknown:
        sys.exit(f"unknown opponent(s): {unknown}. Available: {list(OPPONENTS)}")

    me = load_agent_module("kojimar_lucario")
    my_deck = list(me.my_deck)

    # How many pool decks share each archetype -> the exact-deck ambiguity.
    arch_members: dict[str, list[str]] = defaultdict(list)
    for name, arch in OPPONENTS.items():
        arch_members[arch].append(name)

    print(f"Piloting kojimar_lucario (60 cards, {len(set(my_deck))} unique) vs {len(opponents)} opponents, "
          f"{args.games} games each.\n"
          f"Sniffer uses ONLY opponent public zones (active+bench+discard+stadium).\n")

    confusion: dict[str, Counter] = defaultdict(Counter)
    header = (f"{'opponent':<28}{'true':<11}{'ID rate':<16}{'median turn':<13}"
              f"{'ever-wrong':<12}{'exact-deck?':<12}{'my winrate':<12}errors")
    print(header)
    print("-" * len(header))

    for name in opponents:
        truth = OPPONENTS[name]
        opp = load_agent_module(name)
        opp_deck = [int(x) for x in (REPO / "agents" / name / "deck.csv").read_text().splitlines() if x.strip()][:60]

        n_id = 0            # games whose FINAL prediction == truth
        n_valid = 0         # games that completed without error
        turns_to_id: list[int] = []
        ever_wrong = 0      # games with any non-None wrong prediction before settling
        my_wins = 0
        errors = 0

        for g in range(args.games):
            my_seat = g % 2
            result, timeline, _, err = play_and_sniff(
                me.agent, my_deck, opp.agent, opp_deck, my_seat, args.max_steps)
            if err:
                errors += 1
                continue
            n_valid += 1
            if result == my_seat:
                my_wins += 1

            preds = [p for _, p in timeline]
            final = preds[-1] if preds else None
            confusion[truth][final if final is not None else "unknown"] += 1
            if final == truth:
                n_id += 1
            # first turn at which prediction becomes truth and never changes afterward
            settle_turn = None
            for (turn, p), tail in zip(timeline, range(len(timeline))):
                if p == truth and all(pp == truth for _, pp in timeline[tail:]):
                    settle_turn = turn
                    break
            if settle_turn is not None:
                turns_to_id.append(settle_turn)
            if any(p is not None and p != truth for p in preds):
                ever_wrong += 1

        rate, sigma = wilson_or_sigma(n_id, n_valid)
        med_turn = sorted(turns_to_id)[len(turns_to_id) // 2] if turns_to_id else float("nan")
        wr, wsig = wilson_or_sigma(my_wins, n_valid)
        n_share = len(arch_members[truth])
        exact = "unique" if n_share == 1 else f"1 of {n_share}"
        print(f"{name:<28}{truth:<11}{f'{rate:.2f}±{sigma:.2f}':<16}{str(med_turn):<13}"
              f"{f'{ever_wrong}/{n_valid}':<12}{exact:<12}{f'{wr:.2f}±{wsig:.2f}':<12}{errors}")

    # Confusion matrix (true archetype -> predicted archetype counts).
    print("\nConfusion matrix (true \\ predicted):")
    preds_seen = sorted({p for row in confusion.values() for p in row})
    print(f"{'':<12}" + "".join(f"{p:<12}" for p in preds_seen))
    for truth in sorted(confusion):
        row = confusion[truth]
        print(f"{truth:<12}" + "".join(f"{row.get(p, 0):<12}" for p in preds_seen))

    print("\nInterpretation:")
    print("  * 'ID rate' = fraction of games whose final sniffer prediction matches the")
    print("    opponent's true archetype. 'median turn' = when that call first becomes")
    print("    stable-correct. 'exact-deck?' = whether the archetype pins a single pool")
    print("    deck ('unique') or is shared by several identical-looking lists.")
    print("  * Archetype is recoverable; the exact 60-card list is not, wherever")
    print("    'exact-deck?' says '1 of N' -- those decks share their visible Pokemon and")
    print("    differ only in hidden-zone trainers.")


if __name__ == "__main__":
    main()
