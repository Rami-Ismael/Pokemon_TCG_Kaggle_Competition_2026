"""Deck-pool tests: pool resolution, nesting, and the MIRROR invariant.

The load-bearing assertion is the mirror: with a deck pool active, the
opponent must submit the SAME deck the learner does, every episode. If that
silently breaks, a K>1 pool becomes a cross-deck matchup and the exploitability
sweep measures deck advantage instead of policy exploitability — a failure that
produces plausible numbers rather than an error, which is exactly the kind that
survives to a conclusion. `test_mirror_off_produces_cross_deck` pins the
negative side: without the flag the confound is real, not hypothetical.

Also pinned: subsets are NESTED (P_4 subset-of P_16 for a fixed seed) so K is
the only variable moving between arms; pinning matches by deck CONTENT, since
'all:agents' is content-deduped and drops il_agent's name in favour of an
alphabetically earlier duplicate; and with no pool the env reproduces the
historical single-deck behaviour bit-for-bit.

Requires no pytest: `uv run python tests/test_deck_pool.py` works too.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

os.environ.setdefault("PTCG_DEVICE", "cpu")

from pokemon_tcg.deck_pool import (  # noqa: E402
    DECK_LISTS_DIR,
    DeckPool,
    mirror_deck_agent,
)

# Exploiter-mode env kwargs (frozen module opponent, strict seat alternation).
EXPLOITER = {"mirror_root": None, "league": [("module", "il_agent")],
             "mix": (0.0, 1.0, 0.0), "alternate_seats": True}
POOL_SPEC = "il_agent,wmh_grimmsnarl"


def _gym(**kw):
    from pokemon_tcg.puffer_env import PTCGGym
    return PTCGGym(**EXPLOITER, seed=0, **kw)


def test_resolution_and_dedup():
    # "all:decklists" IS the csv inventory of configs/deck_lists, so assert
    # that identity rather than pinning a count. The literal 7 that used to
    # be here went stale the moment 99541b7 added grimmsnarl_toplayer.csv,
    # and failed on main. A count rots on every new deck; the set does not.
    csvs = sorted(p.stem for p in DECK_LISTS_DIR.glob("*.csv"))
    assert csvs, f"no decklists in {DECK_LISTS_DIR} -- tracked csvs missing"
    decklists = DeckPool.from_spec("all:decklists")
    assert sorted(decklists.names) == csvs
    assert "mega_lucario_ex" in csvs  # the deck our submissions pilot
    # 47 agent deck.csv files, 33 distinct lists: duplicates must not inflate K.
    agents = DeckPool.from_spec("all:agents")
    assert len(agents) == 33, f"expected 33 distinct agent decks, got {len(agents)}"
    assert len(agents.names) == len(set(agents.names))
    assert len(DeckPool.from_spec("mega_lucario_ex")) == 1  # deck_lists stem
    assert len(DeckPool.from_spec("il_agent")) == 1         # agent name


def test_subsets_are_nested_and_pin_by_content():
    for seed in (0, 1):
        pools = [set(DeckPool.from_spec("all:agents", limit=k, seed=seed,
                                        pin=["il_agent"]).names)
                 for k in (1, 4, 16, 33)]
        for small, big in zip(pools, pools[1:]):
            assert small < big, f"pools not nested at seed {seed}: {small} !< {big}"
        # Pinning is by CONTENT: il_agent's list is a duplicate of another
        # agent's, so a name-based pin would raise "not in pool" here.
        assert pools[0] == {"il_agent"}
    a = DeckPool.from_spec("all:agents", limit=4, seed=0, pin=["il_agent"]).names
    b = DeckPool.from_spec("all:agents", limit=4, seed=1, pin=["il_agent"]).names
    assert a != b, "nesting seed must change WHICH decks land in the subset"


def test_k1_pinned_pool_is_the_v1_baseline_deck():
    from pokemon_tcg.selfplay import load_deck
    pool = DeckPool.from_spec("all:agents", limit=1, seed=0, pin=["il_agent"])
    assert pool.entries[0][1] == load_deck(), (
        "K=1 must reproduce the v1 exploiter's deck, or the sweep has no anchor")


def test_mirror_wrapper_substitutes_only_the_deck_step():
    sentinel = [1, 2, 3]
    inner = mirror_deck_agent(lambda obs: ["INNER"], lambda: sentinel)
    assert inner({"select": None}) == sentinel        # deck request -> pooled deck
    assert inner({"select": {"option": [0]}}) == ["INNER"]  # play -> untouched
    assert inner.__code__.co_argcount == 1, (
        "kaggle_environments dispatches on co_argcount; a 2-arg callable "
        "gets (obs, config) and crashes the seat")


def test_mirror_on_keeps_both_seats_on_the_same_deck():
    pool = DeckPool.from_spec(POOL_SPEC)
    assert pool.entries[0][1] != pool.entries[1][1], "test decks must differ"
    env = _gym(deck_pool=pool, mirror_deck=True)
    seen = set()
    for _ in range(6):
        env.reset()
        seen.add(env.deck_name)
        assert list(env.opponent({"select": None})) == list(env.deck), (
            f"MIRROR BROKEN on {env.deck_name}: opponent submitted a "
            f"different deck than the learner")
    assert len(seen) == 2, f"deck never varied across episodes: {seen}"


def test_mirror_off_produces_cross_deck():
    """The confound the flag exists to prevent — pinned so it stays visible."""
    env = _gym(deck_pool=DeckPool.from_spec(POOL_SPEC), mirror_deck=False)
    mismatch = 0
    for _ in range(6):
        env.reset()
        if list(env.opponent({"select": None})) != list(env.deck):
            mismatch += 1
    assert mismatch > 0, (
        "expected cross-deck matchups without mirror_deck; if this now passes "
        "trivially the opponent stopped supplying its own deck")


def test_no_pool_is_unchanged_behaviour():
    from pokemon_tcg.selfplay import load_deck
    env = _gym()
    env.reset()
    assert env.deck == load_deck()
    assert env.deck_pool is None and env.mirror_deck is False


if __name__ == "__main__":
    test_resolution_and_dedup()
    print("resolution + content dedup (33 agents / 7 decklists): OK")
    test_subsets_are_nested_and_pin_by_content()
    print("nested subsets + content-matched pin: OK")
    test_k1_pinned_pool_is_the_v1_baseline_deck()
    print("K=1 == v1 baseline deck: OK")
    test_mirror_wrapper_substitutes_only_the_deck_step()
    print("mirror wrapper scope + co_argcount: OK")
    test_mirror_on_keeps_both_seats_on_the_same_deck()
    print("mirror ON invariant (both seats, deck varies): OK")
    test_mirror_off_produces_cross_deck()
    print("mirror OFF confound still reachable: OK")
    test_no_pool_is_unchanged_behaviour()
    print("no pool == pre-sweep behaviour: OK")
