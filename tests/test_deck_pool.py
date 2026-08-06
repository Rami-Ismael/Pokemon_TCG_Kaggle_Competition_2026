"""Deck-pool tests: pool resolution, nesting, and the BOTH-SEATS invariant.

The load-bearing assertion is that BOTH seats' decks come from the pool. The
learner's deck is submitted by the env, but the opponent answers the engine's
deck request itself — a checkpoint opponent with `selfplay.load_deck()`, a
module opponent with its bundled `deck.csv`. If the opponent-side override
silently breaks, the learner varies its deck while every opponent stays on Mega
Lucario ex forever, which is exactly what happened for three self-play
generations and produces plausible numbers rather than an error.

Two modes are pinned:
  independent (default) — two draws per episode, so cross-deck matchups occur
    AND the opponent still never plays its own bundled list.
  mirror (`--mirror-deck`) — one draw, both seats: the same-deck control.

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
    deck_override_agent,
)

# Exploiter-mode env kwargs (frozen module opponent, strict seat alternation).
# A MODULE opponent is the hardest case for the override: it ships its own
# deck.csv, so if the wrapper is missing the mismatch is guaranteed.
EXPLOITER = {"mirror_root": None, "league": [("module", "il_agent")],
             "mix": (0.0, 1.0), "alternate_seats": True}
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
    # agents/ holds many deck.csv files but far fewer distinct lists;
    # duplicates must not inflate K. Counts are derived, not pinned, for the
    # same reason: dedup now keys on the card multiset, so they move.
    agents = DeckPool.from_spec("all:agents")
    n_files = len(list((REPO / "agents").glob("*/deck.csv")))
    assert len(agents) < n_files, "content dedup did nothing"
    assert len(agents.names) == len(set(agents.names))
    # The union is deduped ACROSS sources too, so it cannot exceed the sum.
    union = DeckPool.from_spec("all:decks")
    assert len(agents) <= len(union) <= len(csvs) + len(agents)
    assert len(union.names) == len(set(union.names))
    assert len(DeckPool.from_spec("mega_lucario_ex")) == 1  # deck_lists stem
    assert len(DeckPool.from_spec("il_agent")) == 1         # agent name


def test_subsets_are_nested_and_pin_by_content():
    k_max = len(DeckPool.from_spec("all:agents"))
    for seed in (0, 1):
        pools = [set(DeckPool.from_spec("all:agents", limit=k, seed=seed,
                                        pin=["il_agent"]).names)
                 for k in (1, 4, 16, k_max)]
        for small, big in zip(pools, pools[1:]):
            assert small < big, f"pools not nested at seed {seed}: {small} !< {big}"
        # Pinning is by CONTENT: il_agent's list is a duplicate of another
        # agent's, so a name-based pin would raise "not in pool" here.
        assert pools[0] == {"il_agent"}
    a = DeckPool.from_spec("all:agents", limit=4, seed=0, pin=["il_agent"]).names
    b = DeckPool.from_spec("all:agents", limit=4, seed=1, pin=["il_agent"]).names
    assert a != b, "nesting seed must change WHICH decks land in the subset"


def test_k1_pinned_pool_is_the_single_deck_baseline():
    from pokemon_tcg.selfplay import load_deck
    pool = DeckPool.from_spec("all:agents", limit=1, seed=0, pin=["il_agent"])
    assert pool.entries[0][1] == load_deck(), (
        "K=1 must reproduce the historical single-deck arm, or the control "
        "arm is not a control")


def test_override_wrapper_substitutes_only_the_deck_step():
    sentinel = [1, 2, 3]
    inner = deck_override_agent(lambda obs: ["INNER"], lambda: sentinel)
    assert inner({"select": None}) == sentinel        # deck request -> our deck
    assert inner({"select": {"option": [0]}}) == ["INNER"]  # play -> untouched
    assert inner.__code__.co_argcount == 1, (
        "kaggle_environments dispatches on co_argcount; a 2-arg callable "
        "gets (obs, config) and crashes the seat")


def test_mirror_mode_keeps_both_seats_on_the_same_deck():
    pool = DeckPool.from_spec(POOL_SPEC)
    assert pool.entries[0][1] != pool.entries[1][1], "test decks must differ"
    env = _gym(deck_pool=pool, mirror_deck=True)
    seen = set()
    for _ in range(8):
        env.reset()
        seen.add(env.deck_name)
        assert list(env.opponent({"select": None})) == list(env.deck), (
            f"MIRROR BROKEN on {env.deck_name}: opponent submitted a "
            f"different deck than the learner")
    assert len(seen) == 2, f"deck never varied across episodes: {seen}"


def test_independent_mode_varies_both_seats_from_the_pool():
    """The core change: the OPPONENT's deck must move too, and must always be
    a pool member — never its own bundled list."""
    pool = DeckPool.from_spec(POOL_SPEC)
    members = [d for _, d in pool.entries]
    env = _gym(deck_pool=pool, mirror_deck=False)
    learner_seen, opp_seen, cross = set(), set(), 0
    for _ in range(40):
        env.reset()
        submitted = list(env.opponent({"select": None}))
        assert submitted in members, (
            "opponent submitted a deck that is NOT in the pool -- the "
            "override is missing and it is playing its own bundled list")
        assert submitted == list(env.opp_deck)
        learner_seen.add(env.deck_name)
        opp_seen.add(env.opp_deck_name)
        cross += env.opp_deck_name != env.deck_name
    assert len(learner_seen) == 2, f"learner deck never varied: {learner_seen}"
    assert len(opp_seen) == 2, (
        f"OPPONENT deck never varied: {opp_seen} -- this is the single-deck "
        f"bug that survived three self-play generations")
    assert cross > 0, "independent draws produced no cross-deck episode in 40"


def test_opponent_hold_still_tracks_the_new_deck():
    """opp_hold keeps the drawn opponent across episodes; the deck override
    must still follow each episode's fresh draw."""
    pool = DeckPool.from_spec(POOL_SPEC)
    env = _gym(deck_pool=pool, mirror_deck=False, opp_hold=8)
    for _ in range(12):
        env.reset()
        assert list(env.opponent({"select": None})) == list(env.opp_deck), (
            "held opponent is serving a stale deck")


def test_no_pool_is_unchanged_behaviour():
    from pokemon_tcg.selfplay import load_deck
    env = _gym()
    env.reset()
    assert env.deck == load_deck()
    assert env.deck_pool is None and env.mirror_deck is False
    # No pool => no override wrapper: the module opponent keeps its own deck.
    assert env.opponent({"select": None}) != []


def test_lineage_only_draw_has_no_third_bucket():
    """TASK 1 regression guard: exactly two buckets, both lineage.

    The old draw fell THROUGH to a hardcoded trio of other competitors'
    agents whenever the first two buckets missed. That bucket is deleted, not
    zeroed — a zeroed bucket reachable by fall-through is how externals would
    quietly re-enter training and stop the evaluation pool being out-of-sample.
    """
    from pokemon_tcg.puffer_env import PTCGGym
    league = [("ckpt", str(REPO / "models" / "il_agent"))]

    # Every draw at any mix must land in one of the two lineage buckets.
    for mix in [(1.0, 0.0), (0.0, 1.0), (0.5, 0.5)]:
        env = PTCGGym(league=league, mirror_root=None, mix=mix, seed=0)
        slugs = {env._draw_opponent()[1] for _ in range(200)}
        assert slugs <= {"mirror", "league_il_agent"}, (
            f"mix={mix} drew something outside the lineage: {slugs}")
        if mix == (1.0, 0.0):
            assert slugs == {"mirror"}
        if mix == (0.0, 1.0):
            assert slugs == {"league_il_agent"}

    env = PTCGGym(league=league, mirror_root=None, mix=(0.5, 0.5), seed=0)
    assert len(env.mix) == 2, "--mix is two shares now (mirror, league)"
    for gone in ("pool_names", "pool_weights", "pool_weights_path",
                 "_maybe_reload_weights"):
        assert not hasattr(env, gone), f"public-pool leftover: {gone}"
    import pokemon_tcg.selfplay as sp
    assert not hasattr(sp, "sample_league"), "selfplay.sample_league not removed"


if __name__ == "__main__":
    test_resolution_and_dedup()
    print("resolution + content dedup + union spec: OK")
    test_subsets_are_nested_and_pin_by_content()
    print("nested subsets + content-matched pin: OK")
    test_k1_pinned_pool_is_the_single_deck_baseline()
    print("K=1 == single-deck baseline: OK")
    test_override_wrapper_substitutes_only_the_deck_step()
    print("override wrapper scope + co_argcount: OK")
    test_mirror_mode_keeps_both_seats_on_the_same_deck()
    print("mirror mode (one draw, both seats): OK")
    test_independent_mode_varies_both_seats_from_the_pool()
    print("independent mode (opponent deck varies, always in-pool): OK")
    test_opponent_hold_still_tracks_the_new_deck()
    print("held opponent tracks each episode's draw: OK")
    test_no_pool_is_unchanged_behaviour()
    print("no pool == pre-pool behaviour: OK")
    test_lineage_only_draw_has_no_third_bucket()
    print("lineage-only draw, no third bucket: OK")
