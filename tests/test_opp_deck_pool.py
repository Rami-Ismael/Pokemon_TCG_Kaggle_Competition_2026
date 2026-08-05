"""Regression tests for opponent-only deck sampling (--opp-deck-pool).

The deck-diversity GENERALIZATION experiment
(notes/experiments/2026-08-05-opponent-deck-diversity-selfplay.md) needs the
inverse of the exploitability sweep's mirror: the learner's deck stays fixed
while checkpoint opponents are re-decked per episode. Three invariants are
pinned here:

1. The learner's deck never moves — otherwise the experiment silently bundles
   "pilot diverse decks" into an arm that is supposed to vary only the
   opponent (one variable per experiment).
2. A ckpt opponent's deck-step submission is drawn from the pool and actually
   varies across episodes — otherwise arm B degenerates into arm A and the
   comparison measures nothing.
3. A MODULE opponent keeps its own bundled deck — rule-based publics are
   written around their cards; re-decking them would replace "diverse
   opponents" with "confused opponents".

Requires no pytest: `uv run python tests/test_opp_deck_pool.py` works too.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

os.environ.setdefault("PTCG_DEVICE", "cpu")

from pokemon_tcg import config  # noqa: E402
from pokemon_tcg.deck_pool import DeckPool  # noqa: E402

POOL_SPEC = "wmh_grimmsnarl,kiyotah_abomasnow"
CKPT_LEAGUE = {"mirror_root": None,
               "league": [("ckpt", str(config.MODELS_DIR / "il_agent"))],
               "mix": (0.0, 1.0, 0.0), "alternate_seats": True}
MODULE_LEAGUE = {"mirror_root": None,
                 "league": [("module", "kiyotah_dragapult")],
                 "mix": (0.0, 1.0, 0.0), "alternate_seats": True}


def _gym(base, **kw):
    from pokemon_tcg.puffer_env import PTCGGym
    return PTCGGym(**base, seed=0, **kw)


def test_learner_fixed_opponent_sampled_from_pool():
    from pokemon_tcg.selfplay import load_deck
    pool = DeckPool.from_spec(POOL_SPEC)
    pool_lists = [sorted(d) for _, d in pool.entries]
    env = _gym(CKPT_LEAGUE, opp_deck_pool=pool)
    fixed = load_deck()
    seen = set()
    for _ in range(8):
        env.reset()
        assert env.deck == fixed, "learner deck moved under opp_deck_pool"
        opp_deck = sorted(env.opponent({"select": None}))
        assert opp_deck in pool_lists, "opponent deck not from the pool"
        assert env.opp_deck_name in pool.names
        seen.add(env.opp_deck_name)
    assert len(seen) > 1, (
        "opponent deck never varied across 8 episodes (K=2; p(all-same) "
        "= 2^-7 under correct sampling)")


def test_module_opponent_keeps_bundled_deck():
    # Identity check, not a fake-obs probe: real module agents (dragapult)
    # parse the full Observation dataclass even at the deck step, so the
    # invariant pinned is "no re-deck wrapper was applied" — the env serves
    # the cached module fn unchanged, which in production supplies its own
    # bundled deck (established by test_mirror_off_produces_cross_deck).
    pool = DeckPool.from_spec(POOL_SPEC)  # excludes kiyotah_dragapult
    env = _gym(MODULE_LEAGUE, opp_deck_pool=pool)
    for _ in range(3):
        env.reset()
        assert env.opp_deck_name is None
        assert env.opponent is env._opp_cache[("module", "kiyotah_dragapult")], (
            "module opponent was wrapped/re-decked under opp_deck_pool")


def test_opp_pool_mutually_exclusive_with_learner_pool():
    pool = DeckPool.from_spec(POOL_SPEC)
    for kw in ({"deck_pool": pool}, {"mirror_deck": True}):
        try:
            _gym(CKPT_LEAGUE, opp_deck_pool=pool, **kw)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError with {list(kw)}")


if __name__ == "__main__":
    for fn in (test_learner_fixed_opponent_sampled_from_pool,
               test_module_opponent_keeps_bundled_deck,
               test_opp_pool_mutually_exclusive_with_learner_pool):
        fn()
        print(f"ok {fn.__name__}")
