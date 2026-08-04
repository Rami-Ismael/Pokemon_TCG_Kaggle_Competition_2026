"""Regression tests for standing rule #2: no opponent-private info in the encoder.

The agent may only see what a real player sees: the board (both sides'
in-play Pokemon), its OWN hand, and public counts (deckCount, handCount,
prize count). Opponent hand / either deck's contents / facedown prizes must
be null in the per-agent observation. The full hidden state DOES exist in
every episode file -- steps[0][0]["visualize"] carries both players' decks,
hands, and prizes per frame -- and must never be read by the pipeline.

These tests pin the two halves of that guarantee:
  1. the recorded per-agent observations are POV-filtered (an engine or
     serialization change that starts leaking fails loudly here), and
  2. the encoder's output is bit-identical with the visualize blob deleted,
     so no code path can be depending on it.

Audit that established these invariants (2026-08-03): 550 episodes /
~192k observations across train + eval, zero violations. Re-audited
2026-08-03 (300 episodes / 107,324 observations, both splits): also zero
violations, plus two invariants now pinned here that the first pass only
checked ad hoc -- select.deck ownership and cross-POV `looking` isolation.

Run directly: uv run python tests/test_privacy_no_leak.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "data" / "external" / "cg-lib"))

import torch  # noqa: E402

from cg.api import AreaType  # noqa: E402
from pokemon_tcg.il_dataset import (  # noqa: E402
    encode_observation,
    iter_decisions,
    resolve_split_dir,
)

# Enough episodes to hit multi-select, deck-search, and looking states
# without making the suite slow (~10k observations, a few seconds). Both
# splits are sampled: the eval day could regress independently if the
# engine version changed between recording days.
N_TRAIN_EPISODES = 20
N_EVAL_EPISODES = 10

OPP_PRIVATE_AREAS = {int(AreaType.DECK), int(AreaType.HAND), int(AreaType.PRIZE)}


def _episode_files():
    train = sorted(resolve_split_dir("train").glob("*.json"))[:N_TRAIN_EPISODES]
    assert train, "no train episodes found -- expected local data on disk"
    eval_ = sorted(resolve_split_dir("eval").glob("*.json"))[:N_EVAL_EPISODES]
    assert eval_, "no eval episodes found -- expected local data on disk"
    return train + eval_


def _iter_pov_observations():
    for path in _episode_files():
        episode = json.loads(path.read_text())
        steps = episode.get("steps") if isinstance(episode, dict) else episode
        for step in steps or []:
            for agent_step in step:
                obs = agent_step.get("observation") or {}
                cur = obs.get("current")
                if cur:
                    yield obs, cur


def _iter_pov_pairs():
    """Yield (current_a, current_b) for steps where BOTH agents have a state."""
    for path in _episode_files():
        episode = json.loads(path.read_text())
        steps = episode.get("steps") if isinstance(episode, dict) else episode
        for step in steps or []:
            curs = [
                (agent_step.get("observation") or {}).get("current")
                for agent_step in step
            ]
            if len(curs) == 2 and all(curs):
                yield curs[0], curs[1]


def test_recorded_observations_are_pov_filtered():
    n = 0
    for _, cur in _iter_pov_observations():
        n += 1
        my = cur["yourIndex"]
        me, opp = cur["players"][my], cur["players"][1 - my]
        assert opp.get("hand") is None, "opponent hand serialized into observation"
        for side in (me, opp):
            assert side.get("deck") is None, "deck contents serialized into observation"
            for prize_card in side.get("prize") or []:
                assert prize_card is None, "facedown prize card serialized into observation"
    assert n > 1000, f"only {n} observations scanned -- sample too small to trust"


def test_opponent_option_refs_never_resolve_private_cards():
    for obs, cur in _iter_pov_observations():
        my = cur["yourIndex"]
        for option in (obs.get("select") or {}).get("option") or []:
            pi = option.get("playerIndex")
            if pi is None or pi == my:
                continue
            if option.get("area") in OPP_PRIVATE_AREAS:
                # A ref into an opponent private zone is only tolerable if the
                # zone is null in this POV obs, so _get_card resolves to zeros.
                area = option["area"]
                if area == int(AreaType.HAND):
                    assert cur["players"][pi].get("hand") is None
                elif area == int(AreaType.PRIZE):
                    idx = option.get("index")
                    prize = cur["players"][pi].get("prize") or []
                    assert idx is None or idx >= len(prize) or prize[idx] is None
                else:
                    raise AssertionError("option references opponent DECK contents")


def test_select_deck_cards_belong_to_the_acting_player():
    """select.deck (deck-search reveal) must only ever hold MY OWN cards.

    2026-08-03 audit: populated 6,489 times across 300 episodes, always the
    acting player's. If this ever fires on a legitimate look-at-opponent's-
    deck card effect, don't just relax it -- re-audit how _get_card's DECK
    branch encodes those refs before deciding the reveal is safe.
    """
    n_populated = 0
    for obs, cur in _iter_pov_observations():
        deck = (obs.get("select") or {}).get("deck")
        if not deck:
            continue
        n_populated += 1
        owners = {c.get("playerIndex") for c in deck if isinstance(c, dict)}
        assert owners <= {cur["yourIndex"]}, (
            f"select.deck contains cards owned by {owners - {cur['yourIndex']}} "
            "-- foreign (opponent) deck contents serialized into a POV select"
        )
    assert n_populated > 0, "no deck-search selects in sample -- check is vacuous"


def test_looking_card_ids_never_mirrored_into_other_pov():
    """Cards one player privately looks at must not appear in the other POV.

    The engine serializes `looking` entries as None-if-facedown per POV;
    2026-08-03 audit found 328 one-sided looks and zero mirrored. A step
    where both POVs carry the identical non-null id list is the leak
    signature this pins against.
    """
    n_looks = 0
    for cur_a, cur_b in _iter_pov_pairs():
        ids_a = [c and c.get("id") for c in cur_a.get("looking") or []]
        ids_b = [c and c.get("id") for c in cur_b.get("looking") or []]
        has_a, has_b = any(ids_a), any(ids_b)
        n_looks += has_a or has_b
        assert not (has_a and has_b and ids_a == ids_b), (
            "identical looked-at card ids serialized into BOTH POVs -- "
            "private look leaked to the opponent's observation"
        )
    assert n_looks > 0, "no looking states in sample -- check is vacuous"


def test_encoder_output_identical_without_visualize_blob():
    path = _episode_files()[0]
    episode = json.loads(path.read_text())
    assert "visualize" in episode["steps"][0][0], (
        "expected the full-hidden-state visualize blob at steps[0][0] -- "
        "if the episode format changed, re-audit privacy before removing this"
    )

    import copy
    import tempfile

    scrubbed = copy.deepcopy(episode)
    scrubbed["steps"][0][0].pop("visualize")

    def encode_all(ep_dict, tmpdir):
        p = Path(tmpdir) / path.name
        p.write_text(json.dumps(ep_dict))
        out = []
        for obs, label, exclude, _meta in iter_decisions(Path(tmpdir)):
            feats = encode_observation(obs, exclude=exclude)
            if feats is not None:
                out.append((label, feats))
        return out

    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        original = encode_all(episode, d1)
        without_blob = encode_all(scrubbed, d2)

    assert len(original) == len(without_blob) > 0
    for (label_a, feats_a), (label_b, feats_b) in zip(original, without_blob):
        assert label_a == label_b
        assert feats_a.keys() == feats_b.keys()
        for key in feats_a:
            a, b = feats_a[key], feats_b[key]
            if isinstance(a, torch.Tensor):
                assert torch.equal(a, b), f"feature {key} depends on the visualize blob"
            else:
                assert a == b


if __name__ == "__main__":
    tests = [
        test_recorded_observations_are_pov_filtered,
        test_opponent_option_refs_never_resolve_private_cards,
        test_select_deck_cards_belong_to_the_acting_player,
        test_looking_card_ids_never_mirrored_into_other_pov,
        test_encoder_output_identical_without_visualize_blob,
    ]
    for t in tests:
        print(f"running {t.__name__}...")
        t()
        print("  OK")
    print("\nall privacy tests passed")
