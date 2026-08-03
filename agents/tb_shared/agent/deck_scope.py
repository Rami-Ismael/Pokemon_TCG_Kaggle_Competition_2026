"""Deck archetype confidence + soft masking for Lucario MCTS lever teaching.

Before an opponent head exists in the NN, we gate matchup-lever bias by how
confident we are in the opponent archetype (deck list + visible board).
Higher confidence → stronger lever blend (scoped rules); low → rely on MCTS/NN.
"""

from __future__ import annotations

from agent.matchup_levers import primary_archetype
from agent.official_registry import official_archetype_for_opponent

# Map visible-board tags (matchup_levers) → official pilot archetype keys.
_BOARD_TO_OFFICIAL: dict[str, str] = {
    "lucario_mirror": "mega_lucario_ex",
    "dragapult_psychic": "dragapult_ex",
    "iono_lightning": "iono",
    "abomasnow_water": "mega_abomasnow_ex",
    "kyogre_water": "mega_abomasnow_ex",  # same water spread family
    "trevenant_control": "trevenant",  # no official pilot; low prior only
    "alakazam_psychic": "alakazam",
    "crustle_wall": "crustle",
    "dunsparce_engine": "dunsparce",
}

# Minimum confidence before any lever bonus applies (soft mask floor).
SCOPE_CONFIDENCE_FLOOR = 0.45


def visible_board_ids(obs_dict: dict) -> set[int]:
    """Opponent active + bench card ids from observation dict."""
    out: set[int] = set()
    try:
        current = obs_dict.get("current") or {}
        yi = int(current.get("yourIndex", 0))
        players = current.get("players") or []
        if len(players) < 2:
            return out
        opp = players[1 - yi]
        for area in (opp.get("active") or [], opp.get("bench") or []):
            for card in area:
                if card is None:
                    continue
                cid = card.get("id") if isinstance(card, dict) else getattr(card, "id", None)
                if cid is not None:
                    out.add(int(cid))
    except (TypeError, ValueError, IndexError, AttributeError):
        pass
    return out


def archetype_confidence(
    opp_name: str,
    opp_deck_ids: list[int],
    board_ids: set[int] | frozenset[int],
) -> float:
    """0..1 confidence that we know which opponent rule family is in play."""
    deck_arch = official_archetype_for_opponent(opp_name, opp_deck_ids)
    board_tag = primary_archetype(board_ids) if board_ids else None
    board_official = _BOARD_TO_OFFICIAL.get(board_tag or "", "")

    conf = 0.40
    if deck_arch:
        conf = 0.82  # deck list is strong prior (field training knows opponent CSV)
    if board_tag and board_official:
        if deck_arch and board_official == deck_arch:
            conf = min(1.0, conf + 0.18)  # board confirms deck
        elif deck_arch:
            conf = max(0.55, conf - 0.10)  # mixed signals
        else:
            conf = min(0.92, conf + 0.25)  # board-only inference
    return max(0.0, min(1.0, conf))


def soft_lever_blend(
    base_blend: float,
    *,
    opp_name: str,
    opp_deck_ids: list[int],
    board_ids: set[int] | frozenset[int],
) -> float:
    """Scale lever teaching by archetype confidence (soft mask on matchup levers)."""
    if base_blend <= 0:
        return 0.0
    conf = archetype_confidence(opp_name, opp_deck_ids, board_ids)
    if conf < SCOPE_CONFIDENCE_FLOOR:
        return 0.0
    # Linear ramp from floor → 1.0
    scale = (conf - SCOPE_CONFIDENCE_FLOOR) / (1.0 - SCOPE_CONFIDENCE_FLOOR)
    return base_blend * max(0.0, min(1.0, scale))
