"""Per-matchup score bonuses for dragapult_agent (Phase 2b).

Uses the same archetype tags as agent/matchup_levers.py. Tune one row at a time;
re-gate via scripts/gate_dragapult.py with harness + Wilson CI.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent.matchup_levers import primary_archetype


@dataclass(frozen=True)
class DragapultLeverDeltas:
    boss_orders_hand: float = 0.0
    boss_orders_play: float = 0.0
    gust_setup: float = 0.0


# Starting hypotheses — each row must pass weighted gate before trust.
DRAGAPULT_LEVERS: dict[str, DragapultLeverDeltas] = {
    "lucario_mirror": DragapultLeverDeltas(
        boss_orders_hand=0.0,
        boss_orders_play=0.0,
    ),
    "alakazam_psychic": DragapultLeverDeltas(),
    "dragapult_psychic": DragapultLeverDeltas(),
    "iono_lightning": DragapultLeverDeltas(),
    "abomasnow_water": DragapultLeverDeltas(),
}


_active_overrides: dict[str, DragapultLeverDeltas] | None = None


def set_dragapult_lever_overrides(overrides: dict[str, DragapultLeverDeltas] | None) -> None:
    global _active_overrides
    _active_overrides = overrides


def _table() -> dict[str, DragapultLeverDeltas]:
    if not _active_overrides:
        return DRAGAPULT_LEVERS
    return {**DRAGAPULT_LEVERS, **_active_overrides}


def levers_for_dragapult(opp_board_ids: set[int] | frozenset[int]) -> DragapultLeverDeltas:
    tag = primary_archetype(opp_board_ids)
    if tag is None:
        return DragapultLeverDeltas()
    return _table().get(tag, DragapultLeverDeltas())


def boss_orders_bonus(opp_board_ids: set[int] | frozenset[int], *, phase: str) -> float:
    """phase: 'hand' (hand_score) or 'play' (MAIN play supporter)."""
    d = levers_for_dragapult(opp_board_ids)
    if phase == "hand":
        return d.boss_orders_hand
    if phase == "play":
        return d.boss_orders_play
    return 0.0
