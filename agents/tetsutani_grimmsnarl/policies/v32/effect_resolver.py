from __future__ import annotations

from . import validated_rule_profile as primitive
from .effect_planner import SelectionIntent, SelectionObjective


class EffectIndexResolver:
    """Resolve a non-MAIN semantic selection plan to legal option indices."""

    def resolve_card(self, view: dict, intents: list[SelectionIntent], baseline: list[int]) -> list[int]:
        for intent in intents:
            if intent.objective is SelectionObjective.COMPATIBILITY:
                return baseline
            if intent.confidence < 90 or intent.desired_card_id is None:
                continue
            hit = primitive._choose_first(
                view,
                lambda s, cid=intent.desired_card_id: int(s.get("source_id", 0) or 0) == cid,
            )
            if hit:
                return hit
        return baseline
