from __future__ import annotations

from . import validated_rule_profile as profile
from .planner import TurnObjective
from .process_queue import ActionIntent


class IndexResolver:
    """Resolve a semantic process queue to a legal engine option index."""

    _OVERRIDE_OBJECTIVES = frozenset({
        TurnObjective.BUILD_ATTACKER,
        TurnObjective.RESET_OPPONENT_HAND,
        TurnObjective.EXPAND_BOARD,
        TurnObjective.REFRESH_SEARCH_ENGINE,
        TurnObjective.REMOVE_TOOL,
        TurnObjective.KEEP_PRESSURE,
        TurnObjective.POWER_ACTIVE_ATTACKER,
        TurnObjective.POWER_ACTIVE_ENGINE,
        TurnObjective.COMPLETE_ATTACKER,
        TurnObjective.BANK_SUPPORTER,
        TurnObjective.POWER_DAMAGE_ENGINE,
        TurnObjective.BUILD_FROSLASS_ENGINE,
        TurnObjective.ACCELERATE_ATTACKER,
        TurnObjective.TACTICAL_SEARCH,
        TurnObjective.BROAD_SEARCH,
        TurnObjective.REPOSITION,
        TurnObjective.POWER_ACTIVE_BASIC,
        TurnObjective.REFILL_HAND,
        TurnObjective.TARGETED_RECOVERY,
        TurnObjective.RECOVER_LOST_ATTACKER,
        TurnObjective.RESTORE_SEARCH_ENGINE,
        TurnObjective.COMPLETE_PASSIVE_ENGINE,
        TurnObjective.CONTINUE_ATTACK_DEVELOPMENT,
        TurnObjective.CONTINUE_EXACT_SEARCH,
    })

    def resolve_main(self, view: dict, queue: list[ActionIntent], baseline: list[int]) -> list[int]:
        for intent in queue:
            if intent.objective is TurnObjective.COMPATIBILITY:
                return baseline
            if intent.objective not in self._OVERRIDE_OBJECTIVES or intent.confidence < 90:
                continue
            idx = profile._find_role(view, intent.role, intent.predicate)
            if idx is not None:
                return profile._legal_fill(view, [idx])
        return baseline
