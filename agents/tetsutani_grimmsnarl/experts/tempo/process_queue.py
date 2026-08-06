from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .planner import PlannedObjective, TurnObjective
from .state_model import TurnState


@dataclass(frozen=True)
class ActionIntent:
    role: str
    objective: TurnObjective
    priority: int
    confidence: int
    reason: str
    predicate: Callable[[dict], bool] | None = None


class ProcessQueueBuilder:
    """Expand strategic objectives into an ordered queue of action intents."""

    OBJECTIVE_ROLES: dict[TurnObjective, tuple[str, ...]] = {
        TurnObjective.MOVE_DAMAGE: ("ABILITY_ADRENA",),
        TurnObjective.ESTABLISH_STADIUM: ("PLAY_SPIKEMUTH",),
        TurnObjective.BUILD_ATTACKER: ("ABILITY_SPIKEMUTH", "PLAY_POFFIN", "PLAY_POKE_PAD", "PLAY_IMPIDIMP"),
        TurnObjective.BUILD_BACKUP_ATTACKER: ("PLAY_POFFIN", "PLAY_POKE_PAD", "PLAY_IMPIDIMP", "ABILITY_SPIKEMUTH"),
        TurnObjective.BUILD_FROSLASS_ENGINE: ("EVOLVE_FROSLASS", "PLAY_SNORUNT", "PLAY_POFFIN"),
        TurnObjective.POWER_DAMAGE_ENGINE: ("ATTACH_MUNKIDORI",),
        TurnObjective.POWER_ACTIVE_ATTACKER: ("ATTACH_IMPIDIMP",),
        TurnObjective.POWER_ACTIVE_ENGINE: ("ATTACH_FROSLASS",),
        TurnObjective.COMPLETE_ATTACKER: ("EVOLVE_GRIMMSNARL", "EVOLVE_MORGREM", "PLAY_RARE_CANDY", "PLAY_POKE_PAD", "ABILITY_SPIKEMUTH"),
        TurnObjective.ACCELERATE_ATTACKER: ("PLAY_RARE_CANDY",),
        TurnObjective.TACTICAL_SEARCH: ("PLAY_DAWN",),
        TurnObjective.BROAD_SEARCH: ("PLAY_POKE_PAD",),
        TurnObjective.REPOSITION: ("RETREAT",),
        TurnObjective.POWER_ACTIVE_BASIC: ("ATTACH_SNORUNT",),
        TurnObjective.REFILL_HAND: ("PLAY_LILLIE", "PLAY_DAWN"),
        TurnObjective.BANK_SUPPORTER: ("PLAY_POKEGEAR",),
        TurnObjective.TARGETED_RECOVERY: ("PLAY_PETREL", "PLAY_NIGHT_STRETCHER", "PLAY_POKEGEAR"),
        TurnObjective.RESET_OPPONENT_HAND: ("PLAY_UNFAIR_STAMP",),
        TurnObjective.EXPAND_BOARD: ("PLAY_POFFIN",),
        TurnObjective.REFRESH_SEARCH_ENGINE: ("PLAY_SPIKEMUTH",),
        TurnObjective.REMOVE_TOOL: ("PLAY_TOOL_SCRAPPER",),
        TurnObjective.DISRUPT: ("PLAY_BOSS", "PLAY_UNFAIR_STAMP", "PLAY_TOOL_SCRAPPER"),
        TurnObjective.KEEP_PRESSURE: ("ATTACK_SHADOW_BULLET",),
        TurnObjective.ATTACK: ("ATTACK_SHADOW_BULLET", "ATTACK_MORGREM", "ATTACK_FILCH", "ATTACK_IMPIDIMP"),
        TurnObjective.END_TURN: ("END",),
        TurnObjective.RECOVER_LOST_ATTACKER: ("EVOLVE_GRIMMSNARL", "ABILITY_SPIKEMUTH", "PLAY_RARE_CANDY", "EVOLVE_MORGREM"),
        TurnObjective.RESTORE_SEARCH_ENGINE: ("PLAY_SPIKEMUTH",),
        TurnObjective.COMPLETE_PASSIVE_ENGINE: ("EVOLVE_FROSLASS",),
        TurnObjective.CONTINUE_ATTACK_DEVELOPMENT: ("EVOLVE_MORGREM", "PLAY_RARE_CANDY", "EVOLVE_GRIMMSNARL", "ABILITY_SPIKEMUTH"),
        TurnObjective.CONTINUE_EXACT_SEARCH: ("ABILITY_SPIKEMUTH",),
    }

    def build_main(
        self,
        state: TurnState,
        objectives: list[PlannedObjective],
        baseline_role: str,
    ) -> list[ActionIntent]:
        queue: list[ActionIntent] = []
        seen: set[tuple[str, TurnObjective]] = set()
        for obj in objectives:
            if obj.objective is TurnObjective.COMPATIBILITY:
                continue
            for role in self.OBJECTIVE_ROLES.get(obj.objective, ()):
                key = (role, obj.objective)
                if role in state.legal_roles and key not in seen:
                    queue.append(ActionIntent(role, obj.objective, obj.priority, obj.confidence, obj.reason))
                    seen.add(key)
        # Validated v20 choice is represented as the final compatibility intent,
        # not as a direct bypass around the hierarchy.
        queue.append(ActionIntent(
            baseline_role,
            TurnObjective.COMPATIBILITY,
            -1,
            0,
            "validated v20 role fallback",
        ))
        return queue
