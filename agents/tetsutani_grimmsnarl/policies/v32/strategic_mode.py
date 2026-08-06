from __future__ import annotations

"""Strategic-mode arbitration for the pure handwritten hierarchical engine.

This layer classifies the public resource ledger into a domain operating mode,
then resolves only validated conflicts between already-legal semantic actions.
No replay identifier, fitted threshold, state hash, rollout, or search is used.
"""

from dataclasses import dataclass
from enum import Enum, auto

from .planner import PlannedObjective, TurnObjective
from .state_model import TurnState


class StrategicMode(Enum):
    OPENING_ESTABLISH = auto()
    EMERGENCY_REBUILD = auto()
    ATTACKER_COMPLETION = auto()
    ENGINE_BUILD = auto()
    BACKUP_BUILD = auto()
    PRESSURE = auto()
    ENDGAME = auto()


@dataclass(frozen=True)
class ModeTrace:
    mode: StrategicMode
    reason: str


class StrategicModeController:
    def classify(self, state: TurnState) -> ModeTrace:
        ledger = state.ledger
        if ledger.ready_attackers == 0:
            if ledger.attack_lines == 0:
                if state.turn >= 5:
                    return ModeTrace(
                        StrategicMode.EMERGENCY_REBUILD,
                        "no remaining attack line after opening",
                    )
                return ModeTrace(
                    StrategicMode.OPENING_ESTABLISH,
                    "opening board has no attack line",
                )
            return ModeTrace(
                StrategicMode.ATTACKER_COMPLETION,
                "attack bodies exist but no Grimmsnarl is ready",
            )
        if state.turn >= 10 or ledger.opponent_prizes <= 2:
            return ModeTrace(
                StrategicMode.ENDGAME,
                "late turn or opponent within two prizes",
            )
        if ledger.froslass_lines < 1 or ledger.munkidori_powered < 1:
            return ModeTrace(
                StrategicMode.ENGINE_BUILD,
                "primary attacker is online but a support engine is missing",
            )
        if ledger.attack_lines < 2:
            return ModeTrace(
                StrategicMode.BACKUP_BUILD,
                "one ready line needs a backup",
            )
        return ModeTrace(
            StrategicMode.PRESSURE,
            "attack and support resources are online",
        )

    def objectives(self, state: TurnState, provisional_role: str) -> list[PlannedObjective]:
        trace = self.classify(state)
        mode = trace.mode
        ledger = state.ledger
        legal = state.legal_roles
        objectives: list[PlannedObjective] = []

        # Exact-search is a missing-resource operation. A generic opponent-hand
        # reset must not spend the supporter/tempo window before the live
        # Spikemuth search, except in a compact endgame hand with a ready attacker.
        if (
            provisional_role == "PLAY_UNFAIR_STAMP"
            and "ABILITY_SPIKEMUTH" in legal
            and not (
                mode is StrategicMode.ENDGAME
                and ledger.hand_size <= 3
                and ledger.ready_attackers >= 1
            )
        ):
            objectives.append(
                PlannedObjective(
                    TurnObjective.CONTINUE_EXACT_SEARCH,
                    1200,
                    100,
                    "finish the live exact-search resource before generic hand disruption",
                )
            )

        # With pressure already online, Petrel is the tactical selector. Choose
        # the exact next function before consuming the supporter on a generic
        # hand reset that is not tied to a missing public board resource.
        if (
            provisional_role == "PLAY_UNFAIR_STAMP"
            and mode in (StrategicMode.PRESSURE, StrategicMode.ENDGAME)
            and "PLAY_PETREL" in legal
        ):
            objectives.append(
                PlannedObjective(
                    TurnObjective.TARGETED_RECOVERY,
                    1199,
                    100,
                    "pressure mode selects an exact tactical trainer before generic disruption",
                )
            )

        # Manual attachment expires at turn end. If a visible Munkidori remains
        # unpowered, consume that public resource before committing the supporter
        # window to draw, disruption, or gust.
        if (
            provisional_role in {"PLAY_UNFAIR_STAMP", "PLAY_BOSS", "PLAY_LILLIE"}
            and ledger.munkidori_unpowered >= 1
            and not ledger.energy_attached
            and "ATTACH_MUNKIDORI" in legal
        ):
            objectives.append(
                PlannedObjective(
                    TurnObjective.POWER_DAMAGE_ENGINE,
                    1201,
                    100,
                    "consume the expiring attachment on an unpowered Munkidori before supporter commitment",
                )
            )

        return sorted(
            objectives,
            key=lambda item: (-item.priority, -item.confidence, item.objective.value),
        )
