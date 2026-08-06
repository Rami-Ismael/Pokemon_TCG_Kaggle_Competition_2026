from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .domain_cards import profile
from .resource_ledger import GRIMMSNARL, ResourceLedger


class TurnPhase(str, Enum):
    SETUP = "setup"
    BUILD = "build"
    STABILIZE = "stabilize"
    PRESSURE = "pressure"
    CLOSE = "close"


@dataclass(frozen=True)
class PhaseAssessment:
    phase: TurnPhase
    reason: str
    prize_transition_imminent: bool
    opponent_requires_multi_attack: bool
    active_role_complete: bool


def infer_turn_phase(ledger: ResourceLedger) -> PhaseAssessment:
    opponent = profile(ledger.opponent_active_id)
    prize_transition = 0 < ledger.opponent_active_hp <= 60
    multi_attack = ledger.opponent_active_hp >= 120
    active_complete = ledger.active_id == GRIMMSNARL and ledger.active_attack_ready

    if ledger.prizes <= 2:
        phase = TurnPhase.CLOSE
        reason = "the remaining prize map is short enough that immediate conversion and gust access dominate"
    elif ledger.attack_ready_grimmsnarl > 0:
        phase = TurnPhase.PRESSURE
        reason = "a primary attacker is online; preserve conversion tempo while funding the replacement attacker"
    elif ledger.control_online or ledger.froslass_online:
        phase = TurnPhase.STABILIZE
        reason = "a support engine is online but the primary attacker still requires completion"
    elif ledger.turn <= 3 or ledger.discard_size == 0:
        phase = TurnPhase.SETUP
        reason = "the public resource cycle is still in its first development window"
    else:
        phase = TurnPhase.BUILD
        reason = "the opening board exists but the attacker/control package is incomplete"

    if opponent.blocks_ex_attack_damage and phase == TurnPhase.PRESSURE:
        reason += "; the Active also blocks direct ex damage, so a bypass job receives an immediate deadline"

    return PhaseAssessment(
        phase=phase,
        reason=reason,
        prize_transition_imminent=prize_transition,
        opponent_requires_multi_attack=multi_attack,
        active_role_complete=active_complete,
    )
