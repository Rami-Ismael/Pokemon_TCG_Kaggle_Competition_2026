from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from .domain_cards import profile
from .resource_ledger import (
    DARK,
    GRIMMSNARL,
    IMPIDIMP,
    MUNKIDORI,
    ResourceLedger,
)
from .turn_phase import PhaseAssessment


class DeadlineClass(IntEnum):
    """Lexicographic deadline classes; lower is resolved first."""

    IMMEDIATE_PRIZE_OR_LOCK = 0
    EXPIRES_THIS_TURN = 1
    ENABLE_ATTACK = 2
    RESERVE_NEXT_TURN = 3
    RESOURCE_COMPRESSION = 4
    OPTIONAL = 5


@dataclass(frozen=True)
class DeadlineLedger:
    phase: PhaseAssessment
    bypass_ex_immunity: bool
    finish_backup_attachment_before_attack: bool
    reserve_attacker_before_prize_transition: bool
    establish_two_role_opening: bool
    recover_after_evolution_commit: bool
    reason: str

    @classmethod
    def from_ledger(cls, ledger: ResourceLedger, phase: PhaseAssessment) -> "DeadlineLedger":
        opponent = profile(ledger.opponent_active_id)
        bench_grim_energies = [
            int(card.get("en", 0) or 0)
            for card in ledger.bench
            if int(card.get("id", 0) or 0) == GRIMMSNARL
        ]
        one_energy_backup = any(value == 1 for value in bench_grim_energies)
        previous_was_dark_selection = ledger.previous_action_source == DARK
        bypass = (
            opponent.blocks_ex_attack_damage
            and ledger.active_attack_ready
            and (ledger.board[IMPIDIMP] == 0 or ledger.board[MUNKIDORI] >= 2)
        )
        finish_backup = (
            not ledger.energy_attached
            and ledger.hand[DARK] > 0
            and one_energy_backup
            and previous_was_dark_selection
            and ledger.active_attack_ready
        )
        reserve_before_prize = (
            phase.prize_transition_imminent
            and ledger.marnie_line_count <= 1
            and ledger.bench_slots > 0
        )
        establish_two_roles = (
            phase.phase.value == "setup"
            and phase.opponent_requires_multi_attack
            and (ledger.discard_size == 0 or ledger.bench_slots >= 4)
            and ledger.bench_slots >= 2
        )
        recover_after_evolution = (
            ledger.last_action_type == 9
            and ledger.recoverable_core >= 3
            and (opponent.is_hand_size_punisher or opponent.is_repeatable_draw_engine)
            and not ledger.supporter_played
        )
        return cls(
            phase=phase,
            bypass_ex_immunity=bypass,
            finish_backup_attachment_before_attack=finish_backup,
            reserve_attacker_before_prize_transition=reserve_before_prize,
            establish_two_role_opening=establish_two_roles,
            recover_after_evolution_commit=recover_after_evolution,
            reason=phase.reason,
        )
