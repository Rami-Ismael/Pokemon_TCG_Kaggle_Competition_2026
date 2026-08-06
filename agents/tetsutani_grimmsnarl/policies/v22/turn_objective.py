from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .domain_cards import profile
from .resource_ledger import (
    FROSLASS,
    GRIMMSNARL,
    IMPIDIMP,
    LILLIE,
    MUNKIDORI,
    PETREL,
    POFFIN,
    POKE_PAD,
    ResourceLedger,
)


class TurnObjective(str, Enum):
    PRESERVE_TEMPO = "preserve_tempo"
    CONVERT_PRIZE = "convert_prize"
    CONTROL_HAND_SIZE = "control_hand_size"
    APPLY_SPREAD_PRESSURE = "apply_spread_pressure"
    ESTABLISH_BOARD = "establish_board"
    BUILD_BACKUP_ATTACKER = "build_backup_attacker"
    COMPLETE_ATTACKER = "complete_attacker"
    ENABLE_CONTROL = "enable_control"
    COMPLETE_FROSLASS = "complete_froslass"
    RECOVER_RESOURCES = "recover_resources"
    PRESERVE_RESOURCE = "preserve_resource"
    GENERAL_PRESSURE = "general_pressure"


@dataclass(frozen=True)
class ObjectiveAssessment:
    primary: TurnObjective
    queue: tuple[TurnObjective, ...]
    hard_reason: str
    committed: bool
    confidence: str


def infer_turn_objective(ledger: ResourceLedger) -> ObjectiveAssessment:
    opponent = profile(ledger.opponent_active_id)

    if (
        ledger.active_id == FROSLASS
        and ledger.active_hp == ledger.active_max_hp == 90
        and ledger.has_attack_ready_bench_grimmsnarl()
        and ledger.hand[PETREL] == 0
    ):
        return ObjectiveAssessment(
            TurnObjective.PRESERVE_TEMPO,
            (TurnObjective.PRESERVE_TEMPO, TurnObjective.CONVERT_PRIZE, TurnObjective.GENERAL_PRESSURE),
            "Froslass has completed its passive role and a ready attacker is waiting",
            True,
            "hard",
        )

    if opponent.is_hand_size_punisher and ledger.hand[LILLIE] and ledger.hand[PETREL]:
        return ObjectiveAssessment(
            TurnObjective.CONTROL_HAND_SIZE,
            (TurnObjective.CONTROL_HAND_SIZE, TurnObjective.RECOVER_RESOURCES),
            "opponent converts our hand size into damage",
            True,
            "hard",
        )

    if (
        opponent.blocks_ex_attack_damage
        and ledger.active_attack_ready
        and (ledger.hand[7] == 0 or ledger.board[FROSLASS] == 0)
    ):
        return ObjectiveAssessment(
            TurnObjective.APPLY_SPREAD_PRESSURE,
            (TurnObjective.APPLY_SPREAD_PRESSURE, TurnObjective.GENERAL_PRESSURE),
            "Active damage is blocked but Shadow Bullet still advances the Bench prize map",
            True,
            "hard",
        )

    if ledger.turn == 1 and ledger.marnie_line_count == 0 and ledger.bench_slots >= 4 and ledger.hand[POFFIN]:
        return ObjectiveAssessment(
            TurnObjective.ESTABLISH_BOARD,
            (TurnObjective.ESTABLISH_BOARD, TurnObjective.ENABLE_CONTROL),
            "primary attacker line is absent and Poffin can establish two jobs",
            True,
            "hard",
        )

    if (
        opponent.is_repeatable_draw_engine
        and ledger.turn <= 2
        and ledger.discard_size == 0
        and ledger.bench_slots >= 3
        and ledger.marnie_line_count <= 1
        and ledger.hand[POFFIN]
    ):
        return ObjectiveAssessment(
            TurnObjective.ESTABLISH_BOARD,
            (TurnObjective.ESTABLISH_BOARD, TurnObjective.COMPLETE_ATTACKER),
            "slow draw-engine matchup leaves an early development window",
            True,
            "hard",
        )

    if (
        ledger.active_attack_ready
        and ledger.active_hp == ledger.active_max_hp
        and ledger.marnie_line_count == 1
        and ledger.bench_slots > 0
        and ledger.hand[IMPIDIMP] > 0
        and ledger.hand[MUNKIDORI] > 0
    ):
        return ObjectiveAssessment(
            TurnObjective.BUILD_BACKUP_ATTACKER,
            (TurnObjective.BUILD_BACKUP_ATTACKER, TurnObjective.ENABLE_CONTROL),
            "first attacker is complete; replacement attacker is the missing resource",
            True,
            "hard",
        )

    if (
        opponent.is_repeatable_draw_engine
        and ledger.prizes == 6
        and ledger.marnie_line_count == 1
        and ledger.bench_slots > 0
        and ledger.hand[IMPIDIMP] > 0
        and ledger.hand[MUNKIDORI] > 0
    ):
        return ObjectiveAssessment(
            TurnObjective.BUILD_BACKUP_ATTACKER,
            (TurnObjective.BUILD_BACKUP_ATTACKER, TurnObjective.ESTABLISH_BOARD),
            "slow opening matchup permits the second primary line before extra support",
            True,
            "hard",
        )

    if ledger.ready_froslass_evolution and ledger.turn <= 6 and ledger.hand[1079] == 0:
        return ObjectiveAssessment(
            TurnObjective.COMPLETE_FROSLASS,
            (TurnObjective.COMPLETE_FROSLASS, TurnObjective.COMPLETE_ATTACKER, TurnObjective.GENERAL_PRESSURE),
            "the passive damage engine can be completed immediately without consuming Rare Candy",
            False,
            "soft",
        )

    if ledger.damaged_own_count > 0 and not ledger.control_online and ledger.hand[7] > 0 and ledger.board[MUNKIDORI] > 0:
        return ObjectiveAssessment(
            TurnObjective.ENABLE_CONTROL,
            (TurnObjective.ENABLE_CONTROL, TurnObjective.COMPLETE_ATTACKER, TurnObjective.GENERAL_PRESSURE),
            "damage counters exist but Adrena-Brain is not yet online",
            False,
            "soft",
        )

    if ledger.attack_ready_grimmsnarl:
        return ObjectiveAssessment(
            TurnObjective.GENERAL_PRESSURE,
            (TurnObjective.CONVERT_PRIZE, TurnObjective.GENERAL_PRESSURE, TurnObjective.RECOVER_RESOURCES),
            "an attacker is online",
            False,
            "soft",
        )
    if ledger.attacker_line_deficit > 0:
        return ObjectiveAssessment(
            TurnObjective.ESTABLISH_BOARD,
            (TurnObjective.ESTABLISH_BOARD, TurnObjective.COMPLETE_ATTACKER, TurnObjective.ENABLE_CONTROL),
            "primary attacker line is underdeveloped",
            False,
            "soft",
        )
    if ledger.evolution_bottleneck:
        return ObjectiveAssessment(
            TurnObjective.COMPLETE_ATTACKER,
            (TurnObjective.COMPLETE_ATTACKER, TurnObjective.COMPLETE_FROSLASS, TurnObjective.GENERAL_PRESSURE),
            "Basic or Stage 1 bodies exist but the next evolution stage is the bottleneck",
            False,
            "soft",
        )
    if ledger.deep_discard:
        return ObjectiveAssessment(
            TurnObjective.RECOVER_RESOURCES,
            (TurnObjective.RECOVER_RESOURCES, TurnObjective.GENERAL_PRESSURE),
            "core resources have accumulated in the discard",
            False,
            "soft",
        )
    return ObjectiveAssessment(
        TurnObjective.COMPLETE_ATTACKER,
        (TurnObjective.COMPLETE_ATTACKER, TurnObjective.GENERAL_PRESSURE),
        "default progression toward the next attacker",
        False,
        "soft",
    )
