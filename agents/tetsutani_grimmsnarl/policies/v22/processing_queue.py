from __future__ import annotations

from dataclasses import dataclass

from .domain_cards import profile
from .resource_ledger import (
    DARK,
    FROSLASS,
    GRIMMSNARL,
    IMPIDIMP,
    LILLIE,
    MORGREM,
    MUNKIDORI,
    NIGHT_STRETCHER,
    PETREL,
    POFFIN,
    POKE_PAD,
    SHADOW_BULLET,
    SPIKEMUTH_GYM,
    ResourceLedger,
    action_signature,
)
from .turn_objective import ObjectiveAssessment, TurnObjective


@dataclass(frozen=True)
class StrategicJob:
    name: str
    objective: TurnObjective
    priority: int
    committed: bool
    completed: bool
    reason: str


@dataclass(frozen=True)
class QueueDecision:
    index: int
    job: str
    reason: str
    mode: str  # committed_change, committed_confirm, structural_confirm


def _sig(t: int, source: int = 0, zone: int = 0, target: int = 0, area: int = 0, attack: int = 0):
    return (t, source, zone, target, area, attack)


def build_processing_queue(ledger: ResourceLedger, objective: ObjectiveAssessment) -> tuple[StrategicJob, ...]:
    opponent = profile(ledger.opponent_active_id)
    jobs = [
        StrategicJob(
            "retreat_to_ready_attacker", TurnObjective.PRESERVE_TEMPO, 120,
            objective.committed and objective.primary == TurnObjective.PRESERVE_TEMPO,
            ledger.active_id == GRIMMSNARL and ledger.active_attack_ready,
            objective.hard_reason,
        ),
        StrategicJob(
            "shrink_hand_before_damage_check", TurnObjective.CONTROL_HAND_SIZE, 115,
            objective.committed and objective.primary == TurnObjective.CONTROL_HAND_SIZE,
            ledger.supporter_played,
            objective.hard_reason,
        ),
        StrategicJob(
            "apply_bench_pressure_through_immunity", TurnObjective.APPLY_SPREAD_PRESSURE, 110,
            objective.committed and objective.primary == TurnObjective.APPLY_SPREAD_PRESSURE,
            False,
            objective.hard_reason,
        ),
        StrategicJob(
            "establish_primary_board", TurnObjective.ESTABLISH_BOARD, 100,
            objective.committed and objective.primary == TurnObjective.ESTABLISH_BOARD,
            ledger.marnie_line_count >= 2 or ledger.bench_slots == 0,
            objective.hard_reason,
        ),
        StrategicJob(
            "establish_replacement_attacker", TurnObjective.BUILD_BACKUP_ATTACKER, 95,
            objective.committed and objective.primary == TurnObjective.BUILD_BACKUP_ATTACKER,
            ledger.marnie_line_count >= 2 or ledger.bench_slots == 0,
            objective.hard_reason,
        ),
        # Structural migration jobs. These are committed only when both the resource
        # bottleneck and the legal action form are unambiguous.
        StrategicJob(
            "finish_evolution_before_control_attachment", TurnObjective.COMPLETE_ATTACKER, 90,
            opponent.is_repeatable_draw_engine
            and ledger.previous_action_type >= 8
            and not ledger.energy_attached,
            ledger.attack_ready_grimmsnarl > 0,
            "after an evolution action in a slow matchup, finish deterministic evolution access before enabling control",
        ),
        StrategicJob(
            "spend_energy_surplus_on_evolution_bottleneck", TurnObjective.COMPLETE_ATTACKER, 88,
            ledger.marnie_line_count == 2 and ledger.dark_surplus,
            ledger.attack_ready_grimmsnarl > 0,
            "Darkness is abundant; evolution access, not energy, is the limiting resource",
        ),
        StrategicJob(
            "acquire_control_body_after_froslass_core", TurnObjective.ENABLE_CONTROL, 86,
            ledger.board[FROSLASS] >= 2 and ledger.prizes >= 4 and ledger.hand[POKE_PAD] > 0,
            False,
            "the Froslass engine is complete; Pokémon search should fill the remaining control role",
        ),
        StrategicJob(
            "acquire_missing_control_body_at_setup_checkpoint", TurnObjective.ENABLE_CONTROL, 84,
            ledger.turn_action_count == 7 and ledger.hand[MUNKIDORI] == 0 and ledger.hand[POKE_PAD] > 0,
            False,
            "the setup queue reached its search checkpoint with no control body in hand",
        ),
        StrategicJob(
            "complete_froslass_engine", TurnObjective.COMPLETE_FROSLASS, 70,
            False,
            ledger.board[FROSLASS] >= 2,
            objective.hard_reason,
        ),
        StrategicJob(
            "enable_damage_control", TurnObjective.ENABLE_CONTROL, 60,
            False,
            ledger.control_online or ledger.damaged_own_count == 0,
            objective.hard_reason,
        ),
        StrategicJob(
            "recover_spent_core", TurnObjective.RECOVER_RESOURCES, 50,
            False,
            ledger.recoverable_core == 0,
            objective.hard_reason,
        ),
    ]
    return tuple(sorted(jobs, key=lambda job: -job.priority))


def _candidate(ledger: ResourceLedger, options: list[dict], signature: tuple[int, int, int, int, int, int]) -> int | None:
    return ledger.find_option(options, signature)


def _resolve_job(job: StrategicJob, ledger: ResourceLedger, options: list[dict], current: tuple[int, int, int, int, int, int]) -> int | None:
    if job.name == "shrink_hand_before_damage_check" and current == _sig(7, LILLIE, 2):
        return _candidate(ledger, options, _sig(7, PETREL, 2))

    if job.name == "apply_bench_pressure_through_immunity" and current == _sig(9, GRIMMSNARL, 2, MORGREM, 5):
        return _candidate(ledger, options, _sig(13, attack=SHADOW_BULLET))

    if job.name == "establish_primary_board" and current in (
        _sig(8, DARK, 2, MUNKIDORI, 5),
        _sig(7, POKE_PAD, 2),
    ):
        return _candidate(ledger, options, _sig(7, POFFIN, 2))

    if job.name == "establish_replacement_attacker" and current == _sig(7, MUNKIDORI, 2):
        return _candidate(ledger, options, _sig(7, IMPIDIMP, 2))

    if job.name == "retreat_to_ready_attacker":
        return next((i for i, option in enumerate(options) if int(option.get("type", -1)) == 12), None)

    if job.name == "finish_evolution_before_control_attachment" and current == _sig(8, DARK, 2, MUNKIDORI, 5):
        return _candidate(ledger, options, _sig(10, zone=7, area=7))

    if job.name == "spend_energy_surplus_on_evolution_bottleneck" and current == _sig(7, POKE_PAD, 2):
        return _candidate(ledger, options, _sig(10, zone=7, area=7))

    if job.name in ("acquire_control_body_after_froslass_core", "acquire_missing_control_body_at_setup_checkpoint"):
        if current == _sig(10, zone=7, area=7):
            return _candidate(ledger, options, _sig(7, POKE_PAD, 2))

    if job.name == "complete_froslass_engine":
        return _candidate(ledger, options, _sig(9, FROSLASS, 2, 860, 5))

    if job.name == "enable_damage_control":
        return _candidate(ledger, options, _sig(8, DARK, 2, MUNKIDORI, 5))

    if job.name == "recover_spent_core":
        for signature in (_sig(7, PETREL, 2), _sig(7, NIGHT_STRETCHER, 2)):
            index = _candidate(ledger, options, signature)
            if index is not None:
                return index
    return None


def _structural_job_for_signature(signature: tuple[int, int, int, int, int, int]) -> str | None:
    t, source, zone, target, area, attack = signature
    if t == 13:
        return "convert_attack"
    if t == 12:
        return "pivot_active"
    if t == 10:
        return "search_evolution"
    if t == 9 and source == FROSLASS:
        return "complete_froslass_engine"
    if t == 9:
        return "advance_attacker_line"
    if t == 8 and target == MUNKIDORI:
        return "enable_damage_control"
    if t == 8:
        return "develop_energy"
    if t == 7 and source in (PETREL, NIGHT_STRETCHER):
        return "recover_spent_core"
    if t == 7 and source == POFFIN:
        return "establish_primary_board"
    if t == 7 and source == IMPIDIMP:
        return "establish_replacement_attacker"
    if t == 7 and source == MUNKIDORI:
        return "establish_control_body"
    if t == 7 and source == POKE_PAD:
        return "search_missing_pokemon_role"
    if t == 7 and source == LILLIE:
        return "refresh_hand"
    return None


def plan_main_action(ledger: ResourceLedger, options: list[dict], baseline: list[int], objective: ObjectiveAssessment) -> QueueDecision | None:
    if len(baseline) != 1:
        return None
    current = action_signature(options[baseline[0]])
    for job in build_processing_queue(ledger, objective):
        if not job.committed or job.completed:
            continue
        index = _resolve_job(job, ledger, options, current)
        if index is not None:
            mode = "committed_change" if index != baseline[0] else "committed_confirm"
            return QueueDecision(index, job.name, job.reason, mode)

    structural = _structural_job_for_signature(current)
    if structural is not None:
        return QueueDecision(
            baseline[0],
            structural,
            "the selected legal action directly satisfies a recognized resource job",
            "structural_confirm",
        )
    return None
