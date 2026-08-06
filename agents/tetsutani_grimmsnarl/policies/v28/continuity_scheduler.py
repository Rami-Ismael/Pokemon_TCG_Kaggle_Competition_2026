from __future__ import annotations

from dataclasses import dataclass

from .deadline_ledger import DeadlineClass
from .domain_cards import profile
from .resource_ledger import FROSLASS, IMPIDIMP, MUNKIDORI, POFFIN, SNORUNT, ResourceLedger
from .strategic_memory import StrategicMemoryView


@dataclass(frozen=True)
class ContinuityDecision:
    index: int
    job: str
    deadline: DeadlineClass
    reason: str
    mode: str


def _find(options: list[dict], predicate) -> int | None:
    return next((i for i, option in enumerate(options) if predicate(option)), None)


def _is_gym(option: dict) -> bool:
    return int(option.get("type", -1)) == 10


def _is_attack(option: dict) -> bool:
    return int(option.get("type", -1)) == 13


def _is_evolve_froslass(option: dict) -> bool:
    return int(option.get("type", -1)) == 9 and int(option.get("source_id", 0) or 0) == FROSLASS


def _is_play_snorunt(option: dict) -> bool:
    return int(option.get("type", -1)) == 7 and int(option.get("source_id", 0) or 0) == SNORUNT



def _is_attach_control(option: dict) -> bool:
    return (
        int(option.get("type", -1)) == 8
        and int(option.get("target_id", 0) or 0) == MUNKIDORI
    )


def _is_play_poffin(option: dict) -> bool:
    return int(option.get("type", -1)) == 7 and int(option.get("source_id", 0) or 0) == POFFIN


def _is_pivot(option: dict) -> bool:
    return int(option.get("type", -1)) == 12


def plan_continuity(
    ledger: ResourceLedger,
    options: list[dict],
    current_indices: list[int],
    memory: StrategicMemoryView,
) -> ContinuityDecision | None:
    if len(current_indices) != 1:
        return None
    current_index = current_indices[0]
    current = options[current_index]
    opponent = profile(ledger.opponent_active_id)

    # The control body was just established and the opposing Active is already in
    # prize-conversion range.  Reserve the next board roles before spending the
    # once-per-turn attachment on control; the attachment remains available after
    # Poffin, while the open Bench roles may disappear after the prize transition.
    if (
        _is_attach_control(current)
        and memory.current_turn_last_job == "bench_control"
        and ledger.opponent_active_hp <= 60
        and ledger.bench_slots >= 2
    ):
        index = _find(options, _is_play_poffin)
        if index is not None:
            return ContinuityDecision(
                index=index,
                job="reserve_board_roles_before_funding_new_control_body",
                deadline=DeadlineClass.RESERVE_NEXT_TURN,
                reason="the control body is already placed and the current target is in prize range; reserve replacement roles before the attachment deadline is spent",
                mode="committed_change" if index != current_index else "committed_confirm",
            )

    # During a persistent attacker-construction objective, an energized Active
    # Impidimp is the only current damage conversion.  Do not spend retreat tempo
    # before completing that attack when no ready Grimmsnarl ex exists.
    if (
        _is_pivot(current)
        and memory.objective_age >= 1
        and ledger.active_id == IMPIDIMP
        and ledger.active_energy == 1
        and ledger.attack_ready_grimmsnarl == 0
    ):
        index = _find(options, _is_attack)
        if index is not None:
            return ContinuityDecision(
                index=index,
                job="complete_basic_attack_before_pivoting_persistent_build_job",
                deadline=DeadlineClass.ENABLE_ATTACK,
                reason="the build objective remains unfinished and the energized Active Basic is the only available damage conversion; attack before spending pivot tempo",
                mode="committed_change" if index != current_index else "committed_confirm",
            )

    # A long-horizon opposing engine makes passive damage a persistent campaign.
    # If the reserved Snorunt can evolve now, complete that job before spending a
    # reusable optional Stadium search.
    if (
        _is_gym(current)
        and ledger.ready_froslass_evolution
        and memory.passive_pressure_campaign
        and (
            opponent.is_repeatable_draw_engine
            or opponent.is_energy_engine
            or opponent.blocks_ex_attack_damage
        )
    ):
        index = _find(options, _is_evolve_froslass)
        if index is not None:
            return ContinuityDecision(
                index=index,
                job="complete_persistent_passive_pressure_before_optional_search",
                deadline=DeadlineClass.ENABLE_ATTACK,
                reason="a multi-turn opposing engine is still active and the reserved Snorunt can complete the passive-pressure job now",
                mode="committed_change" if index != current_index else "committed_confirm",
            )

    # Against hand-size scaling or repeatable energy engines, an open Snorunt slot
    # is a future-turn damage reservation.  Place it before converting the current
    # attack, because the board slot may disappear after the prize transition.
    if (
        _is_attack(current)
        and ledger.bench_slots > 0
        and (opponent.is_hand_size_punisher or opponent.is_energy_engine)
    ):
        index = _find(options, _is_play_snorunt)
        if index is not None:
            return ContinuityDecision(
                index=index,
                job="reserve_passive_pressure_before_prize_conversion",
                deadline=DeadlineClass.RESERVE_NEXT_TURN,
                reason="the current attack remains available, while the open Bench slot for the persistent passive-pressure job may not survive the prize transition",
                mode="committed_change" if index != current_index else "committed_confirm",
            )

    return None
