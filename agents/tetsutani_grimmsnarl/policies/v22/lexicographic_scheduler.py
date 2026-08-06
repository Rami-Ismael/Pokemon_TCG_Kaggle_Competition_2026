from __future__ import annotations

from dataclasses import dataclass

from .deadline_ledger import DeadlineClass, DeadlineLedger
from .resource_ledger import (
    BOSS,
    DARK,
    GRIMMSNARL,
    IMPIDIMP,
    MUNKIDORI,
    PETREL,
    POFFIN,
    POKE_PAD,
    ResourceLedger,
    action_signature,
)


@dataclass(frozen=True)
class ScheduleDecision:
    index: int
    job: str
    deadline: DeadlineClass
    reason: str
    mode: str


def _find(options: list[dict], predicate) -> int | None:
    return next((i for i, option in enumerate(options) if predicate(option)), None)


def _is_play(option: dict, card_id: int) -> bool:
    return int(option.get("type", -1)) == 7 and int(option.get("source_id", 0) or 0) == card_id


def _is_gym(option: dict) -> bool:
    return action_signature(option) == (10, 0, 7, 0, 7, 0)


def _is_attack(option: dict) -> bool:
    return int(option.get("type", -1)) == 13


def _is_attach_bench_grim(option: dict) -> bool:
    return (
        int(option.get("type", -1)) == 8
        and int(option.get("source_id", 0) or 0) == DARK
        and int(option.get("target_id", 0) or 0) == GRIMMSNARL
        and int(option.get("target_area", 0) or 0) == 5
    )


def plan_lexicographic_deadline(
    ledger: ResourceLedger,
    options: list[dict],
    current_indices: list[int],
    deadlines: DeadlineLedger,
) -> ScheduleDecision | None:
    if len(current_indices) != 1:
        return None
    current_index = current_indices[0]
    current = options[current_index]

    # 1. An immunity lock invalidates the normal attack/search queue.  Gusting a
    # vulnerable target is the only available immediate conversion job.
    if _is_gym(current) and deadlines.bypass_ex_immunity:
        index = _find(options, lambda option: _is_play(option, BOSS))
        if index is not None:
            return ScheduleDecision(
                index,
                "bypass_ex_immunity_before_optional_search",
                DeadlineClass.IMMEDIATE_PRIZE_OR_LOCK,
                "direct ex damage is blocked; resolve the gust bypass before spending the optional evolution search",
                "committed_change" if index != current_index else "committed_confirm",
            )

    # 2. Manual attachment expires at turn end.  Punk Up has left the replacement
    # attacker one Energy short, so complete it before converting the current attack.
    if _is_attack(current) and deadlines.finish_backup_attachment_before_attack:
        index = _find(options, _is_attach_bench_grim)
        if index is not None:
            return ScheduleDecision(
                index,
                "finish_backup_attacker_before_attack",
                DeadlineClass.EXPIRES_THIS_TURN,
                "the replacement attacker is one Energy short and the unused manual attachment expires this turn",
                "committed_change" if index != current_index else "committed_confirm",
            )

    # 3. A nearly defeated opposing Active creates a public prize-transition
    # deadline.  Reserve the next attacker Basic before adding a redundant control body.
    if _is_play(current, MUNKIDORI) and deadlines.reserve_attacker_before_prize_transition:
        index = _find(options, lambda option: _is_play(option, IMPIDIMP))
        if index is not None:
            return ScheduleDecision(
                index,
                "reserve_attacker_before_prize_transition",
                DeadlineClass.RESERVE_NEXT_TURN,
                "the current opposing Active is already in conversion range; reserve the next attacker line before duplicating control",
                "committed_change" if index != current_index else "committed_confirm",
            )

    # 4. In the first development window against a multi-attack target, Poffin
    # creates two reserved board roles while Pad creates only one.
    if _is_play(current, POKE_PAD) and deadlines.establish_two_role_opening:
        index = _find(options, lambda option: _is_play(option, POFFIN))
        if index is not None:
            return ScheduleDecision(
                index,
                "establish_two_role_opening_before_single_search",
                DeadlineClass.ENABLE_ATTACK,
                "the opening board has two free role slots and the opponent requires multiple attacks; establish both jobs before single-card search",
                "committed_change" if index != current_index else "committed_confirm",
            )

    # 5. After an evolution commit in a slow control matchup, a live recovery
    # supporter has a turn deadline, while the Stadium search remains reusable.
    if _is_gym(current) and deadlines.recover_after_evolution_commit:
        index = _find(options, lambda option: _is_play(option, PETREL))
        if index is not None:
            return ScheduleDecision(
                index,
                "recover_core_after_evolution_commit",
                DeadlineClass.EXPIRES_THIS_TURN,
                "the evolution job is complete and the recovery supporter expires this turn; the Stadium search can remain queued",
                "committed_change" if index != current_index else "committed_confirm",
            )

    return None
