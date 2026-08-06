from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .domain_cards import profile
from .reservation_ledger import ReservationLedger
from .resource_ledger import (
    DARK,
    GRIMMSNARL,
    IMPIDIMP,
    MORGREM,
    MUNKIDORI,
    PETREL,
    POKE_PAD,
    ResourceLedger,
    action_signature,
)


class JobState(str, Enum):
    BLOCKED = "blocked"
    READY = "ready"
    COMPLETE = "complete"


@dataclass(frozen=True)
class DAGNode:
    name: str
    prerequisites: tuple[str, ...]
    state: JobState
    reason: str


@dataclass(frozen=True)
class DAGDecision:
    index: int
    job: str
    reason: str
    mode: str


def _find(options: list[dict], predicate) -> int | None:
    return next((i for i, option in enumerate(options) if predicate(option)), None)


def _is_play(option: dict, card_id: int) -> bool:
    return int(option.get("type", -1)) == 7 and int(option.get("source_id", 0) or 0) == card_id


def _is_attack(option: dict) -> bool:
    return int(option.get("type", -1)) == 13


def _is_gym(option: dict) -> bool:
    return action_signature(option) == (10, 0, 7, 0, 7, 0)


def _is_attach_to(option: dict, card_id: int, area: int | None = None) -> bool:
    if int(option.get("type", -1)) != 8:
        return False
    if int(option.get("source_id", 0) or 0) != DARK:
        return False
    if int(option.get("target_id", 0) or 0) != card_id:
        return False
    return area is None or int(option.get("target_area", 0) or 0) == area


def _action_job(option: dict) -> str:
    t = int(option.get("type", -1))
    sid = int(option.get("source_id", 0) or 0)
    target = int(option.get("target_id", 0) or 0)
    if t == 14:
        return "close_turn"
    if t == 13:
        return "convert_attack"
    if t == 12:
        return "pivot_active"
    if t == 10:
        return "search_evolution_resource"
    if t == 9:
        return "complete_evolution_job"
    if t == 8 and target == MUNKIDORI:
        return "enable_control"
    if t == 8:
        return "reserve_energy_for_attacker"
    if t == 7:
        return {
            PETREL: "recover_spent_resources",
            POKE_PAD: "search_missing_pokemon_role",
            IMPIDIMP: "establish_attacker_base",
            MUNKIDORI: "establish_control_body",
        }.get(sid, "execute_visible_card_resource")
    return "resolve_protocol_action"


def build_turn_dag(ledger: ResourceLedger, reservations: ReservationLedger) -> tuple[DAGNode, ...]:
    nodes = [
        DAGNode(
            "primary_attacker_ready",
            (),
            JobState.COMPLETE if ledger.attack_ready_grimmsnarl > 0 else JobState.READY,
            "an attack-ready Grimmsnarl ex is required before damage conversion",
        ),
        DAGNode(
            "control_online",
            (),
            JobState.COMPLETE if ledger.control_online else JobState.READY,
            "one energized Munkidori completes the damage-control resource pair",
        ),
        DAGNode(
            "attacker_base_reserved",
            ("control_online",),
            JobState.READY if reservations.reserve_attacker_base else JobState.COMPLETE,
            "after control is committed, the next empty role is an attacker Basic",
        ),
        DAGNode(
            "backup_attacker_energy_reserved",
            ("primary_attacker_ready",),
            JobState.READY
            if ledger.attack_ready_grimmsnarl == 1 and not ledger.energy_attached
            else JobState.COMPLETE,
            "the only ready attacker exists; unused manual attachment belongs to the replacement attacker",
        ),
        DAGNode(
            "post_pivot_evolution_access",
            (),
            JobState.READY
            if ledger.retreated and ledger.board[MORGREM] == 1 and ledger.attack_ready_grimmsnarl <= 1
            else JobState.COMPLETE,
            "after pivoting, the visible middle stage is the next evolution bottleneck",
        ),
        DAGNode(
            "hand_punisher_recovery",
            (),
            JobState.READY
            if profile(ledger.opponent_active_id).is_hand_size_punisher
            and ledger.stadium_played
            and ledger.recoverable_core > 0
            else JobState.COMPLETE,
            "after committing the stadium against hand-size scaling damage, recover spent resources before optional search",
        ),
    ]
    return tuple(nodes)


def plan_turn_dag(
    ledger: ResourceLedger,
    options: list[dict],
    current_indices: list[int],
) -> DAGDecision | None:
    if len(current_indices) != 1:
        return None
    current_index = current_indices[0]
    current = options[current_index]
    reservations = ReservationLedger.from_ledger(ledger)
    nodes = {node.name: node for node in build_turn_dag(ledger, reservations)}

    # A newly completed attacker exists, but the manual attachment is still unused.
    # Finish the replacement-attacker energy reservation before converting the attack.
    if (
        _is_attack(current)
        and ledger.last_action_type == 1
        and ledger.previous_action_type == 9
        and nodes["backup_attacker_energy_reserved"].state == JobState.READY
    ):
        index = _find(options, lambda o: _is_attach_to(o, GRIMMSNARL, 5))
        if index is not None:
            return DAGDecision(
                index,
                "reserve_manual_attachment_for_backup_before_attack",
                nodes["backup_attacker_energy_reserved"].reason,
                "committed_change" if index != current_index else "committed_confirm",
            )

    # Once Darkness was committed to the control job, do not create another control
    # body while the primary attacker-base reservation is still unfinished.
    if (
        _is_play(current, MUNKIDORI)
        and ledger.previous_action_type == 8
        and ledger.previous_action_source == DARK
        and ledger.previous_action_target == MUNKIDORI
        and not ledger.stadium_played
        and ledger.marnie_line_count < 2
    ):
        index = _find(options, lambda o: _is_play(o, IMPIDIMP))
        if index is not None:
            return DAGDecision(
                index,
                "fill_attacker_base_after_control_commit",
                "the control resource was just funded; the remaining open role is the primary attacker base",
                "committed_change" if index != current_index else "committed_confirm",
            )

    # A pivot spends tempo.  When exactly one visible middle stage remains, resolve
    # its deterministic evolution access before using a generic Pokémon search.
    if _is_play(current, POKE_PAD) and nodes["post_pivot_evolution_access"].state == JobState.READY:
        index = _find(options, _is_gym)
        if index is not None:
            return DAGDecision(
                index,
                "resolve_post_pivot_evolution_bottleneck",
                nodes["post_pivot_evolution_access"].reason,
                "committed_change" if index != current_index else "committed_confirm",
            )

    # Against hand-size scaling attackers, the stadium commit defines a control
    # phase.  Recover the spent core before optional evolution search.
    if _is_gym(current) and nodes["hand_punisher_recovery"].state == JobState.READY:
        index = _find(options, lambda o: _is_play(o, PETREL))
        if index is not None:
            return DAGDecision(
                index,
                "recover_core_inside_hand_control_phase",
                nodes["hand_punisher_recovery"].reason,
                "committed_change" if index != current_index else "committed_confirm",
            )

    return DAGDecision(
        current_index,
        _action_job(current),
        "the action is a legal executable node in the turn dependency graph",
        "structural_confirm",
    )
