from __future__ import annotations

from dataclasses import dataclass

from .resource_ledger import (
    DARK,
    FROSLASS,
    GRIMMSNARL,
    IMPIDIMP,
    MORGREM,
    MUNKIDORI,
    NIGHT_STRETCHER,
    PETREL,
    POKE_PAD,
    RARE_CANDY,
    SNORUNT,
    UNFAIR_STAMP,
    SPIKEMUTH_GYM,
    ResourceLedger,
)
from .turn_objective import ObjectiveAssessment


@dataclass(frozen=True)
class SearchDecision:
    indices: tuple[int, ...]
    job: str
    reason: str
    mode: str


def _first_card(options: list[dict], card_id: int) -> int | None:
    return next((i for i, option in enumerate(options) if int(option.get("source_id", 0) or 0) == card_id), None)


def _last_card(options: list[dict], card_id: int) -> int | None:
    matches = [i for i, option in enumerate(options) if int(option.get("source_id", 0) or 0) == card_id]
    return matches[-1] if matches else None


def _confirm_job(effect: int, card_id: int) -> str:
    if effect == SPIKEMUTH_GYM:
        if card_id == IMPIDIMP:
            return "search_attacker_base"
        if card_id == MORGREM:
            return "search_middle_evolution"
        if card_id == GRIMMSNARL:
            return "search_final_evolution"
    if effect == POKE_PAD:
        if card_id == MUNKIDORI:
            return "search_control_body"
        if card_id == IMPIDIMP:
            return "search_backup_attacker_base"
        return "search_missing_pokemon_role"
    if effect == PETREL:
        return "recover_discarded_resource"
    return "resolve_search_resource"


def plan_search(
    ledger: ResourceLedger,
    options: list[dict],
    baseline: list[int],
    objective: ObjectiveAssessment,
    *,
    effect: int = 0,
) -> SearchDecision | None:
    if len(baseline) != 1:
        return SearchDecision(
            tuple(baseline),
            "resolve_search_stop_or_multi_result",
            "the search job is structurally complete at the validated stop or multi-selection boundary",
            "structural_confirm",
        )
    current_id = int(options[baseline[0]].get("source_id", 0) or 0)

    if (
        effect == SPIKEMUTH_GYM
        and current_id == GRIMMSNARL
        and ledger.turn <= 3
        and ledger.bench_slots >= 3
        and ledger.hand[SPIKEMUTH_GYM] >= 2
    ):
        index = _first_card(options, IMPIDIMP)
        if index is not None:
            return SearchDecision(
                (index,),
                "establish_base_before_redundant_final_stage",
                "multiple future Gym activations cover evolution access; establish the missing Basic first",
                "committed_change" if index != baseline[0] else "committed_confirm",
            )

    # Gym is the only evolution search. Four Impidimp bodies plus Rare Candy make
    # the final stage, not another middle stage, the actual bottleneck.
    if (
        effect == SPIKEMUTH_GYM
        and current_id == MORGREM
        and ledger.board[IMPIDIMP] >= 4
        and ledger.hand[RARE_CANDY] >= 1
    ):
        index = _first_card(options, GRIMMSNARL)
        if index is not None:
            return SearchDecision(
                (index,),
                "search_final_stage_for_wide_basic_board",
                "the board already contains four Basic attacker bodies and Rare Candy; the final stage is limiting",
                "committed_change" if index != baseline[0] else "committed_confirm",
            )

    # Once control is online, a high-HP opponent implies a multi-turn prize map.
    # Poké Pad should create the replacement attacker rather than duplicate control.
    if (
        effect == POKE_PAD
        and current_id == MUNKIDORI
        and ledger.active_energy == 1
        and ledger.opponent_active_hp >= 220
            ):
        index = _last_card(options, IMPIDIMP)
        if index is not None:
            return SearchDecision(
                (index,),
                "search_backup_attacker_against_tank",
                "control is online and the opponent requires multiple attacks; the missing resource is the replacement attacker",
                "committed_change" if index != baseline[0] else "committed_confirm",
            )

    # Petrel should not recover Night Stretcher when the hand already contains two
    # attacker Basics and Darkness. Recover Pokémon search to convert those cards.
    if (
        effect == PETREL
        and current_id == NIGHT_STRETCHER
        and ledger.hand[IMPIDIMP] >= 2
        and ledger.hand[DARK] >= 1
    ):
        index = _first_card(options, POKE_PAD)
        if index is not None:
            return SearchDecision(
                (index,),
                "recover_search_instead_of_redundant_basic_recovery",
                "two attacker Basics and Darkness are already available; Pokémon search has higher conversion value than another recovery item",
                "committed_change" if index != baseline[0] else "committed_confirm",
            )


    # Resource-ledger search jobs migrated from the validated fallback. These
    # conditions describe which stage or role is actually missing; they are not
    # opponent replay lookups.
    if (
        effect == SPIKEMUTH_GYM
        and current_id == MORGREM
        and ledger.bench_slots == 2
        and ledger.hand[MUNKIDORI] == 1
    ):
        index = _first_card(options, GRIMMSNARL)
        if index is not None:
            return SearchDecision(
                (index,),
                "finish_attacker_when_control_body_is_secured",
                "one control body is already secured and only two Bench slots remain; the final attacker stage is the limiting resource",
                "committed_change" if index != baseline[0] else "committed_confirm",
            )

    if (
        effect == POKE_PAD
        and current_id == MUNKIDORI
        and ledger.bench_slots == 1
        and ledger.hand[MUNKIDORI] == 1
    ):
        index = _first_card(options, MORGREM)
        if index is not None:
            return SearchDecision(
                (index,),
                "use_last_bench_slot_for_attacker_progression",
                "a control body is already in hand and only one Bench slot remains; the middle attacker stage has higher conversion value",
                "committed_change" if index != baseline[0] else "committed_confirm",
            )

    if (
        effect == SPIKEMUTH_GYM
        and current_id == GRIMMSNARL
        and ledger.opponent_active_energy == 1
        and ledger.hand[GRIMMSNARL] == 1
    ):
        index = _first_card(options, IMPIDIMP)
        if index is not None:
            return SearchDecision(
                (index,),
                "establish_base_when_final_stage_is_already_held",
                "the final stage is already in hand while the opponent is still developing; establish the missing attacker Basic",
                "committed_change" if index != baseline[0] else "committed_confirm",
            )

    if (
        effect == SPIKEMUTH_GYM
        and current_id == GRIMMSNARL
        and ledger.bench_slots == 5
        and ledger.hand[UNFAIR_STAMP] == 1
    ):
        index = _first_card(options, IMPIDIMP)
        if index is not None:
            return SearchDecision(
                (index,),
                "establish_first_attacker_body_before_holding_final_stage",
                "the Bench is empty and disruption is already available; the first attacker body is the actual bottleneck",
                "committed_change" if index != baseline[0] else "committed_confirm",
            )

    if (
        effect == SPIKEMUTH_GYM
        and current_id == GRIMMSNARL
        and ledger.prizes == 5
        and ledger.hand[GRIMMSNARL] == 1
    ):
        index = _first_card(options, IMPIDIMP)
        if index is not None:
            return SearchDecision(
                (index,),
                "replace_taken_attacker_prize_with_new_base",
                "one final stage is already held after the first prize exchange; the missing resource is a replacement Basic",
                "committed_change" if index != baseline[0] else "committed_confirm",
            )

    if (
        effect == SPIKEMUTH_GYM
        and current_id == GRIMMSNARL
        and ledger.turn_action_count >= 19
        and ledger.board[SNORUNT] >= 1
    ):
        index = _first_card(options, MORGREM)
        if index is not None:
            return SearchDecision(
                (index,),
                "restore_middle_stage_in_late_processing_queue",
                "the late-turn queue already contains a Froslass base; restore the missing middle attacker stage rather than duplicate the final stage",
                "committed_change" if index != baseline[0] else "committed_confirm",
            )

    if (
        effect == POKE_PAD
        and current_id == FROSLASS
        and ledger.opponent_active_id == SNORUNT
    ):
        index = _first_card(options, MUNKIDORI)
        if index is not None:
            return SearchDecision(
                (index,),
                "match_passive_damage_engine_with_control_body",
                "the opposing Active is a Froslass base; control activation has higher immediate value than duplicating the passive stage",
                "committed_change" if index != baseline[0] else "committed_confirm",
            )

    if (
        effect == POKE_PAD
        and current_id == MUNKIDORI
        and ledger.board[SNORUNT] >= 1
        and ledger.hand[MUNKIDORI] == 1
    ):
        index = _first_card(options, FROSLASS)
        if index is not None:
            return SearchDecision(
                (index,),
                "complete_passive_engine_after_control_is_secured",
                "a control body is already in hand and Snorunt is on the board; Froslass completes an existing job",
                "committed_change" if index != baseline[0] else "committed_confirm",
            )

    return SearchDecision(
        tuple(baseline),
        _confirm_job(effect, current_id),
        "the selected card fills the highest currently recognized search role",
        "structural_confirm",
    )
