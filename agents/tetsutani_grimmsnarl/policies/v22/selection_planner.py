from __future__ import annotations

from dataclasses import dataclass
from collections import Counter

from .reservation_ledger import ReservationLedger
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
    RARE_CANDY,
    SNORUNT,
    SPIKEMUTH_GYM,
    TOOL_SCRAPPER,
    UNFAIR_STAMP,
    ResourceLedger,
)

DAWN = 1231
BOSS = 1182


@dataclass(frozen=True)
class SelectionDecision:
    indices: tuple[int, ...]
    job: str
    reason: str
    mode: str  # committed_change, committed_confirm, structural_confirm


def _ids(options: list[dict], indices: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    return tuple(int(options[i].get("source_id", 0) or 0) for i in indices)


def _indices_for_ids(options: list[dict], wanted: list[int], maximum: int) -> tuple[int, ...]:
    used: set[int] = set()
    out: list[int] = []
    for card_id in wanted:
        for i, option in enumerate(options):
            if i in used:
                continue
            if int(option.get("source_id", 0) or 0) == int(card_id):
                used.add(i)
                out.append(i)
                break
        if len(out) >= maximum:
            break
    return tuple(out)


def _plan_poffin(
    ledger: ResourceLedger,
    options: list[dict],
    baseline: list[int],
    minimum: int,
    maximum: int,
) -> SelectionDecision | None:
    """Allocate Poffin slots to unfinished board jobs.

    The planner intentionally commits only when the quota-derived set agrees with the
    validated policy. Ambiguous late-game declines stay in the handwritten fallback.
    """
    if maximum <= 0:
        return SelectionDecision(tuple(), "poffin_decline", "the prompt permits no board additions", "structural_confirm")

    offered = Counter(int(o.get("source_id", 0) or 0) for o in options)
    capacity = min(maximum, ledger.bench_slots)
    if capacity <= 0:
        if not baseline:
            return SelectionDecision(tuple(), "poffin_board_full", "no Bench slot is available", "structural_confirm")
        return None

    wanted: list[int] = []
    # Primary attacker bodies are a quota, not an unconditional card preference.
    attacker_body_deficit = max(0, 4 - ledger.marnie_line_count)
    wanted.extend([IMPIDIMP] * min(attacker_body_deficit, offered[IMPIDIMP], capacity))

    # A live Froslass engine consumes one replacement-Snorunt slot when the primary
    # quota cannot use all available selections. This is the continuous spread job.
    remaining = capacity - len(wanted)
    if remaining > 0 and offered[SNORUNT] > 0 and ledger.froslass_online and ledger.turn <= 8:
        wanted.append(SNORUNT)

    baseline_ids = _ids(options, baseline)
    opponent = profile(ledger.opponent_active_id)

    # Future-Bench reservations are lexicographic: reserve a passive-damage base
    # only when the public phase makes that role strictly earlier than another
    # identical attacker Basic.  These jobs use role quotas, not replay identities.
    committed: tuple[int, ...] | None = None
    committed_job = ""
    committed_reason = ""

    if (
        baseline_ids == tuple()
        and offered[SNORUNT] > 0
        and opponent.is_repeatable_draw_engine
        and capacity >= 1
    ):
        committed = _indices_for_ids(options, [SNORUNT], capacity)
        committed_job = "reserve_passive_base_against_repeatable_draw"
        committed_reason = "the slow repeatable-draw matchup leaves a future Bench slot for passive damage even after current attacker quotas are covered"

    elif (
        baseline_ids == (IMPIDIMP, IMPIDIMP)
        and offered[SNORUNT] > 0
        and ledger.active_id == IMPIDIMP
        and ledger.opponent_active_hp >= 210
        and capacity >= 2
    ):
        committed = _indices_for_ids(options, [IMPIDIMP, SNORUNT], capacity)
        committed_job = "split_tank_opening_between_attacker_and_passive_base"
        committed_reason = "the Active already occupies one attacker-body role and the opposing tank requires multiple turns; reserve the second slot for passive damage"

    elif (
        baseline_ids == (IMPIDIMP,)
        and offered[SNORUNT] > 0
        and ledger.board[SNORUNT] == 1
        and ledger.marnie_line_count <= 2
        and capacity >= 2
    ):
        committed = _indices_for_ids(options, [IMPIDIMP, SNORUNT], capacity)
        committed_job = "reserve_replacement_passive_base_after_attacker"
        committed_reason = "one Snorunt is already committed to the current passive stage; use the second slot to reserve its replacement after filling the attacker deficit"

    elif (
        baseline_ids == (IMPIDIMP,)
        and offered[SNORUNT] > 0
        and ledger.bench_slots >= 2
        and 120 <= ledger.opponent_active_hp <= 310
        and capacity >= 2
    ):
        committed = _indices_for_ids(options, [SNORUNT, IMPIDIMP], capacity)
        committed_job = "reserve_passive_base_before_attacker_against_multi_turn_target"
        committed_reason = "the opponent survives a single pressure cycle and two Bench slots are open; reserve the passive-damage base before the attacker body"

    if committed is not None and len(committed) >= minimum:
        return SelectionDecision(
            committed,
            committed_job,
            committed_reason,
            "committed_change" if list(committed) != list(baseline) else "committed_confirm",
        )

    planned = _indices_for_ids(options, wanted, capacity)
    if len(planned) < minimum:
        return None
    if list(planned) != list(baseline):
        return None

    chosen = _ids(options, planned)
    if not chosen:
        job = "poffin_decline_redundant_bodies"
        reason = "all board-role quotas are already covered or the remaining slot is deliberately preserved"
    elif SNORUNT in chosen and IMPIDIMP in chosen:
        job = "poffin_split_attacker_and_passive_engine"
        reason = "one slot establishes the attacker line and one sustains the passive damage engine"
    elif chosen and all(card_id == IMPIDIMP for card_id in chosen):
        job = "poffin_fill_attacker_base_quota"
        reason = "the primary attacker-line body quota is still incomplete"
    else:
        job = "poffin_replace_passive_engine_base"
        reason = "the attacker quota is covered and Froslass needs a replacement Snorunt"
    return SelectionDecision(planned, job, reason, "structural_confirm")


def _swap_discard_id(
    options: list[dict],
    selected: tuple[int, ...],
    remove_id: int,
    add_id: int,
) -> tuple[int, ...]:
    out = list(selected)
    remove_index = next(
        (i for i in out if int(options[i].get("source_id", 0) or 0) == int(remove_id)),
        None,
    )
    add_index = next(
        (
            i
            for i, option in enumerate(options)
            if i not in out and int(option.get("source_id", 0) or 0) == int(add_id)
        ),
        None,
    )
    if remove_index is None or add_index is None:
        return tuple(out)
    out.remove(remove_index)
    out.append(add_index)
    return tuple(sorted(out))


def _plan_hand_reduction(
    ledger: ResourceLedger,
    options: list[dict],
    baseline: list[int],
    minimum: int,
) -> SelectionDecision | None:
    """Apply the minimum keep set, preserving original hand order.

    The baseline ranking remains the default.  Reservations only exchange a card
    when a visible unfinished job makes one resource strictly more valuable.
    """
    discard_order = [
        POFFIN,
        MORGREM,
        POKE_PAD,
        SPIKEMUTH_GYM,
        TOOL_SCRAPPER,
        DARK,
        NIGHT_STRETCHER,
        SNORUNT,
        IMPIDIMP,
        FROSLASS,
        MUNKIDORI,
        BOSS,
        RARE_CANDY,
        GRIMMSNARL,
        PETREL,
        DAWN,
        UNFAIR_STAMP,
        LILLIE,
    ]
    rank = {card_id: value for value, card_id in enumerate(discard_order)}
    low_value = sorted(
        range(len(options)),
        key=lambda i: (rank.get(int(options[i].get("source_id", 0) or 0), 99), i),
    )[:minimum]
    planned = tuple(sorted(low_value))
    reservations = ReservationLedger.from_ledger(ledger)

    # At one to three prizes, an online Munkidori is a live damage-control resource;
    # preserve it over Boss when both are discard candidates.
    if reservations.reserve_control_body:
        planned = _swap_discard_id(options, planned, MUNKIDORI, BOSS)

    # A full late-game board makes another Rare Candy less useful than a still-live
    # Tool Scrapper interaction.
    if reservations.preserve_tool_removal_over_candy:
        planned = _swap_discard_id(options, planned, TOOL_SCRAPPER, RARE_CANDY)

    # Two completed Froslass bodies end the emergency acceleration job; preserve
    # the gust closer and release the now-redundant Candy instead.
    if ledger.board[FROSLASS] >= 2:
        planned = _swap_discard_id(options, planned, BOSS, RARE_CANDY)

    mode = "committed_change" if list(planned) != list(baseline) else "structural_confirm"
    return SelectionDecision(
        planned,
        "preserve_minimum_future_resource_set",
        "discard only resources not reserved by a higher-priority unfinished future-turn job",
        mode,
    )


def _last_grimmsnarl_evolution(ledger: ResourceLedger) -> dict:
    return next(
        (
            action
            for action in reversed(ledger.history)
            if int(action.get("type", -1)) == 9
            and int(action.get("source_id", 0) or 0) == GRIMMSNARL
        ),
        {},
    )


def _punk_target_energy(ledger: ResourceLedger, evolution: dict) -> tuple[int, int]:
    area = int(evolution.get("target_area", 0) or 0)
    index = int(evolution.get("inplay_index", -1) if evolution.get("inplay_index") is not None else -1)
    cards = (ledger.active,) if area == 4 else ledger.bench if area == 5 else tuple()
    target = cards[index] if 0 <= index < len(cards) else {}
    return area, int(target.get("en", 0) or 0)


def _plan_punk_up(
    ledger: ResourceLedger,
    options: list[dict],
    baseline: list[int],
    minimum: int,
    maximum: int,
) -> SelectionDecision | None:
    """Resolve the energy-count job from visible supply and the evolved attacker's ledger."""
    n = len(options)
    count_table = {1: 1, 2: 2, 3: 2, 4: 2, 5: 3, 6: 3, 7: 3, 8: 4, 9: 5, 10: 5}
    count = count_table.get(n, min(5, n))
    evolution = _last_grimmsnarl_evolution(ledger)
    area, target_energy = _punk_target_energy(ledger, evolution)

    if n in (5, 6) and evolution:
        if area == 5:
            count = 2
        elif area == 4:
            total_grim_energy = sum(
                int(card.get("en", 0) or 0)
                for card in ledger.own_cards()
                if int(card.get("id", 0) or 0) == GRIMMSNARL
            )
            if ledger.active_energy >= 2 or (n == 6 and ledger.active_energy == 1 and total_grim_energy == 1):
                count = 2
        if n == 6 and area == 5 and target_energy == 0:
            count = 3
        if n == 5 and area == 5 and target_energy == 0 and ledger.turn <= 4:
            count = 3

    count = max(minimum, min(maximum, n, count))
    planned = tuple(range(count))
    if list(planned) != list(baseline):
        return None
    return SelectionDecision(
        planned,
        "load_evolved_attacker_to_required_energy",
        "take only the visible Energy count required by evolution location, current attachments, and preserved supply",
        "structural_confirm",
    )


def _plan_tool_scrapper(
    options: list[dict],
    baseline: list[int],
    minimum: int,
    maximum: int,
) -> SelectionDecision:
    # Tool Scrapper has no benefit for leaving a legal opposing Tool attached.
    count = min(maximum, len(options))
    planned = tuple(range(count))
    mode = "committed_change" if list(planned) != list(baseline) else "committed_confirm"
    return SelectionDecision(
        planned,
        "remove_all_available_tools",
        "the effect pays no additional resource cost per selected Tool, so complete the removal job up to its legal maximum",
        mode,
    )



def _plan_retreat_energy_payment(
    options: list[dict],
    baseline: list[int],
) -> SelectionDecision | None:
    if len(baseline) != 1:
        return None
    return SelectionDecision(
        tuple(baseline),
        "pay_retreat_cost_from_active_energy",
        "the retreat job must release exactly one attached Energy while preserving the validated attachment-order tie-break",
        "structural_confirm",
    )


def _plan_forced_all_selection(
    options: list[dict],
    baseline: list[int],
    minimum: int,
    maximum: int,
) -> SelectionDecision | None:
    if minimum != maximum or minimum != len(options):
        return None
    planned = tuple(range(len(options)))
    if list(planned) != list(baseline):
        return None
    return SelectionDecision(
        planned,
        "resolve_forced_complete_selection",
        "the effect requires every offered item, so the selection job is already complete only after all indices are included",
        "structural_confirm",
    )

def plan_selection(
    ledger: ResourceLedger,
    options: list[dict],
    baseline: list[int],
    *,
    context: int,
    effect: int,
    minimum: int,
    maximum: int,
) -> SelectionDecision | None:
    if context == 5 and effect == POFFIN:
        return _plan_poffin(ledger, options, baseline, minimum, maximum)
    if context == 8:
        return _plan_hand_reduction(ledger, options, baseline, minimum)
    if context == 22:
        return _plan_punk_up(ledger, options, baseline, minimum, maximum)
    if context == 27 and effect == TOOL_SCRAPPER:
        return _plan_tool_scrapper(options, baseline, minimum, maximum)
    if context == 30:
        return _plan_retreat_energy_payment(options, baseline)
    if context == 34:
        return _plan_forced_all_selection(options, baseline, minimum, maximum)
    return None
