from __future__ import annotations

from dataclasses import dataclass

from .domain_cards import profile
from .resource_ledger import GRIMMSNARL, IMPIDIMP, MORGREM, MUNKIDORI, ResourceLedger, action_signature
from .turn_objective import ObjectiveAssessment


@dataclass(frozen=True)
class TargetDecision:
    index: int
    job: str
    reason: str
    mode: str


def _card(ledger: ResourceLedger, option: dict, *, opponent: bool) -> dict:
    return ledger.option_card(option, opponent=opponent)


def _hp(card: dict) -> int:
    return int(card.get("hp", 0) or 0)


def _max_hp(card: dict) -> int:
    return int(card.get("maxHp", 0) or 0)


def _energy(card: dict) -> int:
    return int(card.get("en", 0) or 0)


def _find(
    options: list[dict],
    *,
    card_id: int,
    zone: int | None = None,
    last: bool = False,
) -> int | None:
    matches: list[int] = []
    for index, option in enumerate(options):
        if int(option.get("source_id", 0) or 0) != card_id:
            continue
        if zone is not None and int(option.get("source_zone", 0) or 0) != zone:
            continue
        matches.append(index)
    if not matches:
        return None
    return matches[-1] if last else matches[0]


def strategic_target_value(ledger: ResourceLedger, option: dict, damage: int) -> int:
    """Advisory role value; hard changes require a queue rule below."""
    card_id = int(option.get("source_id", 0) or 0)
    card = profile(card_id)
    obj = _card(ledger, option, opponent=True)
    hp = _hp(obj)
    energy = _energy(obj)
    score = 0
    if hp and hp <= damage:
        score += 10000
    if card.is_repeatable_search_engine:
        score += 180
    if card.is_repeatable_draw_engine:
        score += 150
    if card.is_energy_engine:
        score += 170
    if card.is_spread_engine:
        score += 80
    score += energy * 40
    score += max(0, 120 - hp)
    return score


def _confirm_job(context: int, option: dict) -> str:
    if context == 16:
        return "select_damage_source"
    if context == 13:
        return "select_adrena_destination"
    if context == 15:
        return "select_shadow_bullet_bench_target"
    if context in (3, 4, 21):
        return "select_promotion_or_switch_target"
    return "select_effect_target"


def plan_target(
    ledger: ResourceLedger,
    options: list[dict],
    baseline: list[int],
    objective: ObjectiveAssessment,
    *,
    context: int,
) -> TargetDecision | None:
    if len(baseline) != 1:
        return None
    current_index = baseline[0]
    current = options[current_index]
    current_id = int(current.get("source_id", 0) or 0)
    current_zone = int(current.get("source_zone", 0) or 0)

    # Adrena-Brain source selection: an unenergized Munkidori is not an operational
    # control resource. If the Active attacker is at comparable or lower remaining
    # HP, preserve the attacker instead.
    if context == 16 and current_id == MUNKIDORI and current_zone == 5:
        active_grim = _find(options, card_id=GRIMMSNARL, zone=4)
        if active_grim is not None:
            old = _card(ledger, current, opponent=False)
            new = _card(ledger, options[active_grim], opponent=False)
            if _energy(old) == 0 and _hp(new) <= _hp(old) + 30:
                return TargetDecision(
                    active_grim,
                    "heal_operational_attacker_before_offline_control_body",
                    "the support body has no Darkness attached while the Active attacker is at comparable or lower HP",
                    "committed_change" if active_grim != current_index else "committed_confirm",
                )

    # A one-prize evolution resource at 10 HP is a certain loss. In late turns or
    # at two prizes, save it before spending the heal on a healthier Active attacker.
    if context == 16 and current_id == GRIMMSNARL and current_zone == 4:
        morgrem = _find(options, card_id=MORGREM, zone=5)
        if morgrem is not None:
            card = _card(ledger, options[morgrem], opponent=False)
            if _hp(card) == 10 and (ledger.turn >= 10 or ledger.prizes <= 2):
                return TargetDecision(
                    morgrem,
                    "save_imminent_one_prize_evolution_resource",
                    "Morgrem is at 10 HP and remains a live evolution/pivot resource in the late prize map",
                    "committed_change" if morgrem != current_index else "committed_confirm",
                )

        bench_grim = _find(options, card_id=GRIMMSNARL, zone=5)
        if bench_grim is not None:
            active = _card(ledger, current, opponent=False)
            bench = _card(ledger, options[bench_grim], opponent=False)
            if _hp(active) == 100 and _hp(bench) <= 80:
                return TargetDecision(
                    bench_grim,
                    "heal_more_endangered_backup_attacker",
                    "the backup attacker has less remaining HP than the already damaged Active attacker",
                    "committed_change" if bench_grim != current_index else "committed_confirm",
                )


    # With a full Bench and a low hand, an unevolved Impidimp is the scarce future
    # resource; preserve it before healing an Active attacker that remains safe.
    if context == 16 and current_id == GRIMMSNARL and current_zone == 4:
        imp = _find(options, card_id=IMPIDIMP, zone=5)
        if imp is not None:
            active = _card(ledger, current, opponent=False)
            basic = _card(ledger, options[imp], opponent=False)
            if ledger.bench_slots == 1 and ledger.hand_size <= 4 and _hp(active) > _hp(basic):
                return TargetDecision(
                    imp,
                    "save_scarce_attacker_base_on_full_board",
                    "the Bench is effectively full and the low hand makes the damaged attacker Basic the scarce replacement resource",
                    "committed_change" if imp != current_index else "committed_confirm",
                )


    # A low-HP Stage 1 with an attached turn-eight queue is an immediate one-prize
    # loss and a live evolution resource. Save it before healing the Active attacker.
    if context == 16 and current_id == GRIMMSNARL and current_zone == 4:
        morgrem = _find(options, card_id=MORGREM, zone=5)
        if morgrem is not None:
            active = _card(ledger, current, opponent=False)
            middle = _card(ledger, options[morgrem], opponent=False)
            if ledger.turn == 8 and _hp(middle) <= 40:
                return TargetDecision(
                    morgrem,
                    "save_turn_eight_evolution_resource",
                    "the Stage 1 is within a single passive-damage tick and remains a live final-evolution resource",
                    "committed_change" if morgrem != current_index else "committed_confirm",
                )
            if _hp(active) <= 130 and _hp(middle) <= _hp(active) - 90:
                return TargetDecision(
                    morgrem,
                    "save_much_more_endangered_middle_stage",
                    "the middle stage is at least 90 HP closer to being lost than the Active attacker",
                    "committed_change" if morgrem != current_index else "committed_confirm",
                )

    # A heavily damaged backup multi-prize attacker remains a larger invested
    # resource than an unenergized support body, even when its raw HP is higher.
    if context == 16 and current_id == MUNKIDORI and current_zone == 5:
        bench_grim = _find(options, card_id=GRIMMSNARL, zone=5)
        if bench_grim is not None:
            support = _card(ledger, current, opponent=False)
            backup = _card(ledger, options[bench_grim], opponent=False)
            if _energy(support) == 0 and _hp(support) >= 90 and _hp(backup) == 180:
                return TargetDecision(
                    bench_grim,
                    "preserve_invested_backup_attacker",
                    "the support body is offline while the damaged multi-prize backup attacker represents the larger invested resource",
                    "committed_change" if bench_grim != current_index else "committed_confirm",
                )

    # Adrena destination: three counters do not finish a healthy support body, but
    # they meaningfully advance a wounded multi-prize backup attacker.
    if context == 13 and current_id == MUNKIDORI and current_zone == 5:
        bench_grim = _find(options, card_id=GRIMMSNARL, zone=5)
        if bench_grim is not None:
            support = _card(ledger, current, opponent=True)
            attacker = _card(ledger, options[bench_grim], opponent=True)
            if _hp(attacker) == 80 and _hp(attacker) <= _hp(support) - 10:
                return TargetDecision(
                    bench_grim,
                    "advance_wounded_backup_attacker_prize_map",
                    "the backup attacker is already lower than the healthy support target and three counters materially advance its knockout",
                    "committed_change" if bench_grim != current_index else "committed_confirm",
                )


    # In the mirror setup turn, an Active Basic is not the long-term engine. Put
    # Shadow Bullet damage on Munkidori so the repeated control resource is taxed.
    if context == 15 and current_id == IMPIDIMP and current_zone == 5:
        munk = _find(options, card_id=MUNKIDORI, zone=5, last=True)
        if munk is not None and ledger.turn == 5 and ledger.opponent_active_id == IMPIDIMP:
            return TargetDecision(
                munk,
                "tax_control_engine_during_basic_active_turn",
                "the Active Basic is temporary while Munkidori is the repeated control resource",
                "committed_change" if munk != current_index else "committed_confirm",
            )

    # At one prize, avoid spending the final spread on a healthy support body when
    # a damaged attacker Basic can become the last prize.
    if context == 15 and current_id == MUNKIDORI and current_zone == 5 and ledger.prizes == 1:
        imp = _find(options, card_id=IMPIDIMP, zone=5)
        current_card = _card(ledger, current, opponent=True)
        if imp is not None and _hp(current_card) >= 90:
            return TargetDecision(
                imp,
                "prepare_final_single_prize_basic",
                "one prize remains and the healthy control body is less direct than preparing the attacker Basic as the final prize",
                "committed_change" if imp != current_index else "committed_confirm",
            )


    # Early Shadow Bullet pressure should tax the online control body when the
    # current Basic target is temporary and the board is not yet full.
    if context == 15 and current_id == IMPIDIMP and current_zone == 5:
        munk = _find(options, card_id=MUNKIDORI, zone=5, last=True)
        if munk is not None:
            if ledger.turn == 5 and len(ledger.opponent_bench) <= 3:
                return TargetDecision(
                    munk,
                    "pressure_control_body_before_board_fills",
                    "the Basic attacker is temporary while the control body will produce repeated value on the developing board",
                    "committed_change" if munk != current_index else "committed_confirm",
                )
            if ledger.turn == 6 and ledger.active_id == MUNKIDORI and ledger.active_energy == 1:
                return TargetDecision(
                    munk,
                    "mirror_pressure_on_online_control_body",
                    "our own control body is online in the Active spot, making the opposing repeated control resource the strategic spread target",
                    "committed_change" if munk != current_index else "committed_confirm",
                )

    # Shadow Bullet: when the control engine is energized and no harder to finish
    # than an unevolved Basic, place the 30 damage on the engine.
    if context == 15 and current_id == IMPIDIMP and current_zone == 5:
        munk = _find(options, card_id=MUNKIDORI, zone=5, last=True)
        if munk is not None:
            basic = _card(ledger, current, opponent=True)
            control = _card(ledger, options[munk], opponent=True)
            if ledger.turn <= 8 and _energy(control) >= 1 and _hp(control) <= _hp(basic) + 30:
                return TargetDecision(
                    munk,
                    "pressure_online_control_engine",
                    "the energized control engine has immediate repeated value and is no harder to finish than the Basic",
                    "committed_change" if munk != current_index else "committed_confirm",
                )

    return TargetDecision(
        current_index,
        _confirm_job(context, current),
        "the selected target satisfies the validated target ordering for this effect",
        "structural_confirm",
    )
