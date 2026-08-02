"""Official Mega Lucario ex sample policy + smart bench integration.

Ports the competition-provided Lucario reference agent (attack plan, supporter
timing, Solrock/Lunatone/Hariyama lines) and layers our bench-depth guard:
  - always keep >= 1 backup basic (empty bench → instant loss on active KO)
  - at most 2 voluntary bench basics (bench snipe / gust exposure)
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass

from agent.agent import HeuristicScorer
from agent.deck_tech import LUCARIO_TECH
from agent.matchup_levers import (
    GUST_SETUP_IDS,
    LUCARIO_LEVERS,
    SNOVER_ID,
    TREVENANT_REVENGE_SETUP_IDS,
    LeverDeltas,
    levers_for_lucario,
    merge_lever_table,
)
from agent.smart_bench import (
    MAX_VOLUNTARY_BENCH,
    bench_counts,
    setup_bench_target_count,
    should_play_basic_to_bench,
    worthwhile_second_bench,
)

# --- deck constants (official sample IDs) ------------------------------------
Makuhita = 673
Hariyama = 674
Lunatone = 675
Solrock = 676
Riolu = 677
Mega_Lucario_ex = 678
Dusk_Ball = 1102
Switch = 1123
Premium_Power_Pro = 1141
Fighting_Gong = 1142
Poke_Pad = 1152
Hero_Cape = 1159
Boss_Orders = 1182
Carmine = 1192
Lillie_Determination = 1227
Gravity_Mountain = 1252
Basic_Fighting_Energy = 6
ATTACK_AURA_JAB = 982
ATTACK_MEGA_BRAVE = 983
LUMIOSE_CITY = 1267

_CARD_TABLE = None


def _load_api():
    import cg.api as api

    global _CARD_TABLE
    if _CARD_TABLE is None:
        _CARD_TABLE = {c.cardId: c for c in api.all_card_data()}
    return api


def _enum_val(x):
    return x.value if hasattr(x, "value") else x


@dataclass
class LucarioPlan:
    attacker: int = -1
    target: int = -1
    attack_index: int = -1
    remain_hp: int = -1
    energy: bool = False


class LucarioScorer(HeuristicScorer):
    """Competition reference Lucario brain with smart bench caps."""

    def __init__(
        self,
        rng=None,
        deck_path: str | None = None,
        lever_overrides: dict[str, LeverDeltas] | None = None,
    ) -> None:
        super().__init__(rng=rng)
        self._fallback = HeuristicScorer(rng=rng)
        self._plan = LucarioPlan()
        self._plan_turn = -1
        self._ability_used = False
        self._deck_path = deck_path
        self._lever_table = merge_lever_table(lever_overrides)

    def rank_options(self, obs_dict, select, current, options) -> list[int]:
        """All option indices sorted best-first (same scoring as choose)."""
        ranked = self._compute_ranked(obs_dict, select, current, options)
        if ranked is not None:
            return ranked
        return list(range(len(options)))

    def choose(self, obs_dict, select, current, options):
        if not options:
            return []
        try:
            ranked = self._compute_ranked(obs_dict, select, current, options)
            if ranked is None:
                return self._fallback.choose(obs_dict, select, current, options)

            api = _load_api()
            obs = api.to_observation_class(obs_dict)
            sel = obs.select
            state = obs.current
            min_count = int(getattr(sel, "minCount", 0) or 0)
            max_count = int(getattr(sel, "maxCount", len(sel.option)) or len(sel.option))

            sc = _enum_val(sel.context)
            if sc == api.SelectContext.SETUP_BENCH_POKEMON.value:
                target = setup_bench_target_count(
                    ranked,
                    sel.option,
                    current,
                    LUCARIO_TECH,
                    min_count,
                    max_count,
                )
                if min_count <= 0 and bench_counts(current)[0] == 0 and ranked:
                    target = max(1, target)
                return ranked[: min(target, max_count)]

            if sc == api.SelectContext.MAIN.value and ranked:
                best = sel.option[ranked[0]]
                if _enum_val(best.type) == api.OptionType.ABILITY.value:
                    card = self._get_card(obs, best.area, best.index, state.yourIndex)
                    if card is not None and card.id == Lunatone:
                        self._ability_used = True

            count = max(min_count, 1) if min_count > 0 else max_count
            return ranked[: min(count, max_count)]
        except Exception:
            return self._fallback.choose(obs_dict, select, current, options)

    def _compute_ranked(self, obs_dict, select, current, options) -> list[int] | None:
        if not options:
            return []
        try:
            api = _load_api()
            obs = api.to_observation_class(obs_dict)
            if obs.select is None or not obs.select.option:
                return None

            state = obs.current
            sel = obs.select
            if state.turn != self._plan_turn:
                self._plan_turn = state.turn
                self._plan = LucarioPlan()
                self._ability_used = False

            ctx = self._scan(obs)
            if _enum_val(sel.context) == api.SelectContext.MAIN.value:
                self._build_plan(obs, ctx)

            scores = [self._score_option(obs, ctx, o) for o in sel.option]
            return [
                i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            ]
        except Exception:
            return None

    # --- scan ----------------------------------------------------------------
    def _scan(self, obs):
        api = _load_api()
        state = obs.current
        my_index = state.yourIndex
        my_state = state.players[my_index]

        field_counts: dict[int, int] = defaultdict(int)
        hand_counts: dict[int, int] = defaultdict(int)
        discard_counts: dict[int, int] = defaultdict(int)
        attacker1 = False
        attacker2 = False
        riolu_min_energy = 99
        mega_min_energy = 99
        has_riolu_in_play = False
        has_mega_lucario = False
        line_needs_energy = False
        op_stage2_count = 0

        for card in my_state.active + my_state.bench:
            if card is None:
                continue
            field_counts[card.id] += 1
            nrg = len(card.energies)
            if card.id == Riolu:
                has_riolu_in_play = True
                riolu_min_energy = min(riolu_min_energy, nrg)
                if nrg < 2:
                    line_needs_energy = True
            elif card.id == Mega_Lucario_ex:
                has_mega_lucario = True
                mega_min_energy = min(mega_min_energy, nrg)
                if nrg < 2:
                    line_needs_energy = True
            if card.id in (Makuhita, Hariyama):
                if len(card.energies) >= 3:
                    attacker2 = True
            elif card.id in (Riolu, Mega_Lucario_ex):
                if len(card.energies) >= 2:
                    attacker1 = True

        op_state = state.players[1 - my_index]
        card_table = _CARD_TABLE or {}
        op_board_ids: set[int] = set()
        for card in op_state.active + op_state.bench:
            if card is None:
                continue
            op_board_ids.add(card.id)
            data = card_table.get(card.id)
            if data is not None and data.stage2:
                op_stage2_count += 1

        levers = levers_for_lucario(op_board_ids, lever_table=self._lever_table)

        for card in my_state.hand:
            hand_counts[card.id] += 1
        for card in my_state.discard:
            discard_counts[card.id] += 1

        stadium_id = 0
        for card in state.stadium:
            stadium_id = card.id

        can_attack = False
        can_switch = False
        can_op_switch = False
        can_use_mega_brave = False
        sel = obs.select
        if _enum_val(sel.context) == api.SelectContext.MAIN.value:
            for o in sel.option:
                ot = _enum_val(o.type)
                if ot == api.OptionType.PLAY.value:
                    card = self._get_card(obs, api.AreaType.HAND, o.index, my_index)
                    if card is not None:
                        if card.id == Switch:
                            can_switch = True
                        elif card.id == Boss_Orders:
                            can_op_switch = True
                elif ot == api.OptionType.EVOLVE.value:
                    card = self._get_card(obs, api.AreaType.HAND, o.index, my_index)
                    if card is not None and card.id == Hariyama:
                        can_op_switch = True
                elif ot == api.OptionType.RETREAT.value:
                    can_switch = True
                elif ot == api.OptionType.ATTACK.value:
                    can_attack = True
                    if o.attackId == ATTACK_MEGA_BRAVE:
                        can_use_mega_brave = True

        return {
            "field_counts": field_counts,
            "hand_counts": hand_counts,
            "discard_counts": discard_counts,
            "attacker1": attacker1,
            "attacker2": attacker2,
            "stadium_id": stadium_id,
            "can_attack": can_attack,
            "can_switch": can_switch,
            "can_op_switch": can_op_switch,
            "can_use_mega_brave": can_use_mega_brave,
            "my_prize": len(my_state.prize),
            "deck_count": int(getattr(my_state, "deckCount", 60) or 60),
            "riolu_line": field_counts[Riolu] + field_counts[Mega_Lucario_ex],
            "has_riolu_in_play": has_riolu_in_play,
            "riolu_min_energy": riolu_min_energy if has_riolu_in_play else 0,
            "has_mega_lucario": has_mega_lucario,
            "mega_min_energy": mega_min_energy if has_mega_lucario else 0,
            "line_needs_energy": line_needs_energy,
            "has_solrock_lunatone_engine": (
                field_counts[Solrock] >= 1 and field_counts[Lunatone] >= 1
            ),
            "discard_energy": discard_counts[Basic_Fighting_Energy],
            "op_stage2_count": op_stage2_count,
            "op_board_ids": op_board_ids,
            "levers": levers,
        }

    # --- attack plan (official sample) ---------------------------------------
    def _build_plan(self, obs, ctx) -> None:
        api = _load_api()
        state = obs.current
        my_index = state.yourIndex
        my_state = state.players[my_index]
        op_state = state.players[1 - my_index]
        sel = obs.select

        if state.turn < 2:
            return

        my_cards = [my_state.active[0]]
        for pokemon in my_state.bench:
            my_cards.append(pokemon)
        op_cards = [op_state.active[0]]
        for pokemon in op_state.bench:
            op_cards.append(pokemon)

        field_counts = ctx["field_counts"]
        hand_counts = ctx["hand_counts"]
        discard_counts = ctx["discard_counts"]
        can_switch = ctx["can_switch"]
        can_op_switch = ctx["can_op_switch"]
        can_use_mega_brave = ctx["can_use_mega_brave"]
        my_prize = ctx["my_prize"]
        levers = ctx.get("levers")

        best_score = -1.0
        plan = LucarioPlan()
        card_table = _CARD_TABLE or {}

        for i, my_pokemon in enumerate(my_cards):
            if my_pokemon is None:
                continue
            if i != 0 and not can_switch:
                break
            for a in range(2):
                energy_required = 0
                base_damage = 0
                base_score = 0.0
                if my_pokemon.id == Mega_Lucario_ex:
                    if a == 0:
                        energy_required = 1
                        base_damage = 130
                        base_score += 60 * min(3, discard_counts[Basic_Fighting_Energy])
                        if discard_counts[Basic_Fighting_Energy] >= 1:
                            base_score += 120.0
                        if ctx.get("has_solrock_lunatone_engine"):
                            base_score += 80.0
                    else:
                        energy_required = 2
                        base_damage = 270
                    if my_prize in (2, 3):
                        base_score -= 500
                elif a == 1:
                    break
                elif my_pokemon.id == Hariyama:
                    energy_required = 3
                    base_damage = 210
                    base_score += 180.0
                elif my_pokemon.id == Makuhita:
                    evolve_ok = False
                    for o in sel.option:
                        if _enum_val(o.type) != api.OptionType.EVOLVE.value:
                            continue
                        index = o.inPlayIndex
                        if _enum_val(o.inPlayArea) == api.AreaType.BENCH.value:
                            index += 1
                        if index == i:
                            evolve_ok = True
                            break
                    if not evolve_ok:
                        break
                    base_score -= 100
                    energy_required = 3
                    base_damage = 210
                elif my_pokemon.id == Solrock:
                    if field_counts[Lunatone] >= 1:
                        energy_required = 1
                        base_damage = 70
                        base_score += 220.0
                else:
                    continue

                if base_damage <= 0:
                    continue

                energy_count = len(my_pokemon.energies)
                more_energy = False
                if a == 1 and i == 0 and energy_count >= 2 and not can_use_mega_brave:
                    break
                if energy_count < energy_required:
                    if hand_counts[Basic_Fighting_Energy] >= 1 and not state.energyAttached:
                        energy_count += 1
                        if energy_count < energy_required:
                            continue
                        more_energy = True
                    else:
                        continue

                for j, op_pokemon in enumerate(op_cards):
                    if op_pokemon is None:
                        continue
                    if j != 0 and not can_op_switch:
                        break
                    damage = base_damage
                    data = card_table.get(op_pokemon.id)
                    if data is not None:
                        if _enum_val(data.weakness) == api.EnergyType.FIGHTING.value:
                            damage *= 2
                        elif _enum_val(data.resistance) == api.EnergyType.FIGHTING.value:
                            damage -= 30
                    op_prize = self._prize_count(op_pokemon)
                    score = self._pokemon_score(op_pokemon)
                    if op_pokemon.hp <= damage:
                        prize = op_prize
                    else:
                        prize = 0
                        score *= damage / op_pokemon.hp
                    line_bonus = base_score
                    # Meta: single-prize targets → Solrock/Hariyama; ex → Mega Brave.
                    if my_pokemon.id == Mega_Lucario_ex and a == 1:
                        if op_prize <= 1 and op_pokemon.hp > 130:
                            if levers and levers.skip_mega_brave_vs_bulk_single_prize > 0:
                                continue
                            continue
                        if op_pokemon.id == SNOVER_ID and levers and levers.skip_mega_brave_vs_bulk_single_prize > 0:
                            continue
                        if op_prize >= 2:
                            line_bonus += 450.0 + (levers.mega_brave_vs_ex if levers else 0.0)
                    elif my_pokemon.id == Mega_Lucario_ex and a == 0 and op_prize >= 2:
                        if op_pokemon.hp <= 130:
                            line_bonus += 300.0
                    elif my_pokemon.id == Solrock and op_prize <= 1 and op_pokemon.hp <= 90:
                        line_bonus += 350.0 + (levers.solrock_vs_single_prize if levers else 0.0)
                    elif my_pokemon.id == Hariyama and op_prize >= 2:
                        line_bonus += 200.0 + (levers.hariyama_vs_ex if levers else 0.0)
                    if levers and j >= 1 and op_pokemon.id in GUST_SETUP_IDS:
                        line_bonus += levers.gust_setup_pokemon
                    if (
                        levers
                        and levers.avoid_ko_trevenant_setup > 0
                        and op_pokemon.id in TREVENANT_REVENGE_SETUP_IDS
                        and op_pokemon.hp <= damage
                    ):
                        line_bonus -= levers.avoid_ko_trevenant_setup
                    score += line_bonus
                    if len(op_state.prize) <= prize:
                        score = 50000.0
                    if i == 0:
                        score += 220
                    if j == 0:
                        score += 300
                    score += energy_count
                    if best_score < score:
                        best_score = score
                        plan = LucarioPlan(i, j, a, op_pokemon.hp - damage, more_energy)

        self._plan = plan

    # --- scoring -------------------------------------------------------------
    def _score_option(self, obs, ctx, o) -> float:
        api = _load_api()
        state = obs.current
        sel = obs.select
        my_index = state.yourIndex
        context = _enum_val(sel.context)
        plan = self._plan
        field_counts = ctx["field_counts"]
        hand_counts = ctx["hand_counts"]
        stadium_id = ctx["stadium_id"]
        can_attack = ctx["can_attack"]
        attacker1 = ctx["attacker1"]
        attacker2 = ctx["attacker2"]

        ot = _enum_val(o.type)
        score = 0.0

        if ot == api.OptionType.NUMBER.value:
            return float(o.number or 0)
        if ot == api.OptionType.YES.value:
            return 1.0

        if ot == api.OptionType.CARD.value:
            card = self._get_card(obs, o.area, o.index, o.playerIndex)
            if card is None:
                return score
            energy_count = len(card.energies) if hasattr(card, "energies") else 0

            if context in (
                api.SelectContext.SWITCH.value,
                api.SelectContext.TO_ACTIVE.value,
            ):
                if o.playerIndex == my_index:
                    score += energy_count * 2
                    if o.index == plan.attacker - 1:
                        score += 100
                    if card.id == Mega_Lucario_ex:
                        score += 8 if ctx["my_prize"] in (2, 3) else 20
                    elif card.id == Hariyama and energy_count >= 2:
                        score += 15
                    elif card.id == Makuhita and energy_count >= 2:
                        score += 10
                    elif card.id == Solrock:
                        score += 5
                    elif card.id == Riolu:
                        score += 4
                elif o.index == plan.target - 1:
                    score += 100
            elif context == api.SelectContext.SETUP_ACTIVE_POKEMON.value:
                levers = ctx.get("levers")
                if card.id == Solrock:
                    score = 2.0 if state.firstPlayer == my_index else 4.0
                    if levers and levers.prefer_solrock_open > 0:
                        score += levers.prefer_solrock_open * 15.0
                elif card.id == Riolu:
                    score = 3.0
                    if levers and levers.prefer_riolu_open > 0:
                        score += levers.prefer_riolu_open * 15.0
                elif card.id == Makuhita:
                    score = 1.0
            elif context == api.SelectContext.SETUP_BENCH_POKEMON.value:
                score = self._setup_bench_score(
                    card.id, field_counts, state, my_index, ctx.get("levers"),
                )
            elif context == api.SelectContext.TO_HAND.value:
                score = self._to_hand_score(
                    card.id, field_counts, hand_counts, state
                )
            elif context == api.SelectContext.ATTACH_FROM.value:
                active = _enum_val(o.area) == api.AreaType.ACTIVE.value
                score = self._energy_score(
                    card, active, attacker1, attacker2, state=state, ctx=ctx,
                )
            return score

        if ot == api.OptionType.PLAY.value:
            card = self._get_card(obs, api.AreaType.HAND, o.index, my_index)
            if card is None:
                return score
            data = _CARD_TABLE.get(card.id) if _CARD_TABLE else None
            if data is not None and _enum_val(data.cardType) == api.CardType.POKEMON.value:
                return self._score_play_pokemon(obs, card.id, field_counts)
            return self._score_play_trainer(
                card.id, plan, can_attack, hand_counts, stadium_id, state, ctx
            )

        if ot == api.OptionType.ATTACH.value:
            card = self._get_card(obs, api.AreaType.HAND, o.index, my_index)
            pokemon = self._get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            if card is None or pokemon is None:
                return score
            if card.id == Hero_Cape:
                score = 7000.0
                if pokemon.id == Riolu:
                    score += 100
                elif pokemon.id == Mega_Lucario_ex:
                    score += 200
            else:
                active = _enum_val(o.inPlayArea) == api.AreaType.ACTIVE.value
                score = float(self._energy_score(
                    pokemon, active, attacker1, attacker2, state=state, ctx=ctx,
                ))
                missing = max(0, 2 - len(pokemon.energies))
                if pokemon.id in (Riolu, Mega_Lucario_ex) and missing > 0:
                    score += 280.0 * missing
                    if state.turn <= 8:
                        score += 120.0 * missing
                if active:
                    if plan.attacker == 0 and plan.energy:
                        score += 200
                elif plan.attacker == 1 + o.inPlayIndex and plan.energy:
                    score += 200
                if pokemon.id == Riolu and len(pokemon.energies) + 1 >= 2:
                    score += 400.0
            return score

        if ot == api.OptionType.EVOLVE.value:
            pokemon = self._get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            if pokemon is None:
                return 9000.0
            score = 9000.0 + len(pokemon.energies)
            evo_card = self._get_card(obs, api.AreaType.HAND, o.index, my_index)
            if pokemon.id == Makuhita and plan.target == 0:
                return -1.0
            if evo_card is not None and evo_card.id == Hariyama and pokemon.id == Makuhita:
                if plan.target >= 1:
                    score += 2200.0
            if pokemon.id == Riolu and len(pokemon.energies) >= 2:
                score += 2500.0
            return score

        if ot == api.OptionType.ABILITY.value:
            card = self._get_card(obs, o.area, o.index, my_index)
            if card is not None and card.id == LUMIOSE_CITY:
                return 1.0
            if card is not None and card.id == Lunatone and field_counts[Solrock] >= 1:
                if ctx["deck_count"] <= 10:
                    return -1.0
                if ctx["deck_count"] <= 15:
                    return 4000.0
                if ctx["discard_energy"] <= 1 or state.turn <= 8:
                    return 35000.0
                return 28000.0
            return 30000.0

        if ot == api.OptionType.RETREAT.value:
            if (
                ctx["has_mega_lucario"]
                and not ctx["can_use_mega_brave"]
                and plan.attack_index == 1
            ):
                return 7800.0
            return 2000.0 if plan.attacker >= 1 else -1.0

        if ot == api.OptionType.ATTACK.value:
            my_state = state.players[my_index]
            bench_count = len([p for p in my_state.bench if p is not None])
            if bench_count == 0 and plan.remain_hp > 0:
                return -1.0
            score = 1000.0
            if plan.attack_index == 1:
                if o.attackId == ATTACK_MEGA_BRAVE:
                    score += 100
            elif plan.attack_index == 0 and o.attackId == ATTACK_AURA_JAB:
                score += 120
                if ctx["discard_energy"] >= 1:
                    score += 80
            elif o.attackId != ATTACK_MEGA_BRAVE:
                score += 100
            return score

        return score

    def _score_play_pokemon(self, obs, card_id: int, field_counts) -> float:
        """Official PLAY scoring + smart bench depth."""
        state = obs.current
        my_state = state.players[state.yourIndex]
        bench_count = len([p for p in my_state.bench if p is not None])
        active = my_state.active[0] if my_state.active else None
        active_id = active.id if active is not None else 0

        if bench_count == 0:
            if card_id in (Lunatone, Solrock):
                if field_counts[card_id] >= 1:
                    return -1.0
                if card_id == Lunatone and field_counts[Solrock] >= 1:
                    return 46000.0
                if card_id == Solrock and field_counts[Lunatone] >= 1:
                    return 44000.0
            if card_id == Riolu:
                if field_counts[Riolu] + field_counts[Mega_Lucario_ex] >= 2:
                    return -1.0
            return 45000.0

        if bench_count >= MAX_VOLUNTARY_BENCH:
            return -1.0

        # Official duplicate / line caps before allowing a 2nd bench basic.
        if card_id in (Lunatone, Solrock):
            if field_counts[card_id] >= 1:
                return -1.0
            if card_id == Lunatone and field_counts[Solrock] >= 1 and state.turn <= 10:
                return 24000.0
            if card_id == Solrock and field_counts[Lunatone] >= 1 and state.turn <= 10:
                return 22000.0
        elif card_id == Riolu:
            if field_counts[Riolu] + field_counts[Mega_Lucario_ex] >= 2:
                return -1.0

        if worthwhile_second_bench(
            card_id,
            bench_count=bench_count,
            turn=state.turn,
            active_id=active_id,
            field_dup_count=field_counts.get(card_id, 0),
            tech=LUCARIO_TECH,
        ):
            return 20000.0
        return -1.0

    def _score_play_trainer(
        self, card_id, plan, can_attack, hand_counts, stadium_id, state, ctx
    ) -> float:
        score = 10000.0
        field_counts = ctx["field_counts"]
        low_deck = ctx["deck_count"] <= 10
        thin_deck = ctx["deck_count"] <= 15
        early = state.turn <= 6
        needs_line = ctx["riolu_line"] == 0
        needs_ex = ctx["has_riolu_in_play"] and not ctx["has_mega_lucario"]
        riolu_underpowered = (
            ctx["has_riolu_in_play"]
            and ctx["riolu_min_energy"] < 2
            and not ctx["has_mega_lucario"]
        )
        mega_underpowered = ctx["has_mega_lucario"] and ctx["mega_min_energy"] < 2
        line_hungry = ctx["line_needs_energy"]

        if card_id == Dusk_Ball:
            if low_deck:
                return -1.0
            if needs_line and early:
                return 9500.0
            if needs_line:
                return 8500.0
            if needs_ex:
                return 7800.0
            if line_hungry and not low_deck:
                return 8600.0
            return 4500.0 if early else 3500.0

        if card_id == Poke_Pad:
            if low_deck:
                return -1.0
            if needs_line:
                return 9800.0
            if needs_ex and ctx["riolu_min_energy"] >= 1:
                return 8200.0
            if line_hungry and ctx["has_riolu_in_play"]:
                return 7600.0
            return 5200.0 if early else 4000.0

        if card_id == Fighting_Gong:
            if state.supporterPlayed:
                return -1.0
            if riolu_underpowered:
                return 9200.0
            if mega_underpowered:
                return 9000.0
            return -1.0

        if card_id == Premium_Power_Pro:
            if state.supporterPlayed and plan.remain_hp <= 0:
                return -1.0
            if can_attack and plan.remain_hp <= 0:
                score = 12000.0
            elif can_attack and plan.attack_index == 1:
                score = 11000.0
            elif line_hungry and not state.supporterPlayed:
                score = 8700.0
            elif not can_attack:
                if (
                    not state.supporterPlayed
                    and hand_counts[Carmine] > 0
                    and hand_counts[Lillie_Determination] == 0
                ):
                    score = 3050.0
                else:
                    return -1.0
            else:
                score = 5000.0
            levers = ctx.get("levers")
            if levers:
                score += levers.premium_power_pro
            return score
        if card_id == Switch:
            if plan.attacker <= 0:
                return -1.0
            if (
                ctx["has_mega_lucario"]
                and not ctx["can_use_mega_brave"]
                and plan.attack_index == 1
            ):
                score = 8800.0
            else:
                score = 6000.0
            levers = ctx.get("levers")
            if levers and plan.attack_index == 1:
                score += levers.switch_after_mega_brave
            return score
        if card_id == Boss_Orders:
            if plan.target < 1:
                return -1.0
            score = 3400.0 + (ctx.get("levers").boss_orders if ctx.get("levers") else 0.0)
            levers = ctx.get("levers")
            if levers and levers.gust_setup_pokemon > 0 and plan.target >= 1:
                score += levers.gust_setup_pokemon * 0.25
            return score
        if card_id == Carmine:
            if low_deck:
                return -1.0
            if thin_deck:
                return 800.0
            if early and not ctx.get("has_solrock_lunatone_engine"):
                return 2800.0
            return 3000.0
        if card_id == Lillie_Determination:
            if low_deck:
                return -1.0
            if thin_deck:
                score = 900.0
            elif early and not ctx.get("has_solrock_lunatone_engine"):
                score = 8600.0
            elif ctx.get("has_solrock_lunatone_engine") and ctx["discard_energy"] <= 1:
                score = 8200.0
            else:
                score = 3100.0
            levers = ctx.get("levers")
            if levers:
                score += levers.lillie_early
            return score
        if card_id == Gravity_Mountain:
            if ctx.get("op_stage2_count", 0) >= 1:
                score = 8900.0
            elif stadium_id == 0:
                return -1.0
            else:
                score = 7000.0
            levers = ctx.get("levers")
            if levers:
                score += levers.gravity_mountain
            return score
        return score

    def _setup_bench_score(self, card_id, field_counts, state, my_index, levers=None) -> float:
        if card_id == Solrock:
            score = 4.5 if field_counts[Lunatone] >= 1 else (
                2.0 if state.firstPlayer == my_index else 4.0
            )
            if levers and levers.prefer_solrock_open > 0:
                score += levers.prefer_solrock_open * 10.0
            return score
        if card_id == Riolu:
            score = 3.5
            if levers and levers.prefer_riolu_open > 0:
                score += levers.prefer_riolu_open * 10.0
            return score
        if card_id == Makuhita:
            score = 1.0
            if levers and levers.bench_hariyama_bonus > 0:
                score += levers.bench_hariyama_bonus / 200.0
            return score
        if card_id == Lunatone:
            if field_counts[Solrock] >= 1:
                return 4.0
            return -1.0 if field_counts[Lunatone] >= 1 else 0.5
        return 0.0

    def _to_hand_score(self, card_id, field_counts, hand_counts, state) -> float:
        score = 200.0 - hand_counts[card_id] * 100.0
        if card_id == Makuhita:
            score += -10.0 if field_counts[card_id] >= 1 else 10.0
        elif card_id == Hariyama:
            score += 20.0 if field_counts[Makuhita] >= 1 else -20.0
        elif card_id == Lunatone:
            score += -250.0 if field_counts[card_id] >= 1 else 80.0
            if field_counts[Solrock] >= 1 and field_counts[Lunatone] == 0:
                score += 60.0
        elif card_id == Solrock:
            score += -250.0 if field_counts[card_id] >= 1 else 70.0
            if field_counts[Lunatone] >= 1 and field_counts[Solrock] == 0:
                score += 50.0
        elif card_id == Riolu:
            riolu_line = field_counts[Riolu] + field_counts[Mega_Lucario_ex]
            if riolu_line >= 2:
                score -= 150.0
            elif riolu_line >= 1:
                score -= 3.0
            else:
                score += 120.0
        elif card_id == Mega_Lucario_ex:
            score += 100.0 if field_counts[Riolu] >= 1 else -30.0
        elif card_id == Basic_Fighting_Energy:
            line_on_field = field_counts[Riolu] + field_counts[Mega_Lucario_ex]
            if not self._ability_used or not state.energyAttached:
                if line_on_field >= 1:
                    score += 90.0
                else:
                    score += 30.0
            else:
                score -= 1.0
        return score

    def _energy_score(
        self,
        pokemon,
        active: bool,
        attacker1: bool,
        attacker2: bool,
        *,
        state=None,
        ctx=None,
    ) -> int:
        energy_count = len(pokemon.energies)
        score = 8000
        if active:
            score += 10
        if pokemon.id in (Makuhita, Hariyama):
            if pokemon.id == Hariyama:
                score += 1
                if ctx and ctx.get("discard_energy", 0) >= 1 and len(pokemon.energies) < 3:
                    score += 80
            if energy_count < 3:
                score += 100
            if attacker2:
                score -= 50
        elif pokemon.id == Lunatone:
            score -= 100
        elif pokemon.id == Solrock:
            if energy_count < 1:
                score += 20
            else:
                score -= 100
        elif pokemon.id in (Riolu, Mega_Lucario_ex):
            missing = max(0, 2 - energy_count)
            early = state is not None and state.turn <= 8
            if missing > 0:
                score += 200 * missing
                if early:
                    score += 140 * missing
                if not active:
                    score += 90 * missing
            if pokemon.id == Mega_Lucario_ex:
                score += 40
                if missing > 0 and active:
                    score += 120
            if attacker1 and energy_count >= 2:
                score -= 50
        return score

    def _prize_count(self, pokemon) -> int:
        data = (_CARD_TABLE or {}).get(pokemon.id)
        count = 3 if data and data.megaEx else 2 if data and data.ex else 1
        for card in pokemon.energyCards:
            if card.id == 12:
                count -= 1
        for card in pokemon.tools:
            if card.id == 1172 and data and "Lillie" in data.name:
                count -= 1
        return max(0, count)

    def _pokemon_score(self, pokemon) -> float:
        data = (_CARD_TABLE or {}).get(pokemon.id)
        score = self._prize_count(pokemon) * 1000
        score += len(pokemon.energies) * 150
        score += len(pokemon.tools) * 100
        if data:
            if data.stage2:
                score += 250
            elif data.stage1:
                score += 130
        pid = pokemon.id
        if pid in (173, 174, 190, 1071):
            score -= 200
        if pid == 112 and len(pokemon.energies) >= 1:
            score += 300
        score += pokemon.hp
        return float(score)

    @staticmethod
    def _get_card(obs, area, index, player_index):
        api = _load_api()
        ps = obs.current.players[player_index]
        a = _enum_val(area)
        if a == api.AreaType.DECK.value:
            return obs.select.deck[index]
        if a == api.AreaType.HAND.value:
            return ps.hand[index]
        if a == api.AreaType.DISCARD.value:
            return ps.discard[index]
        if a == api.AreaType.ACTIVE.value:
            return ps.active[index]
        if a == api.AreaType.BENCH.value:
            return ps.bench[index]
        if a == api.AreaType.PRIZE.value:
            return ps.prize[index]
        if a == api.AreaType.STADIUM.value:
            return obs.current.stadium[index]
        if a == api.AreaType.LOOKING.value:
            return obs.current.looking[index]
        return None


def is_lucario_deck(deck_ids: list[int]) -> bool:
    ids = set(deck_ids)
    return {673, 674, 676, 677, 678}.issubset(ids)
