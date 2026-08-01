import os
import sys
from collections import defaultdict

from cg.api import AreaType, CardType, EnergyType, Observation, SelectContext, OptionType, Card, Pokemon, all_card_data, to_observation_class

"""
Alakazam Deck
This deck uses Alakazam's Powerful Hand attack (20 damage per card in hand)
with a draw engine built around Kadabra/Alakazam Psychic Draw, Dudunsparce's
Run Away Draw, and Fezandipiti ex's Flip the Script.
"""

# Load deck.csv in the dataset
# Fetch card metadata database and create an ID-to-Card lookup table
all_card = all_card_data()
card_table = {c.cardId: c for c in all_card}

# Decklist
Abra = 741              # x4
Kadabra = 742            # x4
Alakazam = 743           # x3
Dunsparce = 65           # x4 (Dig: on heads, blocks all effects next turn)
Dudunsparce = 66         # x2
Fezandipiti_ex = 140     # x1
Genesect = 142           # x1
Psyduck = 858            # x1
Shaymin = 343            # x1
Rare_Candy = 1079        # x3
Enhanced_Hammer = 1081   # x3
Buddy_Buddy_Poffin = 1086  # x4
Night_Stretcher = 1097   # x1
Sacred_Ash = 1129        # x1
Poke_Pad = 1152          # x4
Lucky_Helmet = 1156      # x3
Boss_Orders = 1182       # x2
Hilda = 1225             # x4
Dawn = 1231              # x4
Battle_Cage = 1264       # x4
Basic_Psychic_Energy = 5   # x2
Telepath_Psychic_Energy = 19  # x4
Enriching_Energy = 13    # x1  (ACE SPEC)

# Opponent card IDs to watch for
Duskull = 131
Slowpoke_IDs = (162, 327)
Froakie_IDs = (33, 945)
Wellspring_Mask_Ogerpon_ex = 108
N_Darumaka = 257
Dreepy = 119
Drakloak = 120
Dragapult_ex = 121
Mist_Energy = 11
Rock_Fighting_Energy = 20

# Attack IDs
ATTACK_TELEPORTATION = 1070   # Abra: 10 dmg, cost {P}
ATTACK_SUPER_PSY_BOLT = 1071  # Kadabra: 30 dmg, cost {P}
ATTACK_POWERFUL_HAND = 1072   # Alakazam: 20 per card in hand, cost {P}

# Card ID sets
ABRA_LINE = {Abra, Kadabra, Alakazam}
DUNSPARCE_LINE = {Dunsparce, Dudunsparce}
PSYCHIC_ENERGY_IDS = {Basic_Psychic_Energy, Telepath_Psychic_Energy}

pre_turn = 0
ability_used_dudunsparce = False
ability_used_fezandipiti = False


def get_card(obs: Observation, area: AreaType, index: int, player_index: int) -> Pokemon | Card | None:
    ps = obs.current.players[player_index]
    match area:
        case AreaType.DECK:
            return obs.select.deck[index]
        case AreaType.HAND:
            return ps.hand[index]
        case AreaType.DISCARD:
            return ps.discard[index]
        case AreaType.ACTIVE:
            return ps.active[index]
        case AreaType.BENCH:
            return ps.bench[index]
        case AreaType.PRIZE:
            return ps.prize[index]
        case AreaType.STADIUM:
            return obs.current.stadium[index]
        case AreaType.LOOKING:
            return obs.current.looking[index]
        case _:
            return None


def prize_count(pokemon: Pokemon) -> int:
    data = card_table[pokemon.id]
    count = 3 if data.megaEx else 2 if data.ex else 1
    for card in pokemon.energyCards:
        if card.id == 12:  # Legacy Energy
            count -= 1
    for card in pokemon.tools:
        if card.id == 1172 and "Lillie" in data.name:
            count -= 1
    return max(0, count)


def count_special_defense_energies(pokemon: Pokemon) -> int:
    # KO math: count only the special energies that blank an effect-based attack.
    # Widening this to every special energy misfires on our own Telepath Psychic Energy.
    cnt = 0
    for ec in pokemon.energyCards:
        if ec.id == Mist_Energy or ec.id == Rock_Fighting_Energy:
            cnt += 1
    return cnt


def count_any_special_energies(pokemon: Pokemon) -> int:
    # Hammer targeting: acceleration energies (TR Energy / Telepath) are worth removing too.
    cnt = 0
    for ec in pokemon.energyCards:
        data = card_table.get(ec.id)
        if data is not None and data.cardType == CardType.SPECIAL_ENERGY:
            cnt += 1
    return cnt



# Crash-safety and submission diagnostics. The public policy below is preserved as
# _policy_agent(); the submitted agent() wrapper normalizes its option indices and
# falls back legally on every exception.
_DIAG = defaultdict(int)


def diag_reset() -> None:
    _DIAG.clear()


def diag_snapshot() -> dict:
    total = max(1, _DIAG.get("decisions", 0))
    out = dict(_DIAG)
    out["fallback_rate"] = (_DIAG.get("policy_fallback", 0) + _DIAG.get("obs_fallback", 0)) / total
    return out


def _legal_fallback(select) -> list[int]:
    try:
        n = len(select.option)
        if n == 0 or select.maxCount <= 0:
            return []
        min_count = max(0, min(select.minCount, n))
        max_count = max(min_count, min(select.maxCount, n))
        return list(range(min_count if min_count > 0 else min(1, max_count)))
    except Exception:
        return []


def _normalize_ordered_indices(indices, select) -> list[int]:
    try:
        n = len(select.option)
        if n == 0 or select.maxCount <= 0:
            return []
        min_count = max(0, min(select.minCount, n))
        max_count = max(min_count, min(select.maxCount, n))
        out = []
        seen = set()
        for idx in indices or []:
            if not isinstance(idx, int) or idx in seen or not (0 <= idx < n):
                continue
            seen.add(idx)
            out.append(idx)
            if len(out) >= max_count:
                break
        if len(out) < min_count:
            for idx in range(n):
                if idx not in seen:
                    out.append(idx)
                    seen.add(idx)
                if len(out) >= min_count:
                    break
        return out[:max_count]
    except Exception:
        return _legal_fallback(select)


def _policy_agent(obs_dict, return_scores: bool = False) -> list[int]:
    # Rollouts hand us an Observation dataclass directly instead of a raw dict.
    obs = obs_dict if isinstance(obs_dict, Observation) else to_observation_class(obs_dict)
    if obs.select is None:
        return my_deck

    state = obs.current
    select = obs.select
    context = select.context
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]
    my_prize_count = len(my_state.prize)

    global pre_turn, ability_used_dudunsparce, ability_used_fezandipiti
    if pre_turn != state.turn:
        pre_turn = state.turn
        ability_used_dudunsparce = False
        ability_used_fezandipiti = False

    # ---- Count cards on field / hand / discard ----
    field_counts = defaultdict(int)
    hand_counts = defaultdict(int)
    discard_counts = defaultdict(int)

    my_field = []  # (field_index, pokemon) where 0=active, 1..=bench
    for card in my_state.active:
        if card is not None:
            field_counts[card.id] += 1
            my_field.append((0, card))
    for idx, card in enumerate(my_state.bench):
        if card is not None:
            field_counts[card.id] += 1
            my_field.append((idx + 1, card))

    for card in my_state.hand:
        hand_counts[card.id] += 1

    for card in my_state.discard:
        discard_counts[card.id] += 1

    abra_line_on_field = field_counts[Abra] + field_counts[Kadabra] + field_counts[Alakazam]
    # Energy drought: the attacking Alakazam holds no Psychic energy and neither does the hand,
    # but the discard pile does -> recovering it with Night Stretcher becomes the top priority.
    _p_hand = hand_counts[Basic_Psychic_Energy] + hand_counts[Telepath_Psychic_Energy]
    _p_discard = discard_counts[Basic_Psychic_Energy] + discard_counts[Telepath_Psychic_Energy]
    _zam_dry = any(p.id == Alakazam and not any(ec.id in PSYCHIC_ENERGY_IDS for ec in p.energyCards)
                   for _, p in my_field)
    energy_drought = _zam_dry and _p_hand == 0 and _p_discard >= 1

    # Whiff guard for search cards: count every visible copy (hand, discard, field, and
    # cards sitting under evolutions). If deck + prizes provably hold zero targets left,
    # the search card is not played at all.
    def _visible_copies(cid):
        n = hand_counts.get(cid, 0) + discard_counts.get(cid, 0)
        for _, p in my_field:
            if p.id == cid:
                n += 1
            for pe in (p.preEvolution or []):
                if pe.id == cid:
                    n += 1
        return n

    _poffin_targets_left = max(0, 4 - _visible_copies(Abra)) + max(0, 4 - _visible_copies(Dunsparce))
    _pokepad_targets_left = (_poffin_targets_left + max(0, 4 - _visible_copies(Kadabra))
                             + max(0, 4 - _visible_copies(Alakazam)) + max(0, 3 - _visible_copies(Dudunsparce)))
    dunsparce_line_on_field = field_counts[Dunsparce] + field_counts[Dudunsparce]

    # ---- Opponent field analysis ----
    op_all_pokemon = []
    for card in op_state.active:
        if card is not None:
            op_all_pokemon.append(card)
    for card in op_state.bench:
        if card is not None:
            op_all_pokemon.append(card)

    op_has_duskull = any(p.id == Duskull for p in op_all_pokemon)
    op_has_water_threat = any(
        p.id in Slowpoke_IDs or p.id in Froakie_IDs
        or p.id == Wellspring_Mask_Ogerpon_ex or p.id == N_Darumaka
        for p in op_all_pokemon
    )
    op_has_dragapult_line = any(
        p.id in (Dreepy, Drakloak, Dragapult_ex) for p in op_all_pokemon
    )
    # One-shot pressure: can any Pokemon the opponent has in play already deal 100+ damage?
    # If so, never promote a fragile evolution base (Abra) into the active spot.
    op_oneshot = False
    try:
        for _p in op_all_pokemon:
            _d = card_table.get(_p.id)
            if _d is None:
                continue
            for _aid in _d.attacks:
                _a = ATTACK.get(_aid)
                if _a is not None and (_a.damage or 0) >= 100 and len(_p.energies) >= len(_a.energies):
                    op_oneshot = True
    except NameError:
        pass
    # Hammer insurance: once the opponent is seen using Enhanced Hammer, stop leaning on
    # special energy for the main attacker, which would otherwise be stripped mid-setup.
    op_hammer_seen = any(c.id == Enhanced_Hammer for c in (op_state.discard or []))

    # Detect if opponent has used ACE SPEC
    op_used_ace_spec = False
    for log in obs.logs:
        if hasattr(log, 'cardId') and log.cardId is not None:
            cd = card_table.get(log.cardId)
            if cd and cd.aceSpec and hasattr(log, 'playerIndex') and log.playerIndex == (1 - my_index):
                op_used_ace_spec = True

    stadium_id = 0
    for card in state.stadium:
        stadium_id = card.id

    bench_count = len(my_state.bench)
    bench_max = my_state.benchMax
    bench_free = bench_max - bench_count

    # ---- Active pokemon info ----
    active_pokemon = my_state.active[0] if my_state.active else None
    active_id = active_pokemon.id if active_pokemon else -1
    active_has_psychic = False
    if active_pokemon:
        for ec in active_pokemon.energyCards:
            if ec.id in PSYCHIC_ENERGY_IDS:
                active_has_psychic = True
                break

    # ---- Opponent active info ----
    op_active = op_state.active[0] if op_state.active else None
    op_active_hp = op_active.hp if op_active else 9999

    # ---- Estimate Powerful Hand damage range ----
    hand_size = len(my_state.hand) if my_state.hand else my_state.handCount

    def estimate_hand_increase():
        """Returns (min_increase, max_increase) of hand size this turn from draw effects."""
        min_inc = 0
        max_inc = 0
        for _, p in my_field:
            if p.id == Abra and hand_counts[Kadabra] > 0:
                max_inc += 1  # evolve Kadabra: hand -1, draw +2 = net +1
            elif p.id == Abra and hand_counts[Rare_Candy] > 0 and hand_counts[Alakazam] > 0:
                max_inc += 1  # Rare Candy + Alakazam: hand -2, draw +3 = net +1
            elif p.id == Kadabra and hand_counts[Alakazam] > 0:
                max_inc += 2  # evolve Alakazam: hand -1, draw +3 = net +2
            elif p.id == Dunsparce and hand_counts[Dudunsparce] > 0:
                max_inc += 1  # evolve: hand -1, ability draw +2 = net +1
            elif p.id == Dudunsparce:
                if not ability_used_dudunsparce:
                    max_inc += 3  # Run Away Draw
            elif p.id == Fezandipiti_ex:
                if not ability_used_fezandipiti:
                    max_inc += 3  # Flip the Script
        if hand_counts[Fezandipiti_ex] > 0 and bench_free > 0 and field_counts[Fezandipiti_ex] == 0:
            max_inc += 2  # play -1, ability +3 = net +2

        # Supporter (only 1 can be used)
        supporter_options = []
        if not state.supporterPlayed:
            if hand_counts[Hilda] > 0:
                supporter_options.append(1)   # play -1, search +2 = net +1
            if hand_counts[Dawn] > 0:
                supporter_options.append(2)   # play -1, search +3 = net +2
            if hand_counts[Boss_Orders] > 0:
                supporter_options.append(-1)  # play -1 = net -1
        if supporter_options:
            max_inc += max(supporter_options)

        # Enriching Energy attach: hand -1, draw +4 = net +3
        if hand_counts[Enriching_Energy] > 0 and not state.energyAttached:
            if active_id == Alakazam and active_has_psychic:
                max_inc += 3
        return min_inc, max_inc

    min_hand_inc, max_hand_inc = estimate_hand_increase()
    max_hand_size = hand_size + max_hand_inc
    min_hand_size = hand_size + min_hand_inc
    max_damage = max_hand_size * 20
    min_damage = min_hand_size * 20

    # ---- Target selection for attack ----
    target_idx = -1       # 0 = active, 1.. = bench
    target_pokemon = None
    target_use_boss = False
    target_can_kill = False
    target_prize_gain = 0
    target_hammer_needed = 0
    use_kadabra_finish = False

    if state.turn >= 2 and op_active is not None:
        # Check Kadabra finisher: opponent active HP <= 30
        if op_active_hp <= 30 and (field_counts[Kadabra] >= 1 or active_id == Kadabra):
            target_idx = 0
            target_pokemon = op_active
            target_use_boss = False
            target_can_kill = True
            target_prize_gain = prize_count(op_active)
            use_kadabra_finish = True
        else:
            # Evaluate all opponent pokemon
            all_op = [(0, op_active)]
            for bi, bp in enumerate(op_state.bench):
                if bp is not None:
                    all_op.append((bi + 1, bp))

            candidates = []
            for oi, pkmn in all_op:
                pz = prize_count(pkmn)
                sp_e = count_special_defense_energies(pkmn)
                eff_max_dmg = max_damage
                hm_need = 0
                if sp_e > 0:
                    if hand_counts[Enhanced_Hammer] >= sp_e:
                        hm_need = sp_e
                        eff_max_dmg = (max_hand_size - hm_need) * 20
                    else:
                        eff_max_dmg = 0
                ck = pkmn.hp <= eff_max_dmg and eff_max_dmg > 0
                candidates.append((oi, pkmn, pz, ck, hm_need))

            # Priority 1: kill wins the game
            win_cands = [(oi, pk, pz, ck, hm) for oi, pk, pz, ck, hm in candidates if ck and my_prize_count <= pz]
            if win_cands:
                # Among winners, prefer active (no boss needed), then highest HP
                best = min(win_cands, key=lambda x: (0 if x[0] == 0 else 1, -x[1].hp))
                target_idx, target_pokemon, target_prize_gain, target_can_kill, target_hammer_needed = best
                target_use_boss = target_idx != 0
            else:
                # Priority 2: killable target with most prizes
                killable = [(oi, pk, pz, ck, hm) for oi, pk, pz, ck, hm in candidates if ck]
                if killable:
    # Threat list: engine Pokemon are worth removing ahead of raw prize value.
                    # Munkidori (Adrena-Brain engine), Drakloak / Dreepy (Dragapult ex line).
                    # Kadabra is deliberately excluded: sniping it does not pay off.
                    _SNIPE_IDS = {112, 120, 119}
                    best = max(killable, key=lambda x: (1 if x[1].id in _SNIPE_IDS else 0, x[2], x[1].hp))
                    target_idx, target_pokemon, target_prize_gain, target_can_kill, target_hammer_needed = best
                    target_use_boss = target_idx != 0
                else:
                    # Priority 3: just hit active
                    target_idx = 0
                    target_pokemon = op_active
                    target_use_boss = False
                    target_can_kill = False
                    target_prize_gain = 0

    # Should we use Dudunsparce's ability?
    need_dudunsparce_draw = False
    if target_pokemon is not None and target_can_kill:
        needed = target_pokemon.hp
        current_dmg = (hand_size - target_hammer_needed) * 20
        if current_dmg < needed:
            need_dudunsparce_draw = True

    # Attack lock: energy is attached but MAIN offers no attack option, which means an
    # effect such as Snotted Up sealed it. Switching the active clears the lock.
    attack_locked = False
    if (context == SelectContext.MAIN and state.turn >= 2 and active_pokemon is not None
            and active_id == Alakazam and active_has_psychic):
        attack_locked = not any(o.type == OptionType.ATTACK for o in select.option)

    # Do we need to attach energy to the active to retreat?
    need_retreat_energy = False
    if active_pokemon is not None and state.turn >= 2:
        active_is_attacker = ((active_id == Alakazam and active_has_psychic) or (use_kadabra_finish and active_id == Kadabra)) and not attack_locked
        if not active_is_attacker:
            # Check if there's a better attacker on bench
            has_bench_attacker = False
            if use_kadabra_finish and field_counts[Kadabra] >= 1 and active_id != Kadabra:
                has_bench_attacker = True
            elif field_counts[Alakazam] >= 1 and active_id != Alakazam:
                has_bench_attacker = True
            elif field_counts[Kadabra] >= 1 and active_id != Kadabra:
                has_bench_attacker = True
            if has_bench_attacker:
                retreat_cost = card_table[active_pokemon.id].retreatCost
                active_energy_count = len(active_pokemon.energies)
                if active_energy_count < retreat_cost:
                    need_retreat_energy = True

    # Do we need Fezandipiti ex's Flip the Script to kill the target?
    fez_hand_contribution = 0
    if field_counts[Fezandipiti_ex] >= 1 and not ability_used_fezandipiti:
        fez_hand_contribution = 3
    elif hand_counts[Fezandipiti_ex] > 0 and bench_free > 0 and field_counts[Fezandipiti_ex] == 0:
        fez_hand_contribution = 2  # play -1, ability +3 = net +2
    need_fezandipiti_draw = False
    if target_pokemon is not None and target_can_kill and fez_hand_contribution > 0:
        max_damage_without_fez = (max_hand_size - fez_hand_contribution - target_hammer_needed) * 20
        if max_damage_without_fez < target_pokemon.hp:
            need_fezandipiti_draw = True

    # Also allow Fezandipiti if drawing could find key enablers (Boss, Rare Candy, Alakazam, Energy)
    need_fezandipiti_for_setup = False
    if target_pokemon is not None and target_can_kill and fez_hand_contribution > 0 and not need_fezandipiti_draw:
        # Missing Boss's Orders for bench target
        missing_boss = (target_use_boss and hand_counts[Boss_Orders] == 0
                        and not state.supporterPlayed)
        # Check if we have a ready attacker (Alakazam with psychic energy)
        has_ready_attacker = (active_id == Alakazam and active_has_psychic)
        if not has_ready_attacker:
            for _, p in my_field:
                if p.id == Alakazam and any(ec.id in PSYCHIC_ENERGY_IDS for ec in p.energyCards):
                    has_ready_attacker = True
                    break
        missing_attacker = False
        missing_energy = False
        if not has_ready_attacker:
            # Can we set up Alakazam this turn?
            can_evolve_to_alakazam = (field_counts[Kadabra] >= 1 and hand_counts[Alakazam] >= 1)
            can_rare_candy_alakazam = (field_counts[Abra] >= 1 and hand_counts[Rare_Candy] >= 1
                                       and hand_counts[Alakazam] >= 1)
            if not can_evolve_to_alakazam and not can_rare_candy_alakazam:
                # Missing evolution pieces
                if field_counts[Kadabra] >= 1 and hand_counts[Alakazam] == 0:
                    missing_attacker = True
                elif field_counts[Abra] >= 1 and (hand_counts[Rare_Candy] == 0 or hand_counts[Alakazam] == 0):
                    missing_attacker = True
            # Check if energy is available for the attacker
            energy_in_hand = (hand_counts[Basic_Psychic_Energy] + hand_counts[Telepath_Psychic_Energy]
                              + hand_counts[Enriching_Energy])
            if not state.energyAttached and energy_in_hand == 0:
                has_energized = any(
                    p.id in ABRA_LINE and any(ec.id in PSYCHIC_ENERGY_IDS for ec in p.energyCards)
                    for _, p in my_field
                )
                if not has_energized:
                    missing_energy = True
        if missing_boss or missing_attacker or missing_energy:
            need_fezandipiti_for_setup = True

    # Deck safety: don't let deck count drop to <= prize count unless winning this turn
    can_win_this_turn = target_can_kill and my_prize_count <= target_prize_gain
    deck_count = my_state.deckCount
    # safe_draws: max cards we can draw from deck while keeping deck > prize count
    # We also need 1 card for the draw at start of next turn
    _reserve = my_prize_count + 1
    if state.turn >= 12 and len(op_state.prize) == 6:
        # Against passive / lock decks the usual way we lose is our own deck-out, not damage.
        _reserve = my_prize_count + 6
    safe_draws = deck_count - _reserve if not can_win_this_turn else 999
    # Over-draw brake: 16 cards in hand already means 320 damage, enough to one-shot
    # anything on the board, so further optional draw only burns our own deck.
    # The hard floor at 15 cards left applies only in mill / stall contexts (the opponent
    # has taken no prizes); applying it unconditionally would throttle our own damage.
    overdrawn = hand_size >= 16 or (deck_count <= 15 and not can_win_this_turn and len(op_state.prize) == 6)
    # Hand preservation: Powerful Hand deals 20 per card in hand, so on a developed board a
    # marginal optional play is worth less than the +20 damage of simply holding that card.
    # Still played immediately: removing Mist Energy, setting up a KO, mandatory development,
    # and thin-deck recovery.
    _op_active_mist = op_active is not None and any(
        ec.id == Mist_Energy for ec in (op_active.energyCards or []))
    _preserve_hand = (active_id == Alakazam and active_has_psychic
                      and abra_line_on_field >= 3 and dunsparce_line_on_field >= 2
                      and bench_count >= 2 and not energy_drought
                      and not overdrawn and not target_can_kill and not _op_active_mist)

    # ---- Score each option ----
    scores = []
    for o in select.option:
        score = 0

        if o.type == OptionType.NUMBER:
            score = o.number
            if context == SelectContext.DRAW_COUNT:
                # Thin deck: take the largest draw count still within safe_draws, else the smallest.
                score = o.number if o.number <= max(0, safe_draws) else -o.number

        elif o.type == OptionType.YES:
            score = 1
            if context == SelectContext.ACTIVATE and safe_draws <= 2:
                score = -1  # Thin deck: decline draw / search abilities.

        elif o.type == OptionType.CARD:
            card = get_card(obs, o.area, o.index, o.playerIndex)
            if card is None:
                scores.append(score)
                continue
            energy_count = len(card.energies) if isinstance(card, Pokemon) else 0

            if context == SelectContext.SWITCH or context == SelectContext.TO_ACTIVE:
                if o.playerIndex == my_index:
                    if card.id == Alakazam:
                        score += 100 + energy_count * 10
                    elif card.id == Kadabra:
                        score += 90 if (op_active_hp <= 30) else 30
                    elif card.id == Abra:
                        # Under one-shot pressure, rank Abra below the Dunsparce line (shield first).
                        score += 3 if op_oneshot else 10
                    elif card.id in DUNSPARCE_LINE:
                        score += 5
                    else:
                        score += 1
                else:
                    if target_use_boss and target_pokemon is not None:
                        if o.index == target_idx - 1:
                            score += 100

            elif context == SelectContext.SETUP_ACTIVE_POKEMON:
                if card.id == Abra:
                    score = 10
                elif card.id == Dunsparce:
                    score = 5
                elif card.id == Psyduck:
                    score = 2
                elif card.id == Shaymin:
                    score = 1

            elif context == SelectContext.SETUP_BENCH_POKEMON:
                if card.id == Abra:
                    cur = field_counts[Abra] + field_counts[Kadabra] + field_counts[Alakazam]
                    score = 200 if cur == 0 else 100 + (3 - cur) * 10
                elif card.id == Dunsparce:
                    score = 150 if dunsparce_line_on_field == 0 else 50

            elif context == SelectContext.TO_HAND:
                score = 200 - hand_counts.get(card.id, 0) * 50
                # With one or fewer Pokemon in play, an immediately playable Basic outranks
                # everything else (+800): an empty field is an instant loss.
                _data = card_table.get(card.id)
                if (bench_count == 0 and _data is not None and _data.basic
                        and _data.cardType == CardType.POKEMON):
                    score += 800
                if card.id == Dudunsparce:
                    score += 80 if (field_counts[Dunsparce] >= 1 and field_counts[Dudunsparce] == 0) else -50
                elif card.id == Kadabra:
                    score += 70 if field_counts[Abra] >= 1 else -20
                elif card.id == Alakazam:
                    score += 60 if (field_counts[Kadabra] >= 1 or field_counts[Abra] >= 1) else -20
                elif card.id == Abra:
                    score += 50 if abra_line_on_field < 3 else -50
                elif card.id == Dunsparce:
                    score += 40 if dunsparce_line_on_field < 2 else -50
                elif card.id in PSYCHIC_ENERGY_IDS:
                    score += 30 if not state.energyAttached else -10
                    if energy_drought:
                        score += 500  # During an energy drought, recover Psychic energy first.
                elif card.id == Enriching_Energy:
                    score += 20
                elif card.id == Rare_Candy:
                    score += 40 if field_counts[Abra] >= 1 else -10

            elif context == SelectContext.ATTACH_FROM:
                if isinstance(card, Pokemon):
                    if need_retreat_energy and o.area == AreaType.ACTIVE:
                        score = 150  # Must attach to active to retreat
                    elif len(card.energyCards) >= 1:
                        score = -1  # Don't attach 2+ energy to the same pokemon
                    elif card.id in ABRA_LINE:
                        score = 100
                        if card.id == Alakazam:
                            score += 20
                        elif card.id == Kadabra:
                            score += 10
                        if o.area == AreaType.ACTIVE:
                            score += 5
                    elif card.id in DUNSPARCE_LINE:
                        score = 50
                    else:
                        score = 10

            elif context == SelectContext.TO_BENCH:
                if card.id == Abra:
                    score = 100
                elif card.id == Dunsparce:
                    score = 80
                elif card.id == Psyduck:
                    if op_has_duskull:
                        score = 60
                    else:
                        score = -1
                elif card.id == Shaymin:
                    if op_has_water_threat:
                        score = 40
                    else:
                        score = -1

            elif context == SelectContext.TO_DECK:
                if card.id in ABRA_LINE:
                    score = 100
                elif card.id in DUNSPARCE_LINE:
                    score = 50
                elif card.id in PSYCHIC_ENERGY_IDS or card.id == Enriching_Energy:
                    score = 30  # Return-to-deck order: Pokemon > energy > everything else.
                else:
                    score = 10

            elif context == SelectContext.DISCARD:
                # Forced discards go lowest-value first (higher score = discarded earlier);
                # duplicate copies are the cheapest to let go.
                if card.id in (Alakazam, Rare_Candy):
                    score = 10
                elif card.id in PSYCHIC_ENERGY_IDS or card.id == Enriching_Energy:
                    score = 30
                elif card.id in (Kadabra, Abra, Dunsparce, Dudunsparce):
                    score = 40
                elif card.id in (Hilda, Dawn, Boss_Orders):
                    score = 60
                else:
                    score = 80
                score += hand_counts.get(card.id, 0) * 15

        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            data = card_table[card.id]

            if data.cardType == CardType.POKEMON:
                score = 20000
                is_early = state.turn <= 2

                if card.id == Abra:
                    if is_early:
                        score += 500
                    elif abra_line_on_field < 3:
                        score += 200
                    elif bench_free <= 1:
                        score = -1
                    else:
                        score += 50

                elif card.id == Dunsparce:
                    if dunsparce_line_on_field < 1:
                        score += 400 if is_early else 100
                    elif dunsparce_line_on_field < 2:
                        score += 50
                    else:
                        score = -1

                elif card.id == Fezandipiti_ex:
                    if need_fezandipiti_draw or need_fezandipiti_for_setup:
                        score += 80 if not is_early else 30
                    else:
                        score = -1  # Don't play unless Flip the Script is needed to kill

                elif card.id == Genesect:
                    if not op_used_ace_spec and (hand_counts[Lucky_Helmet] > 0 or hand_counts[Poke_Pad] > 0):
                        score += 100
                    else:
                        score = -1

                elif card.id == Psyduck:
                    if op_has_duskull:
                        score += 300
                    else:
                        score = -1

                elif card.id == Shaymin:
                    if op_has_water_threat:
                        score += 300
                    else:
                        score = -1

                # Keep at least 1 bench slot free
                if bench_free <= 1 and score > 0:
                    score -= 5000

            else:
                score = 10000

                if card.id == Buddy_Buddy_Poffin:
                    if safe_draws < 2 or _poffin_targets_left <= 0:
                        score = -1  # Thin deck, or no target provably left (whiff guard).
                    elif state.turn <= 2:
                        if abra_line_on_field < 3 or dunsparce_line_on_field < 1:
                            score = 18000
                        else:
                            score = 8000
                    else:
                        if abra_line_on_field < 3 or dunsparce_line_on_field < 2:
                            score = 15000
                        elif target_can_kill:
                            score = 8000
                        else:
                            score = -1

                elif card.id == Poke_Pad:
                    if safe_draws < 1 or overdrawn or _pokepad_targets_left <= 0:
                        score = -1  # Thin deck, or no target provably left (whiff guard).
                    elif _preserve_hand and (hand_counts[Hilda] > 0 or hand_counts[Dawn] > 0):
                        score = 800  # Hold it: with draw support already in hand, +20 damage is worth more.
                    elif state.turn <= 2:
                        score = 17000
                    else:
                        score = 14000 if abra_line_on_field < 3 else 12000

                elif card.id == Rare_Candy:
                    # No safe_draws gate here: the draw ability is declined separately, so evolving is deck-neutral.
                    if field_counts[Abra] >= 1 and hand_counts[Alakazam] >= 1:
                        score = 16000
                    else:
                        score = -1

                elif card.id == Night_Stretcher:
                    dis_abra = discard_counts[Abra] + discard_counts[Kadabra] + discard_counts[Alakazam]
                    if energy_drought:
                        score = 16000  # Refuelling a stranded Alakazam outranks further development.
                    elif dis_abra >= 1:
                        score = 13000
                    elif discard_counts[Basic_Psychic_Energy] + discard_counts[Telepath_Psychic_Energy] >= 1:
                        score = 11000
                    else:
                        score = -1

                elif card.id == Sacred_Ash:
                    dis_abra = discard_counts[Abra] + discard_counts[Kadabra] + discard_counts[Alakazam]
                    # Close-race deck recovery: when both sides have taken prizes and our own
                    # deck is thin, shuffle Pokemon from the discard back in to keep draw fuel
                    # available without spending the hand (= damage).
                    _dis_mons = sum(1 for c in my_state.discard
                                    if card_table.get(c.id) is not None
                                    and card_table[c.id].cardType == CardType.POKEMON)
                    _contested = my_prize_count <= 3 and (6 - len(op_state.prize)) >= 3
                    if _contested and deck_count <= 8 and _dis_mons >= 1:
                        score = 15500
                    elif dis_abra >= 2:
                        score = 13500
                    elif dis_abra >= 1:
                        score = 11000
                    else:
                        score = -1

                elif card.id == Enhanced_Hammer:
                    _op_act_mist_h = (op_state.active and op_state.active[0] is not None
                                      and any(ec.id == Mist_Energy for ec in (op_state.active[0].energyCards or [])))
                    if _op_act_mist_h:
                        # Mist Energy on the defending active blanks Powerful Hand: strip it first.
                        score = 13000
                    elif target_hammer_needed > 0:
                        score = 6500
                    elif _preserve_hand:
                        score = 800  # Hold: a marginal energy removal is worth less than +20 attack damage.
                    else:
                        # Defensive special energies stay the top targets; acceleration ones are removed for tempo.
                        any_defense = any(count_special_defense_energies(p) > 0 for p in op_all_pokemon)
                        any_special = any(count_any_special_energies(p) > 0 for p in op_all_pokemon)
                        if any_defense:
                            score = 5000
                        elif any_special:
                            score = 4500
                        else:
                            score = -1

                elif card.id == Lucky_Helmet:
                    score = 7000  # Will be handled via ATTACH

                elif card.id == Boss_Orders:
                    if target_use_boss and target_can_kill:
                        score = 3200
                    else:
                        score = -1

                elif card.id == Hilda:
                    if safe_draws >= 2 and not overdrawn:
                        score = 3000
                    else:
                        score = -1

                elif card.id == Dawn:
                    if safe_draws >= 3 and not overdrawn:
                        score = 3100
                    else:
                        score = -1

                elif card.id == Battle_Cage:
                    if op_has_dragapult_line:
                        score = 19000
                    elif stadium_id != 0:
                        score = 7000
                    else:
                        score = -1

        elif o.type == OptionType.ATTACH:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)

            if card.id == Lucky_Helmet:
                score = 7000
                if pokemon.id == Genesect and not op_used_ace_spec:
                    score += 300
                elif o.inPlayArea == AreaType.ACTIVE:
                    score += 200
                else:
                    score += 50

            elif card.id in PSYCHIC_ENERGY_IDS:
                if need_retreat_energy and o.inPlayArea == AreaType.ACTIVE:
                    score = 9500  # Must attach to active to retreat
                elif len(pokemon.energyCards) >= 1:
                    score = -1  # Don't attach 2+ energy to the same pokemon
                elif op_hammer_seen and pokemon.id == Alakazam:
                    # After a Hammer sighting, attach basic energy to the main attacker (Telepath gets stripped).
                    score = 8330 if card.id == Basic_Psychic_Energy else 7700
                elif pokemon.id in ABRA_LINE:
                    score = 8000
                    if pokemon.id == Alakazam:
                        score += 30
                    elif pokemon.id == Kadabra:
                        score += 20
                    elif pokemon.id == Abra:
                        score += 10
                    if o.inPlayArea == AreaType.ACTIVE:
                        score += 5
                else:
                    score = -1
                # Telepath Psychic Energy searches 2 from deck
                if card.id == Telepath_Psychic_Energy and safe_draws < 2 and score > 0:
                    score = -1

            elif card.id == Enriching_Energy:
                if need_retreat_energy and o.inPlayArea == AreaType.ACTIVE:
                    score = 9500  # Must attach to active to retreat
                elif len(pokemon.energyCards) >= 1:
                    score = -1  # Don't attach 2+ energy to the same pokemon
                elif pokemon.id in DUNSPARCE_LINE:
                    score = 8500
                    if pokemon.id == Dudunsparce:
                        score += 10
                else:
                    score = -1
                # Enriching Energy draws 4 from deck
                if card.id == Enriching_Energy and safe_draws < 4 and score > 0:
                    score = -1

        elif o.type == OptionType.EVOLVE:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score = 9000
            # Tie-break: evolve the copy carrying the least damage.
            if pokemon is not None:
                score -= (pokemon.maxHp - pokemon.hp) // 10

            if card.id == Alakazam:
                # Evolving is deck-neutral: Psychic Draw is a separate ACTIVATE prompt that is
                # declined on a thin deck, so blocking evolution here would only starve us of attackers.
                if overdrawn and field_counts[Alakazam] >= 2:
                    score = -1  # Two attackers are enough; extra evolutions only burn the deck.
                elif o.inPlayArea == AreaType.ACTIVE:
                    score += 200  # Active Alakazam = highest
                else:
                    score += 50  # Bench Alakazam
                score += len(pokemon.energies) * 10

            elif card.id == Kadabra:
                # Same reasoning: the draw prompt is declined separately, so evolving stays deck-neutral.
                score += 100
                if len(pokemon.energies) == 0:
                    score += 50  # Evolve non-energy Abra first
                else:
                    score -= 20
                    if hand_counts[Rare_Candy] > 0 and hand_counts[Alakazam] > 0:
                        score -= 100  # Save energy Abra for Rare Candy -> Alakazam

            elif card.id == Dudunsparce:
                score += 80

        elif o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            if card is None:
                scores.append(score)
                continue

            if card.id == Dudunsparce:
                if need_dudunsparce_draw:
                    if safe_draws >= 3:
                        score = 30000
                    else:
                        score = -1  # Deck too thin
                else:
                    score = -1
            elif card.id == Fezandipiti_ex:
                if (need_fezandipiti_draw or need_fezandipiti_for_setup) and safe_draws >= 3:
                    score = 29000
                else:
                    score = -1  # Don't use unless needed to kill target
            elif card.id == Battle_Cage:
                score = 1
            else:
                score = 28000

        elif o.type == OptionType.RETREAT:
            if attack_locked and (field_counts[Alakazam] >= 2 or field_counts[Kadabra] >= 1):
                score = 2600  # Break the lock: retreat the sealed active and promote another attacker.
            elif active_id == Alakazam and active_has_psychic:
                score = -1
            elif use_kadabra_finish and active_id != Kadabra and field_counts[Kadabra] >= 1:
                score = 2500  # Retreat to bring Kadabra forward for finish
            elif active_id in (Abra, Dunsparce, Dudunsparce, Psyduck, Shaymin, Genesect):
                if field_counts[Alakazam] >= 1 or field_counts[Kadabra] >= 1:
                    score = 2000
                else:
                    score = -1
            else:
                score = -1

        elif o.type == OptionType.ATTACK:
            score = 1000
            # Effective-damage check: Powerful Hand places damage counters (an effect), so it is
            # fully nullified while the defending active carries Mist Energy.
            _op_act_mist = op_active is not None and any(
                ec.id == Mist_Energy for ec in (op_active.energyCards or []))
            if o.attackId == ATTACK_POWERFUL_HAND:
                if _op_act_mist or hand_size == 0:
                    score -= 200  # Blank shot: prefer a damaging attack or a switch instead.
                else:
                    score += 500
            elif o.attackId == ATTACK_SUPER_PSY_BOLT:
                if op_active_hp <= 30:
                    score += 600  # Kadabra finisher
                elif _op_act_mist:
                    score += 400  # Against Mist Energy, only a true damage attack connects.
                else:
                    score += 100
            elif o.attackId == ATTACK_TELEPORTATION:
                score += 50

        scores.append(score)

    if return_scores:
        return scores

    # Select in descending order of score
    desc_indices = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]

    if context == SelectContext.MAIN:
        o = select.option[desc_indices[0]]
        if o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            if card is not None:
                if card.id == Dudunsparce:
                    ability_used_dudunsparce = True
                elif card.id == Fezandipiti_ex:
                    ability_used_fezandipiti = True

    return desc_indices[:select.maxCount]


def agent(obs_dict: dict) -> list[int]:
    try:
        if isinstance(obs_dict, dict) and obs_dict.get("select") is None:
            _DIAG["deck_returns"] += 1
            return my_deck
    except Exception:
        pass

    _DIAG["decisions"] += 1
    try:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            _DIAG["deck_returns"] += 1
            _DIAG["decisions"] -= 1
            return my_deck
        raw = _policy_agent(obs_dict)
        selection = _normalize_ordered_indices(raw, obs.select)
        if selection:
            _DIAG["policy_ok"] += 1
            return selection
        _DIAG["policy_fallback"] += 1
        return _legal_fallback(obs.select)
    except Exception:
        _DIAG["policy_fallback"] += 1
        try:
            obs = to_observation_class(obs_dict)
            if obs.select is None:
                _DIAG["deck_returns"] += 1
                _DIAG["decisions"] -= 1
                return my_deck
            return _legal_fallback(obs.select)
        except Exception:
            _DIAG["obs_fallback"] += 1
            if isinstance(obs_dict, dict) and obs_dict.get("select") is None:
                return my_deck
            return []
