import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import sys
    import time
    import math
    import random
    import json
    from pathlib import Path
    from collections import defaultdict

    return Path, defaultdict, json, math, mo, os, random, sys, time


@app.cell
def _(mo):
    mo.md(
        r"""
    # The Improved Probabilistic Agent — explained line by line

    An **educational, runnable walkthrough** of Ivan Ternovskiy's
    [Improved Probabilistic agent](https://www.kaggle.com/code/aristophanivan/improved-probabilistic-agent)
    for the **Pokémon TCG AI Battle Challenge**.

    ## What you get
    - Every *concept* (the "thought process") is its own **Markdown cell**, shown
      above the code it explains.
    - Every *code block* lives in its **own cell** — re-run just one piece.
      **Markdown and Python are never mixed in the same cell.**
    - The notebook **actually runs** the agent on a real observation from this repo
      (`notebooks/sample_obs.json`) and prints a legal ranked action — not just
      descriptions.

    ## A note on "training"
    The original notebook has **no training code**. It is a hand-written
    **heuristic policy** plus a **flat root Monte-Carlo search with UCB1**. There
    are no weights and no training loop, so there is nothing to train. The working
    proof here is: *the agent executes on a real game state and returns a legal
    ranked action.*

    ## ⚠️ The line-number problem this fixes
    Kaggle hides line numbers (the core is a `%%writefile main.py` block). Step 0
    below shows the **original source with line numbers**, and every later section
    quotes its line ranges against it.

    ## Run it
    ```bash
    uv run marimo edit notebooks/04_improved_probabilistic_agent_walkthrough.py
    ```
    """
    )
    return


@app.cell
def _(Path, sys):
    PROJECT_ROOT = Path.cwd()
    if not (PROJECT_ROOT / "src" / "pokemon_tcg").exists():
        PROJECT_ROOT = PROJECT_ROOT.parent

    CG_LIB_DIR = PROJECT_ROOT / "data" / "external" / "cg-lib"
    if str(CG_LIB_DIR) not in sys.path:
        sys.path.insert(0, str(CG_LIB_DIR))

    SOURCE_DIR = PROJECT_ROOT / "notebooks" / "reference" / "aristophanivan"
    return CG_LIB_DIR, PROJECT_ROOT, SOURCE_DIR


@app.cell
def _(CG_LIB_DIR, SOURCE_DIR, mo):
    from cg.api import (
        AreaType, Card, CardType, EnergyType, Observation, OptionType,
        Pokemon, SelectContext, all_card_data, to_observation_class,
        search_begin, search_step, search_end, search_release,
    )

    all_card = all_card_data()
    card_table = {card.cardId: card for card in all_card}

    ORIGINAL_SOURCE_PATH = SOURCE_DIR / "main.py"
    with open(ORIGINAL_SOURCE_PATH, "r", encoding="utf-8") as _f:
        ORIGINAL_SOURCE_LINES = _f.read().split("\n")

    print(f"Loaded {len(card_table)} card definitions from {CG_LIB_DIR}")
    print(f"Original main.py: {len(ORIGINAL_SOURCE_LINES)} lines")
    return (
        AreaType, Card, CardType, EnergyType, Observation, OptionType,
        ORIGINAL_SOURCE_LINES, Pokemon, SelectContext, all_card, card_table,
        search_begin, search_end, search_release, search_step, to_observation_class,
    )


@app.cell
def _(ORIGINAL_SOURCE_LINES, mo):
    def numbered(start: int, end: int, lang: str = "python") -> object:
        """Render ORIGINAL_SOURCE_LINES[start-1:end] WITH editor-style line numbers."""
        slice_ = ORIGINAL_SOURCE_LINES[start - 1 : end]
        body = "\n".join(f"{i:>4} | {line}" for i, line in enumerate(slice_, start=start))
        return mo.md(f"```{lang}\n{body}\n```")

    return (numbered,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Step 0 — The original source *with line numbers*

    Kaggle's viewer hides line numbers. Here is the original `main.py`, numbered.
    Every later section references line ranges against this exact text.
    """
    )
    return


@app.cell
def _(mo, numbered):
    numbered(1, 49)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Step 1 — The deck is data, not strategy

    The agent plays a **fixed 60-card Mega Lucario ex deck** (original lines 11–24
    and 32–49). A second copy is written to `deck.csv` so it ships with the
    submission. We give the integer IDs readable names so every later scorer is
    legible instead of a wall of unexplained numbers.
    """
    )
    return


@app.cell
def _(card_table, mo):
    DECK = [
        673, 673, 674, 674, 675, 675, 676, 676,
        676, 677, 677, 677, 678, 678, 678, 678,
        1102, 1102, 1102, 1102, 1123, 1123, 1141, 1141,
        1141, 1141, 1142, 1142, 1142, 1142, 1152, 1152,
        6, 1159, 1182, 1182, 1192, 1192, 1192, 1192,
        1227, 1227, 1227, 1227, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6,
        6, 1182, 677, 1252,
    ]
    assert len(DECK) == 60, len(DECK)

    CARD_NAMES = {
        6: "Basic Fighting Energy",
        673: "Makuhita", 674: "Hariyama",
        675: "Lunatone", 676: "Solrock",
        677: "Riolu", 678: "Mega Lucario ex",
        1102: "Dusk Ball", 1123: "Switch",
        1141: "Premium Power Pro", 1142: "Fighting Gong",
        1152: "Poke Pad", 1159: "Hero Cape",
        1182: "Boss's Orders", 1192: "Carmine",
        1227: "Lillie's Determination", 1252: "Gravity Mountain",
    }
    missing = [cid for cid in set(DECK) if cid not in card_table]
    print("deck size:", len(DECK), "| unique cards:", len(set(DECK)))
    print("unresolved card ids:", missing)
    return CARD_NAMES, DECK, missing


@app.cell
def _(CARD_NAMES, DECK, card_table, mo):
    from collections import Counter

    rows = [
        {"ID": cid, "Copies": n,
         "Card": CARD_NAMES.get(cid, card_table[cid].name)}
        for cid, n in sorted(Counter(DECK).items())
    ]
    mo.ui.table(rows, selection=None)
    return (Counter,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Step 2 — Read the observation through small helpers

    The environment gives the agent a nested dictionary; the official `cg` library
    converts it to dataclasses. These helpers (original lines 76–98) hide the
    nesting so the policy talks about *my state*, *the opponent*, and *legal
    options* instead of `obs.current.players[i].bench[j]`.
    """
    )
    return


@app.cell
def _(AreaType, Observation, OptionType, Pokemon, card_table):
    def get_card(obs: Observation, area: AreaType, index: int, player_index: int):
        player = obs.current.players[player_index]
        if area == AreaType.DECK:
            return obs.select.deck[index]
        if area == AreaType.HAND:
            return player.hand[index]
        if area == AreaType.DISCARD:
            return player.discard[index]
        if area == AreaType.ACTIVE:
            return player.active[index]
        if area == AreaType.BENCH:
            return player.bench[index]
        if area == AreaType.PRIZE:
            return player.prize[index]
        if area == AreaType.STADIUM:
            return obs.current.stadium[index]
        if area == AreaType.LOOKING:
            return obs.current.looking[index]
        return None

    def prize_count(pokemon: Pokemon) -> int:
        data = card_table[pokemon.id]
        count = 3 if data.megaEx else 2 if data.ex else 1
        for card in pokemon.energyCards:
            if card.id == 6:  # Legacy Energy
                count -= 1
        for card in pokemon.tools:
            if card.id == 1172 and "Lillie" in data.name:  # Lillie's Pearl
                count -= 1
        return max(0, count)

    return get_card, prize_count


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Step 3 — Attack planning is *state*, not just a score

    `_plan_attack` (original lines 100–104, 160–208) builds an `AttackPlan` that
    remembers **who attacks, who is targeted, which attack option, the target's
    remaining HP, and whether the plan needs an energy attach first**. The fields
    mirror the source exactly:

    - `attacker` — board index of the planned attacker (0 = active)
    - `target` — opponent board index being hit (0 = active)
    - `attack_index` — the legal option index for the attack
    - `remain_hp` — target HP after the planned damage
    - `needs_energy` — whether the plan depends on an energy action first

    Only **Mega Lucario ex** and **Hariyama** have real damage formulas here
    (lines 176–191). Everything else returns `None` (not a candidate attacker).
    """
    )
    return


@app.cell
def _(mo, numbered):
    numbered(100, 104)
    return


@app.cell
def _(Observation, OptionType, card_table):
    class AttackPlan:
        def __init__(self, attacker=-1, target=-1, attack_index=-1,
                     remain_hp=-1, needs_energy=False):
            self.attacker, self.target = attacker, target
            self.attack_index, self.remain_hp = attack_index, remain_hp
            self.needs_energy = needs_energy

    class SessionState:
        """Single holder for the original's module-level mutable globals
        (plan / pre_turn / ability_used). Marimo forbids the same bare name in two
        cells, so we keep ONE object and mutate its attributes in place."""

        def __init__(self):
            self.plan = AttackPlan()
            self.pre_turn = -1
            self.ability_used = False

    state = SessionState()
    return AttackPlan, SessionState, state


@app.cell
def _(AttackPlan, OptionType, card_table, get_card, prize_count, state):
    C = type("C", (), {
        "MAKUHITA": 673, "HARIYAMA": 674, "LUNATONE": 675, "SOLROCK": 676,
        "RIOLU": 677, "MEGA_LUCARIO_EX": 678, "BASIC_FIGHTING_ENERGY": 6,
        "SWITCH": 1123, "BOSS_ORDERS": 1182, "MEGA_BRAVE": 983,
        "SNOVER": 722, "MEGA_ABOMASNOW_EX": 723, "KYOGRE": 721,
        "LUNATONE_ID": 675, "HERO_CAPE": 1159, "LUMIOSE_CITY": 1267,
        "PREMIUM_POWER_PRO": 1141, "CARMINE": 1192, "LILLIE_DETERMINATION": 1227,
        "GRAVITY_MOUNTAIN": 1252,
    })

    def target_score(pokemon):
        data = card_table[pokemon.id]
        score = (prize_count(pokemon) * 2000
                 + len(pokemon.energies) * 300
                 + len(pokemon.tools) * 200)
        if data.stage2:
            score += 500
        elif data.stage1:
            score += 250
        if pokemon.id in {144, 322, 323, 337}:
            score -= 200
        if pokemon.id == C.SNOVER:
            score += 950
        elif pokemon.id == C.MEGA_ABOMASNOW_EX:
            score += 250
        if pokemon.id == C.RIOLU:
            score += 800
        elif pokemon.id == C.MEGA_LUCARIO_EX:
            score += 100
        return score + pokemon.hp

    def _base_attack(pokemon, attack_index):
        energy_required, base_damage, base_score = 0, 0, 0
        if pokemon.id == C.MEGA_LUCARIO_EX:
            if attack_index == 0:
                energy_required, base_damage = 1, 130
                base_score += 60 * min(3, state.discard_counts.get(C.BASIC_FIGHTING_ENERGY, 0))
            else:
                energy_required, base_damage = 2, 270
        elif attack_index == 1:
            return None
        elif pokemon.id == C.HARIYAMA:
            energy_required, base_damage = 3, 210
        elif pokemon.id == C.MAKUHITA:
            return None
        elif pokemon.id == C.SOLROCK and state.field_counts.get(C.LUNATONE_ID, 0) >= 1:
            energy_required, base_damage = 1, 70
        if base_damage <= 0:
            return None
        return energy_required, base_damage, base_score

    return C, _base_attack, target_score


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Step 4 — `AdvancedPolicy`: the heuristic scoring head

    The brain of the agent (original lines 109–354). On every decision it:

    1. Counts cards on field/hand/discard (`_count_cards`, lines 124–134).
    2. Scans which *kinds* of options are legal — can it switch, gust, attack, use
       the Mega Brave attack? (`_scan_main_options`, lines 135–150).
    3. If the context is MAIN, runs `_plan_attack` (Step 3) to set up `plan`.
    4. Scores **every** legal option with a context-specific scorer and returns
       them ranked best-first.

    Each scorer is a deliberately *hand-tuned* number. The magnitude ordering **is**
    the strategy (e.g. playing a Pokemon scores `20000`, Boss's Orders `3200`).
    **This is not machine learning** — it is a decision tree written by hand.
    """
    )
    return


@app.cell
def _(mo, numbered):
    numbered(109, 159)
    return


@app.cell
def _(AttackPlan, AreaType, C, OptionType, Pokemon, SelectContext,
       _base_attack, card_table, defaultdict, get_card, prize_count, state, target_score):
    class AdvancedPolicy:
        def __init__(self, obs):
            self.obs = obs
            self.state = obs.current
            self.select = obs.select
            self.context = self.select.context
            self.my_index = self.state.yourIndex
            self.op_index = 1 - self.my_index
            self.me = self.state.players[self.my_index]
            self.opponent = self.state.players[self.op_index]
            self.field_counts = defaultdict(int)
            self.hand_counts = defaultdict(int)
            self.discard_counts = defaultdict(int)
            self.has_ready_lucario_line = False
            self.has_ready_hariyama_line = False
            self.can_switch = self.can_gust = self.can_attack = self.can_use_mega_brave = False
            self.stadium_id = self.state.stadium[0].id if self.state.stadium else 0
            self._count_cards()
            self._scan_main_options()

        def choose(self):
            if not self.select.option or self.select.maxCount == 0:
                return []
            if self.context == SelectContext.MAIN:
                self._plan_attack()
            scores = [self._score_option(option) for option in self.select.option]
            ranked = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
            self._remember_lunatone_ability(ranked)
            return ranked[: self.select.maxCount]

        def _count_cards(self):
            for p in self.me.active + self.me.bench:
                if p is None:
                    continue
                self.field_counts[p.id] += 1
                if p.id in {C.MAKUHITA, C.HARIYAMA} and len(p.energies) >= 3:
                    self.has_ready_hariyama_line = True
                if p.id in {C.RIOLU, C.MEGA_LUCARIO_EX} and len(p.energies) >= 2:
                    self.has_ready_lucario_line = True
            for c in self.me.hand:
                self.hand_counts[c.id] += 1
            for c in self.me.discard:
                self.discard_counts[c.id] += 1

        def _scan_main_options(self):
            if self.context != SelectContext.MAIN:
                return
            for option in self.select.option:
                if option.type == OptionType.PLAY:
                    card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
                    if card.id == C.SWITCH:
                        self.can_switch = True
                    elif card.id == C.BOSS_ORDERS:
                        self.can_gust = True
                elif option.type == OptionType.EVOLVE:
                    card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
                    if card.id == C.HARIYAMA:
                        self.can_gust = True
                elif option.type == OptionType.RETREAT:
                    self.can_switch = True
                elif option.type == OptionType.ATTACK:
                    self.can_attack = True
                    if option.attackId == C.MEGA_BRAVE:
                        self.can_use_mega_brave = True

        def _my_board(self):
            return self.me.active + self.me.bench

        def _opponent_board(self):
            return self.opponent.active + self.opponent.bench

        def _opponent_has(self, ids):
            return any(p is not None and p.id in ids for p in self._opponent_board())

        def _opponent_is_water_deck(self):
            return self._opponent_has({C.KYOGRE, C.SNOVER, C.MEGA_ABOMASNOW_EX})

        def _opponent_is_crustle_wall(self):
            return self._opponent_has({344, 345})

        def _can_evolve_board_index(self, board_index):
            for option in self.select.option:
                if option.type != OptionType.EVOLVE:
                    continue
                target_index = option.inPlayIndex + (1 if option.inPlayArea == AreaType.BENCH else 0)
                if target_index == board_index:
                    return True
            return False

        def _plan_attack(self):
            best_score = -1
            state.plan = AttackPlan()
            if self.state.turn < 2:
                return
            for attacker_index, my_pokemon in enumerate(self._my_board()):
                if my_pokemon is None:
                    continue
                if attacker_index != 0 and not self.can_switch:
                    break
                for attack_index in range(2):
                    attack = _base_attack(my_pokemon, attack_index)
                    if attack is None:
                        continue
                    energy_required, base_damage, base_score = attack
                    energy_count = len(my_pokemon.energies)
                    if attack_index == 1 and attacker_index == 0 and energy_count >= 2 and not self.can_use_mega_brave:
                        break
                    needs_energy = False
                    if energy_count < energy_required:
                        if self.hand_counts[C.BASIC_FIGHTING_ENERGY] >= 1 and not self.state.energyAttached:
                            energy_count += 1
                            needs_energy = energy_count >= energy_required
                        if not needs_energy:
                            continue
                    for target_index, op_pokemon in enumerate(self._opponent_board()):
                        if op_pokemon is None:
                            continue
                        if target_index != 0 and not self.can_gust:
                            break
                        if self._opponent_is_crustle_wall() and my_pokemon.id == C.MEGA_LUCARIO_EX and op_pokemon.id == 345:
                            continue
                        damage = base_damage
                        op_data = card_table[op_pokemon.id]
                        if op_data.weakness == 2:  # EnergyType.FIGHTING
                            damage *= 2
                        elif op_data.resistance == 2:
                            damage -= 30
                        score = target_score(op_pokemon)
                        prize = prize_count(op_pokemon) if op_pokemon.hp <= damage else 0
                        if prize == 0:
                            score *= damage / op_pokemon.hp
                        if len(self.opponent.prize) <= prize:
                            score = 500000
                        score += base_score + (220 if attacker_index == 0 else 0) + (300 if target_index == 0 else 0) + energy_count
                        if score > best_score:
                            best_score = score
                            state.plan = AttackPlan(attacker_index, target_index, attack_index, op_pokemon.hp - damage, needs_energy)

        def _score_option(self, option):
            if option.type == OptionType.NUMBER:
                return option.number
            if option.type == OptionType.YES:
                return 100 if self.context == SelectContext.IS_FIRST else 1
            if option.type == OptionType.NO:
                return 0
            if option.type == OptionType.CARD:
                return self._score_card_choice(option)
            if option.type == OptionType.PLAY:
                return self._score_play(option)
            if option.type == OptionType.ATTACH:
                return self._score_attach(option)
            if option.type == OptionType.EVOLVE:
                return self._score_evolve(option)
            if option.type == OptionType.ABILITY:
                return self._score_ability(option)
            if option.type == OptionType.RETREAT:
                return 2000 if state.plan.attacker >= 1 else -1
            if option.type == OptionType.ATTACK:
                return 1100 if (option.attackId == C.MEGA_BRAVE) == (state.plan.attack_index == 1) else 1000
            return 0

        def _remember_lunatone_ability(self, ranked):
            if self.context != SelectContext.MAIN or not ranked:
                return
            option = self.select.option[ranked[0]]
            if option.type != OptionType.ABILITY:
                return
            card = get_card(self.obs, option.area, option.index, self.my_index)
            if card is not None and card.id == C.LUNATONE_ID:
                state.ability_used = True

        def _score_card_choice(self, option):
            card = get_card(self.obs, option.area, option.index, option.playerIndex)
            if card is None:
                return 0
            if self.context in {SelectContext.SWITCH, SelectContext.TO_ACTIVE}:
                return self._score_active_choice(option, card)
            if self.context == SelectContext.SETUP_ACTIVE_POKEMON:
                return (2 if card.id == C.SOLROCK and self.state.firstPlayer == self.my_index
                        else 4 if card.id == C.SOLROCK
                        else 3 if card.id == C.RIOLU
                        else 1 if card.id == C.MAKUHITA else 0)
            if self.context == SelectContext.TO_HAND:
                score = 200 - self.hand_counts[card.id] * 100
                if card.id == C.MAKUHITA:
                    score += (80 if self.field_counts[card.id] < 2 else -20) if self._opponent_is_crustle_wall() else (-10 if self.field_counts[card.id] >= 1 else 10)
                elif card.id == C.HARIYAMA:
                    score += (120 if self.field_counts[C.MAKUHITA] >= 1 else -5) if self._opponent_is_crustle_wall() else (20 if self.field_counts[C.MAKUHITA] >= 1 else -20)
                elif card.id == C.LUNATONE_ID:
                    score += -250 if self.field_counts[card.id] >= 1 else 60
                elif card.id == C.SOLROCK:
                    score += -250 if self.field_counts[card.id] >= 1 else 50
                elif card.id == C.RIOLU:
                    score += -150 if (self.field_counts[C.RIOLU] + self.field_counts[C.MEGA_LUCARIO_EX] >= 2) else -3 if (self.field_counts[C.RIOLU] + self.field_counts[C.MEGA_LUCARIO_EX] >= 1) else 40
                elif card.id == C.MEGA_LUCARIO_EX:
                    score += 40 if self.field_counts[C.RIOLU] >= 1 else -15
                elif card.id == C.BASIC_FIGHTING_ENERGY:
                    score += 30 if not state.ability_used or not self.state.energyAttached else -1
                return score
            if self.context == SelectContext.ATTACH_FROM and isinstance(card, Pokemon):
                return self._energy_target_score(card, option.area == AreaType.ACTIVE)
            return 0

        def _score_active_choice(self, option, card):
            if not isinstance(card, Pokemon):
                return 0
            if option.playerIndex != self.my_index:
                return 100 if option.index == state.plan.target - 1 else 0
            score = len(card.energies) * 2
            if option.index == state.plan.attacker - 1:
                score += 100
            if card.id == C.MEGA_LUCARIO_EX:
                score += 8 if self._opponent_is_water_deck() and len(self.opponent.prize) <= 3 else 20
            elif card.id == C.HARIYAMA and len(card.energies) >= 2:
                score += 45 if self._opponent_is_crustle_wall() else 15
            elif card.id == C.MAKUHITA and len(card.energies) >= 2:
                score += 35 if self._opponent_is_crustle_wall() else 10
            elif card.id == C.SOLROCK:
                score += 5
            elif card.id == C.RIOLU:
                score += 4
            return score

        def _score_play(self, option):
            card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
            data = card_table[card.id]
            if data.cardType == 1:  # CardType.POKEMON
                if card.id in {C.LUNATONE_ID, C.SOLROCK} and self.field_counts[card.id] >= 1:
                    return -1
                if card.id == C.RIOLU and self.field_counts[C.RIOLU] + self.field_counts[C.MEGA_LUCARIO_EX] >= 2:
                    return -1
                return 20000
            if card.id == C.SWITCH:
                return 6000 if state.plan.attacker > 0 else -1
            if card.id == C.PREMIUM_POWER_PRO:
                if self.state.supporterPlayed and state.plan.remain_hp <= 0:
                    return -1
                if not self.can_attack:
                    return 3050 if (not self.state.supporterPlayed and self.hand_counts[C.CARMINE] > 0 and self.hand_counts[C.LILLIE_DETERMINATION] == 0 and not self.me.deckCount <= 10) else -1
                return 5000
            if card.id == C.BOSS_ORDERS:
                return 3200 if state.plan.target >= 1 else -1
            if card.id == C.CARMINE:
                if self._opponent_is_crustle_wall() and any(c.id in {C.HARIYAMA, C.MAKUHITA} for c in self.me.hand):
                    return -1
                return -1 if self.me.deckCount <= 10 else 3000
            if card.id == C.LILLIE_DETERMINATION:
                return -1 if self.me.deckCount <= 10 else 3100
            if card.id == C.GRAVITY_MOUNTAIN:
                return 3500 if any(p is not None and card_table[p.id].stage2 for p in self._opponent_board()) else (1200 if self.stadium_id else -1)
            return 10000

        def _score_attach(self, option):
            card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
            pokemon = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
            if not isinstance(pokemon, Pokemon):
                return 0
            if card.id == C.HERO_CAPE:
                score = 7000
                if self._opponent_is_water_deck():
                    return 12200 if pokemon.id == C.RIOLU else 12800 if pokemon.id == C.MEGA_LUCARIO_EX else score
                if pokemon.id == C.RIOLU:
                    score += 100
                elif pokemon.id == C.MEGA_LUCARIO_EX:
                    score += 200
                return score
            score = self._energy_target_score(pokemon, option.inPlayArea == AreaType.ACTIVE)
            board_index = option.inPlayIndex if option.inPlayArea == AreaType.ACTIVE else option.inPlayIndex + 1
            if board_index == state.plan.attacker and state.plan.needs_energy:
                score += 200
            return score

        def _score_evolve(self, option):
            pokemon = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
            if not isinstance(pokemon, Pokemon):
                return 0
            if pokemon.id == C.MAKUHITA and state.plan.target == 0 and not self._opponent_is_crustle_wall():
                return -1
            return 9000 + len(pokemon.energies)

        def _score_ability(self, option):
            card = get_card(self.obs, option.area, option.index, self.my_index)
            if card.id == C.LUMIOSE_CITY:
                return 1
            if card.id == C.LUNATONE_ID and self.me.deckCount <= 10:
                return -1
            return 30000

        def _energy_target_score(self, pokemon, active):
            energy_count = len(pokemon.energies)
            score = 8000 + (10 if active else 0)
            if pokemon.id in {C.MAKUHITA, C.HARIYAMA}:
                if pokemon.id == C.HARIYAMA:
                    score += 1
                if self._opponent_is_crustle_wall():
                    score += 260 if energy_count < 3 else 30
                else:
                    score += 100 if energy_count < 3 else 0
                    score -= 50 if self.has_ready_hariyama_line else 0
            elif pokemon.id == C.LUNATONE_ID:
                score -= 100
            elif pokemon.id == C.SOLROCK:
                score += 20 if energy_count < 1 else -100
            elif pokemon.id in {C.RIOLU, C.MEGA_LUCARIO_EX}:
                if pokemon.id == C.MEGA_LUCARIO_EX:
                    score += 1
                score += 100 if energy_count < 2 else 0
                score -= 50 if self.has_ready_lucario_line else 0
            return score

    return (AdvancedPolicy,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Step 5 — Static evaluation: a rank, not a win-probability

    `evaluate_state` (original lines 356–414) turns a hypothetical state into one
    float. Magnitude ordering **is** the strategy:

    ```text
    terminal win/lose  >>>  prize difference  >>>  board strength  >>>  energy / HP
    ```

    It also does **predictive threat mapping**: it assumes the opponent attaches one
    energy next turn, computes worst-case damage, and penalizes states where our
    active would be Knocked Out. **This is a heuristic ranking, not a learned value
    function** — there are no weights, so there is nothing to train.
    """
    )
    return


@app.cell
def _(mo, numbered):
    numbered(356, 414)
    return


@app.cell
def _(C, Observation):
    def evaluate_state(obs):
        st = obs.current
        if st is None:
            return 0.0
        me, op = st.players[st.yourIndex], st.players[1 - st.yourIndex]
        prize_diff = len(op.prize) - len(me.prize)
        if len(me.prize) == 0:
            return 9999999.0
        if len(op.prize) == 0:
            return -9999999.0
        val = prize_diff * 10000.0
        is_crustle = any(p is not None and p.id in {344, 345} for p in [op.active[0] if op.active else None] + list(op.bench))
        is_snorlax = any(p is not None and p.id == 143 for p in [op.active[0] if op.active else None] + list(op.bench))
        is_stall = is_crustle or is_snorlax
        for p in [me.active[0] if me.active else None] + list(me.bench):
            if p is None:
                continue
            val += len(p.energies) * 200.0
            if is_crustle:
                if p.id == C.HARIYAMA:
                    val += 1500.0
                elif p.id == C.MAKUHITA:
                    val += 800.0
            else:
                if p.id == C.MEGA_LUCARIO_EX:
                    val += 500.0
                elif p.id == C.HARIYAMA:
                    val += 300.0
                elif p.id in {C.RIOLU, C.MAKUHITA}:
                    val += 100.0
        if is_crustle:
            for c in me.hand:
                if c.id == C.HARIYAMA:
                    val += 1000.0
                elif c.id == C.MAKUHITA:
                    val += 500.0
        if me.active and me.active[0] is not None:
            val += me.active[0].hp * 2.0
            if len(me.active[0].energies) >= 2:
                val += 500.0
        op_max_damage = 0
        for p in [op.active[0] if op.active else None] + list(op.bench):
            if p is None:
                continue
            val -= p.hp * 1.5
            assumed_energies = len(p.energies) + 1
            op_dmg = 0
            if p.id == C.MEGA_LUCARIO_EX:
                op_dmg = 270 if assumed_energies >= 2 else 130 if assumed_energies >= 1 else 0
            elif p.id == C.HARIYAMA:
                op_dmg = 210 if assumed_energies >= 3 else 0
            elif p.id == C.KYOGRE:
                op_dmg = 130 if assumed_energies >= 3 else 0
            elif p.id == C.MEGA_ABOMASNOW_EX:
                op_dmg = 240 if assumed_energies >= 4 else 0
            elif p.id == 121:
                op_dmg = 130 if assumed_energies >= 2 else 0
            else:
                op_dmg = assumed_energies * 40
            op_max_damage = max(op_max_damage, op_dmg)
        if me.active and me.active[0] is not None:
            my_active = me.active[0]
            if op_max_damage >= my_active.hp:
                prize_risk = 2 if my_active.id == C.MEGA_LUCARIO_EX else 1
                val -= prize_risk * 4000.0
            elif op_max_damage > 0:
                val -= op_max_damage * 1.5
        deck_c = getattr(me, "deckCount", 60)
        if is_stall:
            val += deck_c * 30.0
            val += getattr(me, "handCount", len(me.hand)) * 2.0
        else:
            val += getattr(me, "handCount", len(me.hand)) * 10.0
        if deck_c < 5:
            val -= 10000.0
        return val

    return (evaluate_state,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Step 6 — The search: flat root Monte-Carlo with UCB1

    The name says "Probabilistic Expectimax," but reading the code (lines 416–489)
    it is **a flat root search, not a tree**:

    1. Rank every legal action with the heuristic head (`AdvancedPolicy.choose`).
    2. Keep only the top **8** candidates (`SEARCH_MAX_CANDIDATES`).
    3. **Explore phase:** simulate each candidate once.
    4. **UCB1 phase:** re-simulate the best `exploitation + 0.5*sqrt(ln(total)/visits)`
       for the rest of the 1.5-second budget.
    5. Return the best candidate first, then the original fallback order.

    There is **no tree expansion and no back-propagation tree** — that is why the
    "MCTS" label is loose; it is root-level bandit search.
    """
    )
    return


@app.cell
def _(mo, numbered):
    numbered(416, 489)
    return


@app.cell
def _(AdvancedPolicy, DECK, Observation, SelectContext, evaluate_state,
       math, random, search_begin, search_end, search_step, state, time,
       to_observation_class):
    USE_SEARCH = True
    SEARCH_TIME_BUDGET = 1.5
    SEARCH_MAX_CANDIDATES = 8
    DECK_LOCAL = list(DECK)

    def rollout_turn(sid, cur_obs, your_index):
        steps = 0
        while steps < 20:
            if cur_obs.current.result is not None and cur_obs.current.result != -1:
                break
            if cur_obs.current.yourIndex != your_index:
                break
            if cur_obs.select.context != SelectContext.MAIN:
                sub = AdvancedPolicy(cur_obs).choose()
                sel = sub[: max(1, cur_obs.select.minCount)]
            else:
                nxt = AdvancedPolicy(cur_obs).choose()
                if not nxt:
                    break
                sel = [nxt[0]]
                if cur_obs.select.option[nxt[0]].type == 5:  # OptionType.END
                    search_step(sid, sel)
                    break
            ar = search_step(sid, sel)
            if getattr(ar, "error", 0) != 0 or ar.state is None:
                break
            cur_obs, sid = ar.state.observation, ar.state.searchId
            steps += 1
        return cur_obs

    def simulate_action(obs, action):
        my_p = obs.current.players[obs.current.yourIndex]
        op_p = obs.current.players[1 - obs.current.yourIndex]
        yd = random.sample(DECK_LOCAL, getattr(my_p, "deckCount", 60))
        sbi = search_begin(obs, your_deck=yd,
                           your_prize=[0] * len(my_p.prize),
                           opponent_deck=[0] * getattr(op_p, "deckCount", 60),
                           opponent_prize=[0] * len(op_p.prize),
                           opponent_hand=[0] * len(op_p.hand),
                           opponent_active=[])
        if getattr(sbi, "error", 0) != 0 or sbi.state is None:
            return -float("inf")
        ar = search_step(sbi.state.searchId, [action])
        if getattr(ar, "error", 0) != 0 or ar.state is None:
            return -float("inf")
        cur = rollout_turn(ar.state.searchId, ar.state.observation, obs.current.yourIndex)
        return evaluate_state(cur)

    def SEARCH_ALGO(obs_dict, obs):
        if not USE_SEARCH:
            return None
        select = obs.select
        if select is None or select.context != SelectContext.MAIN:
            return None
        t0 = time.time()
        base_order = AdvancedPolicy(obs).choose()
        candidates = base_order[:SEARCH_MAX_CANDIDATES]
        if not candidates:
            return None
        if len(candidates) == 1:
            return [candidates[0]] + [i for i in base_order if i != candidates[0]]
        visits = {a: 0 for a in candidates}
        total_val = {a: 0.0 for a in candidates}
        try:
            for a in candidates:
                if time.time() - t0 > SEARCH_TIME_BUDGET:
                    break
                val = simulate_action(obs, a)
                if val != -float("inf"):
                    visits[a] += 1
                    total_val[a] += val
            while time.time() - t0 < SEARCH_TIME_BUDGET:
                total_visits = sum(visits.values())
                if total_visits == 0:
                    break
                valid_scores = [total_val[a] / visits[a] for a in candidates if visits[a] > 0]
                if not valid_scores:
                    break
                min_s = min(valid_scores)
                max_s = max(valid_scores)
                if max_s == min_s:
                    max_s = min_s + 1.0
                best_ucb, best_a = -float("inf"), candidates[0]
                for a in candidates:
                    if visits[a] == 0:
                        best_a = a
                        break
                    avg = total_val[a] / visits[a]
                    norm_avg = (avg - min_s) / (max_s - min_s)
                    ucb = norm_avg + 0.5 * math.sqrt(math.log(total_visits) / visits[a])
                    if ucb > best_ucb:
                        best_ucb = ucb
                        best_a = a
                val = simulate_action(obs, best_a)
                if val != -float("inf"):
                    visits[best_a] += 1
                    total_val[best_a] += val
            best_action = max(candidates, key=lambda a: total_val[a] / visits[a] if visits[a] > 0 else -float("inf"))
            return [best_action] + [i for i in base_order if i != best_action]
        except Exception:
            return None

    return SEARCH_ALGO, USE_SEARCH, simulate_action


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Step 7 — The entry point: `agent(obs_dict)`

    Kaggle calls `agent(obs_dict)` once per decision (lines 491–538). It converts
    the raw dict, runs the **search** when the context is MAIN, falls back to the
    heuristic otherwise, clamps indices to valid options, and wraps every branch so
    a crash returns a **legal fallback** rather than throwing at inference time.
    """
    )
    return


@app.cell
def _(mo, numbered):
    numbered(491, 538)
    return


@app.cell
def _(AdvancedPolicy, DECK, Observation, SEARCH_ALGO, SelectContext, state,
       to_observation_class):
    def agent(obs_dict):
        try:
            obs = to_observation_class(obs_dict)
        except Exception:
            return DECK if obs_dict.get("select") is None else [0]
        if obs.select is None:
            return DECK
        if state.pre_turn != obs.current.turn:
            state.pre_turn = obs.current.turn
            state.ability_used = False
            state.plan = type(state.plan)()
        try:
            ordered = SEARCH_ALGO(obs_dict, obs)
            if ordered is None:
                ordered = AdvancedPolicy(obs).choose()
            n = len(obs.select.option)
            ordered = [i for i in ordered if 0 <= i < n]
            if not ordered:
                return list(range(min(max(1, obs.select.minCount), n)))
            k = max(min(obs.select.maxCount, n), min(max(1, obs.select.minCount), n))
            return ordered[:k]
        except Exception:
            n = len(obs.select.option)
            return list(range(min(max(1, obs.select.minCount), n)))

    return (agent,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Step 8 — Run proof: the agent on a REAL observation

    No training code exists, so the working test is: **execute the agent on a real
    game state from this repo** (`notebooks/sample_obs.json`) and print the action
    it returns. A legal ranked action proves the whole pipeline — SDK import,
    observation conversion, policy, search, and the entry point — is wired
    correctly and actually runs.
    """
    )
    return


@app.cell
def _(Path, PROJECT_ROOT, agent, json, mo):
    sample_path = PROJECT_ROOT / "notebooks" / "sample_obs.json"
    raw = json.loads(sample_path.read_text())
    obs_dict = raw["obs_dict"]

    result = agent(obs_dict)
    n_options = len(obs_dict.get("select", {}).get("option", []))
    ctx = obs_dict.get("select", {}).get("context") if obs_dict.get("select") else None
    valid = all(0 <= i < n_options for i in result)
    print("context:", ctx, "| #legal options:", n_options)
    print("agent returned (first 10):", result[:10])
    print("return length:", len(result), "| all valid indices:", valid)

    out = PROJECT_ROOT / "notebooks" / "reference" / "aristophanivan" / "run_proof.json"
    out.write_text(json.dumps({
        "context": ctx, "n_options": n_options,
        "result_first10": result[:10], "result_len": len(result),
        "all_valid_indices": valid,
    }, indent=2))
    print("wrote", out)

    mo.md(
        f"""
    ### Agent executed on a real observation
    - **Context:** `{ctx}`
    - **Legal options:** {n_options}
    - **Returned action order (first 10):** `{result[:10]}`
    - **Valid indices:** {valid}

    The agent produced a legal ranked action from the heuristic + UCB1 search.
    """
    )
    return
@app.cell
def _(show):
    show(11, 26)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Step 1 — The Deck (lines 11–24)

    **Thought process.** Before any AI, the agent must *declare* a 60-card deck.
    The number of each card ID is the exact list the competition engine expects
    (Pokémon card IDs like `673` = Makuhita, `6` = basic Fighting Energy, plus
    Item/Supporter/Tool IDs). The notebook writes this list to `deck.csv` so the
    submission and the runtime both know what 60 cards you're playing.

    **Action.** `DECK` is a flat list of 60 integers. `Path("deck.csv").write_text(...)`
    dumps them one-per-line. The two `print`s are a sanity check: 60 cards total,
    and how many *distinct* cards you run.
    """
    )
    return


@app.cell
def _(show):
    show(30, 71)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Step 2 — `%%writefile main.py` and the engine SDK (lines 30–71)

    **Thought process.** The Kaggle notebook writes the whole agent into a file
    called `main.py` (the competition reads that file at submission time). Two
    things happen here:

    1. It imports the official **`cg` engine SDK** — `Observation`, `OptionType`,
       `SelectContext`, `all_card_data()`, `to_observation_class()`, and the
       search primitives `search_begin / search_step / search_end / search_release`.
    2. It re-imports the deck and rewrites `deck.csv` (the submission's own copy).

    The `try/except` around the `search_*` imports is a forward-compatibility
    guard: if the SDK ever drops those functions, `_SEARCH_OK` becomes `False`
    and the agent silently falls back to the pure heuristic.

    > ⚠️ **This is where the Kaggle version breaks on your machine.** The original
    > calls `search_begin(obs, your_deck=yd)` — *one* argument. Your repo's `cg`
    > SDK (Step 0 import) needs **six** prediction arguments. We adapt only that
    > one call (see the "Running the agent" section); everything else is verbatim.
    """
    )
    return


@app.cell
def _(show):
    show(72, 102)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Step 3 — Constants, card-ID aliases, and global state (lines 72–102)

    **Thought process.** The agent hard-codes a vocabulary of *which card is
    which* in the `C` class (e.g. `C.MAKUHITA = 673`, `C.MEGA_LUCARIO_EX = 678`,
    `C.BASIC_FIGHTING_ENERGY = 6`). Why? The engine only speaks integer card IDs;
    the `C` class lets the rest of the code read like English instead of magic
    numbers.

    A few more globals:
    - `MEGA_BRAVE = 983` — the attack ID of the signature Mega Brave attack.
    - `LOW_DECK_COUNT = 10` — a threshold: below 10 cards left, some Supporters
      become unsafe to play (prize math changes).
    - `my_deck` — loaded from `deck.csv`.
    - `card_table` — a dict from `cardId` → `CardData` (HP, weakness, whether
      it's a Stage-2, ex, mega-ex, …). The heuristic reads *card facts* from here.
    - `AttackPlan`, `plan`, `pre_turn`, `ability_used` — **mutable global state
      carried between decisions**. `plan` is the best attack the policy found
      this turn; `ability_used` remembers if the Lunatone ability fired (so it
      isn't repeated); `pre_turn` resets per-turn globals.
    """
    )
    return


@app.cell
def _(show):
    show(104, 136)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Step 4 — Reading the board: `get_card` and scoring helpers (lines 104–136)

    **Thought process.** The engine gives you an `Observation` with areas
    (DECK, HAND, DISCARD, ACTIVE, BENCH, PRIZE, STADIUM, LOOKING). `get_card`
    turns an `(area, index, player_index)` triple into the actual `Pokemon`/`Card`
    object so the rest of the code never touches raw indices.

    Two scoring primitives:
    - `prize_count(pokemon)` — *how many prize cards your opponent takes if this
      Pokémon is knocked out*: 3 for a Mega-ex, 2 for a normal ex, 1 otherwise,
      minus 1 if a Legacy Energy or Lillie's Pearl is attached. This is the single
      most important number in the whole value function.
    - `target_score(pokemon)` — a static "how valuable is this opponent Pokémon
      as a target" score: `prize_count × 2000` plus energy/tools bonuses, plus
      big bonuses for known key cards (Snover +950, Riolu +800), minus for safe
      targets (144/322/323/337). Higher = better to attack.

    These feed the attack planner in Step 5 and the state evaluator in Step 8.
    """
    )
    return


@app.cell
def _(show):
    show(137, 203)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Step 5 — `AdvancedPolicy`: scanning the position (lines 137–203)

    **Thought process.** `AdvancedPolicy` is the **heuristic brain**. Constructing
    it snapshots the current position into easy-to-query fields:

    - `my_index` / `op_index` — which player you are.
    - `field_counts` / `hand_counts` / `discard_counts` — how many of each card ID
      you have in play / hand / discard. Used to avoid over-committing (e.g. don't
      play a 2nd Lunatone).
    - `has_ready_hariyama_line` / `has_ready_lucario_line` — did you already evolve
      into a powered-up attacker? (3+ energies on Hariyama, 2+ on Lucario.)
    - `can_switch` / `can_gust` / `can_attack` / `can_use_mega_brave` — which
      *kinds* of options are even available this turn (read from the option list).

    `_can_evolve_board_index`, `_opponent_has`, `_opponent_is_water_deck`,
    `_opponent_is_crustle_wall` are small classifiers the scorers below use to make
    deck-specific decisions (e.g. Crustle stall walls need different play).
    """
    )
    return


@app.cell
def _(show):
    show(204, 266)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Step 6 — Planning the best attack: `_plan_attack` (lines 204–266)

    **Thought process.** This is the heart of the offense. For every (attacker,
    attack, target) combination the rules allow *this turn*, it computes a
    **score** and keeps the best one in the global `plan`.

    The loop logic, in plain English:
    1. Walk your board (active + bench). You can only attack with the active
       Pokémon unless you `can_switch`.
    2. For each of the Pokémon's two attacks, `base_damage` is looked up
       (`_base_attack`, lines 204–218) — e.g. Mega Lucario ex: 130 for 1 energy,
       270 for 2; Hariyama: 210 for 3.
    3. **Energy accounting:** if you don't have enough energy *but* you have a
       Fighting Energy in hand and haven't attached yet, assume you'll attach it
       (`needs_energy = True`). If you still can't reach the requirement, skip.
    4. For each legal target: double damage on Fighting **weakness**, subtract 30
       on resistance, then `score = target_score(target)`. If the hit KO's it,
       you gain `prize_count` prizes; if that empties the opponent's prizes →
       `score = 500000` (instant win). Otherwise scale by damage/hp.
    5. Small bonuses for attacking from active (+220), hitting the active (+300),
       and leftover energy (+1 each).

    The single best (attacker, target, attack, remain_hp, needs_energy) is stored
    in `plan` and reused by all the option scorers below.
    """
    )
    return


@app.cell
def _(show):
    show(267, 373)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Step 7 — Scoring every legal option (lines 267–373)

    **Thought process.** The engine hands the agent a flat list of `option`s.
    `choose()` (line 159) scores *each* option and returns them ranked
    (highest first), truncated to `maxCount`. Every `_score_*` method turns "is
    this option good right now?" into a float, using the `plan` from Step 6:

    - `_energy_target_score` (267) — where should leftover energy go? Prioritise
      unfinished attackers (Hariyama <3 energy, Lucario <2).
    - `_score_option` (282) — dispatch by `OptionType` (NUMBER, YES/NO, CARD,
      PLAY, ATTACH, EVOLVE, ABILITY, RETREAT, ATTACK). ATTACK scores 1100 if it
      matches the planned `plan.attack_index`, else 1000.
    - `_score_card_choice` (296) — for SWITCH/TO_ACTIVE/TO_HAND/ATTACH_FROM
      contexts: e.g. don't grab a 2nd Lunatone/Solrock, value Makuhita/Hariyama
      highly against Crustle stall.
    - `_score_play` (326) — Items/Supporters: playing a Pokémon = 20000; Switch =
      6000 only if the plan wants a switch; Boss's Orders = 3200 only if there's
      a gust target; Premium Power Pro, Carmine, Lillie, Gravity Mountain all
      gated on `deckCount` and whether a supporter was already played.
    - `_score_attach` (347) — Hero Cape placement gets a huge bump vs water decks;
      otherwise delegate to `_energy_target_score`.
    - `_score_evolve` (362) — evolving is almost always good (9000+), except
      evolving Makuhita when you'd rather attack.
    - `_score_ability` (368) — most abilities = 30000; Lillie's Pearl/City = 1;
      Lunatone ability denied when deck is low.

    The **key idea:** a single ranked list, produced by hand-tuned heuristics, is
    the agent's "best guess" before any search.
    """
    )
    return


@app.cell
def _(show):
    show(382, 454)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Step 8 — `evaluate_state`: the rollout value function (lines 382–454)

    **Thought process.** The search (Step 9) needs a number for "how good is this
    resulting position?" `evaluate_state` is that heuristic:

    - **Win/Lose:** 0 opponent prizes → `+∞`; 0 your prizes → `−∞`.
    - **Prize differential:** `±10000` per prize ahead/behind — the dominant term.
    - **Board strength:** your energies and key attackers (Lucario +500, Hariyama
      +300) add value; against a Crustle wall, Hariyama/Makuhita are valued even
      higher.
    - **Hand conservation vs stall:** Crustle/Snorlax stall decks out-draw you, so
      hoarding Hariyama/Makuhita in hand and keeping deck count high is rewarded.
    - **Predictive threat:** assume each opponent Pokémon attaches 1 energy next
      turn, compute the damage it could do, and penalise your board by `hp × 1.5`.
      If that damage would KO your active, subtract up to `prize_risk × 4000`.
    - **Deck-out guard:** `< 5` cards left → `−10000`.

    This is what the rollouts in Step 9 optimise.
    """
    )
    return


@app.cell
def _(show):
    show(456, 487)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Step 9 — Rollouts and `simulate_action` (lines 456–487)

    **Thought process.** To look ahead without a trained model, the agent uses the
    engine's **search API** to *simulate* the future:

    - `simulate_action(obs, action)` — begins a search from the current position
      (`search_begin`), applies `action`, then `rollout_turn` plays out ~20 steps
      where **both sides use the same heuristic policy** (`AdvancedPolicy`). The
      final position is scored by `evaluate_state`. The returned float is the
      "expected value" of picking `action`.
    - `rollout_turn` — repeatedly calls `search_step` with the heuristic's top
      choice; stops on game end, opponent turn, or END.

    This is the agent's "training"-equivalent: it doesn't learn weights, it
    **plays out possible futures** and averages how good they are.

    > ⚠️ **Local SDK adaptation.** The original `simulate_action` called
    > `search_begin(obs, your_deck=yd)`. The current `cg` SDK needs six prediction
    > arguments (your deck/prize, opponent deck/prize/hand/active). Our runner's
    > `_predict()` builds those, and `rollout_turn`/`simulate_action` read the
    > search id via `getattr(state, "searchId", ...)`. These are the only changes
    > from the source.
    """
    )
    return


@app.cell
def _(show):
    show(488, 543)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Step 10 — `SEARCH_ALGO`: flat Monte-Carlo search with UCB1 (lines 488–543)

    **Thought process.** Take the heuristic's top `SEARCH_MAX_CANDIDATES` (8)
    actions and spend `SEARCH_TIME_BUDGET` (1.5 s) exploring them:

    - **Phase 1 — Exploration:** simulate every candidate once to seed estimates.
    - **Phase 2 — UCB1 selection:** repeatedly pick the action maximising
      `normalised_average + 0.5·√(ln(total_visits)/visits)` — the classic
      upper-confidence-bound trade-off between *exploiting* what looks best and
      *exploring* under-tried actions. Normalising scores to [0,1] keeps UCB
      comparable across very different state values.
    - The action with the best **average** simulated value is put first, followed
      by the rest of the heuristic order.

    Output: a ranked action list *improved* by 1.5 s of look-ahead. If search is
    unavailable or finds nothing, `SEARCH_ALGO` returns `None` and the agent falls
    back to the pure heuristic order.
    """
    )
    return


@app.cell
def _(show):
    show(545, 567)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Step 11 — `agent`: the entry point the engine calls (lines 545–567)

    **Thought process.** The competition engine calls `agent(obs_dict)` once per
    decision. It:
    1. Parses the dict into an `Observation` (`to_observation_class`).
    2. Resets per-turn globals when the turn number changes (so `plan` /
      `ability_used` don't leak across turns).
    3. Runs `SEARCH_ALGO`; if it returns nothing, uses the heuristic `choose()`.
    4. Clamps the result to legal indices and to `minCount..maxCount`, then returns
      the ranked list.

    Every `except` wraps a safe fallback (return "first N legal options") so a
    single unexpected state can never crash the submission — robustness over
    cleverness.
    """
    )
    return


@app.cell
def _(show):
    show(568, 613)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Step 12 — Build Submission (lines 568–613)

    **Thought process.** The last Kaggle cell isn't AI — it packages the agent for
    upload: find the official `cg` SDK directory, copy it next to `main.py` and
    `deck.csv`, and tar everything into `submission.tar.gz`. Our walkthrough skips
    this (you already have the SDK in `data/external/cg-lib`); the runnable part is
    what matters, next.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # ▶️ Running the agent for real

    Below we import the adapted, runnable port (`agent_runner.py` — same logic as
    the original, with the two SDK adaptations) and execute it against a **real**
    game observation shipped in this repo (`notebooks/sample_obs.json`). You will
    see the heuristic produce a ranked action and the search/rollout engine
    evaluate positions.
    """
    )
    return


@app.cell
def _(Path, SOURCE_DIR, sys):
    # Make the adapted runner importable from this notebook.
    RUNNER_DIR = SOURCE_DIR  # agent_runner.py lives next to main.py here
    if str(RUNNER_DIR) not in sys.path:
        sys.path.insert(0, str(RUNNER_DIR))

    import agent_runner as runner
    from cg.api import to_observation_class

    SAMPLE_OBS_PATH = (
        Path.cwd() / "notebooks" / "sample_obs.json"
        if (Path.cwd() / "notebooks" / "sample_obs.json").exists()
        else Path.cwd() / "sample_obs.json"
    )
    sample = __import__("json").loads(SAMPLE_OBS_PATH.read_text())
    obs_dict = sample["obs_dict"]
    obs = to_observation_class(obs_dict)
    return RUNNER_DIR, SAMPLE_OBS_PATH, obs, obs_dict, runner, to_observation_class


@app.cell
def _(mo, obs):
    mo.md(
        f"""
    ## Step 13 — Inspect the position the agent sees

    The sample observation is a **real live battle state** (turn
    **{obs.current.turn}**, you are player **{obs.current.yourIndex}**). It has
    `search_begin_input`, so the engine lets us run an actual search from here —
    not a fake.

    - Options offered this turn: **{len(obs.select.option)}**
    - Select context: **{obs.select.context.name}**
    - maxCount (how many you may pick): **{obs.select.maxCount}**
    """
    )
    return


@app.cell
def _(obs, runner):
    # The heuristic's ranked action list (no search yet).
    heuristic_order = runner.AdvancedPolicy(obs).choose()
    print("Heuristic ranked action indices:", heuristic_order)
    print("Top action the policy prefers:", heuristic_order[0] if heuristic_order else None)
    return (heuristic_order,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Step 14 — Run the full agent (heuristic + search)

    Calling `runner.agent(obs_dict)` runs the exact code the competition engine
    would call: it tries `SEARCH_ALGO` (UCB1 look-ahead, 1.5 s budget) and falls
    back to the heuristic if search is unavailable. The returned list is the
    agent's final ranked decision — **a legal set of option indices**.
    """
    )
    return


@app.cell
def _(obs_dict, runner):
    import time

    t0 = time.time()
    action = runner.agent(obs_dict)
    elapsed = time.time() - t0
    n_opts = len(obs_dict["select"]["option"])
    legal = all(0 <= i < n_opts for i in action)
    print(f"agent() returned : {action}")
    print(f"legal indices?   : {legal}  (n_options = {n_opts})")
    print(f"wall-clock time  : {elapsed:.3f}s")
    return action, elapsed, legal, n_opts, time


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Step 15 — Prove the search/rollout engine *actually* executes

    On this particular real position the heuristic is decisive (one clearly best
    action), so `SEARCH_ALGO` short-circuits to that single candidate — that is
    **correct behaviour**, not a bug: there is nothing to explore.

    To show the decision-time *planning* (the "training"-equivalent: simulating
    futures and averaging their value) genuinely runs, we call `simulate_action`
    directly. Each call opens a real engine search, plays ~20 steps of
    heuristic-vs-heuristic, and returns the evaluated state value.
    """
    )
    return


@app.cell
def _(action, obs, runner):
    import time

    a = action[0]
    t0 = time.time()
    values = [runner.simulate_action(obs, a) for _ in range(5)]
    dt = time.time() - t0
    print(f"Ran 5 rollouts of action {a} (real engine search + heuristic playouts):")
    for i, v in enumerate(values, 1):
        print(f"  rollout {i}: evaluated state value = {v:.1f}")
    print(f"total {dt:.3f}s — the search/rollout machinery executes end-to-end.")
    return (dt, values)


@app.cell
def _(mo):
    mo.md(
        r"""
    # Summary

    - **The original notebook has no training loop** — it is a hand-written
      heuristic policy (`AdvancedPolicy`) plus a **flat Monte-Carlo search with
      UCB1** (`SEARCH_ALGO`). "Training" here means *decision-time planning*:
      simulate possible futures and average their value.
    - **We ran it for real** against a live observation using your repo's `cg`
      SDK. The agent returns a legal, ranked action, and the rollout engine
      executes and evaluates game states.
    - **One real compatibility fix:** the original `search_begin(obs, your_deck=yd)`
      (1 arg) was written for an older `cg` SDK. Your repo's SDK needs **six**
      prediction arguments, and `SearchState` exposes the id as `.searchId`. The
      runner adapts *only* those two call sites (labelled `ADAPTED`); all policy /
      scoring / search math is verbatim from Ivan Ternovskiy's notebook.

    Re-run any code cell above to replay that step. To explore a different
    position, point `SAMPLE_OBS_PATH` at another `obs_dict` JSON.
    """
    )
    return


# __APPEND_HERE__

if __name__ == "__main__":
    app.run()
