"""Generalized rule-core scorer: a deck-agnostic OptionScorer that plays the way
the top public agent (makthanithin "1084.5") plays — a two-phase attack *plan*,
opponent-archetype detection, and lethal / prize-trade / weakness scoring — but
WITHOUT hardcoding one deck's attacks and HPs.

Why this is an improvement, not a copy (report/competition_insights.md):
  - The public 1084 `LucarioPolicy` hardcodes Mega Lucario's 130/270 attacks,
    Hariyama 210, the Riolu line, etc. It only pilots the Lucario deck.
  - We read the SAME facts (attack damage, energy cost, weakness, prize value)
    straight from the engine's card/attack tables, so ONE core pilots Lucario,
    Dragapult, Abomasnow, Iono... and we gate it vs the real public field.

The three levers that separate ~600 (rule floor) from ~1084 (top public):
  1. PLAN: each turn, search (my attacker x attack x reachable target) for the
     move that trades best on prizes, doubling for weakness, and recognizing a
     KO that *wins the game* (opponent prizes_left <= prize value -> take it).
  2. READ THE OPPONENT: detect Crustle-wall / water / psychic / ex-heavy from the
     opponent's visible board and switch plan (the 600->1084 lever).
  3. ROUTE RESOURCES to the planned attacker: energy/evolve/gust/switch all serve
     the chosen plan instead of being scored in isolation.

Scope: this scorer owns MAIN (the full main-phase option list) and the few
high-value card-choice contexts (setup / switch / promote / search-to-hand /
energy-from). Every other forced sub-select defers to HeuristicScorer, which
already has careful min/max-count handling and a never-crash fallback.
"""

from __future__ import annotations

import os

from agent.agent import HeuristicScorer
from agent.deck_tech import DEFAULT_TECH, tech_for_deck

# Engine enums / tables are resolved lazily so this module imports even if the
# native cg engine is absent (offline unit tests of the scaffold).
_API = None
_CARD_TABLE = None
_ATTACK_TABLE = None

# Known archetype signatures (card ids visible on the opponent's board).
CRUSTLE_WALL_IDS = {344, 345}          # Dwebble / Crustle: walls Pokemon-ex damage
WATER_DECK_IDS = {721, 722, 723}       # Kyogre / Snover / Mega Abomasnow ex
LEGACY_ENERGY_ID = 12                  # reduces prizes taken when KO'd
LOW_VALUE_SUPPORT_IDS = {144, 322, 323, 337}


def _load_api():
    global _API, _CARD_TABLE, _ATTACK_TABLE
    if _API is not None:
        return _API
    import cg.api as api

    _CARD_TABLE = {c.cardId: c for c in api.all_card_data()}
    _ATTACK_TABLE = {a.attackId: a for a in api.all_attack()}
    _API = api
    return _API


def _enum_val(x):
    """EnergyType/weakness come back as either an IntEnum or a plain int."""
    return x.value if hasattr(x, "value") else x


class _Plan:
    __slots__ = ("attacker", "target", "attack_id", "lethal", "needs_energy", "score")

    def __init__(self, attacker=-1, target=-1, attack_id=-1, lethal=False,
                 needs_energy=False, score=-1.0):
        self.attacker = attacker          # board index: 0=active, 1+=bench[idx-1]
        self.target = target              # board index on opponent side
        self.attack_id = attack_id
        self.lethal = lethal
        self.needs_energy = needs_energy
        self.score = score

    @property
    def valid(self) -> bool:
        return self.attacker >= 0 and self.attack_id >= 0


class RuleCoreScorer(HeuristicScorer):
    """Deck-agnostic plan + opponent-detection brain; HeuristicScorer fallback."""

    def __init__(self, rng=None, deck_path: str | None = None) -> None:
        super().__init__(rng=rng)
        self._fallback = HeuristicScorer(rng=rng)
        self._plan = _Plan()
        self._plan_turn = -1
        self._deck_tech = DEFAULT_TECH
        self._lucario = None
        if deck_path:
            try:
                readable_deck = deck_path
                if not os.path.exists(readable_deck) and os.path.exists("deck.csv"):
                    readable_deck = "deck.csv"
                with open(readable_deck, encoding="utf-8") as fh:
                    deck_ids = [int(x.strip()) for x in fh if x.strip()]
                self._deck_tech = tech_for_deck(deck_ids)
                if self._deck_tech.name == "mega_lucario_ex":
                    from agent.lucario_policy import LucarioScorer

                    self._lucario = LucarioScorer(rng=rng, deck_path=deck_path)
            except Exception:
                self._deck_tech = DEFAULT_TECH

    # --- entry ---------------------------------------------------------------
    def choose(self, obs_dict, select, current, options):
        if not options:
            return []
        if self._lucario is not None:
            return self._lucario.choose(obs_dict, select, current, options)
        try:
            api = _load_api()
            obs = api.to_observation_class(obs_dict)
            if obs.select is None or not obs.select.option:
                return self._fallback.choose(obs_dict, select, current, options)
            ctx = _Read(obs, api, self._deck_tech, self._plan)
            if ctx.is_main:
                self._ensure_plan(ctx)
                scores = [self._score_main_option(ctx, o) for o in obs.select.option]
                best = max(range(len(scores)), key=lambda i: scores[i])
                return [best]
            if ctx.is_card_choice:
                picked = self._choose_card(ctx)
                if picked is not None:
                    return picked
        except Exception:  # never break the never-crash contract
            import os as _os
            if _os.environ.get("RULECORE_DEBUG"):
                import traceback
                traceback.print_exc()
        return self._fallback.choose(obs_dict, select, current, options)

    # --- planning ------------------------------------------------------------
    def _ensure_plan(self, ctx: "_Read") -> None:
        if ctx.turn != self._plan_turn:
            self._plan_turn = ctx.turn
        self._plan = self._build_plan(ctx)

    def _build_plan(self, ctx: "_Read") -> _Plan:
        best = _Plan()
        if ctx.turn < 2:
            return best
        my_board = ctx.my_board
        opp_board = ctx.opp_board
        opp_prizes_left = len(ctx.opponent.prize)
        for a_idx, mine in enumerate(my_board):
            if mine is None:
                continue
            # can only attack from active (idx 0) unless we can move someone up.
            if a_idx != 0 and not ctx.can_reposition:
                continue
            attacker_type = ctx.energy_type(mine.id)
            for atk in ctx.projected_attacks(mine, a_idx):
                if atk.damage <= 0:
                    continue
                cost = len(atk.energies)
                have = len(mine.energies)
                needs_energy = False
                if have < cost:
                    if have + 1 >= cost and ctx.can_attach_energy:
                        needs_energy = True
                    else:
                        continue
                for t_idx, opp in enumerate(opp_board):
                    if opp is None:
                        continue
                    if t_idx != 0 and not ctx.can_gust:
                        continue
                    if ctx.wall_immune(mine.id, opp.id):
                        continue  # e.g. ex attacker into Crustle -> useless
                    dmg = self._effective_damage(ctx, atk.damage, attacker_type, opp.id)
                    kos = opp.hp <= dmg
                    prize = ctx.prize_count(opp) if kos else 0
                    score = ctx.target_score(opp)
                    if not kos:
                        score *= dmg / max(1, opp.hp)   # partial progress
                    lethal = kos and opp_prizes_left <= prize
                    if lethal:
                        score = 1_000_000.0
                    score += 220 if a_idx == 0 else 0   # prefer the active attacker
                    score += 300 if t_idx == 0 else 0   # prefer hitting their active
                    score += have                       # tiny tie-break: readiness
                    if score > best.score:
                        best = _Plan(a_idx, t_idx, atk.attackId, lethal,
                                     needs_energy, score)
        if ctx.archetype["crustle_wall"] and not best.valid:
            focus = ctx.wall_focus_index
            if focus >= 0:
                return _Plan(
                    attacker=focus,
                    target=0,
                    attack_id=-1,
                    lethal=False,
                    needs_energy=True,
                    score=10_000.0,
                )
        return best

    def _effective_damage(self, ctx, base, attacker_type, target_id) -> int:
        data = ctx.card(target_id)
        dmg = base
        if data is not None:
            wk = _enum_val(data.weakness)
            rs = _enum_val(data.resistance)
            if wk and attacker_type and wk == attacker_type:
                dmg *= 2
            elif rs and attacker_type and rs == attacker_type:
                dmg = max(0, dmg - 30)
        return dmg

    # --- MAIN option scoring (unified scale; references the plan) ------------
    def _score_main_option(self, ctx: "_Read", opt) -> float:
        api = ctx.api
        T = api.OptionType
        ot = opt.type
        if ot == T.ATTACK:
            return self._score_attack(ctx, opt)
        if ot == T.ABILITY:
            return 30000.0
        if ot == T.PLAY:
            return self._score_play(ctx, opt)
        if ot == T.EVOLVE:
            return self._score_evolve(ctx, opt)
        if ot == T.ATTACH:
            return self._score_attach(ctx, opt)
        if ot == T.RETREAT:
            if ctx.archetype["crustle_wall"] and ctx.wall_ready_index >= 1:
                return 20000.0
            # retreat only to bring the planned (benched) attacker up
            return 6000.0 if self._plan.attacker >= 1 and ctx.wall_ready_index != -1 else -5.0
        if ot == T.END:
            return 0.0
        if ot in (T.YES, T.NO):
            return 1.0 if ot == T.YES else 0.0
        if ot == T.NUMBER:
            return float(opt.number or 0)
        return 0.0

    def _score_attack(self, ctx, opt) -> float:
        # No backup bench → do not attack into a likely fast loss unless lethal.
        if ctx.bench_count == 0 and not (self._plan.valid and self._plan.lethal):
            return -1.0
        # If the only plan is a walled / useless attack, end the turn instead.
        if not self._plan.valid:
            return -1.0
        # active attacker must match the planned attacker to fire the plan now.
        if self._plan.attacker != 0:
            return -1.0
        if opt.attackId == self._plan.attack_id:
            return 1_000_000.0 if self._plan.lethal else 1100.0
        return 1000.0

    def _score_play(self, ctx, opt) -> float:
        card = ctx.hand_card(opt.index)
        if card is None:
            return 1000.0
        data = ctx.card(card.id)
        if data is None:
            return 10000.0
        if _enum_val(data.cardType) == ctx.api.CardType.POKEMON.value:
            if ctx.bench_count == 0:
                return 45000.0
            if ctx.bench_count >= 2:
                return -1.0
            from agent.smart_bench import worthwhile_second_bench

            active = ctx.my_board[0] if ctx.my_board else None
            active_id = active.id if active is not None else 0
            if worthwhile_second_bench(
                card.id,
                bench_count=ctx.bench_count,
                turn=ctx.turn,
                active_id=active_id,
                field_dup_count=ctx.field_count(card.id),
                tech=ctx.tech,
            ):
                dup = ctx.field_count(card.id)
                return 18000.0 - 1500.0 * max(0, dup - 1)
            return -1.0
        if card.id in ctx.tech.gust_cards:
            return 12000.0 if self._plan.target >= 1 else -1.0
        if card.id in ctx.tech.switch_cards:
            if ctx.archetype["crustle_wall"] and ctx.wall_ready_index >= 1:
                return 22000.0
            return 9000.0 if self._plan.attacker >= 1 and ctx.wall_ready_index != -1 else -1.0
        if card.id in ctx.tech.draw_cards:
            if (
                ctx.archetype["crustle_wall"]
                and ctx.wall_line_count >= 1
                and (ctx.me.deckCount <= 35 or ctx.wall_focus_energy >= 2)
            ):
                return -1.0
            return -1.0 if ctx.low_deck else 8500.0
        if card.id in ctx.tech.search_cards:
            if ctx.archetype["crustle_wall"]:
                if ctx.wall_focus_index >= 0 and ctx.wall_focus_energy >= 2:
                    return -1.0
                if ctx.wall_line_count >= 1 and ctx.me.deckCount <= 35:
                    return -1.0
                return 9500.0
            return -1.0 if ctx.low_deck else 8000.0
        if card.id in ctx.tech.stadium_cards:
            return 7000.0 if ctx.opponent_has_stage2 or ctx.stadium_id else -1.0
        return 10000.0  # trainers: let the engine's offered set drive value

    def _score_evolve(self, ctx, opt) -> float:
        poke = ctx.inplay_card(opt.inPlayArea, opt.inPlayIndex)
        bonus = len(poke.energies) if poke is not None else 0
        card = ctx.hand_card(opt.index)
        if (
            card is not None
            and card.id in ctx.tech.non_ex_wall_attackers
            and poke is not None
            and poke.id in ctx.tech.non_ex_wall_bases
            and ctx.archetype["crustle_wall"]
        ):
            return 12000.0 + bonus * 200.0
        return 9000.0 + bonus  # evolving is near-always progress

    def _score_attach(self, ctx, opt) -> float:
        poke = ctx.inplay_card(opt.inPlayArea, opt.inPlayIndex)
        if poke is None:
            return 7000.0
        active = _enum_val(opt.inPlayArea) == ctx.api.AreaType.ACTIVE.value
        board_index = opt.inPlayIndex if active else opt.inPlayIndex + 1
        score = 8000.0 + (10.0 if active else 0.0)
        if poke.id in ctx.tech.wall_line:
            if ctx.archetype["crustle_wall"]:
                score += 5000.0 if board_index == ctx.wall_focus_index else 1000.0
            else:
                score += 80.0
        # prize-worthy attackers are better energy homes
        score += ctx.prize_count(poke) * 40.0
        score += max(0, 3 - len(poke.energies)) * 30.0
        if board_index == self._plan.attacker:
            score += 250.0 if self._plan.needs_energy else 120.0
        return score

    # --- card-choice contexts (setup / switch / promote / search / energy) ---
    def _choose_card(self, ctx: "_Read") -> "list[int] | None":
        sc = ctx.select_context
        api = ctx.api
        SC = api.SelectContext
        opts = ctx.options
        if sc == SC.SETUP_BENCH_POKEMON:
            return ctx.setup_bench_choices()
        if sc in (SC.SWITCH, SC.TO_ACTIVE, SC.SETUP_ACTIVE_POKEMON):
            best = max(range(len(opts)), key=lambda i: ctx.active_choice_score(opts[i]))
            return [best]
        if sc == SC.TO_HAND:
            best = max(range(len(opts)), key=lambda i: ctx.to_hand_score(opts[i]))
            return [best]
        if sc == SC.ATTACH_FROM:
            best = max(range(len(opts)), key=lambda i: ctx.energy_from_score(opts[i]))
            return [best]
        # everything else here we let the heuristic handle (careful counts)
        return None


class _Read:
    """Typed view over one observation: board access + generalized card facts."""

    def __init__(self, obs, api, tech, plan):
        self.api = api
        self.tech = tech
        self.plan = plan
        self.obs = obs
        self.state = obs.current
        self.select = obs.select
        self.turn = self.state.turn
        self.my_index = self.state.yourIndex
        self.me = self.state.players[self.my_index]
        self.opponent = self.state.players[1 - self.my_index]
        self.select_context = self.select.context
        self.is_main = (self.select.context == api.SelectContext.MAIN)
        self.is_card_choice = (self.select.type == api.SelectType.CARD)
        self.options = self.select.option
        self._fc = None
        self._scan = None
        self._archetype = None
        self._scan_main()

    # board ----------------------------------------------------------------
    @property
    def my_board(self):
        return list(self.me.active) + list(self.me.bench)

    @property
    def opp_board(self):
        return list(self.opponent.active) + list(self.opponent.bench)

    @property
    def bench_count(self) -> int:
        return len([p for p in self.me.bench if p is not None])

    @property
    def wall_line_count(self) -> int:
        return sum(1 for p in self.my_board if p is not None and p.id in self.tech.wall_line)

    @property
    def wall_ready_index(self) -> int:
        if not self.archetype["crustle_wall"]:
            return -1
        for idx, p in enumerate(self.my_board):
            if p is not None and p.id in self.tech.non_ex_wall_attackers and len(p.energies) >= 3:
                return idx
        return -1

    @property
    def wall_focus_index(self) -> int:
        if not self.archetype["crustle_wall"]:
            return -1
        best_idx = -1
        best_score = -1.0
        for idx, p in enumerate(self.my_board):
            if p is None or p.id not in self.tech.wall_line:
                continue
            score = len(p.energies) * 100.0
            if p.id in self.tech.non_ex_wall_attackers:
                score += 500.0
            if idx == 0:
                score += 20.0
            if score > best_score:
                best_idx = idx
                best_score = score
        return best_idx

    @property
    def wall_focus_energy(self) -> int:
        idx = self.wall_focus_index
        if idx < 0:
            return 0
        pokemon = self.my_board[idx]
        return len(pokemon.energies) if pokemon is not None else 0

    def field_count(self, card_id: int) -> int:
        if self._fc is None:
            self._fc = {}
            for p in self.my_board:
                if p is not None:
                    self._fc[p.id] = self._fc.get(p.id, 0) + 1
        return self._fc.get(card_id, 0)

    def hand_card(self, index: int):
        hand = self.me.hand
        return hand[index] if 0 <= index < len(hand) else None

    def inplay_card(self, area, index: int):
        a = _enum_val(area)
        if a == self.api.AreaType.ACTIVE.value:
            seq = self.me.active
        elif a == self.api.AreaType.BENCH.value:
            seq = self.me.bench
        else:
            return None
        return seq[index] if 0 <= index < len(seq) else None

    # generalized card facts ----------------------------------------------
    def card(self, card_id: int):
        return _CARD_TABLE.get(card_id) if _CARD_TABLE else None

    def attacks_of(self, card_id: int):
        data = self.card(card_id)
        if data is None or not _ATTACK_TABLE:
            return []
        return [_ATTACK_TABLE[i] for i in data.attacks if i in _ATTACK_TABLE]

    def projected_attacks(self, pokemon, board_index: int):
        attacks = list(self.attacks_of(pokemon.id))
        if not self.archetype["crustle_wall"] or pokemon.id not in self.tech.non_ex_wall_bases:
            return attacks
        for cid in self.tech.non_ex_wall_attackers:
            if self.can_evolve_to(board_index, cid):
                attacks.extend(self.attacks_of(cid))
        return attacks

    def can_evolve_to(self, board_index: int, card_id: int) -> bool:
        if not self.is_main:
            return False
        T = self.api.OptionType
        for opt in self.options:
            if opt.type != T.EVOLVE:
                continue
            card = self.hand_card(opt.index)
            if card is None or card.id != card_id:
                continue
            target = opt.inPlayIndex
            if _enum_val(opt.inPlayArea) == self.api.AreaType.BENCH.value:
                target += 1
            if target == board_index:
                return True
        return False

    def energy_type(self, card_id: int):
        data = self.card(card_id)
        return _enum_val(data.energyType) if data is not None else None

    def prize_count(self, pokemon) -> int:
        data = self.card(pokemon.id)
        if data is None:
            return 1
        count = 3 if data.megaEx else 2 if data.ex else 1
        for ec in getattr(pokemon, "energyCards", []) or []:
            if ec.id == LEGACY_ENERGY_ID:
                count -= 1
        return max(0, count)

    def target_score(self, pokemon) -> float:
        data = self.card(pokemon.id)
        score = self.prize_count(pokemon) * 1000.0
        score += len(pokemon.energies) * 150.0
        score += len(getattr(pokemon, "tools", []) or []) * 100.0
        if data is not None:
            if data.stage2:
                score += 250.0
            elif data.stage1:
                score += 130.0
        if pokemon.id in LOW_VALUE_SUPPORT_IDS:
            score -= 200.0
        score += pokemon.hp
        return score

    @property
    def low_deck(self) -> bool:
        return self.me.deckCount <= 10

    @property
    def stadium_id(self) -> int:
        return self.state.stadium[0].id if self.state.stadium else 0

    @property
    def opponent_has_stage2(self) -> bool:
        return any(
            p is not None and (self.card(p.id) is not None and self.card(p.id).stage2)
            for p in self.opp_board
        )

    # opponent archetype detection ----------------------------------------
    def _opp_ids(self):
        return {p.id for p in self.opp_board if p is not None}

    @property
    def archetype(self):
        if self._archetype is None:
            ids = self._opp_ids()
            self._archetype = {
                "crustle_wall": bool(ids & CRUSTLE_WALL_IDS),
                "water": bool(ids & WATER_DECK_IDS),
            }
        return self._archetype

    def wall_immune(self, my_id: int, opp_id: int) -> bool:
        """Crustle (345) prevents damage from Pokemon-ex -> don't plan ex into it."""
        if opp_id == 345:
            data = self.card(my_id)
            if data is not None and (data.ex or data.megaEx):
                return True
        return False

    # main-option scan: what repositioning / gust is available this turn ---
    def _scan_main(self):
        self.can_reposition = False
        self.can_gust = False
        self.can_attach_energy = not self.state.energyAttached
        if not self.is_main:
            return
        T = self.api.OptionType
        for opt in self.options:
            if opt.type == T.RETREAT:
                self.can_reposition = True
            elif opt.type == T.PLAY:
                card = self.hand_card(opt.index)
                if card is not None and card.id in self.tech.gust_cards:
                    self.can_gust = True
                if card is not None and card.id in self.tech.switch_cards:
                    self.can_reposition = True
            elif opt.type == T.EVOLVE:
                card = self.hand_card(opt.index)
                if card is not None and card.id in self.tech.gust_cards | self.tech.non_ex_wall_attackers:
                    self.can_gust = True
                target = self.inplay_card(opt.inPlayArea, opt.inPlayIndex)
                if (
                    target is not None
                    and target.id in self.tech.non_ex_wall_bases
                    and self.archetype["crustle_wall"]
                ):
                    self.can_reposition = True

    # card-choice scoring --------------------------------------------------
    def active_choice_score(self, opt) -> float:
        card = self._option_pokemon(opt)
        if card is None:
            return 0.0
        pidx = opt.playerIndex if opt.playerIndex is not None else self.my_index
        if pidx != self.my_index:
            area = _enum_val(opt.area)
            board_index = opt.index + 1 if area == self.api.AreaType.BENCH.value else opt.index
            return 300.0 if board_index == self._planned_target_index() else self.target_score(card) / 20.0
        setup_bonus = dict(self.tech.setup_priority).get(card.id, 0.0)
        # promote the most prize-worthy, energized attacker
        score = self.prize_count(card) * 100.0 + len(card.energies) * 20.0 + card.hp / 10.0
        if self.select_context == self.api.SelectContext.SETUP_ACTIVE_POKEMON:
            score += setup_bonus
        if card.id in self.tech.wall_line and self.archetype["crustle_wall"]:
            score += 80.0 + len(card.energies) * 20.0
            if self.select_context in (self.api.SelectContext.SWITCH, self.api.SelectContext.TO_ACTIVE):
                if self.wall_ready_index >= 0:
                    score += 500.0 if self._board_index_for_option(opt) == self.wall_ready_index else -200.0
                elif self._board_index_for_option(opt) == self.wall_focus_index:
                    score += 120.0
        if card.id in self.tech.non_ex_wall_attackers and self.archetype["crustle_wall"]:
            score += 120.0 + len(card.energies) * 40.0
        data = self.card(card.id)
        if data is not None and data.basic and self.select_context == self.api.SelectContext.SETUP_ACTIVE_POKEMON:
            score += 50.0  # must open with a basic
        return score

    def _board_index_for_option(self, opt) -> int:
        area = _enum_val(opt.area)
        if area == self.api.AreaType.ACTIVE.value:
            return opt.index or 0
        if area == self.api.AreaType.BENCH.value:
            return (opt.index or 0) + 1
        return -1

    def to_hand_score(self, opt) -> float:
        card = self._option_card(opt)
        if card is None:
            return 0.0
        score = 100.0
        if card.id in self.tech.gust_cards:
            score += 180.0 if self.can_gust else 80.0
        if card.id in self.tech.non_ex_wall_attackers and self.archetype["crustle_wall"]:
            score += 240.0 if self.field_count(next(iter(self.tech.non_ex_wall_bases), -1)) else 20.0
        if card.id in self.tech.non_ex_wall_bases and self.archetype["crustle_wall"]:
            score += 180.0 - 40.0 * self.field_count(card.id)
        if card.id in self.tech.switch_cards:
            score += 60.0 if self.can_reposition else 30.0
        if card.id in self.tech.draw_cards and not self.low_deck:
            score += 50.0
        if card.id in self.tech.search_cards:
            if self.archetype["crustle_wall"] and (
                self.me.deckCount <= 35 or self.wall_focus_energy >= 2
            ):
                score -= 200.0
            elif not self.low_deck:
                score += 35.0
        return score

    def energy_from_score(self, opt) -> float:
        card = self._option_pokemon(opt)
        if card is None:
            return 0.0
        board_index = self._board_index_for_option(opt)
        score = 10.0 + len(card.energies) * 2.0
        if self.archetype["crustle_wall"] and self.wall_focus_index >= 0:
            if board_index == self.wall_focus_index:
                return 1000.0 + max(0, 3 - len(card.energies)) * 80.0
            if card.id in self.tech.wall_line:
                return 800.0 + max(0, 3 - len(card.energies)) * 40.0
            if self.wall_focus_energy < 3:
                return -100.0
        if card.id in self.tech.wall_line and self.archetype["crustle_wall"]:
            score += 80.0
        return score

    def setup_bench_choices(self) -> list[int]:
        min_count = int(getattr(self.select, "minCount", 0) or 0)
        max_count = int(getattr(self.select, "maxCount", len(self.options)) or 0)
        if max_count <= 0:
            return []

        ranked = sorted(
            range(len(self.options)),
            key=lambda i: self.setup_bench_score(self.options[i]),
            reverse=True,
        )
        from agent.smart_bench import MAX_VOLUNTARY_BENCH, setup_bench_target_count

        target = setup_bench_target_count(
            ranked,
            self.options,
            self._obs_current_dict(),
            self.tech,
            min_count,
            max_count,
        )
        picks: list[int] = []
        seen_ids: set[int] = set()
        for idx in ranked:
            card = self._option_card(self.options[idx])
            card_id = card.id if card is not None else 0
            if card_id in seen_ids:
                continue
            if self.setup_bench_score(self.options[idx]) <= 0 and len(picks) >= min_count:
                continue
            picks.append(idx)
            seen_ids.add(card_id)
            if len(picks) >= min(target, MAX_VOLUNTARY_BENCH):
                break
        if self.bench_count == 0 and not picks and ranked:
            picks.append(ranked[0])
        if len(picks) < min_count:
            for idx in ranked:
                if idx not in picks:
                    picks.append(idx)
                if len(picks) >= min_count:
                    break
        return picks[:max_count]

    def _obs_current_dict(self):
        """Minimal dict view for smart_bench helpers (setup phase)."""
        try:
            return self.obs.current.to_dict()  # type: ignore[attr-defined]
        except Exception:
            pass
        me = self.me
        opp = self.opponent
        return {
            "turn": self.turn,
            "yourIndex": self.my_index,
            "players": [
                {"active": list(me.active), "bench": list(me.bench), "benchMax": 5},
                {"active": list(opp.active), "bench": list(opp.bench), "benchMax": 5},
            ],
        }

    def setup_bench_score(self, opt) -> float:
        card = self._option_card(opt)
        if card is None:
            return 0.0
        score = dict(self.tech.setup_priority).get(card.id, 0.0)
        if card.id in self.tech.non_ex_wall_bases:
            score += 25.0
        if card.id in self.tech.non_ex_wall_attackers:
            score += 15.0
        return score

    def _planned_target_index(self) -> int:
        return self.plan.target if self.plan and self.plan.target >= 0 else 0

    def _option_pokemon(self, opt):
        area = _enum_val(opt.area)
        idx = opt.index or 0
        pidx = opt.playerIndex if opt.playerIndex is not None else self.my_index
        if not (0 <= pidx < len(self.state.players)):
            return None
        player = self.state.players[pidx]
        if area == self.api.AreaType.ACTIVE.value:
            seq = player.active
        elif area == self.api.AreaType.BENCH.value:
            seq = player.bench
        else:
            return None
        return seq[idx] if 0 <= idx < len(seq) else None

    def _option_card(self, opt):
        area = _enum_val(opt.area)
        idx = opt.index or 0
        pidx = opt.playerIndex if opt.playerIndex is not None else self.my_index
        if area == self.api.AreaType.HAND.value and 0 <= pidx < len(self.state.players):
            hand = self.state.players[pidx].hand
            return hand[idx] if 0 <= idx < len(hand) else None
        if area == self.api.AreaType.DECK.value and self.select.deck:
            return self.select.deck[idx] if 0 <= idx < len(self.select.deck) else None
        if area == self.api.AreaType.DISCARD.value and 0 <= pidx < len(self.state.players):
            discard = self.state.players[pidx].discard
            return discard[idx] if 0 <= idx < len(discard) else None
        return self._option_pokemon(opt)
