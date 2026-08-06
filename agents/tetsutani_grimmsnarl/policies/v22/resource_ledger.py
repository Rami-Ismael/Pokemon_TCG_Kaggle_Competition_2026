from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

DARK = 7
FROSLASS = 104
MUNKIDORI = 112
IMPIDIMP = 646
MORGREM = 647
GRIMMSNARL = 648
SNORUNT = 860
RARE_CANDY = 1079
UNFAIR_STAMP = 1080
POFFIN = 1086
NIGHT_STRETCHER = 1097
POKEGEAR = 1122
TOOL_SCRAPPER = 1137
POKE_PAD = 1152
BOSS = 1182
PETREL = 1219
LILLIE = 1227
DAWN = 1231
SPIKEMUTH_GYM = 1259
SHADOW_BULLET = 937

CORE_POKEMON = (FROSLASS, MUNKIDORI, IMPIDIMP, MORGREM, GRIMMSNARL, SNORUNT)
CORE_RESOURCES = (DARK,) + CORE_POKEMON


def _cards(state: dict, side: str = "me") -> list[dict]:
    player = state.get(side) or {}
    return [c or {} for c in (player.get("active") or []) + (player.get("bench") or [])]


def action_signature(option: dict) -> tuple[int, int, int, int, int, int]:
    return (
        int(option.get("type", -1) if option.get("type") is not None else -1),
        int(option.get("source_id", 0) or 0),
        int(option.get("source_zone", 0) or 0),
        int(option.get("target_id", 0) or 0),
        int(option.get("target_area", 0) or 0),
        int(option.get("attack_id", 0) or 0),
    )


def _hp(card: dict) -> int:
    return int(card.get("hp", 0) or 0)


def _max_hp(card: dict) -> int:
    return int(card.get("maxHp", 0) or 0)


def _energy(card: dict) -> int:
    return int(card.get("en", 0) or 0)


def _card_id(card: dict) -> int:
    return int(card.get("id", 0) or 0)


@dataclass(frozen=True)
class ResourceLedger:
    """Public, deterministic accounting used by every hierarchy layer.

    No fitted value, replay key, hidden card, or stochastic estimate is stored here.
    The ledger only normalizes the current public state and short within-turn history.
    """

    state: dict
    history: tuple[dict, ...]
    turn: int
    turn_action_count: int
    prizes: int
    opponent_prizes: int
    hand_size: int
    bench_slots: int
    discard_size: int
    hand: Counter
    discard: Counter
    board: Counter
    opponent_board: Counter
    active: dict
    bench: tuple[dict, ...]
    opponent_active: dict
    opponent_bench: tuple[dict, ...]
    energy_attached: bool
    supporter_played: bool
    stadium_played: bool
    retreated: bool
    marnie_line_count: int
    ready_impidimp: int
    ready_morgrem: int
    attack_ready_grimmsnarl: int
    energized_munkidori: int
    damaged_own_count: int
    recoverable_core: int
    last_action_type: int
    last_action_source: int
    previous_action_type: int
    previous_action_source: int
    previous_action_target: int

    @classmethod
    def from_state(cls, state: dict, history: Sequence[dict] = ()) -> "ResourceLedger":
        me = state.get("me") or {}
        op = state.get("op") or {}
        own_cards = _cards(state, "me")
        opponent_cards = _cards(state, "op")
        hand = Counter(int(x or 0) for x in (me.get("hand_ids") or []))
        discard = Counter(int(x or 0) for x in (me.get("discard_ids") or []))
        board = Counter(_card_id(c) for c in own_cards)
        opponent_board = Counter(_card_id(c) for c in opponent_cards)
        active = (me.get("active") or [{}])[0] or {}
        bench = tuple(c or {} for c in (me.get("bench") or []))
        opponent_active = (op.get("active") or [{}])[0] or {}
        opponent_bench = tuple(c or {} for c in (op.get("bench") or []))
        ready_impidimp = sum(_card_id(c) == IMPIDIMP and not bool(c.get("appear")) for c in own_cards)
        ready_morgrem = sum(_card_id(c) == MORGREM and not bool(c.get("appear")) for c in own_cards)
        attack_ready_grimmsnarl = sum(_card_id(c) == GRIMMSNARL and _energy(c) >= 2 for c in own_cards)
        energized_munkidori = sum(_card_id(c) == MUNKIDORI and _energy(c) >= 1 for c in own_cards)
        damaged_own_count = sum(_max_hp(c) > _hp(c) for c in own_cards)
        recoverable_core = sum(discard[cid] for cid in CORE_RESOURCES)
        flags = state.get("flags") or {}
        hist = tuple(history or ())
        last = hist[-1] if hist else {}
        prev = hist[-2] if len(hist) > 1 else {}
        return cls(
            state=state,
            history=hist,
            turn=int(state.get("turn", 0) or 0),
            turn_action_count=int(state.get("tac", 0) or 0),
            prizes=int(me.get("prize_n", 0) or 0),
            opponent_prizes=int(op.get("prize_n", 0) or 0),
            hand_size=int(me.get("hand_n", 0) or 0),
            bench_slots=max(0, 5 - len(bench)),
            discard_size=len(me.get("discard_ids") or []),
            hand=hand,
            discard=discard,
            board=board,
            opponent_board=opponent_board,
            active=active,
            bench=bench,
            opponent_active=opponent_active,
            opponent_bench=opponent_bench,
            energy_attached=bool(flags.get("energy")),
            supporter_played=bool(flags.get("supporter")),
            stadium_played=bool(flags.get("stadium")),
            retreated=bool(flags.get("retreated")),
            marnie_line_count=board[IMPIDIMP] + board[MORGREM] + board[GRIMMSNARL],
            ready_impidimp=ready_impidimp,
            ready_morgrem=ready_morgrem,
            attack_ready_grimmsnarl=attack_ready_grimmsnarl,
            energized_munkidori=energized_munkidori,
            damaged_own_count=damaged_own_count,
            recoverable_core=recoverable_core,
            last_action_type=int(last.get("type", -1) if last.get("type") is not None else -1),
            last_action_source=int(last.get("source_id", 0) or 0),
            previous_action_type=int(prev.get("type", -1) if prev.get("type") is not None else -1),
            previous_action_source=int(prev.get("source_id", 0) or 0),
            previous_action_target=int(prev.get("target_id", 0) or 0),
        )

    @property
    def active_id(self) -> int:
        return _card_id(self.active)

    @property
    def active_hp(self) -> int:
        return _hp(self.active)

    @property
    def active_max_hp(self) -> int:
        return _max_hp(self.active)

    @property
    def active_damage(self) -> int:
        return max(0, self.active_max_hp - self.active_hp)

    @property
    def active_energy(self) -> int:
        return _energy(self.active)

    @property
    def opponent_active_id(self) -> int:
        return _card_id(self.opponent_active)

    @property
    def opponent_active_hp(self) -> int:
        return _hp(self.opponent_active)

    @property
    def opponent_active_max_hp(self) -> int:
        return _max_hp(self.opponent_active)

    @property
    def opponent_active_damage(self) -> int:
        return max(0, self.opponent_active_max_hp - self.opponent_active_hp)

    @property
    def opponent_active_energy(self) -> int:
        return _energy(self.opponent_active)

    @property
    def active_attack_ready(self) -> bool:
        return self.active_id == GRIMMSNARL and self.active_energy >= 2

    @property
    def control_online(self) -> bool:
        return self.energized_munkidori > 0

    @property
    def froslass_online(self) -> bool:
        return self.board[FROSLASS] > 0

    @property
    def ready_froslass_evolution(self) -> bool:
        return self.board[SNORUNT] > 0 and self.hand[FROSLASS] > 0

    @property
    def attacker_line_deficit(self) -> int:
        return max(0, 2 - self.marnie_line_count)

    @property
    def evolution_bottleneck(self) -> bool:
        return (
            self.marnie_line_count >= 1
            and self.attack_ready_grimmsnarl == 0
            and (self.ready_impidimp > 0 or self.ready_morgrem > 0)
        )

    @property
    def dark_surplus(self) -> bool:
        return self.hand[DARK] >= 3

    @property
    def deep_discard(self) -> bool:
        return self.discard_size >= 15 or self.recoverable_core >= 7

    def own_cards(self) -> tuple[dict, ...]:
        return (self.active,) + self.bench

    def opponent_cards(self) -> tuple[dict, ...]:
        return (self.opponent_active,) + self.opponent_bench

    def cards_by_id(self, card_id: int, *, opponent: bool = False) -> tuple[dict, ...]:
        cards = self.opponent_cards() if opponent else self.own_cards()
        return tuple(c for c in cards if _card_id(c) == int(card_id))

    def has_attack_ready_bench_grimmsnarl(self) -> bool:
        return any(_card_id(c) == GRIMMSNARL and _energy(c) >= 2 for c in self.bench)

    def lowest_hp(self, card_id: int, *, opponent: bool = False) -> int | None:
        values = [_hp(c) for c in self.cards_by_id(card_id, opponent=opponent)]
        return min(values) if values else None

    def option_card(self, option: dict, *, opponent: bool | None = None) -> dict:
        rel = int(option.get("source_rel", 0) or 0)
        use_opponent = rel == 1 if opponent is None else opponent
        zone = int(option.get("source_zone", 0) or 0)
        index = int(option.get("index", -1) if option.get("index") is not None else -1)
        if use_opponent:
            arr = [self.opponent_active] if zone == 4 else list(self.opponent_bench) if zone == 5 else []
        else:
            arr = [self.active] if zone == 4 else list(self.bench) if zone == 5 else []
        if 0 <= index < len(arr):
            return arr[index] or {}
        card_id = int(option.get("source_id", 0) or 0)
        for card in arr:
            if _card_id(card) == card_id:
                return card
        return {}

    def find_option(self, options: Iterable[dict], signature: tuple[int, int, int, int, int, int]) -> int | None:
        for index, option in enumerate(options):
            if action_signature(option) == signature:
                return index
        return None

    def find_card_option(self, options: Iterable[dict], card_id: int) -> int | None:
        for index, option in enumerate(options):
            if int(option.get("source_id", 0) or 0) == int(card_id):
                return index
        return None
