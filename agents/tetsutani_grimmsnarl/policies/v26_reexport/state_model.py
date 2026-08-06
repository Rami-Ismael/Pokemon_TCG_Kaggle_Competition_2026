from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterable

from . import validated_rule_profile as profile
from .domain import (
    DARK, FROSLASS, MUNKIDORI, IMPIDIMP, MORGREM, GRIMMSNARL, SNORUNT,
    SPIKEMUTH, ATTACK_LINE, FROSLASS_LINE,
)


class TurnPhase(Enum):
    OPENING = auto()
    BUILD = auto()
    PRESSURE = auto()
    ENDGAME = auto()


@dataclass(frozen=True)
class ResourceLedger:
    hand: Counter
    hand_size: int
    deck_size: int
    bench_used: int
    bench_free: int
    attack_lines: int
    ready_attackers: int
    developing_attackers: int
    froslass_lines: int
    munkidori_total: int
    munkidori_powered: int
    munkidori_unpowered: int
    dark_in_hand: int
    stadium_live: bool
    supporter_used: bool
    energy_attached: bool
    active_id: int
    active_energy: int
    active_damage: int
    active_new: bool
    own_prizes: int
    opponent_prizes: int
    opponent_hand_size: int
    opponent_active_damage: int
    opponent_active_energy: int

    @property
    def attack_online(self) -> bool:
        return self.ready_attackers > 0

    @property
    def damage_loop_online(self) -> bool:
        return self.munkidori_powered > 0

    @property
    def board_sparse(self) -> bool:
        return self.bench_free >= 3

    @property
    def dual_engine_complete(self) -> bool:
        return self.attack_lines >= 3 and self.froslass_lines >= 2


@dataclass(frozen=True)
class TurnState:
    view: dict
    turn: int
    action_count: int
    phase: TurnPhase
    ledger: ResourceLedger
    legal_roles: frozenset[str]

    @staticmethod
    def from_view(view: dict) -> "TurnState":
        me = view["me"]
        active = profile._active(me)
        inplay = profile._all_inplay(me)
        ids = [int(p.get("id", 0) or 0) for p in inplay]
        attack_lines = sum(cid in ATTACK_LINE for cid in ids)
        ready = sum(
            p.get("id") == GRIMMSNARL and profile._energy_n(p) >= 2
            for p in inplay
        )
        developing = sum(cid in (IMPIDIMP, MORGREM) for cid in ids)
        froslass_lines = sum(cid in FROSLASS_LINE for cid in ids)
        munk = [p for p in inplay if p.get("id") == MUNKIDORI]
        powered = sum(profile._energy_n(p) > 0 for p in munk)
        bench_used = len(me.get("bench") or [])
        bench_max = int(me.get("bench_max", 5) or 5)
        hand = Counter(int(x or 0) for x in (me.get("hand") or []))
        ledger = ResourceLedger(
            hand=hand,
            hand_size=int(me.get("hand_n", len(me.get("hand") or [])) or 0),
            deck_size=int(me.get("deck", 0) or 0),
            bench_used=bench_used,
            bench_free=max(0, bench_max - bench_used),
            attack_lines=attack_lines,
            ready_attackers=ready,
            developing_attackers=developing,
            froslass_lines=froslass_lines,
            munkidori_total=len(munk),
            munkidori_powered=powered,
            munkidori_unpowered=len(munk) - powered,
            dark_in_hand=hand[DARK],
            stadium_live=int(view.get("stadium", 0) or 0) == SPIKEMUTH,
            supporter_used=bool(view.get("flags", [False, False])[1]),
            energy_attached=bool(view.get("flags", [False])[0]),
            active_id=int(active.get("id", 0) or 0),
            active_energy=profile._energy_n(active),
            active_damage=int(active.get("damage", 0) or 0),
            active_new=bool(active.get("new", False)),
            own_prizes=int(me.get("prize", 0) or 0),
            opponent_prizes=int(view.get("opp", {}).get("prize", 0) or 0),
            opponent_hand_size=int(view.get("opp", {}).get("hand_n", 0) or 0),
            opponent_active_damage=int(profile._active(view.get("opp", {})).get("damage", 0) or 0),
            opponent_active_energy=profile._energy_n(profile._active(view.get("opp", {}))),
        )
        turn = int(view.get("turn", 0) or 0)
        if turn <= 3:
            phase = TurnPhase.OPENING
        elif ready == 0:
            phase = TurnPhase.BUILD
        elif turn >= 10:
            phase = TurnPhase.ENDGAME
        else:
            phase = TurnPhase.PRESSURE
        roles = frozenset(profile._main_role(s) for s in (view.get("opts") or []))
        return TurnState(
            view=view,
            turn=turn,
            action_count=int(view.get("ta", 0) or 0),
            phase=phase,
            ledger=ledger,
            legal_roles=roles,
        )
