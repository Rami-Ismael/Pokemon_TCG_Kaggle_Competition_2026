from __future__ import annotations

from dataclasses import dataclass

from .resource_ledger import (
    DARK,
    FROSLASS,
    GRIMMSNARL,
    IMPIDIMP,
    MORGREM,
    MUNKIDORI,
    RARE_CANDY,
    SNORUNT,
    TOOL_SCRAPPER,
    ResourceLedger,
)

BOSS = 1182


@dataclass(frozen=True)
class ReservationLedger:
    """Future-turn public resource reservations.

    A reservation is not a fitted value.  It is a hard statement that a visible
    resource still has an unfinished strategic job.  The planner may spend only
    resources that are not reserved by a higher-priority unfinished job.
    """

    reserve_attacker_base: bool
    reserve_final_stage: bool
    reserve_control_body: bool
    reserve_control_energy: bool
    reserve_froslass_stage: bool
    reserve_gust_closer: bool
    preserve_tool_removal_over_candy: bool

    @classmethod
    def from_ledger(cls, ledger: ResourceLedger) -> "ReservationLedger":
        return cls(
            reserve_attacker_base=ledger.marnie_line_count < 2 and ledger.bench_slots > 0,
            reserve_final_stage=(ledger.ready_morgrem > 0 or ledger.board[MORGREM] > 0),
            reserve_control_body=(ledger.control_online and ledger.prizes <= 3),
            reserve_control_energy=(not ledger.control_online and ledger.hand[DARK] > 0),
            reserve_froslass_stage=(ledger.board[SNORUNT] > 0),
            reserve_gust_closer=(ledger.prizes <= 3),
            preserve_tool_removal_over_candy=(ledger.bench_slots == 0 and ledger.turn >= 8),
        )

    def reserved_card_ids(self) -> frozenset[int]:
        out: set[int] = set()
        if self.reserve_attacker_base:
            out.add(IMPIDIMP)
        if self.reserve_final_stage:
            out.add(GRIMMSNARL)
        if self.reserve_control_body:
            out.add(MUNKIDORI)
        if self.reserve_control_energy:
            out.add(DARK)
        if self.reserve_froslass_stage:
            out.add(FROSLASS)
        if self.reserve_gust_closer:
            out.add(BOSS)
        if self.preserve_tool_removal_over_candy:
            out.add(TOOL_SCRAPPER)
        return frozenset(out)

    def redundant_card_ids(self, ledger: ResourceLedger) -> frozenset[int]:
        out: set[int] = set()
        if ledger.board[FROSLASS] >= 2:
            out.add(RARE_CANDY)
        if self.preserve_tool_removal_over_candy:
            out.add(RARE_CANDY)
        return frozenset(out)
