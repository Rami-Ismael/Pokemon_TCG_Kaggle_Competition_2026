from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .resource_ledger import ResourceLedger


@dataclass(frozen=True)
class IdentityLedger:
    """Deterministic public provenance for same-name card instances.

    The replay state does not expose hidden information.  This ledger only records
    public board position and serials that already appeared in the within-turn
    action history.  It is used exclusively for deterministic index tie-breaking.
    """

    active_serial: int
    bench_serials: tuple[int, ...]
    last_touched_serial: int
    last_evolved_serial: int
    last_energy_target_serial: int

    @classmethod
    def from_ledger(cls, ledger: ResourceLedger) -> "IdentityLedger":
        active_serial = int(ledger.active.get("serial", -1) or -1)
        bench_serials = tuple(int(c.get("serial", -1) or -1) for c in ledger.bench)
        last_touched = -1
        last_evolved = -1
        last_energy = -1
        for action in ledger.history:
            serial = int(action.get("target_serial", -1) if action.get("target_serial") is not None else -1)
            if serial >= 0:
                last_touched = serial
            t = int(action.get("type", -1) if action.get("type") is not None else -1)
            if t == 9 and serial >= 0:
                last_evolved = serial
            if t == 8 and serial >= 0:
                last_energy = serial
        return cls(active_serial, bench_serials, last_touched, last_evolved, last_energy)

    def board_order_key(self, area: int, index: int) -> tuple[int, int]:
        if area == 4:
            return (0, 0)
        if area == 5:
            return (1, max(0, int(index)))
        return (2, max(0, int(index)))
