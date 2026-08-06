from __future__ import annotations

"""Visible-board Punk Up count refinements."""
import os
from . import validated_rule_profile as profile


class ManualCountProtocol:
    def __init__(self) -> None:
        raw = os.environ.get("PTCG_V32_COUNT_RULES", "ALL").strip()
        self.enabled = None if raw in ("", "ALL") else {x for x in raw.split(",") if x}

    def _on(self, name: str) -> bool:
        return self.enabled is None or name in self.enabled

    @staticmethod
    def _take(view: dict, count: int) -> list[int]:
        count = max(int(view.get("min", 0) or 0), min(int(view.get("max", 0) or 0), len(view.get("opts") or []), count))
        return profile._legal_fill(view, list(range(count)))

    def resolve_punkup(self, view: dict, selected: list[int]) -> list[int]:
        n = len(view.get("opts") or [])
        me = view.get("me") or {}
        opp = view.get("opp") or {}
        inplay = profile._all_inplay(me)
        active = profile._active(me)
        attack_lines = [p for p in inplay if int(p.get("id", 0) or 0) in (profile.IMPIDIMP, profile.MORGREM, profile.GRIMMSNARL)]
        grims = [p for p in inplay if int(p.get("id", 0) or 0) == profile.GRIMMSNARL]
        grim_deficit = sum(max(0, 2 - profile._energy_n(p)) for p in grims)

        # Six available Darkness cards do not imply taking three when the
        # Active Grimmsnarl only needs one Energy to become ready.
        if (
            self._on("SIX_DARK_TWO_FOR_ONE_ENERGY_ACTIVE_GRIM")
            and n == 6
            and int(active.get("id", 0) or 0) == profile.GRIMMSNARL
            and profile._energy_n(active) == 1
        ):
            return self._take(view, 2)

        # With exactly two Darkness cards offered in the opening and only one
        # missing Grimmsnarl Energy, take the single necessary card.
        if (
            self._on("TWO_DARK_ONE_FOR_SINGLE_GRIM_DEFICIT_OPENING")
            and n == 2
            and grim_deficit == 1
            and int(opp.get("prize", 0) or 0) == 6
        ):
            return self._take(view, 1)

        # Five offered cards with exactly two visible attacker-line bodies in
        # the opening are compressed to two Energy, avoiding surplus attachment.
        if (
            self._on("FIVE_DARK_TWO_FOR_TWO_OPENING_LINES")
            and n == 5
            and len(attack_lines) == 2
            and int(opp.get("prize", 0) or 0) == 6
        ):
            return self._take(view, 2)

        return selected
