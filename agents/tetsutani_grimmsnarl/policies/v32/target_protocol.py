from __future__ import annotations

"""Handwritten target-location refinements for damage placement."""

import os
from . import validated_rule_profile as profile


class ManualTargetProtocol:
    def __init__(self) -> None:
        raw = os.environ.get("PTCG_V32_TARGET_RULES", "ALL").strip()
        self.enabled = None if raw in ("", "ALL") else {x for x in raw.split(",") if x}

    def _on(self, name: str) -> bool:
        return self.enabled is None or name in self.enabled

    @staticmethod
    def _state(option: dict) -> dict:
        return option.get("source_obj") or {}

    @classmethod
    def _sig(cls, view: dict, selected: list[int]) -> tuple[int, int]:
        if not selected:
            return (0, 0)
        o = view["opts"][selected[0]]
        return (int(o.get("source_zone", 0) or 0), int(o.get("source_id", 0) or 0))

    @classmethod
    def _damage(cls, option: dict) -> int:
        return int(cls._state(option).get("damage", 0) or 0)

    @classmethod
    def _hp(cls, option: dict) -> int:
        return int(cls._state(option).get("hp", 999) or 999)

    @classmethod
    def _energy(cls, option: dict) -> int:
        return profile._energy_n(cls._state(option))

    @classmethod
    def _choose(cls, view: dict, zone: int, cid: int) -> list[int] | None:
        cand = [
            i for i, o in enumerate(view.get("opts") or [])
            if int(o.get("source_zone", 0) or 0) == zone
            and int(o.get("source_id", 0) or 0) == cid
        ]
        if not cand:
            return None
        # Damage placement selects the most advanced damaged/low-remaining-HP
        # copy when duplicate card IDs are legal.
        best = min(cand, key=lambda i: (cls._hp(view["opts"][i]), -cls._energy(view["opts"][i]), i))
        return profile._legal_fill(view, [best])

    def resolve(self, view: dict, selected: list[int]) -> list[int]:
        if not selected:
            return selected
        ctx = int(view.get("ctx", -1))
        zone, cid = self._sig(view, selected)
        opp = view.get("opp") or {}
        opp_active = (opp.get("active") or [{}])[0]

        # In the Crustle wall matchup, mark the Energy-loaded Active copy rather
        # than an identical Bench copy.  Location is strategically material
        # even though the card ID is the same.
        if (
            self._on("ADRENA_ACTIVE_ENERGIZED_CRUSTLE")
            and ctx == profile.DAMAGE_COUNTER
            and zone == 5 and cid == 345
            and int(opp_active.get("id", 0) or 0) == 345
            and profile._energy_n(opp_active) >= 2
        ):
            hit = self._choose(view, 4, 345)
            if hit:
                return hit

        # A bench Grimmsnarl ex at 110 HP or less is the priority Adrena-Brain
        # mark over a lightly damaged Munkidori utility body.
        if (
            self._on("ADRENA_LOW_HP_BENCH_GRIM_OVER_MUNK")
            and ctx == profile.DAMAGE_COUNTER
            and zone == 5 and cid == profile.MUNKIDORI
        ):
            hit = self._choose(view, 5, profile.GRIMMSNARL)
            if hit and self._hp(view["opts"][hit[0]]) <= 110 and self._damage(view["opts"][selected[0]]) <= 30:
                return hit

        # Shadow Bullet uses the same low-HP cutoff for its Bench damage mark.
        if (
            self._on("SHADOW_LOW_HP_BENCH_GRIM_OVER_MUNK")
            and ctx == profile.DAMAGE
            and zone == 5 and cid == profile.MUNKIDORI
        ):
            hit = self._choose(view, 5, profile.GRIMMSNARL)
            if hit and self._hp(view["opts"][hit[0]]) <= 110 and self._damage(view["opts"][selected[0]]) <= 30:
                return hit

        # At two own prizes, preserve the KO map by marking an Impidimp line
        # before an undamaged/lightly damaged Munkidori.
        if (
            self._on("SHADOW_IMPIDIMP_OVER_MUNK_AT_TWO_PRIZES")
            and ctx == profile.DAMAGE
            and zone == 5 and cid == profile.MUNKIDORI
            and int((view.get("me") or {}).get("prize", 0) or 0) <= 2
            and self._damage(view["opts"][selected[0]]) <= 30
        ):
            hit = self._choose(view, 5, profile.IMPIDIMP)
            if hit:
                return hit

        # In the opening phase, begin pressure on the passive Snorunt line
        # before marking Munkidori.
        if (
            self._on("SHADOW_SNORUNT_OVER_MUNK_OPENING")
            and ctx == profile.DAMAGE
            and zone == 5 and cid == profile.MUNKIDORI
            and int(view.get("turn", 0) or 0) <= 3
        ):
            hit = self._choose(view, 5, profile.SNORUNT)
            if hit:
                return hit

        return selected
