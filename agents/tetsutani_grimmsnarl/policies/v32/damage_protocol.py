from __future__ import annotations

"""Pure handwritten Adrena-Brain damage-source protocol.

The rules in this file are deterministic, local board-state thresholds.  They
use no replay identifiers, no learned parameters, no search and no randomness.
Rule names can be selected with PTCG_DAMAGE_RULES for ablation validation.
"""

import os

from . import validated_rule_profile as profile


ALL_RULES = (
    "DAMAGED_IMPIDIMP_BEFORE_ACTIVE_GRIM",
    "DAMAGED_MUNK_BEFORE_BENCH_GRIM",
    "SIXTY_DAMAGE_MUNK_BEFORE_ACTIVE_GRIM",
)

def _v32_enabled() -> set[str] | None:
    raw = os.environ.get("PTCG_V32_DAMAGE_RULES", "ALL").strip()
    return None if raw in ("", "ALL") else {x for x in raw.split(",") if x}


def _enabled_rules() -> set[str]:
    raw = os.environ.get("PTCG_DAMAGE_RULES", "ALL").strip()
    if not raw or raw.upper() == "ALL":
        return set(ALL_RULES)
    if raw.upper() == "NONE":
        return set()
    return {part.strip().upper() for part in raw.split(",") if part.strip()}


class ManualDamageProtocol:
    """Resolve the single Pokémon from which Adrena-Brain removes damage."""

    @staticmethod
    def _cid(option: dict) -> int:
        return int(option.get("source_id", 0) or 0)

    @staticmethod
    def _zone(option: dict) -> int:
        return int(option.get("source_zone", 0) or 0)

    @staticmethod
    def _damage(option: dict) -> int:
        state = option.get("source_obj") or {}
        return int(state.get("damage", 0) or 0)

    @classmethod
    def _selected_signature(cls, view: dict, selected: list[int]) -> tuple[int, int]:
        if not selected or selected[0] < 0 or selected[0] >= len(view.get("opts") or []):
            return (0, 0)
        option = view["opts"][selected[0]]
        return (cls._zone(option), cls._cid(option))

    @classmethod
    def _choose(cls, view: dict, *, zone: int, card_id: int, min_damage: int = 1) -> list[int] | None:
        candidates = [
            i
            for i, option in enumerate(view.get("opts") or [])
            if cls._zone(option) == zone
            and cls._cid(option) == card_id
            and cls._damage(option) >= min_damage
        ]
        if not candidates:
            return None
        best = max(candidates, key=lambda i: (cls._damage(view["opts"][i]), -i))
        return profile._legal_fill(view, [best])

    def resolve(self, view: dict, selected: list[int]) -> list[int]:
        rules = _enabled_rules()
        selected_zone, selected_id = self._selected_signature(view, selected)
        stadium_live = int(view.get("stadium", 0) or 0) == profile.SPIKEMUTH

        # When the generic source is the Active Grimmsnarl ex, a damaged
        # Impidimp is cleared first while the attacker remains below the
        # observed 220-damage survival boundary.  This preserves the sole basic
        # attacker line without solving future turns.
        if (
            "DAMAGED_IMPIDIMP_BEFORE_ACTIVE_GRIM" in rules
            and selected_zone == 4
            and selected_id == profile.GRIMMSNARL
        ):
            active = (view.get("me") or {}).get("active") or []
            active_damage = int((active[0] if active else {}).get("damage", 0) or 0)
            if active_damage <= 210:
                override = self._choose(
                    view,
                    zone=5,
                    card_id=profile.IMPIDIMP,
                    min_damage=1,
                )
                if override is not None:
                    return override

        # With Spikemuth live, a damaged Munkidori is the first recyclable
        # utility body to protect when the generic selector points at a bench
        # Grimmsnarl ex.
        if (
            "DAMAGED_MUNK_BEFORE_BENCH_GRIM" in rules
            and stadium_live
            and selected_zone == 5
            and selected_id == profile.GRIMMSNARL
        ):
            override = self._choose(
                view,
                zone=5,
                card_id=profile.MUNKIDORI,
                min_damage=30,
            )
            if override is not None:
                return override

        # At two Froslass cycles (60 damage), protect the live Munkidori before
        # removing damage from the Active Grimmsnarl while Spikemuth is active.
        if (
            "SIXTY_DAMAGE_MUNK_BEFORE_ACTIVE_GRIM" in rules
            and stadium_live
            and selected_zone == 4
            and selected_id == profile.GRIMMSNARL
        ):
            override = self._choose(
                view,
                zone=5,
                card_id=profile.MUNKIDORI,
                min_damage=60,
            )
            if override is not None:
                return override

        refined = _v32_enabled()
        def on(name: str) -> bool:
            return refined is None or name in refined

        selected_zone, selected_id = self._selected_signature(view, selected)
        selected_damage = 0
        if selected:
            selected_damage = self._damage(view["opts"][selected[0]])
        opp_active = ((view.get("opp") or {}).get("active") or [{}])[0]
        opp_active_damage = int(opp_active.get("damage", 0) or 0)
        me = view.get("me") or {}

        # Once the opposing Active is already pressured, keep the main attacker
        # out of KO range instead of clearing only 10-30 damage from Munkidori.
        if (
            on("HEAL_ACTIVE_GRIM_OVER_LIGHT_MUNK_ON_PRESSURED_OPP")
            and selected_zone == 5 and selected_id == profile.MUNKIDORI
            and selected_damage <= 30 and opp_active_damage >= 120
        ):
            hit = self._choose(view, zone=4, card_id=profile.GRIMMSNARL, min_damage=1)
            if hit is not None:
                return hit

        # Sixty or more damage on a Morgrem bridge is removed before healing the
        # large Active Grimmsnarl pool.
        if (
            on("HEAL_DAMAGED_MORGREM_OVER_ACTIVE_GRIM")
            and selected_zone == 4 and selected_id == profile.GRIMMSNARL
        ):
            hit = self._choose(view, zone=5, card_id=profile.MORGREM, min_damage=60)
            if hit is not None:
                return hit

        # On a full late-action board, preserve the Active attacker rather than
        # clearing a small amount from a bench Impidimp.
        if (
            on("HEAL_ACTIVE_GRIM_OVER_IMPIDIMP_ON_FULL_BOARD")
            and selected_zone == 5 and selected_id == profile.IMPIDIMP
            and int(view.get("ta", 0) or 0) >= 8
            and len(me.get("bench") or []) >= 5
        ):
            hit = self._choose(view, zone=4, card_id=profile.GRIMMSNARL, min_damage=1)
            if hit is not None:
                return hit

        # A severely damaged bench Grimmsnarl is protected before an Active
        # Munkidori carrying only a small damage mark.
        if (
            on("HEAL_SEVERE_BENCH_GRIM_OVER_ACTIVE_MUNK")
            and selected_zone == 4 and selected_id == profile.MUNKIDORI
            and selected_damage <= 30
        ):
            hit = self._choose(view, zone=5, card_id=profile.GRIMMSNARL, min_damage=120)
            if hit is not None:
                return hit

        return selected
