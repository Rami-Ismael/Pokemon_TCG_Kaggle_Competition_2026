from __future__ import annotations

from dataclasses import dataclass

from . import validated_rule_profile as profile
@dataclass(frozen=True)
class TargetScore:
    index: int
    score: tuple


class TargetEvaluator:
    """Role-aware target evaluation layer.

    Target selection is separated from turn planning.  The primitive selector
    establishes KO and low-HP safety.  This layer then applies fixed public-card
    threat rules: persistent evolved engines outrank nearby low-impact pivots.
    """

    @staticmethod
    def _state(option: dict) -> dict:
        return option.get("source_obj") or option.get("target_obj") or {
            "id": option.get("source_id", option.get("target_id", 0)),
            "hp": 999,
            "energies": [],
        }

    @classmethod
    def _cid(cls, option: dict) -> int:
        p = cls._state(option)
        return int(p.get("id", option.get("source_id", option.get("target_id", 0))) or 0)

    @classmethod
    def _hp(cls, option: dict) -> int:
        return int(cls._state(option).get("hp", 999) or 999)

    @classmethod
    def _energy(cls, option: dict) -> int:
        return profile._energy_n(cls._state(option))

    @classmethod
    def _choose_id(cls, view: dict, card_id: int) -> list[int] | None:
        cand = [i for i, s in enumerate(view.get("opts") or []) if cls._cid(s) == card_id]
        if not cand:
            return None
        best = min(cand, key=lambda i: (cls._hp(view["opts"][i]), -cls._energy(view["opts"][i]), i))
        return profile._legal_fill(view, [best])

    def _engine_threat_override(self, view: dict, selected: list[int]) -> list[int]:
        if not selected or int(view.get("ctx", -1)) not in (profile.DAMAGE_COUNTER, profile.DAMAGE):
            return selected
        ctx = int(view.get("ctx", -1))
        pred = view["opts"][selected[0]]
        pred_id = self._cid(pred)
        pred_hp = self._hp(pred)
        turn = int(view.get("turn", 0) or 0)
        available = {self._cid(s) for s in view.get("opts") or []}

        # Mysterious Rock Inn is a persistent wall.  Adrena-Brain marks the
        # evolved Crustle rather than its low-impact Dwebble pivot.
        if ctx == profile.DAMAGE_COUNTER and pred_id == 344 and 345 in available:
            return self._choose_id(view, 345) or selected

        # Cheer On to Glory is a continuous damage amplifier.  Prefer Roserade
        # to Cynthia's Gabite while the HP gap remains within one 30-damage mark.
        if pred_id == 380 and 342 in available:
            target = self._choose_id(view, 342)
            if target is not None and self._hp(view["opts"][target[0]]) <= pred_hp + 40:
                return target

        # Charging Up recursively restores Energy.  In Adrena-Brain selection,
        # mark Spidops before Articuno's defensive pivot.
        if ctx == profile.DAMAGE_COUNTER and pred_id == 414 and 401 in available and turn >= 4:
            return self._choose_id(view, 401) or selected

        # Recon Directive is a repeatable draw engine.  When Drakloak already
        # carries Energy and is no more than 20 HP healthier, it outranks the
        # generic Munkidori signature mark.
        if pred_id == profile.MUNKIDORI and 120 in available:
            target = self._choose_id(view, 120)
            if (
                target is not None
                and self._energy(view["opts"][target[0]]) >= 1
                and self._hp(view["opts"][target[0]]) <= pred_hp + 20
            ):
                return target

        # Late turns value stopping the Dragapult evolution chain.  Mark Dreepy
        # when it is within 20 HP of the baseline Munkidori target.
        if pred_id == profile.MUNKIDORI and 119 in available and turn >= 9:
            target = self._choose_id(view, 119)
            if target is not None and self._hp(view["opts"][target[0]]) <= pred_hp + 20:
                return target

        return selected

    def select(self, view: dict) -> list[int]:
        ctx = int(view.get("ctx", -1))
        if ctx == profile.REMOVE_DAMAGE:
            return profile._remove_damage_source(view)
        if ctx == profile.DAMAGE_COUNTER:
            base = profile._damage_target(view, int(profile._HANDWRITTEN_STATE.get("adrena_count", 3)))
            return self._engine_threat_override(view, base)
        if ctx == profile.DAMAGE:
            base = profile._damage_target(view, 3, bench_only=True)
            return self._engine_threat_override(view, base)
        if ctx == profile.ATTACH_TO:
            return profile._punkup_count(view)
        if ctx == profile.ATTACH_FROM:
            return profile._punkup_target(view)
        raise ValueError(f"unsupported targeting context: {ctx}")
