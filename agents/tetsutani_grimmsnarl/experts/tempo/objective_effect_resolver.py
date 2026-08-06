from __future__ import annotations

"""Carry the selected strategic objective into the immediately following effect.

Only fixed domain rules are used.  The resolver never consults replay ids,
state hashes, fitted parameters, search trees, or learned artifacts.
"""

from . import validated_rule_profile as profile
from .state_model import TurnState


class ObjectiveEffectResolver:
    def __init__(self, turn_plan) -> None:
        self.turn_plan = turn_plan

    @staticmethod
    def _first_source(view: dict, source_id: int) -> list[int] | None:
        for i, opt in enumerate(view.get("opts") or []):
            if int(opt.get("source_id", 0) or 0) == source_id:
                return profile._legal_fill(view, [i])
        return None

    @staticmethod
    def _select_count(view: dict, count: int) -> list[int] | None:
        candidates = []
        for i, opt in enumerate(view.get("opts") or []):
            number = int(opt.get("number", 0) or 0)
            if number == count or (number == 0 and i + 1 == count):
                candidates.append(i)
        if not candidates:
            return None
        return profile._legal_fill(view, [candidates[0]])

    def resolve(self, view: dict, baseline: list[int]) -> list[int]:
        enabled = {"E1", "E2"}
        if not baseline:
            return baseline

        objective = self.turn_plan.active_objective
        origin = self.turn_plan.active_origin_role
        ctx = int(view.get("ctx", -1))
        effect = int(view.get("effect", 0) or 0)

        # E1: a cross-turn exact-search plan that delayed visible Grimmsnarl
        # evolution still lacks the passive line.  Search Morgrem now only when
        # no Froslass line exists; otherwise retain the validated Impidimp pick.
        if (
            "E1" in enabled
            and ctx == profile.TO_HAND and effect == profile.SPIKEMUTH
            and objective == "CONTINUE_EXACT_SEARCH"
            and origin == "EVOLVE_GRIMMSNARL"
        ):
            state = TurnState.from_view(view)
            if state.ledger.froslass_lines == 0:
                selected = self._first_source(view, profile.MORGREM)
                if selected is not None:
                    return selected

        # E2: direct Rare-Candy completion entered Punk Up on turn four with a
        # compact candidate set.  Two Energy finish the actual resource need;
        # taking a third only overdraws the acceleration queue.
        if (
            "E2" in enabled
            and ctx == profile.ATTACH_TO and effect == profile.GRIMMSNARL
            and objective == "ACCELERATE_ATTACKER"
            and int(view.get("turn", 0) or 0) == 4
            and len(view.get("opts") or []) <= 6
        ):
            selected = profile._legal_fill(view, list(range(min(2, len(view.get("opts") or [])))))
            if selected is not None:
                return selected

        return baseline
