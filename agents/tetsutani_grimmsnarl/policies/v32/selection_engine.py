from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from . import validated_rule_profile as profile
from .effect_planner import EffectObjectivePlanner
from .effect_resolver import EffectIndexResolver
from .state_model import TurnState
from .targeting import TargetEvaluator
from .selection_protocol import ManualSelectionProtocol
from .poffin_protocol import ManualPoffinProtocol


class SelectionPurpose(Enum):
    SETUP_ACTIVE = auto()
    SETUP_BENCH = auto()
    SWITCH_TARGET = auto()
    PROMOTION_TARGET = auto()
    BENCH_SEARCH = auto()
    CARD_SEARCH = auto()
    DISCARD_RESOURCE = auto()
    DAMAGE_TARGET = auto()
    DAMAGE_SOURCE = auto()
    ENERGY_COUNT = auto()
    ENERGY_TARGET = auto()
    ORDER_RESOLUTION = auto()
    BINARY_CHOICE = auto()
    COMPATIBILITY = auto()


@dataclass(frozen=True)
class SelectionTrace:
    purpose: SelectionPurpose
    baseline: tuple[int, ...]
    selected: tuple[int, ...]
    reason: str
    semantic_queue: tuple[str, ...] = ()


class HierarchicalSelectionEngine:
    """Single entry point for every non-MAIN selection context.

    Each context is assigned a domain purpose, obtains a validated primitive
    intent, optionally passes through resource/target planning, and only then
    resolves legal indices.  This keeps setup, search, targeting and multiple
    selections behind the same architectural boundary.
    """

    def __init__(self) -> None:
        self.targets = TargetEvaluator()
        self.effect_planner = EffectObjectivePlanner()
        self.effect_resolver = EffectIndexResolver()
        self.manual_protocol = ManualSelectionProtocol()
        self.poffin_protocol = ManualPoffinProtocol()

    @staticmethod
    def _max_number(view: dict) -> list[int]:
        best = max(range(len(view["opts"])), key=lambda i: (view["opts"][i].get("number", 0), i), default=0)
        return profile._legal_fill(view, [best] if view["opts"] else [])

    def decide(self, view: dict) -> list[int]:
        ctx = int(view.get("ctx", -1))
        if ctx == profile.SETUP_ACTIVE:
            return profile._setup_active(view)
        if ctx == profile.SETUP_BENCH:
            return profile._setup_bench(view)
        if ctx == profile.SWITCH:
            return profile._switch(view)
        if ctx == profile.TO_ACTIVE:
            return profile._choose_promotion(view)
        if ctx == profile.TO_BENCH:
            return self.poffin_protocol.resolve(view, profile._poffin(view))
        if ctx == profile.TO_HAND:
            baseline = profile._to_hand(view)
            if not baseline:
                return baseline
            state = TurnState.from_view(view)
            baseline_card_id = int(view["opts"][baseline[0]].get("source_id", 0) or 0)
            available = {int(s.get("source_id", 0) or 0) for s in view.get("opts") or []}
            intents = self.effect_planner.plan_to_hand(
                state,
                int(view.get("effect", 0) or 0),
                baseline_card_id,
                available,
            )
            selected = self.effect_resolver.resolve_card(view, intents, baseline)
            return self.manual_protocol.resolve_to_hand(view, selected)
        if ctx == profile.DISCARD_CTX:
            return profile._discard(view)
        if ctx in (profile.REMOVE_DAMAGE, profile.DAMAGE_COUNTER, profile.DAMAGE, profile.ATTACH_TO, profile.ATTACH_FROM):
            return self.targets.select(view)
        if ctx == profile.REMOVE_DAMAGE_COUNT:
            selected = self._max_number(view)
            if selected:
                n = view["opts"][selected[0]].get("number", 0) or (selected[0] + 1)
                profile._HANDWRITTEN_STATE["adrena_count"] = int(n)
            return selected
        if ctx in (profile.DISCARD_TOOL, profile.DISCARD_ENERGY, profile.SKILL_ORDER):
            return profile._legal_fill(view, list(range(min(view["max"], len(view["opts"])))))
        if ctx == profile.EVOLVE_CTX:
            return profile._legal_fill(view, [0] if view["opts"] else [])
        if ctx == profile.DRAW_COUNT:
            return self._max_number(view)
        if ctx in (profile.IS_FIRST, profile.MULLIGAN, profile.ACTIVATE) or view.get("stype") == 9:
            return profile._choose_first(view, lambda s: s["type"] == profile.YES) or profile._legal_fill(view, [0])
        return profile._legal_fill(view, list(range(min(view["max"], len(view["opts"])))))

    def explain(self, view: dict) -> SelectionTrace:
        ctx = int(view.get("ctx", -1))
        selected = self.decide(view)
        if ctx == profile.TO_HAND:
            baseline = profile._to_hand(view)
            if baseline:
                state = TurnState.from_view(view)
                baseline_card_id = int(view["opts"][baseline[0]].get("source_id", 0) or 0)
                available = {int(s.get("source_id", 0) or 0) for s in view.get("opts") or []}
                intents = self.effect_planner.plan_to_hand(state, int(view.get("effect", 0) or 0), baseline_card_id, available)
                return SelectionTrace(
                    SelectionPurpose.CARD_SEARCH,
                    tuple(baseline), tuple(selected),
                    intents[0].reason if intents else "card search compatibility",
                    tuple(f"{x.objective.name}:{x.desired_card_id}" for x in intents),
                )
        mapping = {
            profile.SETUP_ACTIVE: SelectionPurpose.SETUP_ACTIVE,
            profile.SETUP_BENCH: SelectionPurpose.SETUP_BENCH,
            profile.SWITCH: SelectionPurpose.SWITCH_TARGET,
            profile.TO_ACTIVE: SelectionPurpose.PROMOTION_TARGET,
            profile.TO_BENCH: SelectionPurpose.BENCH_SEARCH,
            profile.DISCARD_CTX: SelectionPurpose.DISCARD_RESOURCE,
            profile.REMOVE_DAMAGE: SelectionPurpose.DAMAGE_SOURCE,
            profile.DAMAGE_COUNTER: SelectionPurpose.DAMAGE_TARGET,
            profile.DAMAGE: SelectionPurpose.DAMAGE_TARGET,
            profile.ATTACH_TO: SelectionPurpose.ENERGY_COUNT,
            profile.ATTACH_FROM: SelectionPurpose.ENERGY_TARGET,
            profile.SKILL_ORDER: SelectionPurpose.ORDER_RESOLUTION,
            profile.IS_FIRST: SelectionPurpose.BINARY_CHOICE,
            profile.MULLIGAN: SelectionPurpose.BINARY_CHOICE,
            profile.ACTIVATE: SelectionPurpose.BINARY_CHOICE,
        }
        purpose = mapping.get(ctx, SelectionPurpose.COMPATIBILITY)
        return SelectionTrace(purpose, tuple(selected), tuple(selected), "hierarchical context resolver")
