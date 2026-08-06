from __future__ import annotations

"""Five-layer deterministic policy:
turn objective -> process queue -> resource ledger -> target evaluation -> index resolution.
"""

from . import validated_rule_profile as profile
from .domain import *  # re-export fixed public constants
from .index_resolver import IndexResolver
from .planner import ObjectivePlanner
from .process_queue import ProcessQueueBuilder
from .state_model import TurnState
from .targeting import TargetEvaluator
from .selection_engine import HierarchicalSelectionEngine
from .turn_plan import PersistentTurnPlan
from .objective_effect_resolver import ObjectiveEffectResolver
from .strategic_mode import StrategicModeController
from .decision_protocol import ManualProtocolArbiter
from .refinement_protocol import ManualRefinementArbiter

_PLANNER = ObjectivePlanner()
_QUEUE = ProcessQueueBuilder()
_RESOLVER = IndexResolver()
_TARGETS = TargetEvaluator()
_SELECTIONS = HierarchicalSelectionEngine()
_TURN_PLAN = PersistentTurnPlan()
_OBJECTIVE_EFFECTS = ObjectiveEffectResolver(_TURN_PLAN)
_MODES = StrategicModeController()
_PROTOCOL = ManualProtocolArbiter()
_REFINEMENT = ManualRefinementArbiter()


def reset() -> None:
    profile.reset()
    _TURN_PLAN.reset()


def view_from_raw(obs: dict) -> dict:
    return profile.view_from_raw(obs)


def view_from_compact(row: dict) -> dict:
    return profile.view_from_compact(row)


def _sync_state(view: dict) -> None:
    ep = view.get("ep")
    if ep is not None and profile._HANDWRITTEN_STATE.get("episode") != ep:
        profile.reset()
        profile._HANDWRITTEN_STATE["episode"] = ep
    turn = int(view.get("turn", 0) or 0)
    last = profile._HANDWRITTEN_STATE.get("last_turn")
    if last is not None and turn < last:
        profile.reset()
        profile._HANDWRITTEN_STATE["episode"] = ep
    profile._HANDWRITTEN_STATE["last_turn"] = turn


def _resolve_non_main(view: dict) -> list[int]:
    baseline = _SELECTIONS.decide(view)
    return _OBJECTIVE_EFFECTS.resolve(view, baseline)


def decide(view: dict) -> list[int]:
    _sync_state(view)
    if int(view.get("ctx", -1)) != MAIN:
        return _resolve_non_main(view)

    primitive = profile._main(view)
    if not primitive:
        return primitive
    state = TurnState.from_view(view)
    primitive_role = profile._main_role(view["opts"][primitive[0]])

    # Stage 1: stateless hierarchy reproduces the validated v23 decision.
    objectives = _PLANNER.plan_main(state, primitive_role)
    queue = _QUEUE.build_main(state, objectives, primitive_role)
    provisional = _RESOLVER.resolve_main(view, queue, primitive)
    provisional_role = profile._main_role(view["opts"][provisional[0]]) if provisional else "NONE"

    # Stage 2: strategic-horizon planning carries unfinished objectives across
    # the opponent's intervening turn.
    horizon = _TURN_PLAN.horizon_objectives(state, provisional_role)
    horizon_queue = _QUEUE.build_main(state, horizon, provisional_role) if horizon else []
    horizon_selected = _RESOLVER.resolve_main(view, horizon_queue, provisional) if horizon else provisional
    horizon_role = profile._main_role(view["opts"][horizon_selected[0]]) if horizon_selected else "NONE"

    # Stage 3: same-turn persistence may continue an unfinished process after
    # the horizon-aware provisional decision.
    continuation = _TURN_PLAN.continuation_objectives(state, horizon_role)
    if continuation:
        continuation_queue = _QUEUE.build_main(state, continuation, horizon_role)
        selected = _RESOLVER.resolve_main(view, continuation_queue, horizon_selected)
    else:
        continuation_queue = []
        selected = horizon_selected

    # Stage 4: explicit strategic-mode arbitration resolves objective conflicts
    # consistently across rebuild, engine-build, pressure and endgame modes.
    mode_role = profile._main_role(view["opts"][selected[0]]) if selected else "NONE"
    mode_objectives = _MODES.objectives(state, mode_role)
    mode_queue = _QUEUE.build_main(state, mode_objectives, mode_role) if mode_objectives else []
    if mode_objectives:
        selected = _RESOLVER.resolve_main(view, mode_queue, selected)

    # Stage 5: fixed visible-state action-order protocols.  These are narrow,
    # human-readable overrides for repeated sequencing conflicts; no fitting,
    # replay lookup, search, or stochastic component is used.
    protocol = _PROTOCOL.resolve(state, selected)
    if protocol.index is not None:
        selected = profile._legal_fill(view, [protocol.index])

    refinement = _REFINEMENT.resolve(state, selected)
    if refinement.index is not None:
        selected = profile._legal_fill(view, [refinement.index])

    selected_role = profile._main_role(view["opts"][selected[0]]) if selected else "NONE"
    selected_objective = "COMPATIBILITY"
    intent_trace = [*mode_queue, *continuation_queue, *horizon_queue, *queue]
    for intent in intent_trace:
        if intent.role == selected_role and intent.objective.name != "COMPATIBILITY":
            selected_objective = intent.objective.name
            break
    profile._remember_main_role(view, selected)
    _TURN_PLAN.record(state, selected_role, selected_objective, provisional_role)
    return selected


def explain_main(view: dict) -> dict:
    """Return the persistent five-layer decision trace for a MAIN selection."""
    if int(view.get("ctx", -1)) != MAIN:
        return {"context": int(view.get("ctx", -1)), "layer": "non-main selection resolver"}
    primitive = profile._main(view)
    if not primitive:
        return {"context": MAIN, "primitive": [], "selected": []}
    state = TurnState.from_view(view)
    primitive_role = profile._main_role(view["opts"][primitive[0]])
    objectives = _PLANNER.plan_main(state, primitive_role)
    queue = _QUEUE.build_main(state, objectives, primitive_role)
    provisional = _RESOLVER.resolve_main(view, queue, primitive)
    provisional_role = profile._main_role(view["opts"][provisional[0]]) if provisional else "NONE"
    horizon = _TURN_PLAN.horizon_objectives(state, provisional_role)
    horizon_queue = _QUEUE.build_main(state, horizon, provisional_role) if horizon else []
    horizon_selected = _RESOLVER.resolve_main(view, horizon_queue, provisional) if horizon else provisional
    horizon_role = profile._main_role(view["opts"][horizon_selected[0]]) if horizon_selected else "NONE"
    continuation = _TURN_PLAN.continuation_objectives(state, horizon_role)
    continuation_queue = _QUEUE.build_main(state, continuation, horizon_role) if continuation else []
    selected = _RESOLVER.resolve_main(view, continuation_queue, horizon_selected) if continuation else horizon_selected
    mode_role = profile._main_role(view["opts"][selected[0]]) if selected else "NONE"
    mode_trace = _MODES.classify(state)
    mode_objectives = _MODES.objectives(state, mode_role)
    mode_queue = _QUEUE.build_main(state, mode_objectives, mode_role) if mode_objectives else []
    if mode_objectives:
        selected = _RESOLVER.resolve_main(view, mode_queue, selected)
    protocol = _PROTOCOL.resolve(state, selected)
    if protocol.index is not None:
        selected = profile._legal_fill(view, [protocol.index])
    refinement = _REFINEMENT.resolve(state, selected)
    if refinement.index is not None:
        selected = profile._legal_fill(view, [refinement.index])
    return {
        "turn_objective": [{"name": x.objective.name,"priority":x.priority,"confidence":x.confidence,"reason":x.reason} for x in objectives],
        "horizon_objective": [{"name": x.objective.name,"priority":x.priority,"confidence":x.confidence,"reason":x.reason} for x in horizon],
        "persistent_objective": [{"name": x.objective.name,"priority":x.priority,"confidence":x.confidence,"reason":x.reason} for x in continuation],
        "strategic_mode": {"name": mode_trace.mode.name, "reason": mode_trace.reason},
        "mode_objective": [{"name": x.objective.name,"priority":x.priority,"confidence":x.confidence,"reason":x.reason} for x in mode_objectives],
        "process_queue": [{"role":x.role,"objective":x.objective.name,"priority":x.priority,"confidence":x.confidence} for x in queue],
        "horizon_queue": [{"role":x.role,"objective":x.objective.name,"priority":x.priority,"confidence":x.confidence} for x in horizon_queue],
        "persistent_queue": [{"role":x.role,"objective":x.objective.name,"priority":x.priority,"confidence":x.confidence} for x in continuation_queue],
        "mode_queue": [{"role":x.role,"objective":x.objective.name,"priority":x.priority,"confidence":x.confidence} for x in mode_queue],
        "resource_ledger": state.ledger.__dict__,
        "executed_roles": list(_TURN_PLAN.executed_roles),
        "executed_objectives": list(_TURN_PLAN.executed_objectives),
        "last_resource_delta": dict(_TURN_PLAN.last_resource_delta),
        "primitive_role": primitive_role,
        "provisional_role": provisional_role,
        "horizon_role": horizon_role,
        "protocol_rule": protocol.rule,
        "refinement_rule": refinement.rule,
        "selected_role": profile._main_role(view["opts"][selected[0]]) if selected else "NONE",
        "primitive_index": primitive,
        "provisional_index": provisional,
        "selected_index": selected,
    }


def explain_selection(view: dict) -> dict:
    """Return the hierarchical trace for a non-MAIN selection."""
    trace = _SELECTIONS.explain(view)
    return {
        "purpose": trace.purpose.name,
        "baseline_index": list(trace.baseline),
        "selected_index": list(trace.selected),
        "reason": trace.reason,
        "semantic_queue": list(trace.semantic_queue),
    }
