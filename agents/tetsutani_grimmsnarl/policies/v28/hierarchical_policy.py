from __future__ import annotations

from .index_resolver import resolve_index
from .validated_fallback_policy import choose as fallback_choose
from .processing_queue import plan_main_action
from .protocol_resolver import plan_protocol
from .resource_ledger import ResourceLedger
from .search_planner import plan_search
from .selection_planner import plan_selection
from .target_evaluator import plan_target
from .turn_dag import plan_turn_dag
from .turn_phase import infer_turn_phase
from .deadline_ledger import DeadlineLedger
from .lexicographic_scheduler import plan_lexicographic_deadline
from .continuity_scheduler import plan_continuity
from .strategic_memory import StrategicMemory
from .turn_objective import infer_turn_objective

_LAST_TRACE: dict = {}


def choose(state: dict, options: list[dict], history: list[dict], memory: StrategicMemory | None = None) -> list[int]:
    global _LAST_TRACE
    fallback = fallback_choose(state, options, history)
    _LAST_TRACE = {"layer": "fallback", "job": "fallback", "mode": "fallback_only", "context": int(state.get("context", -1)), "changed": False}
    if not options:
        return fallback

    ledger = ResourceLedger.from_state(state, history)
    objective = infer_turn_objective(ledger)
    phase = infer_turn_phase(ledger)
    memory_view = memory.observe(ledger, objective, phase) if memory is not None else None
    context = int(state.get("context", -1))
    selected = fallback

    if context == 0:
        # Preserve all validated v17-v19 committed jobs first, then pass their result
        # through the explicit turn-dependency DAG.  The DAG may complete a missing
        # prerequisite, or structurally confirm the executable node.
        queue_decision = plan_main_action(ledger, options, selected, objective)
        if queue_decision is not None:
            selected = [queue_decision.index]
            _LAST_TRACE.update(layer="processing_queue", job=queue_decision.job, mode=queue_decision.mode)
        dag_decision = plan_turn_dag(ledger, options, selected)
        if dag_decision is not None:
            selected = [dag_decision.index]
            _LAST_TRACE.update(layer="turn_dag", job=dag_decision.job, mode=dag_decision.mode)
        deadlines = DeadlineLedger.from_ledger(ledger, phase)
        schedule_decision = plan_lexicographic_deadline(ledger, options, selected, deadlines)
        if schedule_decision is not None:
            selected = [schedule_decision.index]
            _LAST_TRACE.update(layer="deadline_scheduler", job=schedule_decision.job, mode=schedule_decision.mode)
        if memory_view is not None:
            continuity_decision = plan_continuity(ledger, options, selected, memory_view)
            if continuity_decision is not None:
                selected = [continuity_decision.index]
                _LAST_TRACE.update(layer="continuity_scheduler", job=continuity_decision.job, mode=continuity_decision.mode, objective=memory_view.primary.value, objective_age=memory_view.objective_age, phase_age=memory_view.phase_age, passive_campaign=memory_view.passive_pressure_campaign, passive_campaign_age=memory_view.passive_campaign_age, last_turn_job=memory_view.last_turn_job, current_turn_last_job=memory_view.current_turn_last_job)
    elif context == 7:
        decision = plan_search(
            ledger,
            options,
            selected,
            objective,
            effect=int(state.get("effect", 0) or 0),
        )
        if decision is not None:
            selected = list(decision.indices)
            _LAST_TRACE.update(layer="search_planner", job=decision.job, mode=decision.mode)
    elif context in (5, 8, 22, 27, 30, 34):
        decision = plan_selection(
            ledger,
            options,
            selected,
            context=context,
            effect=int(state.get("effect", 0) or 0),
            minimum=int(state.get("min", 0) or 0),
            maximum=int(state.get("max", 0) or 0),
        )
        if decision is not None:
            selected = list(decision.indices)
            _LAST_TRACE.update(layer="selection_planner", job=decision.job, mode=decision.mode)
    elif context in (3, 4, 13, 14, 15, 16, 17, 18, 19, 20, 21, 37):
        decision = plan_target(ledger, options, selected, objective, context=context)
        if decision is not None:
            selected = [decision.index]
            _LAST_TRACE.update(layer="target_evaluator", job=decision.job, mode=decision.mode)
    else:
        decision = plan_protocol(ledger, options, selected, context=context)
        if decision is not None:
            selected = list(decision.indices)
            _LAST_TRACE.update(layer="protocol_resolver", job=decision.job, mode=decision.mode)

    final = resolve_index(options, selected, fallback)
    _LAST_TRACE["changed"] = final != fallback
    _LAST_TRACE["selected"] = list(final)
    _LAST_TRACE["fallback"] = list(fallback)
    return final
