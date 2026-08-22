from backend.services.agent_rag import route_intent
from backend.services.conversation_state import state_snapshot
from backend.services.support_chat.authorization import authorize_final_support_response
from backend.services.support_chat.response_helpers import (
    append_alert_notice,
    apply_emotional_distress_mode,
    has_tool_action,
    portal_help_reply,
    should_bypass_rag_for_patient_context,
    should_bypass_rag_for_tool_actions,
)
from backend.services.support_chat_context import _recent_patient_context
from backend.services.support_chat_policy import ALLOWED_SUPPORT_INTENTS
from backend.services.support_chat_response import (
    _build_response,
    _should_use_llm_direct_reply,
)
from backend.services.support_chat_safety import (
    _enforce_record_provenance,
    _ensure_complete_response,
    _ensure_complete_safety_reply,
    _immediate_danger_reply,
    _is_out_of_domain_request,
    _out_of_domain_reply,
    _safety_location_followup_reply,
)


def build_support_response(
    *,
    db,
    patient_id,
    user_record,
    normalized,
    actions,
    urgent_flags,
    routing_safety,
    tool_plan,
    safety_followup,
    immediate_danger,
    emotional_distress,
    compound_intent,
    run_patient_agent_pipeline_fn,
    generate_llm_response_fn,
):
    high_risk_alert = None
    alert_action = None
    try:
        from backend.services.high_risk_conversation_alerts import queue_and_dispatch_alert

        high_risk_alert, alert_action = queue_and_dispatch_alert(
            db,
            patient_id=patient_id,
            source_chat_message_id=user_record.id,
            immediate_danger=immediate_danger,
            urgent_flags=sorted(set(urgent_flags)),
            emotional_distress=emotional_distress,
        )
    except Exception:  # noqa: BLE001 - chat safety survives alert-outbox failure
        alert_action = (
            {
                "type": "high_risk_review_alert_failed",
                "message": (
                    "The review-alert queue could not confirm that it recorded this turn. "
                    "Do not wait for a portal reply if you feel unsafe or in immediate danger."
                ),
            }
            if (immediate_danger or urgent_flags)
            else None
        )
    if alert_action is not None:
        actions.append(alert_action)

    patient_context = _recent_patient_context(db, patient_id)
    direct_safety_reply = None
    direct_scope_reply = None
    direct_portal_reply = None
    if safety_followup:
        direct_safety_reply = _safety_location_followup_reply()
    elif immediate_danger:
        direct_safety_reply = _immediate_danger_reply()
    elif _is_out_of_domain_request(
        normalized,
        actions=actions,
        safety=routing_safety,
        emotional_distress=emotional_distress,
    ):
        direct_scope_reply = _out_of_domain_reply()
    elif tool_plan.get("intent") == "portal_help":
        direct_portal_reply = portal_help_reply()

    fallback_response = (
        direct_safety_reply
        or direct_scope_reply
        or direct_portal_reply
        or _build_response(normalized, actions, urgent_flags, patient_context)
    )
    routing_intent = (
        "safety_boundary"
        if direct_safety_reply
        else "scope_boundary"
        if direct_scope_reply
        else "portal_help"
        if direct_portal_reply
        else tool_plan["intent"]
        if tool_plan.get("intent") in ALLOWED_SUPPORT_INTENTS
        else route_intent(normalized, actions=actions, safety=routing_safety)
    )
    from backend.services.live_agentic_shadow import build_live_agentic_shadow

    agentic_shadow = build_live_agentic_shadow(
        normalized,
        patient_context=state_snapshot(patient_id),
        live_intent=routing_intent,
        live_safety=routing_safety,
        live_tools=list(tool_plan.get("selected_tools") or []),
    )
    contextual_record_route = should_bypass_rag_for_patient_context(routing_intent, normalized)
    if (
        _should_use_llm_direct_reply(routing_intent, routing_safety, actions, urgent_flags)
        and not has_tool_action(actions)
        and not contextual_record_route
    ):
        fallback_response = generate_llm_response_fn(
            normalized,
            actions,
            urgent_flags,
            patient_context,
            fallback_response,
        )
    fallback_response = apply_emotional_distress_mode(fallback_response, emotional_distress)
    if (
        direct_safety_reply
        or direct_scope_reply
        or direct_portal_reply
        or contextual_record_route
        or should_bypass_rag_for_tool_actions(actions, routing_intent)
    ):
        direct_reason = (
            "deterministic safety clarification; no RAG generation"
            if direct_safety_reply
            else "out-of-domain request; no LLM or RAG generation"
            if direct_scope_reply
            else "deterministic portal navigation help; no external-evidence RAG generation"
            if direct_portal_reply
            else "patient-scoped record comparison; no external-evidence RAG generation"
            if contextual_record_route
            else "deterministic tool confirmation; no RAG generation"
        )
        agent_result = {
            "reply": fallback_response,
            "intent": routing_intent,
            "safety": routing_safety,
            "citations": [],
            "cache": {"status": "bypassed_for_deterministic_tool_action"},
            "validation": {"status": "not_needed_for_tool_confirmation"},
            "guardrails": {
                "input_passed": routing_safety.get("level") != "blocked",
                "output_passed": True,
                "reason": direct_reason,
            },
            "rag_evaluation": None,
            "pipeline_trace": {
                "steps": [
                    "safety_gate",
                    "intent_routing",
                    (
                        "deterministic_safety_reply"
                        if direct_safety_reply
                        else "scope_boundary_reply"
                        if direct_scope_reply
                        else "portal_help_reply"
                        if direct_portal_reply
                        else "patient_record_context"
                        if contextual_record_route
                        else "deterministic_tool_action"
                    ),
                    (
                        "safety_clarification"
                        if direct_safety_reply
                        else "scope_redirect"
                        if direct_scope_reply
                        else "portal_navigation"
                        if direct_portal_reply
                        else "record_change_explanation"
                        if contextual_record_route
                        else "confirmation_reply"
                    ),
                ],
                "terminal_step": "direct_support",
                "safety_level": routing_safety.get("level"),
                "intent": routing_intent,
                "subquery_count": 0,
                "retrieved_count": 0,
                "reranked_count": 0,
                "compressed_count": 0,
            },
        }
    else:
        agent_result = run_patient_agent_pipeline_fn(
            db=db,
            patient_id=patient_id,
            query=normalized,
            patient_context=patient_context,
            fallback_response=fallback_response,
            actions=actions,
            urgent_flags=urgent_flags,
            preselected_intent=routing_intent,
            compound_intent=compound_intent,
            precomputed_safety=routing_safety,
        )
    agent_result["emotional_distress"] = (
        emotional_distress.to_dict() if emotional_distress is not None else None
    )
    agent_result["reply"] = apply_emotional_distress_mode(agent_result["reply"], emotional_distress)
    response = _enforce_record_provenance(agent_result["reply"], actions)
    response = _ensure_complete_response(response, fallback_response)
    response = _ensure_complete_safety_reply(response, routing_safety)
    response = append_alert_notice(response, actions)
    agent_result["reply"] = response
    agent_result = authorize_final_support_response(
        agent_result,
        query=normalized,
        routing_safety=routing_safety,
        deterministic_tool_confirmation=has_tool_action(actions),
    )
    return agent_result, agent_result["reply"], high_risk_alert, agentic_shadow
