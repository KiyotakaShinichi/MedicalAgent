"""Compatibility facade and transaction coordinator for patient support chat."""

import json
import re

from backend.models import ChatMessage
from backend.processing.radiology_analysis import detect_possible_metastatic_indicators
from backend.services.agent_input_gate import input_guardrail_check
from backend.services.agent_output_gate import output_guardrail_check
from backend.services.agent_post_gen import apply_post_gen_validator
from backend.services.agent_rag import run_patient_agent_pipeline, route_intent, safety_scope_check
from backend.services.app_logging import log_app_event
from backend.services.confirmed_record_write import queue_record_write, resolve_pending_record_write
from backend.services.conversation_state import (
    clear_pending_action,
    get_pending_action,
    remember_turn,
    set_pending_action,
    state_snapshot,
)
from backend.services.input_validation import (
    validate_cbc_values,
    validate_imaging_report_payload,
    validate_symptom_payload,
)
from backend.services.local_llm import select_support_tools_with_local_llm
from backend.services.rag_evidence_envelope import (
    build_fail_closed_error_result,
    enforce_evidence_release,
    enforce_transport_release,
    parse_evidence_envelope,
)
from backend.services.support_chat.authorization import (
    authorize_final_support_response as _authorize_final_support_response,
)
from backend.services.support_chat.conversation_context import (
    extract_candidate_inputs as _extract_candidate_inputs,
    latest_pending_symptom as _latest_pending_symptom,
    resume_pending_symptom_if_possible as _resume_pending_symptom_if_possible,
)
from backend.services.support_chat.persistence import persist_support_turn
from backend.services.support_chat.record_actions import apply_record_actions
from backend.services.support_chat.response_helpers import (
    append_alert_notice as _append_alert_notice,
    apply_emotional_distress_mode as _apply_emotional_distress_mode,
    compound_intent_payload as _compound_intent_payload,
    has_tool_action as _has_tool_action,
    portal_help_reply as _portal_help_reply,
    should_bypass_rag_for_patient_context as _should_bypass_rag_for_patient_context,
    should_bypass_rag_for_tool_actions as _should_bypass_rag_for_tool_actions,
    tool_request_followup_message as _tool_request_followup_message,
)
from backend.services.support_chat.response_pipeline import build_support_response
from backend.services.support_chat.tool_contracts import (
    dedupe_tools as _dedupe_tools,
    has_explicit_record_command as _has_explicit_record_command,
    is_safety_limited_turn as _is_safety_limited_turn,
    normalize_selected_tools as _normalize_selected_tools,
    reconcile_selected_tools as _reconcile_selected_tools,
    rough_chat_intent as _rough_chat_intent,
)
from backend.services.support_chat.tool_planning import (
    deterministic_tool_plan as _deterministic_tool_plan,
    select_tool_plan as _select_tool_plan,
)
from backend.services.support_chat_context import _recent_patient_context
from backend.services.support_chat_extraction import (
    _clinical_lab_alerts,
    _extract_complete_labs,
    _extract_date,
    _extract_imaging_report,
    _extract_medication,
    _extract_severity,
    _extract_symptom,
    _is_general_lab_question,
    _is_short_record_hint,
    _looks_like_partial_imaging,
    _looks_like_partial_labs,
    _should_request_symptom_details,
    _should_save_symptom,
)
from backend.services.support_chat_policy import ALLOWED_SUPPORT_INTENTS, ALLOWED_SUPPORT_TOOLS
from backend.services.support_chat_response import (
    _build_response,
    _generate_llm_response,
    _is_conversational_prompt,
    _is_small_talk,
    _should_use_llm_direct_reply,
)
from backend.services.support_chat_safety import (
    _detect_urgent_flags,
    _enforce_record_provenance,
    _ensure_complete_response,
    _ensure_complete_safety_reply,
    _immediate_danger_reply,
    _is_immediate_danger_statement,
    _is_out_of_domain_request,
    _looks_truncated_reply,
    _out_of_domain_reply,
    _prefer_deterministic_reply,
    _resolve_safety_location_followup,
    _safety_location_followup_reply,
)


__all__ = [
    "ALLOWED_SUPPORT_INTENTS",
    "ALLOWED_SUPPORT_TOOLS",
    "ChatMessage",
    "apply_post_gen_validator",
    "build_fail_closed_error_result",
    "clear_pending_action",
    "detect_possible_metastatic_indicators",
    "enforce_evidence_release",
    "enforce_transport_release",
    "get_pending_action",
    "json",
    "log_app_event",
    "output_guardrail_check",
    "parse_evidence_envelope",
    "queue_record_write",
    "re",
    "route_intent",
    "select_support_tools_with_local_llm",
    "set_pending_action",
    "validate_cbc_values",
    "validate_imaging_report_payload",
    "validate_symptom_payload",
    "_append_alert_notice",
    "_apply_emotional_distress_mode",
    "_authorize_final_support_response",
    "_build_response",
    "_clinical_lab_alerts",
    "_compound_intent_payload",
    "_dedupe_tools",
    "_detect_urgent_flags",
    "_deterministic_tool_plan",
    "_enforce_record_provenance",
    "_ensure_complete_response",
    "_ensure_complete_safety_reply",
    "_extract_candidate_inputs",
    "_extract_complete_labs",
    "_extract_date",
    "_extract_imaging_report",
    "_extract_medication",
    "_extract_severity",
    "_extract_symptom",
    "_generate_llm_response",
    "_has_explicit_record_command",
    "_has_tool_action",
    "_immediate_danger_reply",
    "_is_conversational_prompt",
    "_is_general_lab_question",
    "_is_immediate_danger_statement",
    "_is_out_of_domain_request",
    "_is_safety_limited_turn",
    "_is_short_record_hint",
    "_is_small_talk",
    "_latest_pending_symptom",
    "_looks_like_partial_imaging",
    "_looks_like_partial_labs",
    "_looks_truncated_reply",
    "_normalize_selected_tools",
    "_out_of_domain_reply",
    "_portal_help_reply",
    "_prefer_deterministic_reply",
    "_recent_patient_context",
    "_reconcile_selected_tools",
    "_resolve_safety_location_followup",
    "_resume_pending_symptom_if_possible",
    "_rough_chat_intent",
    "_safety_location_followup_reply",
    "_select_tool_plan",
    "_should_bypass_rag_for_patient_context",
    "_should_bypass_rag_for_tool_actions",
    "_should_request_symptom_details",
    "_should_save_symptom",
    "_should_use_llm_direct_reply",
    "_tool_request_followup_message",
    "handle_patient_chat",
]


def handle_patient_chat(db, patient_id, message):
    normalized = message.strip()
    if not normalized:
        raise ValueError("Message cannot be empty")

    prior_state = state_snapshot(patient_id)
    prior_user_messages = [
        str(row.get("message") or "")
        for row in prior_state.get("recent_messages") or []
        if row.get("role") == "user"
    ][-3:]
    safety_followup = _resolve_safety_location_followup(db, patient_id, normalized)
    immediate_danger = _is_immediate_danger_statement(normalized)
    remember_turn(patient_id, "user", normalized)

    urgent_flags = _detect_urgent_flags(normalized)
    if immediate_danger:
        urgent_flags.append("immediate_danger_statement")
    if safety_followup:
        urgent_flags.append("safety_location_followup")
    urgent_flags = sorted(set(urgent_flags))
    routing_safety = safety_scope_check(
        normalized,
        urgent_flags,
        previous_user_messages=prior_user_messages,
    )
    input_guardrail_preview = input_guardrail_check(normalized, routing_safety)
    terminal_input_block = input_guardrail_preview.get("status") == "failed"
    try:
        from backend.services.emotional_distress_detection import detect_emotional_distress

        emotional_distress = detect_emotional_distress(normalized, safety=routing_safety)
        if emotional_distress.response_mode == "crisis_support":
            urgent_flags.append("emotional_crisis")
            routing_safety = safety_scope_check(
                normalized,
                urgent_flags,
                previous_user_messages=prior_user_messages,
            )
            emotional_distress = detect_emotional_distress(normalized, safety=routing_safety)
    except Exception:  # noqa: BLE001 - affective detection must never block chat
        emotional_distress = None

    compound_intent = None
    llm_compound_verdict: dict | None = None
    try:
        if terminal_input_block:
            raise RuntimeError("terminal input safety block")
        from backend.services.compound_intent_router import detect_compound_intents_with_llm

        compound_intent, llm_compound_verdict = detect_compound_intents_with_llm(message)
    except Exception:  # noqa: BLE001 - never break chat on compound routing
        try:
            if terminal_input_block:
                raise RuntimeError("terminal input safety block")
            from backend.services.compound_intent_router import detect_compound_intents

            compound_intent = detect_compound_intents(message)
        except Exception:  # noqa: BLE001
            compound_intent = None

    extracted = _extract_candidate_inputs(normalized)
    resumed_symptom = _resume_pending_symptom_if_possible(db, patient_id, normalized, extracted)
    if resumed_symptom:
        extracted["symptom"] = resumed_symptom
    deterministic_plan = _deterministic_tool_plan(normalized, extracted, routing_safety)
    tool_plan = _select_tool_plan(normalized, extracted, deterministic_plan, routing_safety)

    user_record = ChatMessage(
        patient_id=patient_id,
        role="user",
        message=normalized,
        intent="patient_support",
    )
    db.add(user_record)
    db.flush()

    confirmation_actions = (
        None if terminal_input_block else resolve_pending_record_write(db, patient_id, normalized)
    )
    actions = list(confirmation_actions or [])
    selected_tools = (
        set()
        if terminal_input_block or confirmation_actions is not None
        else set(tool_plan["selected_tools"])
    )
    if terminal_input_block:
        tool_plan = {
            **tool_plan,
            "intent": "safety_boundary",
            "selected_tools": ["none"],
            "forced_tools": [],
            "source": "terminal_input_safety_block",
            "confidence": 1.0,
            "reason": (
                "input safety boundary is terminal; no record action or tool follow-up is allowed"
            ),
        }
    if confirmation_actions is not None:
        tool_plan = {
            **tool_plan,
            "intent": "data_entry_confirmation",
            "selected_tools": ["none"],
            "source": "confirmed_write_state",
            "confidence": 1.0,
            "reason": (
                "resolved an explicit confirm/cancel turn against patient-scoped pending state"
            ),
        }

    actions, urgent_flags, routing_safety = apply_record_actions(
        db=db,
        patient_id=patient_id,
        normalized=normalized,
        extracted=extracted,
        selected_tools=selected_tools,
        actions=actions,
        urgent_flags=urgent_flags,
        terminal_input_block=terminal_input_block,
        tool_plan=tool_plan,
        compound_intent=compound_intent,
        user_record=user_record,
        routing_safety=routing_safety,
        prior_user_messages=prior_user_messages,
    )
    agent_result, response, high_risk_alert, agentic_shadow = build_support_response(
        db=db,
        patient_id=patient_id,
        user_record=user_record,
        normalized=normalized,
        actions=actions,
        urgent_flags=urgent_flags,
        routing_safety=routing_safety,
        tool_plan=tool_plan,
        safety_followup=safety_followup,
        immediate_danger=immediate_danger,
        emotional_distress=emotional_distress,
        compound_intent=compound_intent,
        run_patient_agent_pipeline_fn=run_patient_agent_pipeline,
        generate_llm_response_fn=_generate_llm_response,
    )
    return persist_support_turn(
        db=db,
        patient_id=patient_id,
        response=response,
        actions=actions,
        tool_plan=tool_plan,
        agent_result=agent_result,
        agentic_shadow=agentic_shadow,
        compound_intent=compound_intent,
        llm_compound_verdict=llm_compound_verdict,
        urgent_flags=urgent_flags,
        high_risk_alert=high_risk_alert,
    )
