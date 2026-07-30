"""Planner-executor-verifier scaffold for bounded agentic turns.

This module simulates the control contract that a future live agent can use:
plan first, execute only allowed workflow tools, verify, then package a final
answer.  The executor is intentionally lightweight and side-effect free unless
the caller explicitly marks a turn as confirmed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from backend.services.agent_execution_policy import (
    build_confirmation_contract,
    enforce_agent_execution_policy,
)
from backend.services.agent_verifier import verify_agent_turn
from backend.services.agent_text_normalization import normalize_agent_text
from backend.services.bounded_agentic_workflow import WRITE_TOOLS, plan_patient_agent_workflow


AGENTIC_ORCHESTRATOR_VERSION = "agentic_turn_orchestrator_v1_2026_05"


def run_agentic_turn(
    message: str,
    *,
    patient_context: dict[str, Any] | None = None,
    confirmed_by_user: bool = False,
    patient_scope_id: str = "synthetic_demo_patient",
    require_bound_confirmation: bool = False,
) -> dict[str, Any]:
    """Run a side-effect-free bounded agentic turn."""

    context = dict(patient_context or {})
    plan = plan_patient_agent_workflow(message, patient_context=context)
    pending_confirmation = dict(context.get("pending_confirmation") or {})
    action_payload = (
        dict(pending_confirmation.get("action_payload") or {})
        if confirmed_by_user and pending_confirmation
        else _confirmation_action_payload(message, plan, context)
    )
    execution_policy = enforce_agent_execution_policy(
        plan,
        confirmed_by_user=confirmed_by_user,
        memory_entries=list(context.get("memory_entries") or []),
        patient_scope_id=patient_scope_id,
        action_payload=action_payload,
        confirmation_contract=pending_confirmation.get("contract"),
        consumed_confirmation_ids=set(context.get("consumed_confirmation_ids") or []),
        require_bound_confirmation=require_bound_confirmation,
    )
    execution = _simulate_execution(
        plan,
        confirmed_by_user=confirmed_by_user,
        execution_policy=execution_policy,
    )
    final_response = _package_final_response(plan, execution)
    verifier = verify_agent_turn(plan=plan, execution=execution, final_response=final_response)
    state_update = _state_update_from_turn(
        message,
        plan,
        execution,
        context,
        patient_scope_id=patient_scope_id,
        action_payload=action_payload,
    )
    return {
        "schema_version": AGENTIC_ORCHESTRATOR_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "message": message,
        "plan": plan,
        "execution_policy": execution_policy,
        "execution": execution,
        "final_response": final_response,
        "verifier": verifier,
        "state_update": state_update,
        "trace_diagnostics": {
            "route": plan.get("route"),
            "evidence_needed": "retrieve_sources" in plan.get("allowed_tools", []),
            "tool_required": _primary_tool(plan),
            "confirmation_required": bool(plan.get("requires_confirmation_before_write")),
            "blocked_authority": plan.get("prohibited_medical_authority", []),
            "execution_policy_decision": execution_policy["decision"],
            "execution_terminal_state": execution_policy["terminal_state"],
            "confirmation_binding_required": bool(require_bound_confirmation),
            "confirmation_binding_valid": execution_policy["confirmation_validation"]["valid"],
            "final_verifier_passed": verifier["passed"],
        },
    }


def run_agentic_conversation(turns: list[dict[str, Any]]) -> dict[str, Any]:
    """Run a stateful multi-turn conversation through the scaffold."""

    state: dict[str, Any] = {}
    results: list[dict[str, Any]] = []
    for turn in turns:
        result = run_agentic_turn(
            turn["message"],
            patient_context=state,
            confirmed_by_user=bool(turn.get("confirmed_by_user", False)),
            patient_scope_id=str(turn.get("patient_scope_id") or "synthetic_demo_patient"),
            require_bound_confirmation=True,
        )
        state.update(result.get("state_update") or {})
        results.append(result)
    return {
        "schema_version": "agentic_conversation_trace_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "turn_count": len(turns),
        "turns": results,
        "final_state": state,
    }


def _simulate_execution(
    plan: dict[str, Any],
    *,
    confirmed_by_user: bool,
    execution_policy: dict[str, Any],
) -> dict[str, Any]:
    allowed_tools = list(execution_policy.get("effective_tools") or [])
    executed_tools: list[str] = []
    confirmation_prompted = bool(plan.get("requires_confirmation_before_write"))
    for tool in allowed_tools:
        if tool in WRITE_TOOLS and not confirmed_by_user:
            continue
        if tool == "confirm_before_save" and confirmed_by_user:
            executed_tools.append(tool)
            continue
        executed_tools.append(tool)
    records_written = sorted(tool for tool in executed_tools if tool in WRITE_TOOLS)
    return {
        "executed_tools": executed_tools,
        "confirmed_by_user": bool(confirmed_by_user),
        "confirmation_prompted": confirmation_prompted,
        "records_written": records_written,
        "side_effect_mode": "simulated_confirmed_write" if records_written else "no_write",
        "policy_decision": execution_policy.get("decision"),
        "policy_violations": list(execution_policy.get("violations") or []),
    }


def _package_final_response(plan: dict[str, Any], execution: dict[str, Any]) -> dict[str, Any]:
    route = str(plan.get("route") or "")
    if "refusal" in route or route in {"security_refusal", "crisis_support", "urgent_clinician_review"}:
        return {
            "response_type": "safe_refusal",
            "citation_status": "not_applicable_safe_refusal",
            "citations": [],
            "blocked_claim_categories": [],
            "message_intent": "boundary_or_escalation",
        }
    if route == "source_backed_education":
        return {
            "response_type": "source_backed_education",
            "citation_status": "not_applicable_safe_stub",
            "citations": ["source_stub"],
            "blocked_claim_categories": [],
            "message_intent": "education",
        }
    if route.startswith("record_"):
        return {
            "response_type": "confirmation_request" if not execution.get("records_written") else "saved_record_ack",
            "citation_status": "not_applicable_safe_stub",
            "citations": [],
            "blocked_claim_categories": [],
            "message_intent": "record_organization",
        }
    if route.startswith("request_"):
        return {
            "response_type": "missing_details_request",
            "citation_status": "not_applicable_safe_stub",
            "citations": [],
            "blocked_claim_categories": [],
            "message_intent": "missing_data_explanation",
        }
    if route == "clinician_summary":
        return {
            "response_type": "clinician_review_summary",
            "citation_status": "not_applicable_safe_stub",
            "citations": [],
            "blocked_claim_categories": [],
            "message_intent": "record_organization",
        }
    return {
        "response_type": "conversation",
        "citation_status": "not_applicable_safe_stub",
        "citations": [],
        "blocked_claim_categories": [],
        "message_intent": "conversation",
    }


def _state_update_from_turn(
    message: str,
    plan: dict[str, Any],
    execution: dict[str, Any],
    state: dict[str, Any],
    *,
    patient_scope_id: str,
    action_payload: dict[str, Any],
) -> dict[str, Any]:
    route = str(plan.get("route") or "")
    update: dict[str, Any] = {}
    boundary_routes = {
        "security_refusal",
        "medical_boundary_refusal",
        "diagnosis_boundary_refusal",
        "treatment_boundary_refusal",
        "prognosis_boundary_refusal",
        "genetics_boundary_refusal",
        "tumor_marker_boundary_refusal",
        "urgent_clinician_review",
        "crisis_support",
    }
    reused_boundary = bool((plan.get("trace") or {}).get("boundary_context_reused"))
    if route in boundary_routes and not reused_boundary:
        update["active_safety_boundary"] = {
            "route": route,
            "review_route": plan.get("review_route"),
            "turns_remaining": 3,
            "created_by": "agentic_turn_orchestrator",
        }
    elif reused_boundary:
        active = dict(state.get("active_safety_boundary") or {})
        remaining = max(0, int(active.get("turns_remaining") or 1) - 1)
        update["active_safety_boundary"] = {**active, "turns_remaining": remaining} if remaining else None
    elif state.get("active_safety_boundary") and route != "conversation":
        update["active_safety_boundary"] = None
    if route == "request_symptom_details":
        symptom = _extract_symptom_name(message)
        if symptom:
            update["pending_symptom"] = symptom
    if route.startswith("record_") and not execution.get("records_written") and execution.get("confirmation_prompted"):
        contract = build_confirmation_contract(
            plan,
            patient_scope_id=patient_scope_id,
            action_payload=action_payload,
        )
        update["pending_confirmation"] = {
            "tool": _primary_tool(plan),
            "route": route,
            "created_by": "agentic_turn_orchestrator",
            "patient_scope_id": patient_scope_id,
            "action_payload": action_payload,
            "contract": contract,
        }
    if route.startswith("record_") and execution.get("records_written"):
        consumed = list(state.get("consumed_confirmation_ids") or [])
        prior_contract = dict((state.get("pending_confirmation") or {}).get("contract") or {})
        if prior_contract.get("confirmation_id"):
            consumed.append(str(prior_contract["confirmation_id"]))
        update["consumed_confirmation_ids"] = sorted(set(consumed))
        update["pending_confirmation"] = None
    if route == "record_symptom" and execution.get("records_written"):
        update["pending_symptom"] = None
    return update


def _confirmation_action_payload(
    message: str,
    plan: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "route": str(plan.get("route") or ""),
        "tool": _primary_tool(plan),
        "normalized_request": normalize_agent_text(message),
        "pending_symptom": state.get("pending_symptom"),
    }


def _primary_tool(plan: dict[str, Any]) -> str | None:
    for tool in plan.get("allowed_tools") or []:
        if tool not in {"classify_intent", "detect_safety_boundary", "confirm_before_save"}:
            return tool
    return None


def _extract_symptom_name(message: str) -> str | None:
    lowered = normalize_agent_text(message)
    for symptom in ["nausea", "nauseous", "fatigue", "pain", "fever", "mouth sores", "neuropathy", "vomiting", "bleeding"]:
        if symptom in lowered:
            return "nausea" if symptom == "nauseous" else symptom
    if "tired" in lowered or "weak" in lowered:
        return "fatigue"
    return None


__all__ = ["AGENTIC_ORCHESTRATOR_VERSION", "run_agentic_conversation", "run_agentic_turn"]
