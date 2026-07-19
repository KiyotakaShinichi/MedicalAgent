"""Side-effect-free planner diagnostics for live patient-support turns.

The bounded orchestrator remains an evaluation scaffold. This adapter lets the
live pipeline record its planner/verifier opinion without changing routing,
executing tools, writing patient records, or replacing the patient-facing
answer. Only concise decision metadata is returned; user text and hidden
reasoning are intentionally excluded.
"""

from __future__ import annotations

import os
from typing import Any

from backend.services.agentic_turn_orchestrator import run_agentic_turn


SHADOW_SCHEMA_VERSION = "live_agentic_shadow_v1_2026_07"


def shadow_enabled() -> bool:
    value = os.getenv("NLCARE_AGENTIC_SHADOW_ENABLED", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def build_live_agentic_shadow(
    message: str,
    *,
    patient_context: dict[str, Any] | None,
    live_intent: str,
    live_safety: dict[str, Any] | None,
    live_tools: list[str] | None,
) -> dict[str, Any]:
    if not shadow_enabled():
        return {
            "schema_version": SHADOW_SCHEMA_VERSION,
            "status": "disabled",
            "shadow_only": True,
            "changed_live_behavior": False,
        }

    try:
        result = run_agentic_turn(
            message,
            patient_context=dict(patient_context or {}),
            confirmed_by_user=False,
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics must never break chat
        return {
            "schema_version": SHADOW_SCHEMA_VERSION,
            "status": "diagnostic_error",
            "shadow_only": True,
            "changed_live_behavior": False,
            "error_type": type(exc).__name__,
        }

    plan = result.get("plan") or {}
    execution = result.get("execution") or {}
    verifier = result.get("verifier") or {}
    shadow_route = str(plan.get("route") or "unknown")
    planned_tool = (result.get("trace_diagnostics") or {}).get("tool_required")
    normalized_live_tools = sorted(
        tool for tool in (live_tools or []) if tool and tool != "none"
    )
    shadow_family = _shadow_route_family(shadow_route)
    live_family = _live_route_family(live_intent, live_safety or {}, normalized_live_tools)

    return {
        "schema_version": SHADOW_SCHEMA_VERSION,
        "status": "observed",
        "shadow_only": True,
        "changed_live_behavior": False,
        "planner_route": shadow_route,
        "planner_route_family": shadow_family,
        "live_intent": live_intent,
        "live_route_family": live_family,
        "route_family_aligned": shadow_family == live_family,
        "planner_primary_tool": planned_tool,
        "live_selected_tools": normalized_live_tools,
        "tool_alignment": _tool_alignment(planned_tool, normalized_live_tools),
        "confirmation_required": bool(plan.get("requires_confirmation_before_write")),
        "simulated_records_written": list(execution.get("records_written") or []),
        "verifier_passed": bool(verifier.get("passed")),
        "review_required": shadow_family != live_family or not bool(verifier.get("passed")),
        "claim_boundary": (
            "Planner comparison metadata only. It does not change the live reply, "
            "execute a tool, validate clinical safety, or establish clinical readiness."
        ),
    }


def _shadow_route_family(route: str) -> str:
    if route.startswith("record_") or route.startswith("request_"):
        return "data_entry"
    if route in {"source_backed_education"}:
        return "education"
    if route in {"conversation", "clinician_summary"}:
        return "conversation"
    if route in {"scope_boundary", "security_refusal"}:
        return "scope_or_security_boundary"
    if "refusal" in route or route in {"urgent_clinician_review", "crisis_support"}:
        return "medical_safety_boundary"
    return "other"


def _live_route_family(intent: str, safety: dict[str, Any], tools: list[str]) -> str:
    if tools or intent == "data_entry_confirmation":
        return "data_entry"
    if intent in {"education", "patient_timeline_monitoring"}:
        return "education"
    if intent == "scope_boundary":
        return "scope_or_security_boundary"
    if intent in {"safety_boundary", "treatment_decision_boundary"}:
        return "medical_safety_boundary"
    if str(safety.get("level") or "").lower() in {"blocked", "high_risk", "urgent"}:
        return "medical_safety_boundary"
    if intent in {"conversation", "portal_help", "patient_memory", "emotional_support", "general_support"}:
        return "conversation"
    return "other"


def _tool_alignment(planned_tool: Any, live_tools: list[str]) -> str:
    if not planned_tool and not live_tools:
        return "both_no_tool"
    if planned_tool in live_tools:
        return "aligned"
    if planned_tool and live_tools:
        return "different_tool"
    return "one_sided_tool"


__all__ = ["SHADOW_SCHEMA_VERSION", "build_live_agentic_shadow", "shadow_enabled"]
