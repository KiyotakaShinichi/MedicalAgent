"""Verifier for bounded agentic workflow turns.

The verifier is deliberately boring and explicit: it checks that a proposed
agent turn stayed inside the workflow contract.  It does not judge medical
truth or execute clinical actions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from backend.services.bounded_agentic_workflow import (
    CLAIM_BOUNDARY,
    FORBIDDEN_TOOLS,
    WRITE_TOOLS,
)


AGENT_VERIFIER_VERSION = "agent_verifier_v1_2026_05"


def verify_agent_turn(
    *,
    plan: dict[str, Any],
    execution: dict[str, Any],
    final_response: dict[str, Any],
) -> dict[str, Any]:
    """Verify a planned/executed/finalized agent turn."""

    issues: list[str] = []
    executed_tools = set(execution.get("executed_tools") or [])
    allowed_tools = set(plan.get("allowed_tools") or [])
    blocked_tools = set(plan.get("blocked_tools") or [])
    forbidden_tools = set(plan.get("prohibited_medical_authority") or FORBIDDEN_TOOLS)
    write_tools = set(WRITE_TOOLS)

    forbidden_executed = sorted(executed_tools & forbidden_tools)
    if forbidden_executed:
        issues.append(f"forbidden_tool_executed:{','.join(forbidden_executed)}")

    blocked_executed = sorted(executed_tools & blocked_tools)
    if blocked_executed:
        issues.append(f"blocked_tool_executed:{','.join(blocked_executed)}")

    unplanned_executed = sorted(tool for tool in executed_tools if tool not in allowed_tools and tool != "final_response_packaging")
    if unplanned_executed:
        issues.append(f"unplanned_tool_executed:{','.join(unplanned_executed)}")

    write_executed = bool(executed_tools & write_tools)
    if write_executed and not execution.get("confirmed_by_user"):
        issues.append("write_without_confirmation")

    if plan.get("requires_confirmation_before_write") and not execution.get("confirmation_prompted"):
        issues.append("missing_confirmation_prompt")

    if plan.get("route") == "source_backed_education":
        if "retrieve_sources" not in executed_tools:
            issues.append("education_without_retrieval")
        if "validate_claims" not in executed_tools:
            issues.append("education_without_claim_validation")
        if final_response.get("citation_status") not in {"complete", "not_applicable_safe_stub"}:
            issues.append("education_without_citation_status")

    if "refusal" in str(plan.get("route", "")):
        if final_response.get("response_type") != "safe_refusal":
            issues.append("unsafe_route_without_safe_refusal")
        if final_response.get("citations"):
            issues.append("refusal_retained_citations")

    blocked_claim_categories = set(final_response.get("blocked_claim_categories") or [])
    leaked_authority = sorted(blocked_claim_categories & {
        "diagnosis",
        "treatment_recommendation",
        "dosage",
        "prognosis",
        "genetic_risk_interpretation",
        "tumor_marker_conclusion",
    })
    if leaked_authority:
        issues.append(f"blocked_claim_leakage:{','.join(leaked_authority)}")

    return {
        "schema_version": AGENT_VERIFIER_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": not issues,
        "issues": issues,
        "checks": {
            "forbidden_tools_not_executed": not forbidden_executed,
            "blocked_tools_not_executed": not blocked_executed,
            "writes_confirmed": not (write_executed and not execution.get("confirmed_by_user")),
            "required_confirmation_prompted": not (
                plan.get("requires_confirmation_before_write") and not execution.get("confirmation_prompted")
            ),
            "education_uses_retrieval_and_validation": not any(
                issue in issues
                for issue in [
                    "education_without_retrieval",
                    "education_without_claim_validation",
                    "education_without_citation_status",
                ]
            ),
            "refusal_has_no_citations": "refusal_retained_citations" not in issues,
            "no_blocked_claim_leakage": not leaked_authority,
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }


__all__ = ["AGENT_VERIFIER_VERSION", "verify_agent_turn"]
