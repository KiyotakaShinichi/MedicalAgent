"""Typed execution policy for bounded patient-support agent plans.

The policy is deliberately narrower than the planner. It does not decide what
medical content is correct; it verifies that a proposed plan stays inside the
engineering action boundary before any tool is simulated or executed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from backend.services.bounded_agentic_workflow import (
    FORBIDDEN_TOOLS,
    READ_TOOLS,
    WRITE_TOOLS,
)


class AgentState(StrEnum):
    INPUT_GUARD = "input_guard"
    PLAN = "plan"
    EVIDENCE = "evidence"
    CONFIRM = "confirm"
    EXECUTE = "execute"
    VERIFY = "verify"
    COMPLETE = "complete"
    NO_VALID_ACTION = "no_valid_action"


TRUSTED_MEMORY_PROVENANCE = {
    "current_turn",
    "patient_record",
    "confirmed_write_state",
    "system_policy",
}
CONTROL_TOOLS = {
    "confirm_before_save",
}
KNOWN_TOOLS = READ_TOOLS | WRITE_TOOLS | CONTROL_TOOLS


@dataclass(frozen=True)
class AgentBudget:
    max_tool_calls: int = 6
    max_workflow_steps: int = 10
    max_write_tools: int = 1


def enforce_agent_execution_policy(
    plan: dict[str, Any],
    *,
    confirmed_by_user: bool,
    memory_entries: list[dict[str, Any]] | None = None,
    budget: AgentBudget = AgentBudget(),
) -> dict[str, Any]:
    """Return the effective, fail-closed execution envelope for a plan."""

    requested_tools = _dedupe(str(tool) for tool in (plan.get("allowed_tools") or []))
    requested_steps = list(plan.get("workflow_steps") or [])
    violations: list[str] = []
    unknown_tools = sorted(set(requested_tools) - KNOWN_TOOLS)
    forbidden_requested = sorted(set(requested_tools) & FORBIDDEN_TOOLS)
    write_tools = [tool for tool in requested_tools if tool in WRITE_TOOLS]

    if unknown_tools:
        violations.append("unknown_tool_requested")
    if forbidden_requested:
        violations.append("forbidden_medical_authority_tool_requested")
    if len(requested_tools) > budget.max_tool_calls:
        violations.append("tool_call_budget_exceeded")
    if len(requested_steps) > budget.max_workflow_steps:
        violations.append("workflow_step_budget_exceeded")
    if len(write_tools) > budget.max_write_tools:
        violations.append("write_tool_budget_exceeded")
    if write_tools and not bool(plan.get("requires_confirmation_before_write")):
        violations.append("write_without_confirmation_contract")

    memory_audit = _audit_memory(memory_entries or [])
    if memory_audit["untrusted_authority_attempt_count"]:
        violations.append("untrusted_memory_authority_attempt")

    state_path = [AgentState.INPUT_GUARD, AgentState.PLAN]
    if "retrieve_sources" in requested_tools:
        state_path.append(AgentState.EVIDENCE)
    if write_tools and not confirmed_by_user:
        state_path.append(AgentState.CONFIRM)

    effective_tools: list[str] = []
    terminal_state = AgentState.COMPLETE
    decision = "allow"
    if violations:
        decision = "block"
        terminal_state = AgentState.NO_VALID_ACTION
        state_path.append(terminal_state)
    else:
        state_path.append(AgentState.EXECUTE)
        for tool in requested_tools:
            if tool in WRITE_TOOLS and not confirmed_by_user:
                continue
            effective_tools.append(tool)
        state_path.extend([AgentState.VERIFY, AgentState.COMPLETE])

    return {
        "schema_version": "agent_execution_policy_v1",
        "decision": decision,
        "terminal_state": terminal_state.value,
        "state_path": [state.value for state in state_path],
        "requested_tools": requested_tools,
        "effective_tools": effective_tools,
        "write_tools_requested": write_tools,
        "confirmed_by_user": bool(confirmed_by_user),
        "violations": violations,
        "budget": {
            "max_tool_calls": budget.max_tool_calls,
            "max_workflow_steps": budget.max_workflow_steps,
            "max_write_tools": budget.max_write_tools,
            "requested_tool_count": len(requested_tools),
            "requested_step_count": len(requested_steps),
            "requested_write_tool_count": len(write_tools),
        },
        "memory_audit": memory_audit,
        "clinical_authority_allowed": False,
        "clinical_validation": False,
        "claim_boundary": (
            "This execution policy constrains software actions only. It does not "
            "validate medical correctness and cannot authorize diagnosis, treatment, "
            "dosage, prognosis, genetic-risk, or tumor-marker conclusions."
        ),
    }


def _audit_memory(entries: list[dict[str, Any]]) -> dict[str, Any]:
    untrusted = 0
    authority_attempts = 0
    for entry in entries:
        provenance = str(entry.get("provenance") or "untrusted")
        trusted = provenance in TRUSTED_MEMORY_PROVENANCE and bool(
            entry.get("trusted", provenance in TRUSTED_MEMORY_PROVENANCE)
        )
        if not trusted:
            untrusted += 1
            requested_action = str(entry.get("requested_action") or "")
            if requested_action in WRITE_TOOLS or requested_action in FORBIDDEN_TOOLS:
                authority_attempts += 1
    return {
        "entry_count": len(entries),
        "untrusted_entry_count": untrusted,
        "untrusted_authority_attempt_count": authority_attempts,
        "trusted_provenance_values": sorted(TRUSTED_MEMORY_PROVENANCE),
        "memory_is_non_authoritative_by_default": True,
    }


def _dedupe(values: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


__all__ = [
    "AgentBudget",
    "AgentState",
    "enforce_agent_execution_policy",
]
