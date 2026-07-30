"""Typed execution policy for bounded patient-support agent plans.

The policy is deliberately narrower than the planner. It does not decide what
medical content is correct; it verifies that a proposed plan stays inside the
engineering action boundary before any tool is simulated or executed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4

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


def build_confirmation_contract(
    plan: dict[str, Any],
    *,
    patient_scope_id: str,
    action_payload: dict[str, Any],
    now: datetime | None = None,
    ttl_seconds: int = 300,
    confirmation_id: str | None = None,
) -> dict[str, Any]:
    """Bind a pending confirmation to one patient-scoped software action.

    This is an integrity contract stored in trusted application state, not a
    bearer credential or cryptographic proof of user identity.
    """

    issued_at = _as_utc(now or datetime.now(timezone.utc))
    write_tools = sorted(
        str(tool)
        for tool in (plan.get("allowed_tools") or [])
        if str(tool) in WRITE_TOOLS
    )
    bounded_ttl = max(30, min(int(ttl_seconds), 900))
    payload_hash = _stable_hash(action_payload)
    return {
        "schema_version": "agent_confirmation_contract_v1",
        "confirmation_id": confirmation_id or str(uuid4()),
        "patient_scope_id": str(patient_scope_id),
        "route": str(plan.get("route") or ""),
        "write_tools": write_tools,
        "action_payload_hash": payload_hash,
        "issued_at": issued_at.isoformat(),
        "expires_at": (issued_at + timedelta(seconds=bounded_ttl)).isoformat(),
        "revoked": False,
        "consumed": False,
        "stored_in_trusted_application_state": True,
        "cryptographic_identity_proof": False,
    }


def validate_confirmation_contract(
    contract: dict[str, Any] | None,
    *,
    plan: dict[str, Any],
    patient_scope_id: str,
    action_payload: dict[str, Any],
    consumed_confirmation_ids: set[str] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate scope, action binding, freshness, revocation, and replay."""

    issues: list[str] = []
    payload = dict(contract or {})
    confirmation_id = str(payload.get("confirmation_id") or "")
    expected_tools = sorted(
        str(tool)
        for tool in (plan.get("allowed_tools") or [])
        if str(tool) in WRITE_TOOLS
    )
    if payload.get("schema_version") != "agent_confirmation_contract_v1":
        issues.append("missing_or_unknown_confirmation_contract")
    if not confirmation_id:
        issues.append("missing_confirmation_id")
    if str(payload.get("patient_scope_id") or "") != str(patient_scope_id):
        issues.append("confirmation_patient_scope_mismatch")
    if str(payload.get("route") or "") != str(plan.get("route") or ""):
        issues.append("confirmation_route_mismatch")
    if sorted(str(tool) for tool in (payload.get("write_tools") or [])) != expected_tools:
        issues.append("confirmation_tool_mismatch")
    if str(payload.get("action_payload_hash") or "") != _stable_hash(action_payload):
        issues.append("confirmation_payload_mismatch")
    if payload.get("revoked") is True:
        issues.append("confirmation_revoked")
    if payload.get("consumed") is True or confirmation_id in (consumed_confirmation_ids or set()):
        issues.append("confirmation_replayed")

    current = _as_utc(now or datetime.now(timezone.utc))
    try:
        issued_at = _parse_datetime(payload.get("issued_at"))
        expires_at = _parse_datetime(payload.get("expires_at"))
    except (TypeError, ValueError):
        issues.append("confirmation_time_invalid")
    else:
        if current < issued_at - timedelta(seconds=5):
            issues.append("confirmation_not_yet_valid")
        if current > expires_at:
            issues.append("confirmation_expired")

    return {
        "valid": not issues,
        "issues": issues,
        "confirmation_id": confirmation_id or None,
        "patient_scope_id": str(patient_scope_id),
        "expected_write_tools": expected_tools,
        "cryptographic_identity_proof": False,
    }


def enforce_agent_execution_policy(
    plan: dict[str, Any],
    *,
    confirmed_by_user: bool,
    memory_entries: list[dict[str, Any]] | None = None,
    budget: AgentBudget = AgentBudget(),
    patient_scope_id: str | None = None,
    action_payload: dict[str, Any] | None = None,
    confirmation_contract: dict[str, Any] | None = None,
    consumed_confirmation_ids: set[str] | None = None,
    require_bound_confirmation: bool = False,
    now: datetime | None = None,
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

    memory_audit = _audit_memory(
        memory_entries or [],
        patient_scope_id=patient_scope_id,
    )
    if memory_audit["untrusted_authority_attempt_count"]:
        violations.append("untrusted_memory_authority_attempt")
    if memory_audit["cross_patient_entry_count"]:
        violations.append("cross_patient_memory_scope_mismatch")

    confirmation_validation = {
        "valid": not write_tools or not require_bound_confirmation,
        "issues": [],
        "confirmation_id": None,
        "patient_scope_id": patient_scope_id,
        "expected_write_tools": sorted(write_tools),
        "cryptographic_identity_proof": False,
        "strict_binding_required": bool(require_bound_confirmation),
    }
    if write_tools and confirmed_by_user and require_bound_confirmation:
        if not patient_scope_id:
            confirmation_validation["valid"] = False
            confirmation_validation["issues"] = ["missing_patient_scope_id"]
        else:
            confirmation_validation = {
                **validate_confirmation_contract(
                    confirmation_contract,
                    plan=plan,
                    patient_scope_id=patient_scope_id,
                    action_payload=action_payload or {},
                    consumed_confirmation_ids=consumed_confirmation_ids,
                    now=now,
                ),
                "strict_binding_required": True,
            }
        if not confirmation_validation["valid"]:
            violations.append("invalid_or_stale_confirmation_contract")

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
        "confirmation_validation": confirmation_validation,
        "clinical_authority_allowed": False,
        "clinical_validation": False,
        "claim_boundary": (
            "This execution policy constrains software actions only. It does not "
            "validate medical correctness and cannot authorize diagnosis, treatment, "
            "dosage, prognosis, genetic-risk, or tumor-marker conclusions."
        ),
    }


def _audit_memory(
    entries: list[dict[str, Any]],
    *,
    patient_scope_id: str | None = None,
) -> dict[str, Any]:
    untrusted = 0
    authority_attempts = 0
    cross_patient = 0
    for entry in entries:
        provenance = str(entry.get("provenance") or "untrusted")
        trusted = provenance in TRUSTED_MEMORY_PROVENANCE and bool(
            entry.get("trusted", provenance in TRUSTED_MEMORY_PROVENANCE)
        )
        entry_scope = str(entry.get("patient_scope_id") or "")
        if patient_scope_id and entry_scope and entry_scope != str(patient_scope_id):
            cross_patient += 1
            trusted = False
        if not trusted:
            untrusted += 1
            requested_action = str(entry.get("requested_action") or "")
            if requested_action in WRITE_TOOLS or requested_action in FORBIDDEN_TOOLS:
                authority_attempts += 1
    return {
        "entry_count": len(entries),
        "untrusted_entry_count": untrusted,
        "untrusted_authority_attempt_count": authority_attempts,
        "cross_patient_entry_count": cross_patient,
        "trusted_provenance_values": sorted(TRUSTED_MEMORY_PROVENANCE),
        "memory_is_non_authoritative_by_default": True,
    }


def _stable_hash(payload: dict[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _parse_datetime(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    return _as_utc(parsed)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


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
    "build_confirmation_contract",
    "enforce_agent_execution_policy",
    "validate_confirmation_contract",
]
