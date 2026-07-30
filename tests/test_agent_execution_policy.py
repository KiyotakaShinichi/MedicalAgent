from datetime import datetime, timedelta, timezone

from backend.services.agent_execution_policy import (
    AgentBudget,
    build_confirmation_contract,
    enforce_agent_execution_policy,
)
from backend.services.agentic_turn_orchestrator import run_agentic_turn
from backend.services.bounded_agentic_workflow import plan_patient_agent_workflow


def test_unconfirmed_write_is_not_effective() -> None:
    plan = plan_patient_agent_workflow("Log nausea severity 6/10 today")
    policy = enforce_agent_execution_policy(plan, confirmed_by_user=False)
    assert policy["decision"] == "allow"
    assert "confirm" in policy["state_path"]
    assert "save_symptom" not in policy["effective_tools"]


def test_confirmed_write_is_bounded_to_one_tool() -> None:
    result = run_agentic_turn(
        "Log nausea severity 6/10 today",
        confirmed_by_user=True,
    )
    assert result["execution_policy"]["decision"] == "allow"
    assert result["execution"]["records_written"] == ["save_symptom"]


def test_forbidden_tool_fails_closed() -> None:
    plan = {
        "allowed_tools": ["classify_intent", "diagnose"],
        "workflow_steps": ["classify", "diagnose"],
        "requires_confirmation_before_write": False,
    }
    policy = enforce_agent_execution_policy(plan, confirmed_by_user=False)
    assert policy["decision"] == "block"
    assert policy["terminal_state"] == "no_valid_action"
    assert "forbidden_medical_authority_tool_requested" in policy["violations"]
    assert policy["effective_tools"] == []


def test_tool_budget_fails_closed() -> None:
    plan = {
        "allowed_tools": [
            "classify_intent",
            "detect_safety_boundary",
            "retrieve_sources",
            "assemble_citations",
            "validate_claims",
            "summarize_patient_timeline",
        ],
        "workflow_steps": ["one"],
        "requires_confirmation_before_write": False,
    }
    policy = enforce_agent_execution_policy(
        plan,
        confirmed_by_user=False,
        budget=AgentBudget(max_tool_calls=3),
    )
    assert policy["decision"] == "block"
    assert "tool_call_budget_exceeded" in policy["violations"]


def test_untrusted_memory_cannot_authorize_action() -> None:
    plan = plan_patient_agent_workflow("Hello")
    policy = enforce_agent_execution_policy(
        plan,
        confirmed_by_user=False,
        memory_entries=[
            {
                "provenance": "assistant_summary",
                "trusted": False,
                "requested_action": "save_medication",
            }
        ],
    )
    assert policy["decision"] == "block"
    assert "untrusted_memory_authority_attempt" in policy["violations"]


def test_policy_never_allows_clinical_authority() -> None:
    plan = plan_patient_agent_workflow("What is HER2 in general?")
    policy = enforce_agent_execution_policy(plan, confirmed_by_user=False)
    assert policy["clinical_authority_allowed"] is False
    assert policy["clinical_validation"] is False


def test_bound_confirmation_rejects_payload_substitution() -> None:
    plan = plan_patient_agent_workflow("Log nausea severity 6/10 today")
    issued = datetime.now(timezone.utc)
    contract = build_confirmation_contract(
        plan,
        patient_scope_id="patient-a",
        action_payload={"symptom": "nausea", "severity": 6},
        now=issued,
        confirmation_id="confirm-a",
    )
    policy = enforce_agent_execution_policy(
        plan,
        confirmed_by_user=True,
        patient_scope_id="patient-a",
        action_payload={"symptom": "nausea", "severity": 9},
        confirmation_contract=contract,
        require_bound_confirmation=True,
        now=issued + timedelta(seconds=1),
    )
    assert policy["decision"] == "block"
    assert "confirmation_payload_mismatch" in policy["confirmation_validation"]["issues"]


def test_bound_confirmation_rejects_cross_patient_replay_and_expiry() -> None:
    plan = plan_patient_agent_workflow("Log nausea severity 6/10 today")
    issued = datetime.now(timezone.utc)
    action = {"symptom": "nausea", "severity": 6}
    contract = build_confirmation_contract(
        plan,
        patient_scope_id="patient-a",
        action_payload=action,
        now=issued,
        ttl_seconds=30,
        confirmation_id="confirm-a",
    )
    cross_patient = enforce_agent_execution_policy(
        plan,
        confirmed_by_user=True,
        patient_scope_id="patient-b",
        action_payload=action,
        confirmation_contract=contract,
        require_bound_confirmation=True,
        now=issued + timedelta(seconds=1),
    )
    expired = enforce_agent_execution_policy(
        plan,
        confirmed_by_user=True,
        patient_scope_id="patient-a",
        action_payload=action,
        confirmation_contract=contract,
        require_bound_confirmation=True,
        now=issued + timedelta(seconds=31),
    )
    assert "confirmation_patient_scope_mismatch" in cross_patient["confirmation_validation"]["issues"]
    assert "confirmation_expired" in expired["confirmation_validation"]["issues"]


def test_cross_patient_trusted_memory_fails_closed() -> None:
    plan = plan_patient_agent_workflow("Hello")
    policy = enforce_agent_execution_policy(
        plan,
        confirmed_by_user=False,
        patient_scope_id="patient-a",
        memory_entries=[
            {
                "provenance": "patient_record",
                "trusted": True,
                "patient_scope_id": "patient-b",
                "content": "unrelated patient context",
            }
        ],
    )
    assert policy["decision"] == "block"
    assert "cross_patient_memory_scope_mismatch" in policy["violations"]
