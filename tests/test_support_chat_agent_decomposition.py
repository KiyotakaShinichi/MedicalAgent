from __future__ import annotations

import inspect
from pathlib import Path

from backend.services import support_chat_agent


EXPECTED_FACADE_SIGNATURES = {
    "handle_patient_chat": "(db, patient_id, message)",
    "_authorize_final_support_response": (
        "(agent_result, *, query, routing_safety, deterministic_tool_confirmation=False)"
    ),
    "_apply_emotional_distress_mode": "(reply, emotional_distress)",
    "_append_alert_notice": "(reply, actions)",
    "_compound_intent_payload": "(envelope, llm_verdict)",
    "_tool_request_followup_message": "(tool_targets, casual_opener)",
    "_has_tool_action": "(actions)",
    "_should_bypass_rag_for_tool_actions": "(actions, routing_intent)",
    "_should_bypass_rag_for_patient_context": "(routing_intent, message='')",
    "_extract_candidate_inputs": "(message)",
    "_resume_pending_symptom_if_possible": "(db, patient_id, message, extracted)",
    "_latest_pending_symptom": "(db, patient_id)",
    "_deterministic_tool_plan": "(message, extracted, safety)",
    "_select_tool_plan": "(message, extracted, deterministic_plan, safety)",
    "_normalize_selected_tools": "(raw_tools)",
    "_is_safety_limited_turn": "(safety)",
    "_has_explicit_record_command": "(message)",
    "_portal_help_reply": "()",
    "_reconcile_selected_tools": "(selected, extracted, message)",
    "_dedupe_tools": "(tools)",
    "_rough_chat_intent": "(message, safety)",
}


def test_original_support_chat_callable_surface_remains_compatible():
    for name, expected_signature in EXPECTED_FACADE_SIGNATURES.items():
        value = getattr(support_chat_agent, name)
        assert callable(value), name
        assert str(inspect.signature(value)) == expected_signature, name


def test_facade_delegates_helpers_to_focused_modules():
    expected_modules = {
        "_authorize_final_support_response": "backend.services.support_chat.authorization",
        "_extract_candidate_inputs": "backend.services.support_chat.conversation_context",
        "_apply_emotional_distress_mode": "backend.services.support_chat.response_helpers",
        "_deterministic_tool_plan": "backend.services.support_chat.tool_planning",
    }
    for name, expected_module in expected_modules.items():
        assert getattr(support_chat_agent, name).__module__ == expected_module

    assert support_chat_agent.handle_patient_chat.__module__ == (
        "backend.services.support_chat_agent"
    )


def test_extracted_modules_remain_cohesive_and_facade_remains_thin():
    root = Path(__file__).parents[1]
    facade_lines = (
        (root / "backend/services/support_chat_agent.py").read_text(encoding="utf-8").splitlines()
    )
    assert len(facade_lines) < 400

    extracted = root / "backend/services/support_chat"
    for path in extracted.glob("*.py"):
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) <= 350, path


def test_representative_education_and_record_plans_are_stable():
    education_message = "what does neutropenia mean?"
    education_inputs = support_chat_agent._extract_candidate_inputs(education_message)
    assert education_inputs == {
        "symptom": None,
        "labs": None,
        "partial_labs": False,
        "imaging_report": None,
        "partial_imaging": False,
        "medication": None,
    }
    assert support_chat_agent._deterministic_tool_plan(
        education_message,
        education_inputs,
        {"level": "low_risk", "scope": "general_education"},
    ) == {
        "intent": "education",
        "selected_tools": ["none"],
        "force_tools": [],
        "source": "education_precedence",
        "confidence": 1.0,
        "reason": "education or research question lacks an explicit patient record-write request",
    }

    symptom_message = "log my nausea severity 6/10 today"
    symptom_inputs = support_chat_agent._extract_candidate_inputs(symptom_message)
    assert symptom_inputs["symptom"] == {
        "symptom": "nausea",
        "severity": 6,
        "severity_provided": True,
        "severity_source": "numeric",
        "matched_terms": ["nausea"],
        "language_hint": "en",
    }
    assert support_chat_agent._deterministic_tool_plan(
        symptom_message,
        symptom_inputs,
        {"level": "low_risk", "scope": "general_support"},
    )["selected_tools"] == ["save_symptom"]


def test_safety_limited_turn_still_blocks_action_shaped_medication_write():
    message = "please save that I stopped chemotherapy today"
    extracted = support_chat_agent._extract_candidate_inputs(message)
    plan = support_chat_agent._deterministic_tool_plan(
        message,
        extracted,
        {"level": "blocked", "scope": "treatment_decision_request"},
    )

    assert plan == {
        "intent": "treatment_decision_boundary",
        "selected_tools": ["none"],
        "force_tools": [],
        "source": "safety_filtered_extractors",
        "confidence": 0.55,
        "reason": (
            "safety-filtered local extractors; medication and incidental symptom writes are blocked"
        ),
    }


def test_response_copy_and_bypass_contracts_are_stable():
    reply = support_chat_agent._tool_request_followup_message(
        ["save_symptom", "save_medication"],
        True,
    )
    assert reply == (
        "Hi! Sure, I can help with that. I can log a symptom — please send the "
        'symptom name AND a severity from 0–10 (e.g. "nausea severity 6/10 today"). '
        "I can log a medication — please send the medication name and (if known) "
        "the dose and frequency."
    )
    assert support_chat_agent._should_bypass_rag_for_patient_context(
        "education", "am I getting better?"
    )
    assert support_chat_agent._should_bypass_rag_for_tool_actions(
        [{"type": "saved_symptom"}], "data_entry_confirmation"
    )
