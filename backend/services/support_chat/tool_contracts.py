import re

from backend.services.support_chat_extraction import (
    _should_request_symptom_details,
    _should_save_symptom,
)
from backend.services.support_chat_policy import ALLOWED_SUPPORT_TOOLS
from backend.services.support_chat_response import _is_conversational_prompt


def normalize_selected_tools(raw_tools):
    if isinstance(raw_tools, str):
        raw_tools = [raw_tools]
    tools = []
    for tool in raw_tools or []:
        normalized = str(tool or "").strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in ALLOWED_SUPPORT_TOOLS:
            tools.append(normalized)
    return dedupe_tools(tools) or ["none"]


def is_safety_limited_turn(safety):
    safety = safety or {}
    return safety.get("level") in {"high_risk", "blocked"} or safety.get("scope") in {
        "treatment_decision_request",
        "urgent_or_safety_related",
        "diagnosis_or_outcome_claim",
    }


def has_explicit_record_command(message):
    return re.search(r"\b(?:log|save|record|add|enter)\b", str(message or "").lower()) is not None


def reconcile_selected_tools(selected, extracted, message):
    reconciled = []
    selected = set(selected or [])
    symptom = extracted.get("symptom")

    if "save_symptom" in selected:
        if symptom and _should_save_symptom(message, symptom):
            reconciled.append("save_symptom")
        elif symptom and _should_request_symptom_details(message, symptom):
            reconciled.append("request_symptom_details")
    if (
        "request_symptom_details" in selected
        and symptom
        and _should_request_symptom_details(message, symptom)
    ):
        reconciled.append("request_symptom_details")

    if "save_complete_cbc" in selected:
        if extracted.get("labs"):
            reconciled.append("save_complete_cbc")
        elif extracted.get("partial_labs"):
            reconciled.append("request_missing_cbc_fields")
    if "request_missing_cbc_fields" in selected and extracted.get("partial_labs"):
        reconciled.append("request_missing_cbc_fields")

    if "save_imaging_report" in selected:
        if extracted.get("imaging_report"):
            reconciled.append("save_imaging_report")
        elif extracted.get("partial_imaging"):
            reconciled.append("request_missing_imaging_details")
    if "request_missing_imaging_details" in selected and extracted.get("partial_imaging"):
        reconciled.append("request_missing_imaging_details")

    if "save_medication" in selected and extracted.get("medication"):
        reconciled.append("save_medication")

    return dedupe_tools(reconciled) or ["none"]


def dedupe_tools(tools):
    seen = set()
    deduped = []
    for tool in tools or []:
        if tool == "none" and len(tools) > 1:
            continue
        if tool not in seen:
            seen.add(tool)
            deduped.append(tool)
    return deduped


def rough_chat_intent(message, safety):
    lower = message.lower()
    if safety.get("scope") == "treatment_decision_request":
        return "treatment_decision_boundary"
    if safety.get("scope") in {"urgent_or_safety_related", "diagnosis_or_outcome_claim"}:
        return "safety_boundary"
    if _is_conversational_prompt(message):
        return "conversation"
    if any(
        term in lower
        for term in [
            "remember",
            "what did i tell",
            "what did i say",
            "last message",
            "previous message",
            "chat history",
        ]
    ):
        return "patient_memory"
    if any(
        term in lower
        for term in [
            "last 14",
            "timeline",
            "cycle",
            "toxicity",
            "score",
            "my treatment",
            "working",
            "progress",
        ]
    ):
        return "patient_timeline_monitoring"
    if any(
        term in lower
        for term in ["upload", "site", "portal", "dashboard", "where can i", "how do i add"]
    ):
        return "portal_help"
    if any(
        term in lower
        for term in [
            "pcr",
            "response",
            "mri",
            "cbc",
            "wbc",
            "hemoglobin",
            "platelets",
            "chemo",
            "chemotherapy",
            "side effect",
            "breast cancer",
            "neutropenia",
            "infection risk",
        ]
    ):
        return "education"
    if any(term in lower for term in ["anxious", "worried", "sad", "scared", "depressed"]):
        return "emotional_support"
    return "general_support"
