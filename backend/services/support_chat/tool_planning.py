import re

from backend.services.agent_rag import route_intent
from backend.services.local_llm import select_support_tools_with_local_llm
from backend.services.support_chat.tool_contracts import (
    dedupe_tools,
    has_explicit_record_command,
    is_safety_limited_turn,
    normalize_selected_tools,
    reconcile_selected_tools,
    rough_chat_intent,
)
from backend.services.support_chat_extraction import (
    _is_short_record_hint,
    _should_request_symptom_details,
    _should_save_symptom,
)
from backend.services.support_chat_policy import ALLOWED_SUPPORT_INTENTS


def deterministic_tool_plan(message, extracted, safety):
    safety = safety or {}
    safety_limited = safety.get("level") in {"high_risk", "blocked"} or safety.get("scope") in {
        "treatment_decision_request",
        "urgent_or_safety_related",
        "diagnosis_or_outcome_claim",
    }

    lower = message.lower()
    explicit_write_cues = (
        "log my",
        "save my",
        "record my",
        "add my",
        "enter my",
        "i have",
        "i had",
        "i took",
        "my result",
        "my report says",
    )
    education_only_cues = (
        "paper",
        "research",
        "study",
        "article",
        "publication",
        "knowledge base",
        "what does",
        "explain",
        "how does",
        "why ",
    )
    has_structured_candidate = any(
        extracted.get(key)
        for key in (
            "symptom",
            "labs",
            "partial_labs",
            "imaging_report",
            "partial_imaging",
            "medication",
        )
    )
    if (
        route_intent(message, actions=[], safety=safety) == "education"
        and not any(cue in lower for cue in explicit_write_cues)
        and (not has_structured_candidate or any(cue in lower for cue in education_only_cues))
    ):
        return {
            "intent": "education",
            "selected_tools": ["none"],
            "force_tools": [],
            "source": "education_precedence",
            "confidence": 1.0,
            "reason": "education or research question lacks an explicit patient record-write request",
        }

    tools = []
    force_tools = []

    if extracted.get("symptom"):
        if _should_save_symptom(message, extracted["symptom"]):
            tools.append("save_symptom")
            force_tools.append("save_symptom")
        elif _should_request_symptom_details(message, extracted["symptom"]):
            tools.append("request_symptom_details")

    if extracted.get("labs"):
        tools.append("save_complete_cbc")
        force_tools.append("save_complete_cbc")
    elif extracted.get("partial_labs"):
        tools.append("request_missing_cbc_fields")
        if _is_short_record_hint(message):
            force_tools.append("request_missing_cbc_fields")

    if extracted.get("imaging_report"):
        tools.append("save_imaging_report")
        force_tools.append("save_imaging_report")
    elif extracted.get("partial_imaging"):
        tools.append("request_missing_imaging_details")
        if _is_short_record_hint(message):
            force_tools.append("request_missing_imaging_details")

    if extracted.get("medication"):
        tools.append("save_medication")
        force_tools.append("save_medication")

    if safety_limited:
        blocked_tools = {"save_medication"}
        explicit_record_command = has_explicit_record_command(message)
        explicit_imaging_record = bool(
            extracted.get("imaging_report")
            and re.search(r"\b(?:mri|ct|ultrasound|mammogram|imaging|scan)\b", lower)
            and re.search(r"\b(?:report|impression|findings?)\b", lower)
        )
        if not explicit_imaging_record:
            blocked_tools.update({"save_imaging_report", "request_missing_imaging_details"})
        if not explicit_record_command:
            blocked_tools.update({"save_symptom", "request_symptom_details"})
        tools = [tool for tool in tools if tool not in blocked_tools]
        force_tools = [tool for tool in force_tools if tool not in blocked_tools]

    tools = dedupe_tools(tools) or ["none"]
    return {
        "intent": (
            "data_entry_confirmation" if tools != ["none"] else rough_chat_intent(message, safety)
        ),
        "selected_tools": tools,
        "force_tools": dedupe_tools(force_tools),
        "source": "safety_filtered_extractors" if safety_limited else "deterministic_extractors",
        "confidence": 1.0 if tools != ["none"] else 0.55,
        "reason": (
            "safety-filtered local extractors; medication and incidental symptom writes are blocked"
            if safety_limited
            else "validated local extractors and record-hint heuristics"
        ),
    }


def select_tool_plan(message, extracted, deterministic_plan, safety):
    deterministic_tools = [
        tool for tool in deterministic_plan.get("selected_tools", []) if tool != "none"
    ]
    llm = {"available": False}
    if not deterministic_tools:
        try:
            from backend.services.multilingual_tool_router import tool_intent_hints_from_text

            hints = tool_intent_hints_from_text(message)
        except Exception:  # noqa: BLE001 - never break chat on hint routing
            hints = []
        if hints:
            llm = select_support_tools_with_local_llm(
                message,
                deterministic_tools=deterministic_plan["selected_tools"],
                deterministic_intent=deterministic_plan["intent"],
                safety=safety,
            )
    selected = deterministic_plan["selected_tools"]
    source = deterministic_plan["source"]
    confidence = deterministic_plan["confidence"]
    reason = deterministic_plan["reason"]
    planned_intent = deterministic_plan["intent"]

    if llm.get("available") and float(llm.get("confidence") or 0) >= 0.6:
        selected = normalize_selected_tools(llm.get("selected_tools") or llm.get("tools") or [])
        selected = reconcile_selected_tools(selected, extracted, message)
        candidate_intent = str(llm.get("intent") or "").strip()
        if candidate_intent in ALLOWED_SUPPORT_INTENTS:
            planned_intent = candidate_intent
        source = f"llm_{llm.get('provider')}"
        confidence = float(llm.get("confidence") or 0)
        reason = llm.get("reason") or "LLM support-tool router"

    selected = reconcile_selected_tools(selected, extracted, message)
    selected = dedupe_tools(
        [tool for tool in selected if tool != "none"] + deterministic_plan.get("force_tools", [])
    )
    if is_safety_limited_turn(safety) and not has_explicit_record_command(message):
        selected = [
            tool
            for tool in selected
            if tool not in {"save_symptom", "request_symptom_details", "save_medication"}
        ]
    if not selected:
        selected = ["none"]
    if selected != ["none"]:
        planned_intent = "data_entry_confirmation"
    elif planned_intent == "data_entry_confirmation":
        planned_intent = rough_chat_intent(message, safety)
    elif safety.get("scope") == "treatment_decision_request":
        planned_intent = "treatment_decision_boundary"
    elif safety.get("scope") in {"urgent_or_safety_related", "diagnosis_or_outcome_claim"}:
        planned_intent = "safety_boundary"

    return {
        "intent": (
            planned_intent
            if planned_intent in ALLOWED_SUPPORT_INTENTS
            else rough_chat_intent(message, safety)
        ),
        "selected_tools": selected,
        "deterministic_tools": deterministic_plan["selected_tools"],
        "forced_tools": deterministic_plan.get("force_tools", []),
        "source": source,
        "confidence": round(confidence, 3),
        "reason": reason,
    }
