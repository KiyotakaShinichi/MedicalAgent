import json
import re
from datetime import date, timedelta

from groq import Groq

from backend.config import get_groq_api_key, get_groq_model, get_llm_adjudication_enabled
from backend.models import (
    ChatMessage,
    ClinicalIntervention,
    ImagingReport,
    LabResult,
    MedicationLog,
    SymptomReport,
    Treatment,
    TreatmentOutcome,
)
from backend.processing.radiology_analysis import detect_possible_metastatic_indicators
from backend.services.agent_rag import run_patient_agent_pipeline, route_intent, safety_scope_check
from backend.services.app_logging import log_app_event
from backend.services.input_validation import validate_cbc_values, validate_imaging_report_payload, validate_symptom_payload
from backend.services.local_llm import select_support_tools_with_local_llm
from backend.services.security_guardrails import detect_multilingual_medical_danger, normalize_security_text


SYMPTOM_KEYWORDS = {
    "fatigue": ["fatigue", "tired", "weak", "exhausted"],
    "nausea": ["nausea", "nauseous", "vomit", "vomiting"],
    "pain": ["pain", "ache", "aching", "sore"],
    "bloody discharge": ["blood discharge", "bloody discharge", "bleeding discharge", "blood-stained discharge"],
    "bleeding": ["bleeding", "blood loss", "spotting"],
    "discharge": ["discharge"],
    "fever": ["fever", "temperature", "chills"],
    "shortness of breath": ["shortness of breath", "breathless", "difficulty breathing"],
    "neuropathy": ["neuropathy", "tingling", "numbness"],
    "low appetite": ["low appetite", "no appetite", "not eating"],
    "anxiety": ["anxiety", "anxious", "scared", "worried", "panic"],
    "sadness": ["sad", "depressed", "crying", "hopeless"],
}

URGENT_TERMS = [
    "chest pain",
    "cannot breathe",
    "difficulty breathing",
    "faint",
    "fainted",
    "confused",
    "bleeding",
    "blood discharge",
    "bloody discharge",
    "fever",
    "suicidal",
    "self harm",
    "kill myself",
]

CHAT_SYSTEM_PROMPT = """\
You are a patient support assistant for a breast cancer treatment monitoring portal.

Rules:
- Do not diagnose, stage, confirm recurrence/metastasis, or decide treatment.
- Do not tell the patient to start, stop, increase, or decrease medications.
- Do not invent patient facts. Use recent_context only when directly helpful.
- Explain what was logged, ask for missing tracking details when useful, and encourage oncology team review for concerning symptoms.
- For greetings, identity questions, and "how are you" style messages, answer naturally and briefly as a warm portal assistant.
- If saved_actions contains a saved item, acknowledge the item clearly and do not add unrelated oncology education.
- If the message asks about prior chat, summarize only patient-scoped recent_context / chat messages.
- If urgent wording is present, advise contacting the oncology team or emergency services now.
- Keep the tone calm and practical. Maximum 120 words. Plain text only.
"""

ALLOWED_SUPPORT_TOOLS = {
    "none",
    "save_symptom",
    "request_symptom_details",
    "save_complete_cbc",
    "request_missing_cbc_fields",
    "save_medication",
    "save_imaging_report",
    "request_missing_imaging_details",
}

ALLOWED_SUPPORT_INTENTS = {
    "conversation",
    "education",
    "portal_help",
    "patient_memory",
    "emotional_support",
    "patient_timeline_monitoring",
    "general_support",
    "data_entry_confirmation",
    "safety_boundary",
    "treatment_decision_boundary",
}


def handle_patient_chat(db, patient_id, message):
    normalized = message.strip()
    if not normalized:
        raise ValueError("Message cannot be empty")

    urgent_flags = _detect_urgent_flags(normalized)
    routing_safety = safety_scope_check(normalized, urgent_flags)
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

    actions = []
    selected_tools = set(tool_plan["selected_tools"])

    symptom = extracted["symptom"]
    if "save_symptom" in selected_tools and symptom:
        # Honest-save rule: only persist when severity was explicitly provided.
        # Auto-defaulting silently caused "I've logged your symptom" replies
        # for messages that gave no severity, which is misleading.
        if not symptom.get("severity_provided") or symptom["severity"] is None:
            actions.append({
                "type": "partial_symptom_detected",
                "symptom": symptom["symptom"],
                "message": (
                    f"I noticed you mentioned {symptom['symptom']}, but I need a severity "
                    f"before I can save it. Please send it on a 0-10 scale, for example: "
                    f"\"{symptom['symptom']} severity 6/10 today\"."
                ),
            })
        else:
            severity = int(symptom["severity"])
            try:
                validate_symptom_payload(symptom["symptom"], severity)
                db.add(SymptomReport(
                    patient_id=patient_id,
                    date=_extract_date(normalized),
                    symptom=symptom["symptom"],
                    severity=severity,
                    notes=f"Captured from support chat: {normalized}",
                ))
                db.flush()  # surface DB errors here, before we claim a save happened
                actions.append({
                    "type": "saved_symptom",
                    "symptom": symptom["symptom"],
                    "severity": severity,
                    "resumed_from_memory": bool(symptom.get("resumed_from_memory")),
                })
            except Exception as exc:
                # Validation or DB write failed.  Roll back the pending change
                # and surface a truthful "I couldn't save it yet" action.
                db.rollback()
                actions.append({
                    "type": "symptom_save_failed",
                    "symptom": symptom["symptom"],
                    "severity": severity,
                    "reason": str(exc)[:200],
                    "message": (
                        f"I couldn't save the {symptom['symptom']} entry just now — "
                        f"there was a problem with the record. Please try again or "
                        f"log it from the portal manually."
                    ),
                })
    elif "request_symptom_details" in selected_tools and symptom:
        actions.append({
            "type": "partial_symptom_detected",
            "symptom": symptom["symptom"],
            "message": (
                f"I noticed you mentioned {symptom['symptom']}. If you want me to log it, "
                f"send the severity from 0-10, for example: "
                f"\"{symptom['symptom']} severity 6/10 today\"."
            ),
        })

    labs = extracted["labs"]
    if "save_complete_cbc" in selected_tools and labs:
        validate_cbc_values(labs["wbc"], labs["hemoglobin"], labs["platelets"])
        lab_alerts = _clinical_lab_alerts(labs)
        db.add(LabResult(
            patient_id=patient_id,
            date=_extract_date(normalized),
            wbc=labs["wbc"],
            hemoglobin=labs["hemoglobin"],
            platelets=labs["platelets"],
            source="chat_agent",
            source_note="Captured from patient support chat.",
        ))
        actions.append({"type": "saved_labs", **labs})
        if lab_alerts:
            actions.append({
                "type": "clinical_rule_alert",
                "alerts": lab_alerts,
                "message": "CBC safety rule triggered before RAG retrieval.",
            })
            urgent_flags.extend([alert["rule"] for alert in lab_alerts])
    elif "request_missing_cbc_fields" in selected_tools and extracted["partial_labs"]:
        actions.append({
            "type": "partial_labs_detected",
            "message": "I saw lab information, but I need WBC, hemoglobin, and platelets together to save a CBC row.",
        })

    imaging_report = extracted["imaging_report"]
    if "save_imaging_report" in selected_tools and imaging_report:
        validate_imaging_report_payload(
            imaging_report["modality"],
            imaging_report["report_type"],
            imaging_report["findings"],
            imaging_report["impression"],
            body_site=imaging_report["body_site"],
        )
        db.add(ImagingReport(
            patient_id=patient_id,
            date=imaging_report["date"],
            modality=imaging_report["modality"],
            report_type=imaging_report["report_type"],
            body_site=imaging_report["body_site"],
            findings=imaging_report["findings"],
            impression=imaging_report["impression"],
        ))
        actions.append({
            "type": "saved_imaging_report",
            "modality": imaging_report["modality"],
            "date": imaging_report["date"].isoformat(),
            "report_type": imaging_report["report_type"],
        })
        indicators = detect_possible_metastatic_indicators(
            f"{imaging_report['findings']} {imaging_report['impression']}"
        )
        if indicators:
            sites = sorted({indicator["site"] for indicator in indicators})
            actions.append({
                "type": "possible_metastatic_indicator",
                "sites": sites,
                "message": (
                    "Report wording mentions possible distant-disease indicators. "
                    "This is not a diagnosis and should be reviewed by the oncology team."
                ),
            })
            urgent_flags.extend([f"imaging_{site}" for site in sites])
    elif "request_missing_imaging_details" in selected_tools and extracted["partial_imaging"]:
        actions.append({
            "type": "partial_imaging_detected",
            "message": "I saw imaging wording. To save it as a report, paste the report date plus findings or impression text.",
        })

    medication = extracted["medication"]
    if "save_medication" in selected_tools and medication:
        db.add(MedicationLog(
            patient_id=patient_id,
            date=_extract_date(normalized),
            medication=medication["medication"],
            dose=medication.get("dose"),
            frequency=medication.get("frequency"),
            notes=f"Captured from support chat: {normalized}",
        ))
        actions.append({"type": "saved_medication", **medication})

    patient_context = _recent_patient_context(db, patient_id)
    fallback_response = _build_response(normalized, actions, urgent_flags, patient_context)
    routing_intent = tool_plan["intent"] if tool_plan.get("intent") in ALLOWED_SUPPORT_INTENTS else route_intent(normalized, actions=actions, safety=routing_safety)
    if _should_use_llm_direct_reply(routing_intent, routing_safety, actions, urgent_flags) and not _has_tool_action(actions):
        fallback_response = _generate_llm_response(normalized, actions, urgent_flags, patient_context, fallback_response)
    if _should_bypass_rag_for_tool_actions(actions, routing_intent):
        agent_result = {
            "reply": fallback_response,
            "intent": routing_intent,
            "safety": routing_safety,
            "citations": [],
            "cache": {"status": "bypassed_for_deterministic_tool_action"},
            "validation": {"status": "not_needed_for_tool_confirmation"},
            "guardrails": {
                "input_passed": routing_safety.get("level") != "blocked",
                "output_passed": True,
                "reason": "deterministic tool confirmation; no RAG generation",
            },
            "rag_evaluation": None,
            "pipeline_trace": [
                "safety_gate",
                "intent_routing",
                "deterministic_tool_action",
                "confirmation_reply",
            ],
        }
    else:
        agent_result = run_patient_agent_pipeline(
            db=db,
            patient_id=patient_id,
            query=normalized,
            patient_context=patient_context,
            fallback_response=fallback_response,
            actions=actions,
            urgent_flags=urgent_flags,
            preselected_intent=routing_intent,
        )
    response = agent_result["reply"]
    assistant_record = ChatMessage(
        patient_id=patient_id,
        role="assistant",
        message=response,
        intent="patient_support_response",
        saved_actions_json=json.dumps({
            "saved_actions": actions,
            "tool_plan": tool_plan,
            "agent_pipeline": {
                "intent": agent_result.get("intent"),
                "safety": agent_result.get("safety"),
                "citations": agent_result.get("citations") or [],
                "cache": agent_result.get("cache"),
                "validation": agent_result.get("validation"),
                "guardrails": agent_result.get("guardrails"),
                "rag_evaluation": agent_result.get("rag_evaluation"),
            },
        }),
    )
    db.add(assistant_record)
    db.commit()
    db.refresh(assistant_record)
    log_app_event(
        db=db,
        event_type="agent_rag",
        patient_id=patient_id,
        route="/me/chat",
        status="ok",
        input_payload={
            "intent": agent_result.get("intent"),
            "safety_level": (agent_result.get("safety") or {}).get("level"),
            "cache": agent_result.get("cache"),
            "tool_plan": tool_plan,
        },
        output_payload={
            "citation_count": len(agent_result.get("citations") or []),
            "validation": agent_result.get("validation"),
        },
    )

    return {
        "reply": response,
        "saved_actions": actions,
        "tool_plan": tool_plan,
        "urgent_flags": urgent_flags,
        "agent_pipeline": {
            "intent": agent_result.get("intent"),
            "safety": agent_result.get("safety"),
            "citations": agent_result.get("citations") or [],
            "cache": agent_result.get("cache"),
            "validation": agent_result.get("validation"),
            "guardrails": agent_result.get("guardrails"),
            "rag_evaluation": agent_result.get("rag_evaluation"),
            "pipeline_trace": agent_result.get("pipeline_trace"),
        },
        "assistant_message_id": assistant_record.id,
        "safety_note": "This assistant logs and summarizes information only. It does not diagnose or give treatment instructions.",
    }


def _has_tool_action(actions):
    return any(
        action.get("type") in {
            "saved_symptom",
            "saved_labs",
            "saved_medication",
            "saved_imaging_report",
            "partial_symptom_detected",
            "partial_labs_detected",
            "partial_imaging_detected",
            "symptom_save_failed",
        }
        for action in actions
    )


def _should_bypass_rag_for_tool_actions(actions, routing_intent):
    if routing_intent != "data_entry_confirmation":
        return False
    return _has_tool_action(actions)


def _extract_candidate_inputs(message):
    labs = _extract_complete_labs(message)
    imaging_report = _extract_imaging_report(message)
    symptom = _extract_symptom(message)
    return {
        "symptom": symptom,
        "labs": labs,
        "partial_labs": bool(not labs and _looks_like_partial_labs(message) and not _is_general_lab_question(message)),
        "imaging_report": imaging_report,
        "partial_imaging": bool(not imaging_report and _looks_like_partial_imaging(message)),
        "medication": _extract_medication(message),
    }


def _resume_pending_symptom_if_possible(db, patient_id, message, extracted):
    """Resume a prior partial symptom save when the user replies with severity only."""
    if extracted.get("symptom"):
        return None
    severity = _extract_severity(message.lower())
    if severity is None:
        return None
    pending_symptom = _latest_pending_symptom(db, patient_id)
    if not pending_symptom:
        return None
    return {
        "symptom": pending_symptom,
        "severity": severity,
        "severity_provided": True,
        "resumed_from_memory": True,
    }


def _latest_pending_symptom(db, patient_id):
    rows = (
        db.query(ChatMessage)
        .filter(ChatMessage.patient_id == patient_id, ChatMessage.role == "assistant")
        .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
        .limit(8)
        .all()
    )
    for row in rows:
        try:
            payload = json.loads(row.saved_actions_json or "{}")
        except (TypeError, ValueError):
            continue
        for action in payload.get("saved_actions") or []:
            action_type = action.get("type")
            if action_type == "saved_symptom":
                return None
            if action_type == "partial_symptom_detected" and action.get("symptom"):
                return str(action["symptom"])
    return None


def _deterministic_tool_plan(message, extracted, safety):
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

    tools = _dedupe_tools(tools) or ["none"]
    return {
        "intent": "data_entry_confirmation" if tools != ["none"] else _rough_chat_intent(message, safety),
        "selected_tools": tools,
        "force_tools": _dedupe_tools(force_tools),
        "source": "deterministic_extractors",
        "confidence": 1.0 if tools != ["none"] else 0.55,
        "reason": "validated local extractors and record-hint heuristics",
    }


def _select_tool_plan(message, extracted, deterministic_plan, safety):
    deterministic_tools = [tool for tool in deterministic_plan.get("selected_tools", []) if tool != "none"]
    llm = {"available": False}
    if not deterministic_tools:
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
        selected = _normalize_selected_tools(llm.get("selected_tools") or llm.get("tools") or [])
        selected = _reconcile_selected_tools(selected, extracted, message)
        candidate_intent = str(llm.get("intent") or "").strip()
        if candidate_intent in ALLOWED_SUPPORT_INTENTS:
            planned_intent = candidate_intent
        source = f"llm_{llm.get('provider')}"
        confidence = float(llm.get("confidence") or 0)
        reason = llm.get("reason") or "LLM support-tool router"

    selected = _reconcile_selected_tools(selected, extracted, message)
    selected = _dedupe_tools([tool for tool in selected if tool != "none"] + deterministic_plan.get("force_tools", []))
    if not selected:
        selected = ["none"]
    if selected != ["none"]:
        planned_intent = "data_entry_confirmation"
    elif planned_intent == "data_entry_confirmation":
        planned_intent = _rough_chat_intent(message, safety)
    elif safety.get("scope") == "treatment_decision_request":
        planned_intent = "treatment_decision_boundary"
    elif safety.get("scope") in {"urgent_or_safety_related", "diagnosis_or_outcome_claim"}:
        planned_intent = "safety_boundary"

    return {
        "intent": planned_intent if planned_intent in ALLOWED_SUPPORT_INTENTS else _rough_chat_intent(message, safety),
        "selected_tools": selected,
        "deterministic_tools": deterministic_plan["selected_tools"],
        "forced_tools": deterministic_plan.get("force_tools", []),
        "source": source,
        "confidence": round(confidence, 3),
        "reason": reason,
    }


def _normalize_selected_tools(raw_tools):
    if isinstance(raw_tools, str):
        raw_tools = [raw_tools]
    tools = []
    for tool in raw_tools or []:
        normalized = str(tool or "").strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in ALLOWED_SUPPORT_TOOLS:
            tools.append(normalized)
    return _dedupe_tools(tools) or ["none"]


def _reconcile_selected_tools(selected, extracted, message):
    reconciled = []
    selected = set(selected or [])
    symptom = extracted.get("symptom")

    if "save_symptom" in selected:
        if symptom and _should_save_symptom(message, symptom):
            reconciled.append("save_symptom")
        elif symptom and _should_request_symptom_details(message, symptom):
            reconciled.append("request_symptom_details")
    if "request_symptom_details" in selected and symptom and _should_request_symptom_details(message, symptom):
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

    return _dedupe_tools(reconciled) or ["none"]


def _dedupe_tools(tools):
    seen = set()
    deduped = []
    for tool in tools or []:
        if tool == "none" and len(tools) > 1:
            continue
        if tool not in seen:
            seen.add(tool)
            deduped.append(tool)
    return deduped


def _rough_chat_intent(message, safety):
    lower = message.lower()
    if safety.get("scope") == "treatment_decision_request":
        return "treatment_decision_boundary"
    if safety.get("scope") in {"urgent_or_safety_related", "diagnosis_or_outcome_claim"}:
        return "safety_boundary"
    if _is_conversational_prompt(message):
        return "conversation"
    if any(term in lower for term in ["remember", "what did i tell", "what did i say", "last message", "previous message", "chat history"]):
        return "patient_memory"
    if any(term in lower for term in ["last 14", "timeline", "cycle", "toxicity", "score", "my treatment", "working", "progress"]):
        return "patient_timeline_monitoring"
    if any(term in lower for term in ["upload", "site", "portal", "dashboard", "where can i", "how do i add"]):
        return "portal_help"
    if any(term in lower for term in ["pcr", "response", "mri", "cbc", "wbc", "hemoglobin", "platelets", "chemo", "chemotherapy", "side effect", "breast cancer", "neutropenia", "infection risk"]):
        return "education"
    if any(term in lower for term in ["anxious", "worried", "sad", "scared", "depressed"]):
        return "emotional_support"
    return "general_support"


def _extract_symptom(message):
    lower = message.lower()
    symptom_name = None
    for canonical, terms in SYMPTOM_KEYWORDS.items():
        if any(term in lower for term in terms):
            symptom_name = canonical
            break

    if not symptom_name:
        return None

    severity = _extract_severity(lower)
    return {
        "symptom": symptom_name,
        "severity": severity,
        "severity_provided": severity is not None,
    }


def _should_save_symptom(message, symptom):
    lower = message.lower()
    if _is_general_symptom_question(lower):
        return False
    if symptom.get("severity_provided"):
        return True
    if _explicit_tracking_request(lower) and _has_self_scoped_symptom_wording(lower):
        return True
    if symptom.get("symptom") == "fever" and _has_self_scoped_symptom_wording(lower):
        return True
    return False


def _should_request_symptom_details(message, symptom):
    lower = message.lower()
    if not symptom or _should_save_symptom(message, symptom) or _is_general_symptom_question(lower):
        return False
    if symptom.get("symptom") in {"anxiety", "sadness"} and not _explicit_tracking_request(lower):
        return False
    return _has_self_scoped_symptom_wording(lower) or _explicit_tracking_request(lower)


def _explicit_tracking_request(lower_message):
    return any(term in lower_message for term in ["log ", "save ", "record ", "report ", "track ", "add "])


def _has_self_scoped_symptom_wording(lower_message):
    patterns = [
        "i have",
        "i feel",
        "i am feeling",
        "i'm feeling",
        "i am having",
        "i'm having",
        "my ",
        "ako",
        "may ",
        "masakit",
        "nakakaramdam",
    ]
    return any(pattern in lower_message for pattern in patterns)


def _is_general_symptom_question(lower_message):
    stripped = lower_message.strip()
    if "my " in stripped or " i " in f" {stripped} ":
        return False
    question_starts = ("what is", "what are", "what does", "how does", "why", "can you explain", "explain", "tell me about")
    return stripped.endswith("?") or stripped.startswith(question_starts)


def _extract_severity(lower_message):
    patterns = [
        r"(\d{1,2})\s*/\s*10",
        r"severity\s*(?:is|:)?\s*(\d{1,2})",
        r"pain\s*(?:is|:)?\s*(\d{1,2})",
        r"level\s*(?:is|:)?\s*(\d{1,2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, lower_message)
        if match:
            return max(0, min(10, int(match.group(1))))
    return None


def _extract_complete_labs(message):
    lower = message.lower()
    wbc = _extract_number_after(lower, ["wbc", "white blood"])
    hemoglobin = _extract_number_after(lower, ["hemoglobin", "hgb", "hb"])
    platelets = _extract_number_after(lower, ["platelets", "platelet", "plt"])
    if wbc is None or hemoglobin is None or platelets is None:
        return None
    return {
        "wbc": wbc,
        "hemoglobin": hemoglobin,
        "platelets": platelets,
    }


def _looks_like_partial_labs(message):
    lower = message.lower()
    return any(term in lower for term in ["wbc", "white blood", "hemoglobin", "hgb", "platelet", "platelets", "plt"])


def _is_general_lab_question(message):
    lower = message.lower().strip()
    if any(term in lower for term in ["my wbc", "my hemoglobin", "my hgb", "my platelet", "my cbc", "i got", "i have"]):
        return False
    question_starts = ("what is", "what are", "what does", "how does", "why", "can you explain", "explain", "tell me about")
    return lower.endswith("?") or lower.startswith(question_starts)


def _is_short_record_hint(message):
    return len(re.findall(r"[a-zA-Z0-9]+", message)) <= 4


def _extract_imaging_report(message):
    lower = message.lower()
    if not _has_imaging_term(lower):
        return None
    report_terms = [
        "findings",
        "impression",
        "mass",
        "tumor",
        "lesion",
        "bi-rads",
        "birads",
        "cm",
        "decreased",
        "increased",
        "stable",
        "interval",
        "response",
        "ct",
        "pet/ct",
        "pet ct",
        "hepatic",
        "liver",
        "lung",
        "pleural",
        "pericardial",
        "ascites",
        "free fluid",
        "peritoneal",
        "omental",
        "carcinomatosis",
        "lymph node",
        "adenopathy",
        "metastatic",
        "metastasis",
        "metastases",
    ]
    if not any(term in lower for term in report_terms):
        return None
    if _is_general_imaging_question(lower):
        return None

    modality, body_site = _infer_imaging_modality_and_body_site(lower)

    report_type = "Patient-entered imaging note"
    if any(term in lower for term in ["follow-up", "follow up", "interval", "decreased", "increased", "stable"]):
        report_type = "Follow-up"
    elif "baseline" in lower:
        report_type = "Baseline"
    elif any(term in lower for term in ["staging", "restaging", "metastatic", "metastasis", "metastases", "ascites", "peritoneal"]):
        report_type = "Staging/follow-up"

    impression = _extract_report_section(message, "impression") or _best_imaging_sentence(message) or message
    findings = _extract_report_section(message, "findings") or message
    return {
        "date": _extract_date(message),
        "modality": modality,
        "report_type": report_type,
        "body_site": body_site,
        "findings": findings.strip()[:12000],
        "impression": impression.strip()[:4000],
    }


def _looks_like_partial_imaging(message):
    lower = message.lower()
    if _is_general_imaging_question(lower):
        return False
    if not _has_imaging_term(lower):
        return False
    self_scoped_terms = ["my", "report", "scan", "result", "uploaded", "impression", "findings", "ct", "pet", "ultrasound"]
    return len(lower.split()) <= 6 or any(term in lower for term in self_scoped_terms)


def _has_imaging_term(lower_message):
    return any(term in lower_message for term in [
        "mri",
        "imaging",
        "scan",
        "bi-rads",
        "birads",
        "mammogram",
        "ultrasound",
        "sonogram",
        "ct ",
        " ct",
        "ct/",
        "pet/ct",
        "pet ct",
        "cat scan",
    ])


def _is_general_imaging_question(lower_message):
    stripped = lower_message.strip()
    question_starts = ("what is", "what does", "how does", "why", "can you explain", "explain")
    return stripped.endswith("?") and stripped.startswith(question_starts) and "my " not in stripped


def _extract_report_section(message, label):
    pattern = rf"{label}\s*(?:is|:|-)?\s*(.+?)(?:\b(?:findings|impression)\s*(?:is|:|-)|$)"
    match = re.search(pattern, message, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    section = re.sub(r"\s+", " ", match.group(1)).strip(" .,:;-")
    return section if section else None


def _infer_imaging_modality_and_body_site(lower):
    abdominal_terms = [
        "abdomen",
        "abdominal",
        "pelvis",
        "pelvic",
        "liver",
        "hepatic",
        "ascites",
        "free fluid",
        "peritoneal",
        "omental",
        "carcinomatosis",
        "retroperitoneal",
    ]
    chest_terms = ["chest", "lung", "pulmonary", "pleural", "mediastinal", "pericardial"]

    if "pet/ct" in lower or "pet ct" in lower:
        return "FDG PET/CT", "Whole body"
    if re.search(r"\bct\b", lower) or "cat scan" in lower:
        if any(term in lower for term in abdominal_terms) and any(term in lower for term in chest_terms):
            return "CT chest/abdomen/pelvis", "Chest/abdomen/pelvis"
        if any(term in lower for term in abdominal_terms):
            return "CT abdomen/pelvis", "Abdomen/pelvis"
        if any(term in lower for term in chest_terms):
            return "CT chest", "Chest"
        return "CT scan", "Unspecified"
    if "mammogram" in lower or "mammography" in lower:
        return "Mammogram", "Breast"
    if "ultrasound" in lower or "sonogram" in lower:
        if any(term in lower for term in abdominal_terms):
            return "Abdominal ultrasound", "Abdomen"
        return "Breast ultrasound", "Breast"
    return "Breast MRI", "Breast"


def _best_imaging_sentence(message):
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|[;\n]", message) if part.strip()]
    terms = [
        "mass",
        "tumor",
        "lesion",
        "bi-rads",
        "birads",
        "cm",
        "decreased",
        "increased",
        "stable",
        "interval",
        "ascites",
        "free fluid",
        "peritoneal",
        "omental",
        "pleural",
        "hepatic",
        "liver",
        "metastatic",
        "metastasis",
        "metastases",
        "adenopathy",
    ]
    for sentence in sentences:
        if any(term in sentence.lower() for term in terms):
            return sentence
    return None


def _extract_number_after(lower_message, labels):
    for label in labels:
        match = re.search(rf"{re.escape(label)}\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)", lower_message)
        if match:
            return float(match.group(1))
    return None


def _clinical_lab_alerts(labs):
    checks = [
        {
            "rule": "very_low_wbc",
            "label": "WBC",
            "value": labs["wbc"],
            "threshold": "<2.0",
            "severity": "high",
            "triggered": labs["wbc"] < 2.0,
        },
        {
            "rule": "very_low_hemoglobin",
            "label": "hemoglobin",
            "value": labs["hemoglobin"],
            "threshold": "<8.0",
            "severity": "high",
            "triggered": labs["hemoglobin"] < 8.0,
        },
        {
            "rule": "very_low_platelets",
            "label": "platelets",
            "value": labs["platelets"],
            "threshold": "<50",
            "severity": "high",
            "triggered": labs["platelets"] < 50,
        },
    ]
    return [
        {key: value for key, value in item.items() if key != "triggered"}
        for item in checks
        if item["triggered"]
    ]


def _extract_medication(message):
    lower = message.lower()
    patterns = [
        r"(?:i am taking|i'm taking|taking|started|start taking|took)\s+([a-zA-Z0-9\- ]{3,40})",
        r"(?:medication|medicine|drug)\s*(?:is|:)?\s*([a-zA-Z0-9\- ]{3,40})",
    ]
    for pattern in patterns:
        match = re.search(pattern, lower)
        if not match:
            continue
        medication = _clean_medication_name(match.group(1))
        if medication:
            return {
                "medication": medication,
                "dose": _extract_dose(message),
                "frequency": _extract_frequency(lower),
            }
    return None


def _clean_medication_name(raw):
    cleaned = re.split(r"\b(?:and|but|because|for|since|with|today|yesterday|dose|mg|mcg|ml)\b|[,.]", raw, maxsplit=1)[0]
    cleaned = re.sub(r"\s+\d+(?:\.\d+)?\s*$", "", cleaned)
    cleaned = cleaned.strip(" .,:;-")
    if cleaned in {"my medication", "medicine", "medication"}:
        return None
    return cleaned[:80] if cleaned else None


def _extract_dose(message):
    match = re.search(r"(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|units?)", message, flags=re.IGNORECASE)
    return match.group(0) if match else None


def _extract_frequency(lower_message):
    for phrase in ["twice a day", "once a day", "daily", "weekly", "every week", "every night", "every morning"]:
        if phrase in lower_message:
            return phrase
    return None


def _extract_date(message):
    lower = message.lower()
    iso_match = re.search(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", lower)
    if iso_match:
        year, month, day = [int(part) for part in iso_match.groups()]
        return date(year, month, day)
    slash_match = re.search(r"\b(\d{1,2})/(\d{1,2})/(20\d{2})\b", lower)
    if slash_match:
        month, day, year = [int(part) for part in slash_match.groups()]
        return date(year, month, day)
    if "yesterday" in lower:
        return date.today() - timedelta(days=1)

    month_match = re.search(
        r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
        r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s*"
        r"(\d{1,2})(?:st|nd|rd|th)?(?:,?\s*(20\d{2}))?\b",
        lower,
    )
    if month_match:
        month_name, day_text, year_text = month_match.groups()
        month_lookup = {
            "jan": 1, "january": 1,
            "feb": 2, "february": 2,
            "mar": 3, "march": 3,
            "apr": 4, "april": 4,
            "may": 5,
            "jun": 6, "june": 6,
            "jul": 7, "july": 7,
            "aug": 8, "august": 8,
            "sep": 9, "sept": 9, "september": 9,
            "oct": 10, "october": 10,
            "nov": 11, "november": 11,
            "dec": 12, "december": 12,
        }
        year = int(year_text) if year_text else date.today().year
        return date(year, month_lookup[month_name], int(day_text))
    return date.today()


def _detect_urgent_flags(message):
    normalized = normalize_security_text(message)
    flags = [term for term in URGENT_TERMS if term in normalized]
    danger = detect_multilingual_medical_danger(message)
    flags.extend(danger.get("matches") or [])
    return sorted(set(flags))


def _prefer_deterministic_reply(message):
    lower = message.lower()
    deterministic_terms = [
        "last 14",
        "last fourteen",
        "what changed",
        "timeline",
        "tumor board",
        "toxicity",
        "cycle 2",
        "score",
        "model",
        "why",
    ]
    return any(term in lower for term in deterministic_terms)


def _build_response(message, actions, urgent_flags, patient_context):
    if not actions and not urgent_flags and _is_conversational_prompt(message):
        return _conversation_reply(patient_context)

    parts = []
    if urgent_flags:
        parts.append(
            "I noticed possible urgent wording. If symptoms feel severe, sudden, or unsafe, contact your oncology team or local emergency services now."
        )

    saved = [action for action in actions if action["type"].startswith("saved_")]
    failed = [action for action in actions if action["type"].endswith("_save_failed")]
    partial_actions = [action for action in actions if action["type"].startswith("partial_")]
    if saved:
        labels = []
        for action in saved:
            if action["type"] == "saved_symptom":
                labels.append(f"symptom: {action['symptom']} severity {action['severity']}/10")
            elif action["type"] == "saved_labs":
                labels.append("CBC values")
            elif action["type"] == "saved_medication":
                labels.append(f"medication: {action['medication']}")
            elif action["type"] == "saved_imaging_report":
                labels.append(f"{action['modality']} report from {action['date']}")
        parts.append("I saved this to your patient record: " + "; ".join(labels) + ".")
    if failed:
        parts.extend(action["message"] for action in failed if action.get("message"))
    if not saved and not failed:
        if partial_actions:
            parts.extend(action["message"] for action in partial_actions if action.get("message"))
        else:
            contextual = _contextual_reply(message, patient_context)
            parts.append(contextual or "I heard you. I can chat normally, answer low-risk education questions with sources when needed, or help log symptoms, CBC values, medications, and imaging report text.")

    partial_labs = [action for action in actions if action["type"] == "partial_labs_detected"]
    if partial_labs and saved:
        parts.append(partial_labs[0]["message"])

    partial_imaging = [action for action in actions if action["type"] == "partial_imaging_detected"]
    if partial_imaging and saved:
        parts.append(partial_imaging[0]["message"])

    lab_alerts = [action for action in actions if action["type"] == "clinical_rule_alert"]
    if lab_alerts:
        labels = [
            f"{alert['label']} {alert['value']} ({alert['severity']}, threshold {alert['threshold']})"
            for alert in lab_alerts[0]["alerts"]
        ]
        parts.append(
            "A deterministic CBC safety rule flagged this for clinician review: "
            + "; ".join(labels)
            + ". Please contact your oncology care team for medical guidance."
        )

    imaging_alerts = [action for action in actions if action["type"] == "possible_metastatic_indicator"]
    if imaging_alerts:
        sites = ", ".join(imaging_alerts[0].get("sites") or ["unspecified"])
        parts.append(
            "The imaging text includes wording that may need clinician review "
            f"({sites}). I am only logging the report text and cannot diagnose metastasis."
        )

    parts.append(
        "I can help track what you are feeling and summarize it for review, but I cannot diagnose or decide treatment."
    )
    return " ".join(parts)


def _is_small_talk(message):
    cleaned = re.sub(r"[^a-z0-9\s]", " ", message.lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    small_talk = {
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "kumusta",
        "kamusta",
        "thanks",
        "thank you",
        "salamat",
    }
    return cleaned in small_talk or cleaned.startswith(("hi ", "hello ", "hey "))


def _is_conversational_prompt(message):
    return _is_small_talk(message) or _is_identity_or_capability_question(message) or _is_social_checkin(message)


def _is_identity_or_capability_question(message):
    cleaned = re.sub(r"[^a-z0-9\s]", " ", message.lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    patterns = [
        "who are you",
        "what are you",
        "what can you do",
        "what do you do",
        "how can you help",
        "help me",
        "can you help",
        "are you a doctor",
        "are you ai",
        "are you an ai",
    ]
    return any(pattern in cleaned for pattern in patterns)


def _is_social_checkin(message):
    cleaned = re.sub(r"[^a-z0-9\s]", " ", message.lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    patterns = [
        "how are you",
        "how are u",
        "how you doing",
        "how are you doing",
        "are you ok",
        "what s up",
        "whats up",
    ]
    return any(pattern in cleaned for pattern in patterns)


def _conversation_reply(patient_context):
    memory_hint = _latest_memory_hint(patient_context)
    base = (
        "I am the portal support agent for this breast cancer monitoring demo. "
        "I can chat normally, remember recent patient-scoped messages, log symptoms, save complete CBC values, "
        "record medications, and save report-like MRI/imaging notes for clinician review."
    )
    if memory_hint:
        base += f" I can also refer back to recent portal context, like {memory_hint}."
    base += " I cannot diagnose or choose treatment."
    return base


def _should_use_llm_direct_reply(intent, safety, actions, urgent_flags):
    if urgent_flags or safety.get("level") == "high_risk":
        return False
    if intent == "data_entry_confirmation":
        return bool(actions)
    return intent in {
        "conversation",
        "patient_memory",
        "patient_timeline_monitoring",
        "general_support",
        "emotional_support",
    }


def _generate_llm_response(message, actions, urgent_flags, patient_context, fallback_response):
    if not get_llm_adjudication_enabled():
        return fallback_response
    api_key = get_groq_api_key()
    if not api_key:
        return fallback_response

    user_prompt = {
        "patient_message": message,
        "saved_actions": actions,
        "urgent_flags": urgent_flags,
        "recent_context": patient_context,
        "fallback_reply": fallback_response,
    }
    try:
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=get_groq_model(),
            temperature=0.2,
            max_tokens=220,
            messages=[
                {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_prompt, default=str)},
            ],
        )
        reply = completion.choices[0].message.content.strip()
    except Exception:
        return fallback_response

    if not reply:
        return fallback_response
    if urgent_flags and "emergency" not in reply.lower() and "oncology" not in reply.lower():
        return fallback_response
    # Honest-save guard: the LLM is allowed to rephrase the deterministic
    # fallback, but it must never claim to have logged/saved/recorded something
    # when no save action actually succeeded.  If the LLM hallucinates a save,
    # discard its output and return the deterministic reply.
    saved_count = sum(1 for action in actions if str(action.get("type", "")).startswith("saved_"))
    if saved_count == 0:
        lower_reply = reply.lower()
        save_claim_terms = (
            "i logged", "i've logged", "i have logged",
            "i saved", "i've saved", "i have saved",
            "i recorded", "i've recorded", "i have recorded",
            "i added", "i've added", "i have added",
            "added to your record", "saved to your record", "logged to your record",
        )
        if any(term in lower_reply for term in save_claim_terms):
            return fallback_response
    return reply


def _recent_patient_context(db, patient_id):
    lab_rows = (
        db.query(LabResult)
        .filter(LabResult.patient_id == patient_id)
        .order_by(LabResult.date.desc(), LabResult.id.desc())
        .limit(50)
        .all()
    )
    latest_lab = (
        lab_rows[0] if lab_rows else None
    )
    symptoms = (
        db.query(SymptomReport)
        .filter(SymptomReport.patient_id == patient_id)
        .order_by(SymptomReport.date.desc(), SymptomReport.id.desc())
        .limit(3)
        .all()
    )
    medications = (
        db.query(MedicationLog)
        .filter(MedicationLog.patient_id == patient_id)
        .order_by(MedicationLog.date.desc(), MedicationLog.id.desc())
        .limit(3)
        .all()
    )
    treatments = (
        db.query(Treatment)
        .filter(Treatment.patient_id == patient_id)
        .order_by(Treatment.date.desc(), Treatment.id.desc())
        .limit(6)
        .all()
    )
    imaging_reports = (
        db.query(ImagingReport)
        .filter(ImagingReport.patient_id == patient_id)
        .order_by(ImagingReport.date.desc(), ImagingReport.id.desc())
        .limit(3)
        .all()
    )
    chat_messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.patient_id == patient_id)
        .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
        .limit(8)
        .all()
    )
    interventions = (
        db.query(ClinicalIntervention)
        .filter(ClinicalIntervention.patient_id == patient_id)
        .order_by(ClinicalIntervention.date.desc(), ClinicalIntervention.id.desc())
        .limit(4)
        .all()
    )
    outcome = (
        db.query(TreatmentOutcome)
        .filter(TreatmentOutcome.patient_id == patient_id)
        .first()
    )
    synthetic_prediction, synthetic_xai = _synthetic_model_context(patient_id)
    timeline_context = _timeline_context(
        lab_rows=lab_rows,
        symptoms=symptoms,
        treatments=treatments,
        imaging_reports=imaging_reports,
        interventions=interventions,
        outcome=outcome,
        synthetic_prediction=synthetic_prediction,
    )

    return {
        "latest_lab": {
            "date": latest_lab.date,
            "wbc": latest_lab.wbc,
            "hemoglobin": latest_lab.hemoglobin,
            "platelets": latest_lab.platelets,
            "source": latest_lab.source,
        } if latest_lab else None,
        "recent_symptoms": [
            {"date": row.date, "symptom": row.symptom, "severity": row.severity}
            for row in symptoms
        ],
        "recent_medications": [
            {"date": row.date, "medication": row.medication, "dose": row.dose, "frequency": row.frequency}
            for row in medications
        ],
        "recent_treatments": [
            {"date": row.date, "cycle": row.cycle, "drug": row.drug}
            for row in treatments
        ],
        "recent_imaging": [
            {
                "date": row.date,
                "modality": row.modality,
                "report_type": row.report_type,
                "impression": row.impression[:240],
            }
            for row in imaging_reports
        ],
        "recent_chat": [
            {
                "role": row.role,
                "message": row.message[:240],
                "created_at": row.created_at,
            }
            for row in reversed(chat_messages)
        ],
        "recent_interventions": [
            {
                "date": row.date,
                "type": row.intervention_type,
                "reason": row.reason,
                "medication_or_product": row.medication_or_product,
            }
            for row in interventions
        ],
        "treatment_outcome": {
            "assessment_date": outcome.assessment_date,
            "response_category": outcome.response_category,
            "cancer_status": outcome.cancer_status,
            "maintenance_plan": outcome.maintenance_plan,
        } if outcome else None,
        "synthetic_model_prediction": synthetic_prediction,
        "synthetic_model_explanation": synthetic_xai,
        "timeline_context": timeline_context,
    }


def _contextual_reply(message, context):
    lower = message.lower()
    status_terms = ["how am i", "how is my treatment", "am i improving", "working", "progress", "score"]
    explain_terms = ["why", "explain", "factor", "contribute", "model"]
    doctor_terms = ["doctor", "oncologist", "tell them", "bring", "ask"]
    timeline_terms = ["last 14", "last fourteen", "what changed", "timeline", "tumor board", "toxicity", "cycle 2"]
    memory_terms = ["remember", "what did i tell", "what did i say", "last message", "previous message", "chat history"]

    if any(term in lower for term in memory_terms):
        return _memory_reply(message, context)

    if any(term in lower for term in timeline_terms):
        timeline = context.get("timeline_context") or {}
        if "tumor board" in lower:
            return timeline.get("tumor_board_brief")
        if "toxicity" in lower or "cycle 2" in lower:
            return timeline.get("toxicity_summary")
        if "last 14" in lower or "last fourteen" in lower or "what changed" in lower:
            changes = timeline.get("last_14_day_changes") or []
            if changes:
                return "In the last represented 14 days: " + "; ".join(changes[:5]) + "."
            return "I do not see timeline events in the last represented 14-day window. This may mean the record is missing recent updates."

    if any(term in lower for term in status_terms):
        prediction = context.get("synthetic_model_prediction") or {}
        outcome = context.get("treatment_outcome")
        latest_lab = context.get("latest_lab")
        recent_imaging = context.get("recent_imaging") or []
        probability = prediction.get("logistic_regression_probability")
        if probability is None:
            probability = prediction.get("random_forest_probability")
        if probability is not None:
            percent = round(float(probability) * 100, 1)
            lab_text = (
                f"WBC {latest_lab.get('wbc')}, hemoglobin {latest_lab.get('hemoglobin')}, platelets {latest_lab.get('platelets')}"
                if latest_lab else "not available"
            )
            return (
                f"Based on the demo model, your synthetic treatment-response score is {percent}%. "
                "This is a simulator-trained tracking signal, not a clinical prediction. "
                f"Latest CBC: {lab_text}. "
                f"Latest imaging note: {recent_imaging[0].get('impression') if recent_imaging else 'not available'}."
            )
        if outcome:
            return (
                f"Your record lists final response as {outcome.get('response_category')} with status {outcome.get('cancer_status')}. "
                "Use this as a portal summary only and confirm meaning with your oncology team."
            )

    if any(term in lower for term in explain_terms):
        xai = context.get("synthetic_model_explanation") or {}
        positives = xai.get("positive_contributions") or []
        negatives = xai.get("negative_contributions") or []
        if positives or negatives:
            toward = ", ".join(item["feature"] for item in positives[:3]) or "none"
            away = ", ".join(item["feature"] for item in negatives[:3]) or "none"
            return (
                f"The demo explanation says features pushing toward response include: {toward}. "
                f"Features pushing away include: {away}. "
                "These explain model behavior on synthetic data, not medical causality."
            )

    if any(term in lower for term in doctor_terms):
        latest_lab = context.get("latest_lab")
        interventions = context.get("recent_interventions") or []
        symptoms = context.get("recent_symptoms") or []
        items = []
        if latest_lab:
            items.append(f"latest CBC WBC {latest_lab.get('wbc')}, hemoglobin {latest_lab.get('hemoglobin')}, platelets {latest_lab.get('platelets')}")
        if symptoms:
            items.append(f"recent symptom {symptoms[0].get('symptom')} severity {symptoms[0].get('severity')}/10")
        if interventions:
            items.append(f"recent support intervention: {interventions[0].get('type')}")
        if items:
            return "For your care team, bring up " + "; ".join(items) + "."

    return None


def _memory_reply(message, context):
    current = message.strip().lower()
    recent_chat = context.get("recent_chat") or []
    previous_user_messages = []
    seen = set()
    for item in recent_chat:
        if item.get("role") != "user":
            continue
        text = item.get("message", "").strip()
        key = re.sub(r"\s+", " ", text.lower())
        if not text or key == current or key in seen or _is_small_talk(text):
            continue
        seen.add(key)
        previous_user_messages.append(text)
    if previous_user_messages:
        snippets = [text.strip() for text in previous_user_messages[-3:] if text.strip()]
        return (
            "From this chat, the recent things you told me were: "
            + " | ".join(snippets)
            + ". I use this only as portal memory for tracking and review, not diagnosis."
        )

    hint = _latest_memory_hint(context)
    if hint:
        return f"I do not see earlier chat notes yet, but your portal context includes {hint}. This is for tracking and clinician review only."
    return "I do not see earlier chat details for this patient yet. Tell me symptoms, CBC values, medications, or imaging report text and I can help log them."


def _latest_memory_hint(context):
    latest_lab = context.get("latest_lab")
    symptoms = context.get("recent_symptoms") or []
    imaging = context.get("recent_imaging") or []
    medications = context.get("recent_medications") or []
    if latest_lab:
        return f"latest CBC WBC {latest_lab.get('wbc')}, hemoglobin {latest_lab.get('hemoglobin')}, platelets {latest_lab.get('platelets')}"
    if symptoms:
        return f"recent symptom {symptoms[0].get('symptom')} severity {symptoms[0].get('severity')}/10"
    if imaging:
        return f"recent imaging note: {imaging[0].get('impression')}"
    if medications:
        return f"recent medication: {medications[0].get('medication')}"
    return None


def _synthetic_model_context(patient_id):
    try:
        from backend.services.complete_synthetic_xai import (
            load_complete_synthetic_patient_prediction,
            load_complete_synthetic_patient_xai,
        )

        return (
            load_complete_synthetic_patient_prediction(patient_id),
            load_complete_synthetic_patient_xai(patient_id),
        )
    except Exception:
        return None, None


def _timeline_context(lab_rows, symptoms, treatments, imaging_reports, interventions, outcome, synthetic_prediction):
    events = []
    for row in lab_rows:
        events.append((row.date, f"CBC WBC {row.wbc}, hemoglobin {row.hemoglobin}, platelets {row.platelets}"))
    for row in symptoms:
        events.append((row.date, f"symptom {row.symptom} severity {row.severity}/10"))
    for row in treatments:
        events.append((row.date, f"treatment cycle {row.cycle}: {row.drug}"))
    for row in imaging_reports:
        events.append((row.date, f"{row.modality} impression: {row.impression[:100]}"))
    for row in interventions:
        events.append((row.date, f"support intervention {row.intervention_type}: {row.reason[:90]}"))

    last_14 = _last_14_day_changes(events)
    toxicity = _chat_toxicity_summary(lab_rows, symptoms)
    probability = None
    if synthetic_prediction:
        probability = synthetic_prediction.get("logistic_regression_probability") or synthetic_prediction.get("gradient_boosting_probability")
    probability_text = f"Demo response probability {round(float(probability) * 100, 1)}%. " if probability is not None else ""
    outcome_text = (
        f"Recorded outcome {outcome.response_category} / {outcome.cancer_status}. "
        if outcome else ""
    )
    tumor_board = (
        f"{probability_text}{toxicity} "
        f"Recent changes: {'; '.join(last_14[:4]) if last_14 else 'no recent timeline events represented'}. "
        f"{outcome_text}For clinician review only; this is not diagnosis or treatment advice."
    )
    return {
        "last_14_day_changes": last_14,
        "toxicity_summary": toxicity,
        "tumor_board_brief": tumor_board,
    }


def _last_14_day_changes(events):
    if not events:
        return []
    latest = max(date_value for date_value, _ in events)
    start = latest - timedelta(days=14)
    return [
        text for date_value, text in sorted(events, key=lambda item: item[0], reverse=True)
        if date_value >= start
    ]


def _chat_toxicity_summary(lab_rows, symptoms):
    if not lab_rows:
        return "CBC toxicity trend is unavailable because no CBC rows are present."
    sorted_labs = sorted(lab_rows, key=lambda row: row.date)
    early = sorted_labs[:max(1, len(sorted_labs) // 2)]
    late = sorted_labs[max(1, len(sorted_labs) // 2):] or sorted_labs[-1:]
    early_min_wbc = min(row.wbc for row in early)
    late_min_wbc = min(row.wbc for row in late)
    late_min_platelets = min(row.platelets for row in late)
    high_symptoms = [row for row in symptoms if row.severity >= 7]
    if late_min_wbc < early_min_wbc * 0.8 or late_min_platelets < 75 or high_symptoms:
        return (
            f"CBC/symptom toxicity needs review: late minimum WBC {round(late_min_wbc, 2)}, "
            f"late minimum platelets {round(late_min_platelets, 1)}, high symptom reports {len(high_symptoms)}."
        )
    return (
        f"CBC toxicity does not look worse in the latest represented window: "
        f"late minimum WBC {round(late_min_wbc, 2)}, late minimum platelets {round(late_min_platelets, 1)}."
    )
