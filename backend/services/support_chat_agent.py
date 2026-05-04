import json
import re
from datetime import date, timedelta

from groq import Groq

from backend.config import get_groq_api_key
from backend.models import ChatMessage, LabResult, MedicationLog, SymptomReport


SYMPTOM_KEYWORDS = {
    "fatigue": ["fatigue", "tired", "weak", "exhausted"],
    "nausea": ["nausea", "nauseous", "vomit", "vomiting"],
    "pain": ["pain", "ache", "aching", "sore"],
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
- Explain what was logged, ask for missing tracking details when useful, and encourage oncology team review for concerning symptoms.
- If urgent wording is present, advise contacting the oncology team or emergency services now.
- Keep the tone calm and practical. Maximum 120 words. Plain text only.
"""


def handle_patient_chat(db, patient_id, message):
    normalized = message.strip()
    if not normalized:
        raise ValueError("Message cannot be empty")

    user_record = ChatMessage(
        patient_id=patient_id,
        role="user",
        message=normalized,
        intent="patient_support",
    )
    db.add(user_record)

    actions = []
    urgent_flags = _detect_urgent_flags(normalized)

    symptom = _extract_symptom(normalized)
    if symptom:
        db.add(SymptomReport(
            patient_id=patient_id,
            date=_extract_date(normalized),
            symptom=symptom["symptom"],
            severity=symptom["severity"],
            notes=f"Captured from support chat: {normalized}",
        ))
        actions.append({
            "type": "saved_symptom",
            "symptom": symptom["symptom"],
            "severity": symptom["severity"],
        })

    labs = _extract_complete_labs(normalized)
    if labs:
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
    elif _looks_like_partial_labs(normalized):
        actions.append({
            "type": "partial_labs_detected",
            "message": "I saw lab information, but I need WBC, hemoglobin, and platelets together to save a CBC row.",
        })

    medication = _extract_medication(normalized)
    if medication:
        db.add(MedicationLog(
            patient_id=patient_id,
            date=_extract_date(normalized),
            medication=medication["medication"],
            dose=medication.get("dose"),
            frequency=medication.get("frequency"),
            notes=f"Captured from support chat: {normalized}",
        ))
        actions.append({"type": "saved_medication", **medication})

    fallback_response = _build_response(actions, urgent_flags)
    response = _generate_llm_response(
        db=db,
        patient_id=patient_id,
        message=normalized,
        actions=actions,
        urgent_flags=urgent_flags,
        fallback_response=fallback_response,
    )
    db.add(ChatMessage(
        patient_id=patient_id,
        role="assistant",
        message=response,
        intent="patient_support_response",
        saved_actions_json=json.dumps(actions),
    ))
    db.commit()

    return {
        "reply": response,
        "saved_actions": actions,
        "urgent_flags": urgent_flags,
        "safety_note": "This assistant logs and summarizes information only. It does not diagnose or give treatment instructions.",
    }


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
        "severity": severity if severity is not None else 5,
    }


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


def _extract_number_after(lower_message, labels):
    for label in labels:
        match = re.search(rf"{re.escape(label)}\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)", lower_message)
        if match:
            return float(match.group(1))
    return None


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
    if "yesterday" in lower:
        return date.today() - timedelta(days=1)
    return date.today()


def _detect_urgent_flags(message):
    lower = message.lower()
    return [term for term in URGENT_TERMS if term in lower]


def _build_response(actions, urgent_flags):
    parts = []
    if urgent_flags:
        parts.append(
            "I noticed possible urgent wording. If symptoms feel severe, sudden, or unsafe, contact your oncology team or local emergency services now."
        )

    saved = [action for action in actions if action["type"].startswith("saved_")]
    if saved:
        labels = []
        for action in saved:
            if action["type"] == "saved_symptom":
                labels.append(f"symptom: {action['symptom']} severity {action['severity']}/10")
            elif action["type"] == "saved_labs":
                labels.append("CBC values")
            elif action["type"] == "saved_medication":
                labels.append(f"medication: {action['medication']}")
        parts.append("I saved this to your patient record: " + "; ".join(labels) + ".")
    else:
        parts.append("I saved your message in the chat history.")

    partial_labs = [action for action in actions if action["type"] == "partial_labs_detected"]
    if partial_labs:
        parts.append(partial_labs[0]["message"])

    parts.append(
        "I can help track what you are feeling and summarize it for review, but I cannot diagnose or decide treatment."
    )
    return " ".join(parts)


def _generate_llm_response(db, patient_id, message, actions, urgent_flags, fallback_response):
    api_key = get_groq_api_key()
    if not api_key:
        return fallback_response

    context = _recent_patient_context(db, patient_id)
    user_prompt = {
        "patient_message": message,
        "saved_actions": actions,
        "urgent_flags": urgent_flags,
        "recent_context": context,
        "fallback_reply": fallback_response,
    }
    try:
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
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
    return reply


def _recent_patient_context(db, patient_id):
    latest_lab = (
        db.query(LabResult)
        .filter(LabResult.patient_id == patient_id)
        .order_by(LabResult.date.desc(), LabResult.id.desc())
        .first()
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
    }
