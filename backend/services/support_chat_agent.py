import json
import re
from datetime import date, timedelta

from groq import Groq

from backend.config import get_groq_api_key
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

    patient_context = _recent_patient_context(db, patient_id)
    fallback_response = _build_response(normalized, actions, urgent_flags, patient_context)
    response = _generate_llm_response(
        message=normalized,
        actions=actions,
        urgent_flags=urgent_flags,
        patient_context=patient_context,
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


def _build_response(message, actions, urgent_flags, patient_context):
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
        contextual = _contextual_reply(message, patient_context)
        parts.append(contextual or "I saved your message in the chat history.")

    partial_labs = [action for action in actions if action["type"] == "partial_labs_detected"]
    if partial_labs:
        parts.append(partial_labs[0]["message"])

    parts.append(
        "I can help track what you are feeling and summarize it for review, but I cannot diagnose or decide treatment."
    )
    return " ".join(parts)


def _generate_llm_response(message, actions, urgent_flags, patient_context, fallback_response):
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
    }


def _contextual_reply(message, context):
    lower = message.lower()
    status_terms = ["how am i", "how is my treatment", "am i improving", "working", "progress", "score"]
    explain_terms = ["why", "explain", "factor", "contribute", "model"]
    doctor_terms = ["doctor", "oncologist", "tell them", "bring", "ask"]

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
