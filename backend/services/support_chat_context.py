from __future__ import annotations

from datetime import timedelta

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
