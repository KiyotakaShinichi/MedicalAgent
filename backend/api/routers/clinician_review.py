"""Clinician review endpoints.

This router owns the clinician-facing review surface that previously lived
inside ``patient.py``:

- POST /patients/{patient_id}/timeline-question — ask a structured question
  against the patient's evidence timeline.
- POST /patients/{patient_id}/summary-review — record a clinician decision on
  an AI-generated summary (approve / edit / reject / unsafe / etc.).
- GET  /summary-reviews — list previously recorded reviews.
- GET  /clinician/review-queue — priority-sorted patient queue for review.

All route paths are preserved exactly as they were on the patient router so
existing clients continue to work. Splitting these endpoints into their own
router keeps clinician-review-as-audit-data cleanly separated from
patient-self-service code in ``patient.py``.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.api.deps import get_clinician_or_admin_context, get_db
from backend.crud import get_all_patients, get_patient
from backend.models import HighRiskConversationAlert
from backend.services.high_risk_conversation_alerts import build_alert_automation_status, serialize_alert
from backend.services.timeline_intelligence import answer_timeline_question


router = APIRouter(tags=["clinician-review"])


# ─── Request schemas ──────────────────────────────────────────────────────────


class TimelineQuestionRequest(BaseModel):
    question: str


class ClinicianSummaryReviewRequest(BaseModel):
    decision: str
    clinician_notes: str | None = None
    edited_patient_summary: str | None = None
    explanation_quality_score: int | None = None
    model_usefulness_score: int | None = None
    review_target: str | None = None
    reason_category: str | None = None
    model_version: str | None = None
    rag_version: str | None = None


class AlertAcknowledgementRequest(BaseModel):
    note: str | None = None


# ─── Endpoints ───────────────────────────────────────────────────────────────


@router.post("/patients/{patient_id}/timeline-question")
def answer_patient_timeline_question_endpoint(
    patient_id: str,
    payload: TimelineQuestionRequest,
    context=Depends(get_clinician_or_admin_context),
    db: Session = Depends(get_db),
):
    # Lazy import so this router does not pull in the heavy patient_report
    # builder at module load time.
    from backend.api.routers.patient import build_patient_report_response

    if not get_patient(db, patient_id):
        raise HTTPException(status_code=404, detail="Patient not found")
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")
    report = build_patient_report_response(patient_id, db)
    return answer_timeline_question(report, payload.question)


@router.post("/patients/{patient_id}/summary-review")
def create_patient_summary_review_endpoint(
    patient_id: str,
    payload: ClinicianSummaryReviewRequest,
    context=Depends(get_clinician_or_admin_context),
    db: Session = Depends(get_db),
):
    from backend.api.routers.patient import build_patient_report_response
    from backend.services.clinician_feedback import create_clinical_summary_review

    if not get_patient(db, patient_id):
        raise HTTPException(status_code=404, detail="Patient not found")

    report = build_patient_report_response(patient_id, db)
    try:
        review = create_clinical_summary_review(
            db=db,
            patient_id=patient_id,
            reviewer_role=context.role,
            decision=payload.decision,
            summary_snapshot=report.get("patient_timeline_summary") or {},
            clinician_notes=payload.clinician_notes,
            edited_patient_summary=payload.edited_patient_summary,
            explanation_quality_score=payload.explanation_quality_score,
            model_usefulness_score=payload.model_usefulness_score,
            review_target=payload.review_target,
            reason_category=payload.reason_category,
            model_version=payload.model_version,
            rag_version=payload.rag_version,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "message": "Clinician summary review saved",
        "review": review,
        "safety_note": (
            "Review feedback is audit data. "
            "It does not change the patient record automatically."
        ),
    }


@router.get("/summary-reviews")
def list_summary_reviews_endpoint(
    patient_id: str | None = None,
    limit: int = 50,
    context=Depends(get_clinician_or_admin_context),
    db: Session = Depends(get_db),
):
    from backend.services.clinician_feedback import list_clinical_summary_reviews

    return {
        "summary_reviews": list_clinical_summary_reviews(
            db, patient_id=patient_id, limit=limit
        )
    }


@router.get("/clinician/high-risk-conversation-alerts")
def list_high_risk_conversation_alerts(
    status: str | None = None,
    limit: int = 50,
    context=Depends(get_clinician_or_admin_context),
    db: Session = Depends(get_db),
):
    query = db.query(HighRiskConversationAlert)
    if status:
        query = query.filter(HighRiskConversationAlert.status == status)
    rows = (
        query.order_by(
            HighRiskConversationAlert.created_at.desc(),
            HighRiskConversationAlert.id.desc(),
        )
        .limit(max(1, min(limit, 200)))
        .all()
    )
    return {
        "alerts": [serialize_alert(row) for row in rows],
        "count": len(rows),
        "clinical_validation": False,
        "safety_note": (
            "This is an engineering review queue, not a monitored emergency service. "
            "Notification status does not prove that a clinician saw or acted on an item."
        ),
    }


@router.get("/clinician/high-risk-conversation-alerts/status")
def get_high_risk_conversation_alert_status(
    context=Depends(get_clinician_or_admin_context),
    db: Session = Depends(get_db),
):
    return build_alert_automation_status(db)


@router.get("/clinician/high-risk-conversation-alerts/{alert_id}")
def get_high_risk_conversation_alert(
    alert_id: int,
    context=Depends(get_clinician_or_admin_context),
    db: Session = Depends(get_db),
):
    alert = db.query(HighRiskConversationAlert).filter(HighRiskConversationAlert.id == alert_id).first()
    if alert is None:
        raise HTTPException(status_code=404, detail="Review alert not found")
    return {
        "alert": serialize_alert(alert),
        "safety_note": (
            "Review the linked patient-scoped chat in NLCare. Do not infer a diagnosis or treatment action "
            "from the alert reason code alone."
        ),
    }


@router.post("/clinician/high-risk-conversation-alerts/{alert_id}/acknowledge")
def acknowledge_high_risk_conversation_alert(
    alert_id: int,
    payload: AlertAcknowledgementRequest,
    context=Depends(get_clinician_or_admin_context),
    db: Session = Depends(get_db),
):
    alert = db.query(HighRiskConversationAlert).filter(HighRiskConversationAlert.id == alert_id).first()
    if alert is None:
        raise HTTPException(status_code=404, detail="Review alert not found")
    alert.status = "acknowledged"
    alert.acknowledged_by_role = context.role
    alert.acknowledgement_note = (payload.note or "").strip()[:1000] or None
    alert.acknowledged_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(alert)
    return {
        "message": "Review alert acknowledged in NLCare.",
        "alert": serialize_alert(alert),
        "safety_note": "Acknowledgement is workflow audit data, not a clinical decision or patient-contact claim.",
    }


@router.get("/clinician/review-queue")
def clinician_review_queue_endpoint(
    limit: int = 25,
    context=Depends(get_clinician_or_admin_context),
    db: Session = Depends(get_db),
):
    from backend.api.routers.patient import build_patient_report_response

    patients = get_all_patients(db)[: max(1, min(limit, 100))]
    rows = []
    for patient in patients:
        report = build_patient_report_response(patient.id, db)
        assessment = report.get("multimodal_assessment") or {}
        summary = report.get("patient_timeline_summary") or {}
        intelligence = report.get("timeline_intelligence") or {}
        latest_review = report.get("latest_clinician_review")
        risks = report.get("risks") or []
        urgent_count = sum(1 for risk in risks if risk.get("severity") == "urgent_review")
        review_flags = summary.get("review_flags") or []
        missing = intelligence.get("missing_data_warnings") or []
        status = assessment.get("overall_status") or "unknown"
        priority_score = (
            urgent_count * 5
            + len(review_flags) * 2
            + (0 if latest_review else 2)
            + (2 if status in {"needs_clinician_review", "watch_closely"} else 0)
            + min(len(missing), 3)
        )
        rows.append(
            {
                "patient_id": patient.id,
                "patient_name": patient.name,
                "overall_status": status,
                "priority_score": priority_score,
                "urgent_flag_count": urgent_count,
                "review_flag_count": len(review_flags),
                "missing_data_count": len(missing),
                "latest_review_decision": (
                    latest_review.get("decision") if latest_review else None
                ),
                "headline": summary.get("headline"),
                "top_review_flags": review_flags[:3],
                "top_missing_data_warnings": missing[:3],
                "recommended_action": assessment.get("recommended_action"),
            }
        )

    rows = sorted(
        rows,
        key=lambda row: (row["priority_score"], row["urgent_flag_count"]),
        reverse=True,
    )
    return {
        "queue": rows,
        "safety_note": (
            "Review queue prioritizes monitoring signals for clinician attention. "
            "It does not diagnose or choose treatment."
        ),
    }


@router.get("/clinician/patients/{patient_id}/prediction-traces")
def clinician_patient_prediction_traces_endpoint(
    patient_id: str,
    limit: int = 25,
    abstained_only: bool = False,
    context=Depends(get_clinician_or_admin_context),
    db: Session = Depends(get_db),
):
    """Return the patient's prediction-trace history with provenance fields.

    Each row carries the same envelope the patient sees on their dashboard
    plus the model + feature-set + threshold + calibration provenance, so a
    clinician reviewing the patient can audit which model produced which
    decision and which modalities were available at the time.
    """
    from backend.services.prediction_trace import (
        list_recent_traces,
        summarise_traces,
    )

    if not get_patient(db, patient_id):
        raise HTTPException(status_code=404, detail="Patient not found")

    traces = list_recent_traces(
        db,
        limit=limit,
        patient_id=patient_id,
        abstained_only=abstained_only,
    )
    # Filter the global summary down to just this patient so the dashboard's
    # headline tiles describe the patient under review, not the cohort.
    from backend.models import PredictionTrace
    patient_summary_base = (
        db.query(PredictionTrace)
        .filter(PredictionTrace.patient_id == patient_id)
        .order_by(PredictionTrace.created_at.desc(), PredictionTrace.id.desc())
        .limit(500)
        .all()
    )
    abstained = sum(1 for row in patient_summary_base if row.abstained)
    decisions: dict[str, int] = {}
    for row in patient_summary_base:
        decisions[row.decision] = decisions.get(row.decision, 0) + 1

    return {
        "patient_id": patient_id,
        "traces": traces,
        "patient_summary": {
            "total": len(patient_summary_base),
            "abstention_rate": (
                round(abstained / len(patient_summary_base), 4)
                if patient_summary_base else None
            ),
            "decision_counts": decisions,
        },
        # Cohort-level context — useful when a reviewer wants to know
        # whether this patient's abstention rate is unusual.
        "cohort_summary": summarise_traces(db),
        "claim_boundary": (
            "Engineering trace data only.  Each row records what the model "
            "decided and why, not whether the decision was clinically correct."
        ),
    }
