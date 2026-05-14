import json

from backend.models import ClinicalSummaryReview
from backend.services.review_constants import REVIEW_DECISIONS, REVIEW_TARGETS


VALID_REVIEW_DECISIONS = set(REVIEW_DECISIONS)
VALID_REVIEW_TARGETS = set(REVIEW_TARGETS)


def create_clinical_summary_review(
    db,
    patient_id,
    reviewer_role,
    decision,
    summary_snapshot,
    clinician_notes=None,
    edited_patient_summary=None,
    explanation_quality_score=None,
    model_usefulness_score=None,
    review_target=None,
    reason_category=None,
    model_version=None,
    rag_version=None,
):
    normalized = decision.lower().strip()
    if normalized not in VALID_REVIEW_DECISIONS:
        raise ValueError(f"decision must be one of {sorted(VALID_REVIEW_DECISIONS)}")
    target = (review_target or "summary").lower().strip()
    if target not in VALID_REVIEW_TARGETS:
        raise ValueError(f"review_target must be one of {sorted(VALID_REVIEW_TARGETS)}")
    _validate_score(explanation_quality_score, "explanation_quality_score")
    _validate_score(model_usefulness_score, "model_usefulness_score")

    row = ClinicalSummaryReview(
        patient_id=patient_id,
        reviewer_role=reviewer_role,
        decision=normalized,
        review_target=target,
        reason_category=reason_category,
        clinician_notes=clinician_notes,
        edited_patient_summary=edited_patient_summary,
        summary_snapshot_json=json.dumps(summary_snapshot, default=str),
        model_version=model_version,
        rag_version=rag_version,
        explanation_quality_score=explanation_quality_score,
        model_usefulness_score=model_usefulness_score,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return _review_to_dict(row)


def list_clinical_summary_reviews(db, patient_id=None, limit=50):
    query = db.query(ClinicalSummaryReview)
    if patient_id:
        query = query.filter(ClinicalSummaryReview.patient_id == patient_id)
    rows = (
        query.order_by(ClinicalSummaryReview.created_at.desc(), ClinicalSummaryReview.id.desc())
        .limit(max(1, min(limit, 200)))
        .all()
    )
    return [_review_to_dict(row) for row in rows]


def latest_clinical_summary_review(db, patient_id):
    row = (
        db.query(ClinicalSummaryReview)
        .filter(ClinicalSummaryReview.patient_id == patient_id)
        .order_by(ClinicalSummaryReview.created_at.desc(), ClinicalSummaryReview.id.desc())
        .first()
    )
    return _review_to_dict(row) if row else None


def clinical_feedback_summary(db):
    rows = db.query(ClinicalSummaryReview).all()
    if not rows:
        return {
            "review_count": 0,
            "decision_counts": {},
            "reason_category_counts": {},
            "review_target_counts": {},
            "average_explanation_quality_score": None,
            "average_model_usefulness_score": None,
        }

    decision_counts = {}
    reason_counts = {}
    target_counts = {}
    explanation_scores = []
    usefulness_scores = []
    for row in rows:
        decision_counts[row.decision] = decision_counts.get(row.decision, 0) + 1
        if getattr(row, "reason_category", None):
            reason_counts[row.reason_category] = reason_counts.get(row.reason_category, 0) + 1
        target_key = getattr(row, "review_target", None) or "summary"
        target_counts[target_key] = target_counts.get(target_key, 0) + 1
        if row.explanation_quality_score is not None:
            explanation_scores.append(row.explanation_quality_score)
        if row.model_usefulness_score is not None:
            usefulness_scores.append(row.model_usefulness_score)

    return {
        "review_count": len(rows),
        "decision_counts": decision_counts,
        "reason_category_counts": reason_counts,
        "review_target_counts": target_counts,
        "average_explanation_quality_score": _mean(explanation_scores),
        "average_model_usefulness_score": _mean(usefulness_scores),
    }


def _validate_score(value, name):
    if value is None:
        return
    if value < 1 or value > 5:
        raise ValueError(f"{name} must be between 1 and 5")


def _mean(values):
    if not values:
        return None
    return round(sum(values) / len(values), 2)


def _review_to_dict(row):
    return {
        "id": row.id,
        "patient_id": row.patient_id,
        "reviewer_role": row.reviewer_role,
        "decision": row.decision,
        "review_target": getattr(row, "review_target", None),
        "reason_category": getattr(row, "reason_category", None),
        "clinician_notes": row.clinician_notes,
        "edited_patient_summary": row.edited_patient_summary,
        "summary_snapshot": json.loads(row.summary_snapshot_json),
        "model_version": getattr(row, "model_version", None),
        "rag_version": getattr(row, "rag_version", None),
        "explanation_quality_score": row.explanation_quality_score,
        "model_usefulness_score": row.model_usefulness_score,
        "created_at": str(row.created_at),
    }
