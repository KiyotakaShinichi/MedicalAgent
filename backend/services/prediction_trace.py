"""Prediction traceability service.

Every evidence-aware prediction the system makes can opt into writing a
`PredictionTrace` row that captures *everything a reviewer would need* to
reconstruct the call: which question was asked, which model+config produced
the answer, which modalities were present, what the abstention layer decided,
whether the validator allowed it, and which RAG sources (if any) the chat
layer was using when the prediction fired.

The trace table is the single audit surface a clinician or admin can query
to answer: "what did the system tell this patient, when, under which model
version, and with what evidence?"  It is engineering provenance — not a
clinical record.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from sqlalchemy.orm import Session

from backend.models import PredictionTrace
from backend.services.predict_with_abstention import (
    DEFAULT_CALIBRATOR_PATH,
    DEFAULT_MODEL_PATH,
    EvidenceAwarePrediction,
    LOWER_DECISION_THRESHOLD,
    UPPER_DECISION_THRESHOLD,
    predict_with_abstention,
)


# Feature-set version is a constant for now — bump it when NUMERIC_FEATURES
# or CATEGORICAL_FEATURES changes in complete_synthetic_training.
FEATURE_SET_VERSION = "synthetic_v2_2026_05"


@dataclass
class TraceContext:
    """Optional context the caller can attach to a trace row.  None of these
    are required — abstention decisions can be recorded without any of them
    — but they are how the trace links back to chat sessions, clinicians,
    and the wider system."""

    patient_id: str | None = None
    request_id: str | None = None
    actor_role: str | None = None
    safety_triggers: list[str] = field(default_factory=list)
    validator_decision: str | None = None
    rag_source_ids: list[str] = field(default_factory=list)
    timeline_snapshot_hash: str | None = None
    notes: str | None = None


# ─── Recording ───────────────────────────────────────────────────────────────


def record_prediction_trace(
    db: Session,
    prediction: EvidenceAwarePrediction,
    *,
    context: TraceContext | None = None,
    threshold_config: Mapping[str, float] | None = None,
    calibration_config: Mapping[str, Any] | None = None,
) -> PredictionTrace:
    """Persist a `PredictionTrace` row for one prediction call.

    Defaults to the project's threshold + calibration configs so the caller
    doesn't have to repeat them on every call.  The function returns the
    persisted row (with its id populated) so the caller can echo the trace
    id back in API responses for end-to-end correlation.
    """
    ctx = context or TraceContext()
    thresholds = threshold_config or {
        "lower": LOWER_DECISION_THRESHOLD,
        "upper": UPPER_DECISION_THRESHOLD,
    }
    calibration = calibration_config or {
        "method": "isotonic" if prediction.calibrated else None,
        "applied": prediction.calibrated,
    }

    row = PredictionTrace(
        patient_id=ctx.patient_id,
        request_id=ctx.request_id,
        actor_role=ctx.actor_role,
        question=prediction.question,
        decision=prediction.decision,
        probability=prediction.probability,
        raw_probability=prediction.raw_probability,
        calibrated=1 if prediction.calibrated else 0,
        confidence=prediction.confidence,
        evidence_sufficiency=prediction.evidence.sufficiency,
        abstained=1 if prediction.evidence.abstain else 0,
        abstain_reason=prediction.evidence.reason,
        modalities_present_json=json.dumps(prediction.evidence.modalities_present),
        modalities_missing_json=json.dumps(prediction.evidence.modalities_missing),
        confidence_modifier=prediction.evidence.confidence_modifier,
        model_version=prediction.model_version,
        feature_set_version=FEATURE_SET_VERSION,
        threshold_config_json=json.dumps(dict(thresholds)),
        calibration_config_json=json.dumps(dict(calibration)),
        safety_triggers_json=json.dumps(list(ctx.safety_triggers)),
        validator_decision=ctx.validator_decision,
        rag_source_ids_json=json.dumps(list(ctx.rag_source_ids)),
        timeline_snapshot_hash=ctx.timeline_snapshot_hash,
        notes=ctx.notes,
    )
    db.add(row)
    db.flush()
    return row


def predict_and_trace(
    db: Session,
    row: Mapping[str, Any],
    *,
    question: str = "response_classification",
    context: TraceContext | None = None,
    model_path: str = DEFAULT_MODEL_PATH,
    calibrator_path: str | None = DEFAULT_CALIBRATOR_PATH,
    commit: bool = True,
) -> tuple[EvidenceAwarePrediction, PredictionTrace]:
    """One-shot helper: run prediction, persist the trace, return both.

    Commits by default so callers that just want "fire and log" don't have
    to manage transactions.  Pass ``commit=False`` when wrapping this in a
    larger unit-of-work the caller controls.
    """
    prediction = predict_with_abstention(
        row,
        question=question,
        model_path=model_path,
        calibrator_path=calibrator_path,
    )
    trace = record_prediction_trace(db, prediction, context=context)
    if commit:
        db.commit()
        db.refresh(trace)
    return prediction, trace


# ─── Reading ─────────────────────────────────────────────────────────────────


def list_recent_traces(
    db: Session,
    *,
    limit: int = 50,
    patient_id: str | None = None,
    decision: str | None = None,
    abstained_only: bool = False,
) -> list[dict[str, Any]]:
    """Return the most recent traces as plain dicts ready for JSON encoding."""
    query = db.query(PredictionTrace).order_by(
        PredictionTrace.created_at.desc(), PredictionTrace.id.desc(),
    )
    if patient_id is not None:
        query = query.filter(PredictionTrace.patient_id == patient_id)
    if decision is not None:
        query = query.filter(PredictionTrace.decision == decision)
    if abstained_only:
        query = query.filter(PredictionTrace.abstained == 1)

    safe_limit = max(1, min(limit, 200))
    rows = query.limit(safe_limit).all()
    return [_row_to_dict(r) for r in rows]


def summarise_traces(db: Session, *, lookback: int = 500) -> dict[str, Any]:
    """Aggregate stats over the last `lookback` traces — used by the admin
    card to summarise without paging through every individual row."""
    rows = (
        db.query(PredictionTrace)
        .order_by(PredictionTrace.created_at.desc(), PredictionTrace.id.desc())
        .limit(max(1, min(lookback, 5000)))
        .all()
    )
    if not rows:
        return {
            "total": 0,
            "abstention_rate": None,
            "decision_counts": {},
            "evidence_sufficiency_counts": {},
            "model_versions": [],
        }

    decision_counts: dict[str, int] = {}
    sufficiency_counts: dict[str, int] = {}
    abstained = 0
    model_versions: set[str] = set()
    for r in rows:
        decision_counts[r.decision] = decision_counts.get(r.decision, 0) + 1
        if r.evidence_sufficiency:
            sufficiency_counts[r.evidence_sufficiency] = (
                sufficiency_counts.get(r.evidence_sufficiency, 0) + 1
            )
        if r.abstained:
            abstained += 1
        if r.model_version:
            model_versions.add(r.model_version)

    return {
        "total": len(rows),
        "abstention_rate": round(abstained / len(rows), 4),
        "decision_counts": decision_counts,
        "evidence_sufficiency_counts": sufficiency_counts,
        "model_versions": sorted(model_versions),
    }


# ─── Helpers ─────────────────────────────────────────────────────────────────


def hash_input_row(row: Mapping[str, Any]) -> str:
    """Stable fingerprint of an input row.  Use as `timeline_snapshot_hash`
    when you want two predictions on the same patient state to be visibly
    linked in the audit log."""
    payload = json.dumps(row, default=str, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _loads(value: str | None) -> Any:
    if not value:
        return []
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return []


def _row_to_dict(row: PredictionTrace) -> dict[str, Any]:
    """One trace row → JSON-safe dict.  Keys are stable; the frontend
    `PredictionTraceRow` type mirrors this shape exactly."""
    return {
        "id": row.id,
        "created_at": str(row.created_at) if row.created_at else None,
        "patient_id": row.patient_id,
        "request_id": row.request_id,
        "actor_role": row.actor_role,
        "question": row.question,
        "decision": row.decision,
        "probability": row.probability,
        "raw_probability": row.raw_probability,
        "calibrated": bool(row.calibrated),
        "confidence": row.confidence,
        "evidence_sufficiency": row.evidence_sufficiency,
        "abstained": bool(row.abstained),
        "abstain_reason": row.abstain_reason,
        "modalities_present": _loads(row.modalities_present_json),
        "modalities_missing": _loads(row.modalities_missing_json),
        "confidence_modifier": row.confidence_modifier,
        "model_version": row.model_version,
        "feature_set_version": row.feature_set_version,
        "threshold_config": _loads(row.threshold_config_json),
        "calibration_config": _loads(row.calibration_config_json),
        "safety_triggers": _loads(row.safety_triggers_json),
        "validator_decision": row.validator_decision,
        "rag_source_ids": _loads(row.rag_source_ids_json),
        "timeline_snapshot_hash": row.timeline_snapshot_hash,
        "notes": row.notes,
    }


__all__ = [
    "FEATURE_SET_VERSION",
    "TraceContext",
    "hash_input_row",
    "list_recent_traces",
    "predict_and_trace",
    "record_prediction_trace",
    "summarise_traces",
]
