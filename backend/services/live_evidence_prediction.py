"""Live evidence-aware prediction service for the patient report.

Bridges the synthetic ML pipeline to the live patient flow: looks up the
patient's most recent cycle row, runs the hybrid prediction (classification
+ regression + toxicity), records one ``PredictionTrace`` row per head,
and returns both the per-head envelope and a hybrid bundle ready to embed
in the patient report payload.

Patients without a synthetic-CSV row return ``None`` cleanly — the report
build path treats that as "no live prediction available" rather than an
error, so the dashboard's existing pre-computed prediction stays intact.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from sqlalchemy.orm import Session

from backend.services.hybrid_prediction import (
    DEFAULT_REGRESSION_MODEL_PATH,
    DEFAULT_TOXICITY_MODEL_PATH,
    EvidenceAwareRegression,
    HybridPrediction,
    predict_hybrid,
    predict_response_score_with_abstention,
    predict_toxicity_with_abstention,
)
from backend.services.evidence_sufficiency import EvidenceAssessment
from backend.services.predict_with_abstention import (
    DEFAULT_CALIBRATOR_PATH,
    DEFAULT_MODEL_PATH,
    EvidenceAwarePrediction,
)
from backend.services.prediction_trace import (
    TraceContext,
    hash_input_row,
    predict_and_trace,
    record_prediction_trace,
)
from backend.services.realtime_ood_gate import assess_realtime_ood


DEFAULT_TIMELINE_CSV = "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv"


@lru_cache(maxsize=2)
def _load_timeline_index(csv_path: str) -> pd.DataFrame | None:
    """Load the temporal training rows once and cache them in memory.

    Returns ``None`` (cached) when the CSV is absent so the report path
    can short-circuit without touching disk every time.
    """
    path = Path(csv_path)
    if not path.exists():
        return None
    return pd.read_csv(path)


def _latest_cycle_row(timeline: pd.DataFrame, patient_id: str) -> dict[str, Any] | None:
    """Return the patient's most-recent cycle row as a plain dict, or
    ``None`` if they have no rows in the synthetic timeline."""
    matches = timeline[timeline["patient_id"] == patient_id]
    if matches.empty:
        return None
    sort_keys: list[str] = []
    if "treatment_date" in matches.columns:
        sort_keys.append("treatment_date")
    if "cycle" in matches.columns:
        sort_keys.append("cycle")
    if sort_keys:
        matches = matches.sort_values(sort_keys)
    return matches.iloc[-1].to_dict()


def build_evidence_aware_prediction(
    patient_id: str,
    db: Session,
    *,
    timeline_csv: str = DEFAULT_TIMELINE_CSV,
    actor_role: str | None = None,
    request_id: str | None = None,
    model_path: str = DEFAULT_MODEL_PATH,
    calibrator_path: str | None = DEFAULT_CALIBRATOR_PATH,
    record_trace: bool = True,
) -> dict[str, Any] | None:
    """Compute the evidence-aware prediction for a patient, persist a trace,
    and return the envelope ready for JSON serialisation.  Returns ``None``
    when the patient is not present in the synthetic timeline.

    This is the **classification-only** entry point preserved for backward
    compatibility with the existing patient-dashboard card.  New code should
    call ``build_hybrid_prediction`` to also surface the regression score and
    the toxicity signal.
    """
    timeline = _load_timeline_index(timeline_csv)
    if timeline is None:
        return None
    row = _latest_cycle_row(timeline, patient_id)
    if row is None:
        return None
    if not Path(model_path).exists():
        return None

    context = TraceContext(
        patient_id=patient_id,
        request_id=request_id,
        actor_role=actor_role,
        validator_decision="allowed",
        timeline_snapshot_hash=hash_input_row(row),
        notes="live patient-report inference",
    )
    prediction, _trace = predict_and_trace(
        db,
        row,
        question="response_classification",
        context=context,
        model_path=model_path,
        calibrator_path=calibrator_path,
        commit=record_trace,
    )
    return prediction.to_dict()


def build_hybrid_prediction(
    patient_id: str,
    db: Session,
    *,
    timeline_csv: str = DEFAULT_TIMELINE_CSV,
    actor_role: str | None = None,
    request_id: str | None = None,
    classification_model_path: str = DEFAULT_MODEL_PATH,
    calibrator_path: str | None = DEFAULT_CALIBRATOR_PATH,
    regression_model_path: str = DEFAULT_REGRESSION_MODEL_PATH,
    toxicity_model_path: str = DEFAULT_TOXICITY_MODEL_PATH,
    record_trace: bool = True,
) -> dict[str, Any] | None:
    """Compute the full hybrid envelope (classification + regression +
    toxicity) for a patient, persist one trace row per head, and return
    a JSON-serialisable bundle ready for the patient/clinician report.

    Returns ``None`` when the patient is not in the synthetic cohort or the
    classification artifact is missing — same null contract as
    ``build_evidence_aware_prediction``.
    """
    timeline = _load_timeline_index(timeline_csv)
    if timeline is None:
        return None
    row = _latest_cycle_row(timeline, patient_id)
    if row is None:
        return None
    if not Path(classification_model_path).exists():
        return None
    ood_gate = assess_realtime_ood(row)
    ood_trigger = f"ood_gate:{ood_gate.severity}:{','.join(ood_gate.reasons[:3])}" if ood_gate.severity != "none" else None

    if ood_gate.severity == "severe":
        bundle = _severe_ood_hybrid_bundle(
            ood_gate.to_dict(),
            classification_model_path=classification_model_path,
            regression_model_path=regression_model_path,
            toxicity_model_path=toxicity_model_path,
        )
    else:
        bundle = predict_hybrid(
            row,
            classification_model_path=classification_model_path,
            calibrator_path=calibrator_path,
            regression_model_path=regression_model_path,
            toxicity_model_path=toxicity_model_path,
        )

    # One trace row per head: same patient_id + timeline_snapshot_hash so a
    # reviewer can group the three traces produced by a single report build.
    # The regression head writes a trace with `decision` mirroring its
    # `EvidenceAwareRegression.decision` and `probability=None` (the
    # underlying value is a score, not a probability) so the existing
    # trace schema doesn't need new columns.
    snapshot_hash = hash_input_row(row)
    base_context = TraceContext(
        patient_id=patient_id,
        request_id=request_id,
        actor_role=actor_role,
        safety_triggers=[ood_trigger] if ood_trigger else [],
        validator_decision="allowed" if ood_gate.severity != "severe" else "ood_gate_abstained",
        timeline_snapshot_hash=snapshot_hash,
        notes=f"live patient-report inference (hybrid); ood_gate={ood_gate.severity}",
    )

    record_prediction_trace(db, bundle.classification, context=base_context)
    record_prediction_trace(db, bundle.toxicity, context=base_context)
    # Persist a synthetic-trace row for the regression head by reusing the
    # classification trace shape — the regression decision is what matters
    # for audit, and ``probability=None`` accurately reflects that this
    # head's primary output is a score, not a probability.
    regression_trace_view = EvidenceAwarePrediction(
        decision=bundle.response_score.decision,
        probability=None,
        raw_probability=(
            bundle.response_score.raw_response_score
            if bundle.response_score.raw_response_score is not None else None
        ),
        calibrated=False,
        confidence=bundle.response_score.confidence,
        evidence=bundle.response_score.evidence,
        model_version=bundle.response_score.model_version,
        question=bundle.response_score.question,
    )
    record_prediction_trace(db, regression_trace_view, context=base_context)

    if record_trace:
        db.commit()

    payload = bundle.to_dict()
    payload["ood_gate"] = ood_gate.to_dict()
    return payload


def _severe_ood_hybrid_bundle(
    ood_gate: Mapping[str, Any],
    *,
    classification_model_path: str,
    regression_model_path: str,
    toxicity_model_path: str,
) -> HybridPrediction:
    reason = "severe_ood_or_data_quality_gate:" + ",".join(ood_gate.get("reasons") or ["unknown"])
    evidence = EvidenceAssessment(
        modalities_present=[],
        modalities_missing=["data_quality_or_ood_review_required"],
        sufficiency="insufficient",
        abstain=True,
        reason=reason,
        confidence_modifier=0.0,
    )
    classification = EvidenceAwarePrediction(
        decision="insufficient_evidence",
        probability=None,
        raw_probability=None,
        calibrated=False,
        confidence="low",
        evidence=evidence,
        model_version=Path(classification_model_path).stem,
        question="response_classification",
    )
    toxicity = EvidenceAwarePrediction(
        decision="insufficient_evidence",
        probability=None,
        raw_probability=None,
        calibrated=False,
        confidence="low",
        evidence=evidence,
        model_version=Path(toxicity_model_path).stem,
        question="toxicity_signal",
    )
    regression = EvidenceAwareRegression(
        decision="insufficient_evidence",
        response_score=None,
        raw_response_score=None,
        uncertainty_band=None,
        confidence="low",
        evidence=evidence,
        model_version=Path(regression_model_path).stem,
        question="response_score_regression",
        uncertainty_method="abstained_realtime_ood_gate",
    )
    return HybridPrediction(
        classification=classification,
        response_score=regression,
        toxicity=toxicity,
    )


__all__ = [
    "build_evidence_aware_prediction",
    "build_hybrid_prediction",
]
