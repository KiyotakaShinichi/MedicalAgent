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
import math
import re
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from sqlalchemy.orm import Session

from backend.models import (
    BreastCancerProfile,
    ClinicalIntervention,
    ImagingReport,
    LabResult,
    SymptomReport,
    Treatment,
)
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


def _live_patient_row_from_db(db: Session, patient_id: str) -> dict[str, Any] | None:
    """Build a best-effort inference row from the live demo patient record.

    The trained models consume the synthetic timeline feature schema.  Demo
    patients such as ``P001`` live in SQL tables instead of the CSV, so they
    need a narrow adapter before the evidence-aware heads can do their normal
    abstention/confidence logic.  This adapter does not create clinical facts:
    unknown fields stay missing, and the output remains a synthetic-only
    monitoring signal.
    """
    profile = (
        db.query(BreastCancerProfile)
        .filter(BreastCancerProfile.patient_id == patient_id)
        .first()
    )
    labs = (
        db.query(LabResult)
        .filter(LabResult.patient_id == patient_id)
        .order_by(LabResult.date.asc(), LabResult.id.asc())
        .all()
    )
    treatments = (
        db.query(Treatment)
        .filter(Treatment.patient_id == patient_id)
        .order_by(Treatment.date.asc(), Treatment.cycle.asc(), Treatment.id.asc())
        .all()
    )
    symptoms = (
        db.query(SymptomReport)
        .filter(SymptomReport.patient_id == patient_id)
        .order_by(SymptomReport.date.asc(), SymptomReport.id.asc())
        .all()
    )
    imaging_reports = (
        db.query(ImagingReport)
        .filter(ImagingReport.patient_id == patient_id)
        .order_by(ImagingReport.date.asc(), ImagingReport.id.asc())
        .all()
    )
    interventions = (
        db.query(ClinicalIntervention)
        .filter(ClinicalIntervention.patient_id == patient_id)
        .order_by(ClinicalIntervention.date.asc(), ClinicalIntervention.id.asc())
        .all()
    )

    if not any((profile, labs, treatments, symptoms, imaging_reports, interventions)):
        return None

    latest_treatment = treatments[-1] if treatments else None
    treatment_date = latest_treatment.date if latest_treatment else (labs[-1].date if labs else None)
    cycle = latest_treatment.cycle if latest_treatment else max(len(treatments), 1)

    pre_lab = _latest_lab_on_or_before(labs, treatment_date) or (labs[0] if labs else None)
    recovery_lab = labs[-1] if labs else None
    nadir_labs = _nadir_window_labs(labs, treatment_date) if treatment_date else labs

    baseline_size = _first_report_size(imaging_reports)
    latest_size = _last_report_size(imaging_reports)
    percent_change = None
    if baseline_size is not None and latest_size is not None and baseline_size > 0:
        percent_change = round(((latest_size - baseline_size) / baseline_size) * 100.0, 2)

    recent_symptoms = (
        [row for row in symptoms if treatment_date and row.date >= treatment_date]
        or symptoms
    )
    intervention_count = len(interventions)

    return {
        "patient_id": patient_id,
        "cycle": cycle,
        "treatment_date": treatment_date.isoformat() if treatment_date else None,
        "age": None,
        "stage": _normalise_stage(profile.cancer_stage if profile else None),
        "molecular_subtype": _normalise_subtype(profile.molecular_subtype if profile else None),
        "regimen": _normalise_regimen([t.drug for t in treatments]),
        "pre_wbc": _lab_value(pre_lab, "wbc"),
        "pre_anc": None,
        "pre_hemoglobin": _lab_value(pre_lab, "hemoglobin"),
        "pre_platelets": _lab_value(pre_lab, "platelets"),
        "nadir_wbc": _min_lab_value(nadir_labs, "wbc"),
        "nadir_anc": None,
        "nadir_hemoglobin": _min_lab_value(nadir_labs, "hemoglobin"),
        "nadir_platelets": _min_lab_value(nadir_labs, "platelets"),
        "recovery_wbc": _lab_value(recovery_lab, "wbc"),
        "recovery_hemoglobin": _lab_value(recovery_lab, "hemoglobin"),
        "recovery_platelets": _lab_value(recovery_lab, "platelets"),
        "mri_tumor_size_cm": latest_size,
        "mri_percent_change_from_baseline": percent_change,
        "max_symptom_severity": max((s.severity for s in recent_symptoms), default=None),
        "symptom_count": len(recent_symptoms),
        "intervention_count": intervention_count,
        "dose_delayed": int(any(_contains_any(i.intervention_type, ("delay", "held", "hold", "defer")) for i in interventions)),
        "dose_reduced": int(any(_contains_any(i.intervention_type, ("reduc", "lower")) for i in interventions)),
        "live_record_adapter": True,
    }


def _latest_lab_on_or_before(labs: list[LabResult], target_date: Any | None) -> LabResult | None:
    if not labs or target_date is None:
        return None
    eligible = [row for row in labs if row.date <= target_date]
    return eligible[-1] if eligible else None


def _nadir_window_labs(labs: list[LabResult], treatment_date: Any | None) -> list[LabResult]:
    if not labs or treatment_date is None:
        return []
    window_end = treatment_date + timedelta(days=21)
    window = [row for row in labs if treatment_date <= row.date <= window_end]
    return window or [row for row in labs if row.date >= treatment_date] or labs


def _lab_value(row: LabResult | None, field: str) -> float | None:
    if row is None:
        return None
    value = getattr(row, field, None)
    return float(value) if value is not None and math.isfinite(float(value)) else None


def _min_lab_value(rows: list[LabResult], field: str) -> float | None:
    values = [_lab_value(row, field) for row in rows]
    values = [v for v in values if v is not None]
    return min(values) if values else None


def _first_report_size(reports: list[ImagingReport]) -> float | None:
    for report in reports:
        size = _largest_cm(report)
        if size is not None:
            return size
    return None


def _last_report_size(reports: list[ImagingReport]) -> float | None:
    for report in reversed(reports):
        size = _largest_cm(report)
        if size is not None:
            return size
    return None


def _largest_cm(report: ImagingReport) -> float | None:
    text = f"{report.findings or ''} {report.impression or ''}"
    values = []
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*cm\b", text, flags=re.IGNORECASE):
        try:
            values.append(float(match.group(1)))
        except ValueError:
            continue
    return max(values) if values else None


def _normalise_stage(value: str | None) -> str | None:
    if not value:
        return None
    upper = value.upper().replace("STAGE", "").strip()
    if upper in {"II", "2"}:
        return "IIA"
    if upper in {"III", "3"}:
        return "IIIA"
    if upper in {"IV", "4"}:
        return "IV"
    return upper


def _normalise_subtype(value: str | None) -> str | None:
    if not value:
        return None
    text = value.lower().replace(" ", "")
    if "triple" in text:
        return "triple-negative"
    hr_pos = "hr-positive" in text or "hr+" in text or ("er-positive" in text and "pr-positive" in text)
    her2_pos = "her2-positive" in text or "her2+" in text
    her2_neg = "her2-negative" in text or "her2-" in text
    if hr_pos and her2_pos:
        return "HR+/HER2+"
    if hr_pos and her2_neg:
        return "HR+/HER2-"
    if her2_pos:
        return "HER2+"
    return value


def _normalise_regimen(drugs: list[str]) -> str | None:
    joined = " ".join(drugs).lower()
    if "tchp" in joined:
        return "TCHP"
    if "carboplatin" in joined and "paclitaxel" in joined:
        return "paclitaxel + carboplatin then AC"
    if "paclitaxel" in joined and ("doxorubicin" in joined or "cyclophosphamide" in joined or "ac" in joined):
        return "dose-dense AC then paclitaxel"
    if "paclitaxel" in joined:
        return "dose-dense AC then paclitaxel"
    return drugs[-1] if drugs else None


def _contains_any(value: str | None, needles: tuple[str, ...]) -> bool:
    text = (value or "").lower()
    return any(needle in text for needle in needles)


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
    row = _latest_cycle_row(timeline, patient_id) if timeline is not None else None
    if row is None:
        row = _live_patient_row_from_db(db, patient_id)
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
    row = _latest_cycle_row(timeline, patient_id) if timeline is not None else None
    if row is None:
        row = _live_patient_row_from_db(db, patient_id)
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

    # Persist a synthetic-trace row for the regression head by reusing the
    # classification trace shape — the regression decision is what matters
    # for audit, and ``probability=None`` accurately reflects that this
    # head's primary output is a score, not a probability.
    if record_trace:
        record_prediction_trace(db, bundle.classification, context=base_context)
        record_prediction_trace(db, bundle.toxicity, context=base_context)
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
        db.commit()

    payload = bundle.to_dict()
    payload["ood_gate"] = ood_gate.to_dict()
    payload["inference_source"] = (
        "live_patient_record_adapter"
        if row.get("live_record_adapter") else "synthetic_timeline_row"
    )
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
