from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_OUTPUT_PATH = "Data/mle_monitoring/current_vs_realism_candidate.json"


def build_current_vs_candidate_report(
    current_metrics_path: str = "Data/complete_synthetic_training/complete_synthetic_model_metrics.json",
    candidate_metrics_path: str = "Data/complete_synthetic_training_realism_v2/complete_synthetic_model_metrics.json",
    current_realism_path: str = "Data/mle_monitoring/synthetic_realism_report.json",
    candidate_realism_path: str = "Data/mle_monitoring/synthetic_realism_candidate_report.json",
    output_path: str | None = DEFAULT_OUTPUT_PATH,
) -> dict:
    current_metrics = _load_json(current_metrics_path)
    candidate_metrics = _load_json(candidate_metrics_path)
    current_realism = _load_json(current_realism_path)
    candidate_realism = _load_json(candidate_realism_path)

    current = _model_snapshot(current_metrics, current_realism)
    candidate = _model_snapshot(candidate_metrics, candidate_realism)
    recommendation = _recommendation(current, candidate)
    report = {
        "schema_version": "current_vs_realism_candidate_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "current": current,
        "candidate": candidate,
        "recommendation": recommendation,
        "claim_boundary": (
            "This compares synthetic engineering candidates only. It does not establish clinical validity."
        ),
    }
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def _model_snapshot(metrics: dict, realism: dict) -> dict:
    best = metrics.get("best_model_by_patient_level_roc_auc")
    model = ((metrics.get("models") or {}).get(best) or {}) if best else {}
    return {
        "best_classifier": best,
        "patient_level_roc_auc": _metric(model, "patient_level_roc_auc"),
        "patient_level_average_precision": _metric(model, "patient_level_average_precision"),
        "patient_level_brier_score": _metric(model, "patient_level_brier_score"),
        "calibrated_validation_ece": _metric(model, "calibrated_validation_ece"),
        "realism_status": realism.get("status"),
        "realism_alignment_score": (realism.get("realism_alignment_score") or {}).get("score"),
        "sim_to_real_status": (realism.get("sim_to_real_comparison") or {}).get("status"),
        "threshold_coverage_status": (realism.get("threshold_coverage") or {}).get("status"),
    }


def _recommendation(current: dict, candidate: dict) -> dict:
    current_auc = current.get("patient_level_roc_auc")
    candidate_auc = candidate.get("patient_level_roc_auc")
    current_realism = current.get("realism_alignment_score") or 0
    candidate_realism = candidate.get("realism_alignment_score") or 0
    auc_delta = None
    if current_auc is not None and candidate_auc is not None:
        auc_delta = round(float(candidate_auc) - float(current_auc), 4)
    promote = (
        (candidate_realism >= current_realism)
        and (auc_delta is None or auc_delta >= -0.03)
        and candidate.get("sim_to_real_status") in {"strong", "passed", "acceptable"}
    )
    return {
        "decision": "promote_candidate_after_review" if promote else "keep_current_and_continue_tuning",
        "auc_delta": auc_delta,
        "realism_delta": round(float(candidate_realism - current_realism), 4),
        "rationale": (
            "Candidate is preferable when it materially improves realism without losing more than "
            "0.03 AUROC on synthetic evaluation. Clinician-safety and external-validation boundaries still apply."
        ),
    }


def _metric(model: dict, key: str):
    value = model.get(key)
    if value is None:
        return None
    try:
        return round(float(value), 4)
    except Exception:
        return value


def _load_json(path: str) -> dict:
    artifact = Path(path)
    if not artifact.exists():
        return {}
    try:
        return json.loads(artifact.read_text(encoding="utf-8"))
    except Exception:
        return {}
