from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from backend.services.app_logging import build_app_monitoring_summary
from backend.services.clinician_feedback import clinical_feedback_summary
from backend.services.drift_monitoring import DEFAULT_OUTPUT_PATH as DRIFT_OUTPUT_PATH
from backend.services.failure_case_gallery import load_failure_case_gallery
from backend.services.patient_data_quality import audit_patient_data_coherence
from backend.services.rag_analytics import build_rag_evaluation_summary
from backend.services.rag_eval_suite import DEFAULT_OUTPUT_PATH as RAG_EVAL_OUTPUT_PATH
from backend.services.safety_red_team import DEFAULT_OUTPUT_PATH as SAFETY_OUTPUT_PATH


DEFAULT_SYNTHETIC_METRICS_PATH = "Data/complete_synthetic_training/complete_synthetic_model_metrics.json"


def build_safety_evaluation_center(db) -> dict:
    safety_red_team = _load_artifact(SAFETY_OUTPUT_PATH)
    rag_eval = _load_artifact(RAG_EVAL_OUTPUT_PATH)
    drift_report = _load_artifact(DRIFT_OUTPUT_PATH)
    failure_gallery = load_failure_case_gallery()

    calibration = _calibration_snapshot(DEFAULT_SYNTHETIC_METRICS_PATH)
    data_quality = audit_patient_data_coherence(db, output_path=None, include_examples=False)
    rag_trace_summary = build_rag_evaluation_summary(db)
    clinician_feedback = clinical_feedback_summary(db)
    audit_summary = build_app_monitoring_summary(db)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "safety_red_team": safety_red_team,
        "prompt_injection_defense": _category_summary(safety_red_team, ["prompt_injection", "multilingual_attack", "encoded_attack"]),
        "urgent_symptom_escalation": _category_summary(safety_red_team, ["urgent_symptom_report", "low_cbc_infection_risk", "emotional_distress"]),
        "medication_refusal": _category_summary(safety_red_team, ["treatment_decision_request", "medication_change_request"]),
        "privacy_exfiltration": _category_summary(safety_red_team, ["cross_patient_privacy"]),
        "rag_eval": rag_eval,
        "rag_trace_summary": rag_trace_summary,
        "calibration_metrics": calibration,
        "drift_report": drift_report,
        "data_quality": data_quality,
        "clinician_feedback": clinician_feedback,
        "failure_case_gallery": failure_gallery,
        "audit_log_summary": audit_summary,
        "safety_note": "Monitoring and governance evidence only. This page does not imply clinical validation.",
    }


def _load_artifact(path: str) -> dict:
    artifact_path = Path(path)
    if not artifact_path.exists():
        return {
            "status": "not_generated",
            "message": f"Artifact not generated: {path}",
            "path": path,
        }
    try:
        return json.loads(artifact_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {
            "status": "error",
            "message": f"Artifact could not be parsed: {path}",
            "path": path,
        }


def _category_summary(safety_red_team: dict, categories: list[str]) -> dict:
    if not safety_red_team or safety_red_team.get("status") == "not_generated":
        return {"status": "not_generated", "pass_rate": None, "case_count": 0, "categories": categories}

    cases = safety_red_team.get("cases") or []
    filtered = [case for case in cases if case.get("category") in categories]
    if not filtered:
        return {"status": "unavailable", "pass_rate": None, "case_count": 0, "categories": categories}

    passed = sum(1 for case in filtered if case.get("pass"))
    return {
        "status": "passed" if passed == len(filtered) else "needs_attention",
        "pass_rate": round(passed / len(filtered), 3) if filtered else None,
        "case_count": len(filtered),
        "categories": categories,
    }


def _calibration_snapshot(path: str) -> dict:
    metrics_path = Path(path)
    if not metrics_path.exists():
        return {"status": "unavailable", "message": "Calibration metrics file not available."}
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "error", "message": "Calibration metrics file could not be parsed."}

    best = (metrics or {}).get("best_model_by_patient_level_roc_auc")
    models = (metrics or {}).get("models") or {}
    best_metrics = models.get(best) or {}
    calibration = best_metrics.get("calibration") or {}
    before = calibration.get("before_temperature_scaling") or {}
    after = calibration.get("after_temperature_scaling") or {}

    return {
        "status": "available" if best_metrics else "unavailable",
        "best_model": best,
        "brier_score": best_metrics.get("patient_level_brier_score") or best_metrics.get("brier_score"),
        "ece_before": before.get("ece"),
        "ece_after": after.get("ece"),
        "temperature": after.get("temperature"),
        "method": calibration.get("method"),
        "note": "Calibration metrics are computed on synthetic evaluation splits.",
    }
