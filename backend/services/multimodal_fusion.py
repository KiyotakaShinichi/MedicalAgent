from pathlib import Path
import json

import pandas as pd


def build_multimodal_assessment(
    patient_id,
    report,
    model_predictions_csv_path="Data/breastdcedl_spy1_model_predictions.csv",
    shap_explanations_json_path="Data/breastdcedl_spy1_shap_explanations.json",
):
    mri_signal = _mri_response_signal(
        patient_id,
        report,
        model_predictions_csv_path,
        shap_explanations_json_path,
    )
    clinical_signal = _clinical_monitoring_signal(report)
    symptom_signal = _symptom_signal(report)
    score_breakdown = _treatment_monitoring_score_breakdown(
        mri_signal,
        clinical_signal,
        symptom_signal,
    )
    score = score_breakdown["final_score"]
    overall = _overall_status(score, mri_signal, clinical_signal, symptom_signal)

    return {
        "overall_status": overall["status"],
        "overall_message": overall["message"],
        "treatment_monitoring_score": score,
        "score_interpretation": _score_interpretation(score),
        "score_breakdown": score_breakdown,
        "signals": {
            "mri_response": mri_signal,
            "clinical_monitoring": clinical_signal,
            "symptoms": symptom_signal,
        },
        "recommended_action": overall["recommended_action"],
        "patient_next_steps": [
            "Check that recent symptoms, CBC values, medications, and imaging dates are complete.",
            "Open the review queue and prepare questions about the listed record items for your care team.",
            "Use NLCare to add missing records or summarize what changed; do not change treatment based on this index.",
        ],
        "safety_note": (
            "Exploratory monitoring support only. This does not diagnose cancer, choose treatment, "
            "or replace clinician review."
        ),
    }


def _mri_response_signal(patient_id, report, predictions_csv_path, shap_explanations_json_path):
    prediction = _load_patient_prediction(patient_id, predictions_csv_path)
    if prediction is not None:
        probability = float(prediction["best_model_pcr_probability"])
        explanation = load_patient_shap_explanation(patient_id, shap_explanations_json_path)
        if probability >= 0.66:
            status = "favorable_response_signal"
            message = "The MRI-derived fields group with the higher simulator response-pattern class."
        elif probability <= 0.40:
            status = "lower_response_signal"
            message = "The MRI-derived fields group with the lower simulator response-pattern class."
        else:
            status = "indeterminate_response_signal"
            message = "The MRI-derived synthetic grouping is uncertain."

        return {
            "status": status,
            "source": "breastdcedl_cross_validated_baseline",
            "pcr_probability": round(probability, 3),
            "response_signal_score": round(probability * 100),
            "model": prediction.get("best_model", "unknown"),
            "xai": explanation,
            "message": message,
            "caveat": "Model is a PoC baseline and is not clinically validated.",
        }

    synthetic_prediction = report.get("synthetic_model_prediction") or {}
    hybrid_signal = synthetic_prediction.get("hybrid_mle_signal")
    if hybrid_signal:
        hybrid_score = float(hybrid_signal.get("hybrid_score", 50))
        status = hybrid_signal.get("status") or _status_from_score(hybrid_score)
        if status == "favorable_response_signal":
            message = "The available fields group with the higher synthetic monitoring-pattern class."
        elif "lower" in status:
            message = "The available fields group with the lower synthetic monitoring-pattern class."
        else:
            message = "The synthetic monitoring-pattern grouping is mixed or uncertain."

        return {
            "status": status,
            "source": "hybrid_complete_synthetic_classification_regression",
            "response_probability": hybrid_signal.get("classification_probability"),
            "response_score_percent": hybrid_signal.get("response_score_percent"),
            "hybrid_score": round(hybrid_score, 1),
            "response_signal_score": round(hybrid_score),
            "model": "hybrid_mle_signal_v1",
            "hybrid_mle_signal": hybrid_signal,
            "xai": report.get("synthetic_model_explanation"),
            "message": message,
            "caveat": "Hybrid signal is trained on synthetic simulator data and is not clinically validated.",
        }

    synthetic_probability = _synthetic_response_probability(synthetic_prediction)
    if synthetic_probability is not None:
        explanation = report.get("synthetic_model_explanation")
        if synthetic_probability >= 0.66:
            status = "favorable_response_signal"
            message = "The longitudinal fields group with the higher simulator monitoring-pattern class."
        elif synthetic_probability <= 0.40:
            status = "lower_response_signal"
            message = "The longitudinal fields group with the lower simulator monitoring-pattern class."
        else:
            status = "indeterminate_response_signal"
            message = "The longitudinal synthetic grouping is uncertain."

        return {
            "status": status,
            "source": "complete_synthetic_longitudinal_model",
            "response_probability": round(synthetic_probability, 3),
            "response_signal_score": round(synthetic_probability * 100),
            "model": _synthetic_probability_source(synthetic_prediction),
            "xai": explanation,
            "message": message,
            "caveat": "Model is trained on synthetic simulator data and is not clinically validated.",
        }

    radiology = report.get("radiology_summary") or {}
    size_status = radiology.get("size_status")
    if size_status == "decreased":
        return {
            "status": "favorable_response_signal",
            "source": "imaging_report_nlp",
            "response_signal_score": 70,
            "message": "Available imaging report wording suggests interval decrease.",
            "caveat": "Report NLP signal, not raw-image model output.",
        }
    if size_status == "increased":
        return {
            "status": "lower_response_signal",
            "source": "imaging_report_nlp",
            "response_signal_score": 30,
            "message": "Available imaging report wording suggests interval increase.",
            "caveat": "Report NLP signal, not raw-image model output.",
        }
    if size_status:
        return {
            "status": "indeterminate_response_signal",
            "source": "imaging_report_nlp",
            "response_signal_score": 50,
            "message": f"Available imaging report size status is {size_status}.",
            "caveat": "Report NLP signal, not raw-image model output.",
        }

    return {
        "status": "unavailable",
        "source": "none",
        "response_signal_score": None,
        "message": "No MRI response model or imaging trend signal is available for this patient.",
        "caveat": "Upload or import MRI/imaging data to enable this branch.",
    }


def _synthetic_response_probability(prediction):
    for key in [
        "gradient_boosting_calibrated_probability",
        "gradient_boosting_probability",
        "extra_trees_calibrated_probability",
        "extra_trees_probability",
        "random_forest_calibrated_probability",
        "random_forest_probability",
        "logistic_regression_calibrated_probability",
        "logistic_regression_probability",
        "temporal_gru_probability",
        "temporal_1d_cnn_probability",
        "temporal_baseline_cnn_probability",
    ]:
        value = prediction.get(key)
        if value is not None:
            return float(value)
    return None


def _synthetic_probability_source(prediction):
    for key in [
        "gradient_boosting_calibrated_probability",
        "gradient_boosting_probability",
        "extra_trees_calibrated_probability",
        "extra_trees_probability",
        "random_forest_calibrated_probability",
        "random_forest_probability",
        "logistic_regression_calibrated_probability",
        "logistic_regression_probability",
        "temporal_gru_probability",
        "temporal_1d_cnn_probability",
        "temporal_baseline_cnn_probability",
    ]:
        if prediction.get(key) is not None:
            return key.replace("_calibrated_probability", "").replace("_probability", "")
    return "unknown"


def _status_from_score(score):
    if score >= 70:
        return "favorable_response_signal"
    if score < 45:
        return "lower_response_signal_or_review_needed"
    return "mixed_response_signal"


def _clinical_monitoring_signal(report):
    risks = report.get("risks") or []
    urgent = [risk for risk in risks if risk.get("severity") == "urgent_review"]
    watch = [risk for risk in risks if risk.get("severity") == "watch"]
    has_labs = bool(report.get("latest_labs") or report.get("lab_history"))

    if urgent:
        status = "needs_review"
        message = f"{len(urgent)} urgent clinical risk flag(s) are present."
    elif watch:
        status = "watch_closely"
        message = f"{len(watch)} watch-level clinical risk flag(s) are present."
    elif not has_labs:
        status = "unavailable"
        message = "No CBC trend is available, so the absence of flags cannot be interpreted as reassuring."
    else:
        status = "stable_or_no_flags"
        message = "No urgent CBC/treatment/radiology risk flags are present."

    return {
        "status": status,
        "message": message,
        "risk_count": len(risks),
        "urgent_count": len(urgent),
        "watch_count": len(watch),
        "has_labs": has_labs,
        "evidence_available": has_labs or bool(risks),
        "has_synthetic_labs": bool(report.get("has_synthetic_labs")),
        "lab_sources": report.get("lab_sources", []),
    }


def _symptom_signal(report):
    symptoms = report.get("symptoms") or []
    if not symptoms:
        return {
            "status": "not_reported",
            "message": "No symptom reports are available.",
            "max_severity": None,
        }

    max_severity = max(int(item.get("severity", 0)) for item in symptoms)
    if max_severity >= 8:
        status = "needs_review"
        message = "High-severity symptoms are reported."
    elif max_severity >= 5:
        status = "watch_closely"
        message = "Moderate symptoms are reported."
    else:
        status = "low_symptom_burden"
        message = "Only low-severity symptoms are reported."

    return {
        "status": status,
        "message": message,
        "max_severity": max_severity,
        "symptom_count": len(symptoms),
    }


def _treatment_monitoring_score(mri_signal, clinical_signal, symptom_signal):
    return _treatment_monitoring_score_breakdown(
        mri_signal,
        clinical_signal,
        symptom_signal,
    )["final_score"]


def _treatment_monitoring_score_breakdown(mri_signal, clinical_signal, symptom_signal):
    base = mri_signal.get("response_signal_score")
    max_severity = symptom_signal.get("max_severity")
    available_modalities = [
        name
        for name, available in (
            ("imaging_or_model", base is not None),
            ("cbc_or_review_flags", bool(clinical_signal.get("evidence_available"))),
            ("symptoms", max_severity is not None),
        )
        if available
    ]
    if not available_modalities:
        return {
            "base_signal": None,
            "urgent_review_flags": 0,
            "urgent_flag_deduction": 0.0,
            "watch_flags": 0,
            "watch_flag_deduction": 0.0,
            "peak_recorded_symptom_severity": None,
            "symptom_deduction": 0.0,
            "synthetic_lab_provenance_deduction": 0,
            "total_deduction": 0.0,
            "final_score": None,
            "evidence_sufficiency": "insufficient",
            "available_modalities": [],
            "abstained": True,
            "abstain_reason": "No imaging/model, CBC/review-flag, or symptom evidence is available.",
            "formula": None,
            "claim_boundary": "No synthetic monitoring index is generated when evidence is insufficient.",
        }
    if base is None:
        base = 50

    urgent_count = int(clinical_signal.get("urgent_count", 0) or 0)
    watch_count = int(clinical_signal.get("watch_count", 0) or 0)
    urgent_deduction = min(35, urgent_count * 12)
    watch_deduction = min(20, watch_count * 5)

    symptom_deduction = 0.0
    if max_severity is not None:
        symptom_deduction = min(12, int(max_severity) * 1.2)

    synthetic_lab_deduction = 3 if clinical_signal.get("has_synthetic_labs") else 0
    total_deduction = urgent_deduction + watch_deduction + symptom_deduction + synthetic_lab_deduction
    final_score = max(0, min(100, round(float(base) - total_deduction)))

    return {
        "base_signal": round(float(base), 1),
        "urgent_review_flags": urgent_count,
        "urgent_flag_deduction": round(float(urgent_deduction), 1),
        "watch_flags": watch_count,
        "watch_flag_deduction": round(float(watch_deduction), 1),
        "peak_recorded_symptom_severity": max_severity,
        "symptom_deduction": round(float(symptom_deduction), 1),
        "synthetic_lab_provenance_deduction": synthetic_lab_deduction,
        "total_deduction": round(float(total_deduction), 1),
        "final_score": final_score,
        "evidence_sufficiency": "sufficient_for_synthetic_index",
        "available_modalities": available_modalities,
        "abstained": False,
        "abstain_reason": None,
        "formula": "clamp(base signal - capped review-flag deductions - symptom deduction - synthetic-lab provenance deduction, 0, 100)",
        "claim_boundary": "Record-based synthetic engineering index; not cancer status, treatment success, prognosis, or a treatment recommendation.",
    }


def _score_interpretation(score):
    if score is None:
        return {
            "scale": "not_displayed",
            "meaning": "No monitoring index was generated because the recorded evidence is insufficient.",
            "bands": {},
            "caveat": "Missing data must not be converted into a neutral or reassuring score.",
        }
    return {
        "scale": "0-100",
        "meaning": "Higher combines a higher simulator grouping with fewer fixed portal review-rule matches.",
        "bands": {
            "70-100": "higher synthetic grouping / fewer fixed review matches",
            "45-69": "mixed record pattern",
            "0-44": "lower synthetic grouping / more fixed review matches",
        },
        "caveat": "Legacy engineering index only; not health status, treatment effectiveness, prognosis, or medical advice.",
    }


def _overall_status(score, mri_signal, clinical_signal, symptom_signal):
    if score is None:
        return {
            "status": "insufficient_evidence",
            "message": "There is not enough recorded evidence to generate a monitoring index.",
            "recommended_action": "Add available records and bring any current concerns to the care team for review.",
        }
    if clinical_signal["status"] == "needs_review" or symptom_signal["status"] == "needs_review":
        return {
            "status": "needs_clinician_review",
            "message": "Combined signals suggest clinician review should be prioritized.",
            "recommended_action": "Contact the oncology care team or route for medical review.",
        }
    if score < 45 or clinical_signal["status"] == "watch_closely" or symptom_signal["status"] == "watch_closely":
        return {
            "status": "watch_closely",
            "message": "The available record contains one or more fixed review-rule matches.",
            "recommended_action": "Discuss the listed record items with the care team; do not alter treatment from this index.",
        }
    if score >= 70 and mri_signal["status"] == "favorable_response_signal":
        return {
            "status": "favorable_response_signal",
            "message": "The simulator grouping is higher and no major fixed portal review rule matched.",
            "recommended_action": "Continue clinician-directed care and treat this only as an engineering summary.",
        }
    return {
        "status": "on_track_or_no_major_flags",
        "message": "No major combined portal review rule matched in the available data.",
        "recommended_action": "Continue clinician-directed care; absence of a rule match is not reassurance.",
    }


def _load_patient_prediction(patient_id, predictions_csv_path):
    path = Path(predictions_csv_path)
    if not path.exists():
        return None

    predictions = pd.read_csv(path)
    row = predictions[predictions["patient_id"] == patient_id]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def load_patient_shap_explanation(patient_id, explanation_json_path):
    path = Path(explanation_json_path)
    if not path.exists():
        return None
    explanations = json.loads(path.read_text(encoding="utf-8"))
    return explanations.get(patient_id)
