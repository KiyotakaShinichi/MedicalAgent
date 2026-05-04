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
    score = _treatment_monitoring_score(mri_signal, clinical_signal, symptom_signal)
    overall = _overall_status(score, mri_signal, clinical_signal, symptom_signal)

    return {
        "overall_status": overall["status"],
        "overall_message": overall["message"],
        "treatment_monitoring_score": score,
        "score_interpretation": _score_interpretation(score),
        "signals": {
            "mri_response": mri_signal,
            "clinical_monitoring": clinical_signal,
            "symptoms": symptom_signal,
        },
        "recommended_action": overall["recommended_action"],
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
            message = "MRI model signal leans toward complete treatment response."
        elif probability <= 0.40:
            status = "lower_response_signal"
            message = "MRI model signal leans away from complete treatment response."
        else:
            status = "indeterminate_response_signal"
            message = "MRI model signal is uncertain."

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
    synthetic_probability = _synthetic_response_probability(synthetic_prediction)
    if synthetic_probability is not None:
        explanation = report.get("synthetic_model_explanation")
        if synthetic_probability >= 0.66:
            status = "favorable_response_signal"
            message = "Synthetic longitudinal model leans toward favorable treatment response."
        elif synthetic_probability <= 0.40:
            status = "lower_response_signal"
            message = "Synthetic longitudinal model leans away from favorable treatment response."
        else:
            status = "indeterminate_response_signal"
            message = "Synthetic longitudinal model signal is uncertain."

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
        "logistic_regression_probability",
        "extra_trees_probability",
        "random_forest_probability",
        "gradient_boosting_probability",
        "temporal_gru_probability",
        "temporal_1d_cnn_probability",
    ]:
        value = prediction.get(key)
        if value is not None:
            return float(value)
    return None


def _synthetic_probability_source(prediction):
    for key in [
        "logistic_regression_probability",
        "extra_trees_probability",
        "random_forest_probability",
        "gradient_boosting_probability",
        "temporal_gru_probability",
        "temporal_1d_cnn_probability",
    ]:
        if prediction.get(key) is not None:
            return key.replace("_probability", "")
    return "unknown"


def _clinical_monitoring_signal(report):
    risks = report.get("risks") or []
    urgent = [risk for risk in risks if risk.get("severity") == "urgent_review"]
    watch = [risk for risk in risks if risk.get("severity") == "watch"]

    if urgent:
        status = "needs_review"
        message = f"{len(urgent)} urgent clinical risk flag(s) are present."
    elif watch:
        status = "watch_closely"
        message = f"{len(watch)} watch-level clinical risk flag(s) are present."
    else:
        status = "stable_or_no_flags"
        message = "No urgent CBC/treatment/radiology risk flags are present."

    return {
        "status": status,
        "message": message,
        "risk_count": len(risks),
        "urgent_count": len(urgent),
        "watch_count": len(watch),
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
    base = mri_signal.get("response_signal_score")
    if base is None:
        base = 50

    penalty = 0
    penalty += min(35, clinical_signal.get("urgent_count", 0) * 12)
    penalty += min(20, clinical_signal.get("watch_count", 0) * 5)

    max_severity = symptom_signal.get("max_severity")
    if max_severity is not None:
        penalty += min(12, int(max_severity) * 1.2)

    if clinical_signal.get("has_synthetic_labs"):
        penalty += 3

    return max(0, min(100, round(base - penalty)))


def _score_interpretation(score):
    return {
        "scale": "0-100",
        "meaning": "Higher means stronger available evidence of favorable response with fewer monitoring concerns.",
        "bands": {
            "70-100": "favorable/on-track signal",
            "45-69": "mixed or watch closely",
            "0-44": "lower response signal or clinical concerns",
        },
        "caveat": "Score is an exploratory PoC signal, not a clinical treatment recommendation.",
    }


def _overall_status(score, mri_signal, clinical_signal, symptom_signal):
    if clinical_signal["status"] == "needs_review" or symptom_signal["status"] == "needs_review":
        return {
            "status": "needs_clinician_review",
            "message": "Combined signals suggest clinician review should be prioritized.",
            "recommended_action": "Contact the oncology care team or route for medical review.",
        }
    if score < 45 or clinical_signal["status"] == "watch_closely" or symptom_signal["status"] == "watch_closely":
        return {
            "status": "watch_closely",
            "message": "Combined signals suggest closer monitoring is reasonable.",
            "recommended_action": "Continue monitoring trends and review at the next clinical touchpoint.",
        }
    if score >= 70 and mri_signal["status"] == "favorable_response_signal":
        return {
            "status": "favorable_response_signal",
            "message": "Available signals lean toward favorable treatment response with no major clinical warning flags.",
            "recommended_action": "Continue routine monitoring and clinician-directed care.",
        }
    return {
        "status": "on_track_or_no_major_flags",
        "message": "No major combined warning pattern is present in the available data.",
        "recommended_action": "Continue routine monitoring and clinician-directed care.",
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
