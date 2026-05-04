import json
from pathlib import Path

import pandas as pd

from backend.models import ModelRegistry, PredictionAuditLog
from backend.services.clinician_feedback import clinical_feedback_summary


DEFAULT_SYNTHETIC_METRICS_PATH = "Data/complete_synthetic_training/complete_synthetic_model_metrics.json"
DEFAULT_SYNTHETIC_PREDICTIONS_PATH = "Data/complete_synthetic_training/complete_synthetic_model_predictions.csv"
DEFAULT_SYNTHETIC_TRAINING_CSV = "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv"
DEFAULT_BREASTDCEDL_METRICS_PATH = "Data/breastdcedl_spy1_baseline_metrics.json"


def build_admin_analytics(db):
    synthetic_metrics = _load_json(DEFAULT_SYNTHETIC_METRICS_PATH)
    breastdcedl_metrics = _load_json(DEFAULT_BREASTDCEDL_METRICS_PATH)
    predictions = _load_csv(DEFAULT_SYNTHETIC_PREDICTIONS_PATH)
    training_rows = _load_csv(DEFAULT_SYNTHETIC_TRAINING_CSV)

    return {
        "roles": {
            "patient": "Personal portal, uploads, symptom/CBC/medication logging, support agent.",
            "clinician": "Patient list, timeline review, AI summary approval/edit/reject workflow.",
            "admin": "Model evaluation, drift monitoring, A/B comparison, audit and feedback analytics.",
        },
        "model_performance": _model_performance(synthetic_metrics, breastdcedl_metrics),
        "metric_interpretation_guide": _metric_interpretation_guide(),
        "drift_monitoring": _drift_monitoring(training_rows),
        "ab_testing": _ab_testing(synthetic_metrics, predictions),
        "audit_and_feedback": _audit_and_feedback(db),
        "data_quality": _data_quality(training_rows),
        "safety_positioning": (
            "Admin analytics are for ML engineering monitoring only. They do not diagnose or make treatment decisions."
        ),
    }


def _model_performance(synthetic_metrics, breastdcedl_metrics):
    synthetic_models = (synthetic_metrics or {}).get("models") or {}
    best = (synthetic_metrics or {}).get("best_model_by_patient_level_roc_auc")
    best_metrics = synthetic_models.get(best, {}) if best else {}
    synthetic_payload = {
        "task": (synthetic_metrics or {}).get("task"),
        "best_model": best,
        "patient_level_roc_auc": best_metrics.get("patient_level_roc_auc"),
        "patient_level_average_precision": best_metrics.get("patient_level_average_precision"),
        "patient_level_brier_score": best_metrics.get("patient_level_brier_score"),
        "patient_level_sensitivity": best_metrics.get("patient_level_sensitivity"),
        "patient_level_specificity": best_metrics.get("patient_level_specificity"),
        "warning": (synthetic_metrics or {}).get("warning"),
    }
    synthetic_payload["metric_statuses"] = _score_model_metric_set(synthetic_payload)

    return {
        "synthetic_longitudinal_response": {
            **synthetic_payload,
        },
        "real_breastdcedl_baseline": breastdcedl_metrics or {
            "status": "not_available",
            "message": "BreastDCEDL baseline metrics file was not found.",
        },
    }


def _drift_monitoring(training_rows):
    if training_rows is None or training_rows.empty or "patient_id" not in training_rows.columns:
        return {"status": "unavailable", "features": []}

    patient_order = sorted(training_rows["patient_id"].dropna().unique())
    if len(patient_order) < 4:
        return {"status": "insufficient_data", "features": []}

    midpoint = len(patient_order) // 2
    reference_ids = set(patient_order[:midpoint])
    current_ids = set(patient_order[midpoint:])
    reference = training_rows[training_rows["patient_id"].isin(reference_ids)]
    current = training_rows[training_rows["patient_id"].isin(current_ids)]

    feature_rows = []
    for feature in ["age", "nadir_wbc", "nadir_hemoglobin", "nadir_platelets", "mri_percent_change_from_baseline", "max_symptom_severity"]:
        if feature not in training_rows.columns:
            continue
        ref_mean = float(reference[feature].dropna().mean())
        cur_mean = float(current[feature].dropna().mean())
        ref_std = float(reference[feature].dropna().std() or 1.0)
        standardized_mean_shift = abs(cur_mean - ref_mean) / max(ref_std, 1e-6)
        status = _standardized_shift_status(standardized_mean_shift)
        feature_rows.append({
            "feature": feature,
            "reference_mean": round(ref_mean, 3),
            "current_mean": round(cur_mean, 3),
            "standardized_mean_shift": round(standardized_mean_shift, 3),
            "status": status,
            "meaning": _status_meaning(status),
        })

    watch_count = sum(1 for row in feature_rows if row["status"] in {"unideal", "failed"})
    return {
        "status": "watch" if watch_count else "stable",
        "method": "reference/current split by synthetic patient id; standardized mean shift.",
        "watch_feature_count": watch_count,
        "features": feature_rows,
    }


def _ab_testing(synthetic_metrics, predictions):
    models = (synthetic_metrics or {}).get("models") or {}
    candidates = []
    for name, metrics in models.items():
        candidate = {
            "model": name,
            "patient_level_roc_auc": metrics.get("patient_level_roc_auc"),
            "patient_level_average_precision": metrics.get("patient_level_average_precision"),
            "patient_level_brier_score": metrics.get("patient_level_brier_score"),
            "patient_level_sensitivity": metrics.get("patient_level_sensitivity"),
            "patient_level_specificity": metrics.get("patient_level_specificity"),
        }
        candidate["metric_statuses"] = _score_model_metric_set(candidate)
        candidates.append(candidate)
    candidates = sorted(
        candidates,
        key=lambda row: row.get("patient_level_roc_auc") if row.get("patient_level_roc_auc") is not None else -1,
        reverse=True,
    )

    disagreement = None
    if predictions is not None and not predictions.empty:
        probability_columns = [column for column in predictions.columns if column.endswith("_probability")]
        if len(probability_columns) >= 2:
            label_frame = predictions[probability_columns].apply(lambda column: column >= 0.5)
            disagreement = round(float(label_frame.nunique(axis=1).gt(1).mean()), 3)

    return {
        "champion": candidates[0] if candidates else None,
        "challengers": candidates[1:4],
        "prediction_disagreement_rate": disagreement,
        "recommendation": (
            "Use champion/challenger evaluation offline until clinician-feedback and real-world monitoring data are strong enough."
        ),
    }


def _audit_and_feedback(db):
    return {
        "registered_model_count": db.query(ModelRegistry).count(),
        "prediction_audit_count": db.query(PredictionAuditLog).count(),
        "clinical_feedback": clinical_feedback_summary(db),
    }


def _data_quality(training_rows):
    if training_rows is None or training_rows.empty:
        return {"status": "unavailable", "missingness": []}

    missingness = []
    for column in training_rows.columns:
        rate = float(training_rows[column].isna().mean())
        if rate:
            status = _missing_rate_status(rate)
            missingness.append({
                "column": column,
                "missing_rate": round(rate, 3),
                "status": status,
                "meaning": _status_meaning(status),
            })
    return {
        "status": "watch" if any(row["status"] in {"unideal", "failed"} for row in missingness) else "ok",
        "rows": int(len(training_rows)),
        "patients": int(training_rows["patient_id"].nunique()) if "patient_id" in training_rows.columns else None,
        "missingness": sorted(missingness, key=lambda row: row["missing_rate"], reverse=True)[:20],
    }


def _load_json(path):
    json_path = Path(path)
    if not json_path.exists():
        return None
    return json.loads(json_path.read_text(encoding="utf-8"))


def _load_csv(path):
    csv_path = Path(path)
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def _metric_interpretation_guide():
    return {
        "status_levels": [
            {
                "status": "failed",
                "meaning": "Below the minimum engineering gate. Do not present as a reliable model signal.",
            },
            {
                "status": "unideal",
                "meaning": "Works weakly or has important risk. Useful for debugging, not for headline claims.",
            },
            {
                "status": "acceptable",
                "meaning": "Reasonable for a PoC if limitations are explicit and humans stay in the loop.",
            },
            {
                "status": "strong",
                "meaning": "Good engineering result on the current validation setup.",
            },
            {
                "status": "passed",
                "meaning": "Passes this project gate. This is not clinical validation.",
            },
        ],
        "model_metric_bands": {
            "AUROC": "Failed <0.60, unideal 0.60-0.70, acceptable 0.70-0.80, strong 0.80-0.90, passed >=0.90.",
            "Average precision / AUPRC": "Interpreted relative to class prevalence. For this PoC: failed <0.50, unideal 0.50-0.65, acceptable 0.65-0.80, strong 0.80-0.90, passed >=0.90.",
            "Sensitivity": "False negatives matter in monitoring. Failed <0.75, unideal 0.75-0.85, acceptable 0.85-0.90, strong 0.90-0.95, passed >=0.95.",
            "Specificity": "Controls false alarms. Failed <0.60, unideal 0.60-0.70, acceptable 0.70-0.80, strong 0.80-0.90, passed >=0.90.",
            "Brier score": "Lower is better. Failed >0.25, unideal 0.18-0.25, acceptable 0.12-0.18, strong 0.08-0.12, passed <=0.08.",
            "Drift standardized mean shift": "Lower is better. Passed <0.20, acceptable 0.20-0.50, unideal 0.50-0.80, failed >=0.80.",
            "Missingness": "Lower is better. Passed <=5%, acceptable <=10%, unideal <=20%, failed >20%.",
        },
        "what_current_metrics_do_not_prove": [
            "They do not prove clinical safety.",
            "They do not prove generalization to real hospitals.",
            "They do not prove fairness across age, stage, subtype, or scanner/site groups.",
            "Synthetic-data metrics mostly prove that the model learned the simulator.",
        ],
        "recommended_next_metrics": [
            "Calibration curve and expected calibration error.",
            "Decision curve analysis / net benefit.",
            "False-negative case review table.",
            "Subgroup performance by stage, molecular subtype, age band, and data source.",
            "Confidence intervals via bootstrap resampling.",
            "Alert precision: how many clinician-review flags were accepted by clinicians.",
            "Time-to-review and override-rate metrics for clinician workflow.",
            "Data freshness and missing-baseline indicators.",
            "Scanner/site/protocol drift when real MRI metadata is available.",
            "LLM summary quality: factuality, completeness, safety, and clinician edit distance.",
        ],
        "next_steps": [
            "Add calibration plots and ECE to the training output.",
            "Add subgroup metric tables for synthetic stage/subtype and BreastDCEDL real-data groups.",
            "Add false-negative review cards to the admin dashboard.",
            "Start logging clinician decisions and compare AI flags against accepted/rejected reviews.",
            "Separate synthetic simulator metrics from real-dataset metrics visually in the dashboard.",
        ],
    }


def _score_model_metric_set(metrics):
    return {
        "patient_level_roc_auc": _higher_is_better_status(metrics.get("patient_level_roc_auc"), [0.60, 0.70, 0.80, 0.90]),
        "patient_level_average_precision": _higher_is_better_status(metrics.get("patient_level_average_precision"), [0.50, 0.65, 0.80, 0.90]),
        "patient_level_sensitivity": _higher_is_better_status(metrics.get("patient_level_sensitivity"), [0.75, 0.85, 0.90, 0.95]),
        "patient_level_specificity": _higher_is_better_status(metrics.get("patient_level_specificity"), [0.60, 0.70, 0.80, 0.90]),
        "patient_level_brier_score": _lower_is_better_status(metrics.get("patient_level_brier_score"), [0.25, 0.18, 0.12, 0.08]),
    }


def _higher_is_better_status(value, thresholds):
    if value is None:
        return "unavailable"
    if value < thresholds[0]:
        return "failed"
    if value < thresholds[1]:
        return "unideal"
    if value < thresholds[2]:
        return "acceptable"
    if value < thresholds[3]:
        return "strong"
    return "passed"


def _lower_is_better_status(value, thresholds):
    if value is None:
        return "unavailable"
    if value > thresholds[0]:
        return "failed"
    if value > thresholds[1]:
        return "unideal"
    if value > thresholds[2]:
        return "acceptable"
    if value > thresholds[3]:
        return "strong"
    return "passed"


def _standardized_shift_status(value):
    if value < 0.20:
        return "passed"
    if value < 0.50:
        return "acceptable"
    if value < 0.80:
        return "unideal"
    return "failed"


def _missing_rate_status(value):
    if value <= 0.05:
        return "passed"
    if value <= 0.10:
        return "acceptable"
    if value <= 0.20:
        return "unideal"
    return "failed"


def _status_meaning(status):
    meanings = {
        "failed": "Below gate; fix before relying on this signal.",
        "unideal": "Weak or risky; investigate before presenting as strong.",
        "acceptable": "Usable for PoC with clear caveats.",
        "strong": "Good current engineering signal.",
        "passed": "Meets this project gate, not clinical validation.",
        "unavailable": "Metric could not be computed.",
    }
    return meanings.get(status, "Status not recognized.")
