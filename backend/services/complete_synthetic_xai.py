import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backend.services.complete_synthetic_training import CATEGORICAL_FEATURES, NUMERIC_FEATURES


DEFAULT_SYNTHETIC_XAI_PATH = "Data/complete_synthetic_training/synthetic_xai_explanations.json"


def generate_complete_synthetic_xai(
    ml_csv_path: str = "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv",
    model_path: str = "Data/complete_synthetic_training/logistic_regression_treatment_success_binary.joblib",
    predictions_csv_path: str = "Data/complete_synthetic_training/complete_synthetic_model_predictions.csv",
    output_json_path: str = DEFAULT_SYNTHETIC_XAI_PATH,
    top_n: int = 6,
):
    model = joblib.load(model_path)
    rows = pd.read_csv(ml_csv_path)
    predictions = pd.read_csv(predictions_csv_path) if Path(predictions_csv_path).exists() else pd.DataFrame()

    preprocessor = model.named_steps["preprocess"]
    classifier = model.named_steps["classifier"]
    transformed = preprocessor.transform(rows[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed)
    feature_names = [_clean_feature_name(name) for name in preprocessor.get_feature_names_out()]

    if not hasattr(classifier, "coef_"):
        raise ValueError("Synthetic XAI currently expects a linear classifier with coef_")
    coefficients = classifier.coef_[0]
    row_contributions = transformed * coefficients

    explanations = {}
    for patient_id, group in rows.reset_index(drop=True).groupby("patient_id"):
        indices = group.index.to_numpy()
        patient_contrib = np.mean(row_contributions[indices], axis=0)
        contributions = [
            {
                "feature": feature_names[index],
                "contribution": round(float(value), 6),
                "direction": "toward_success" if value > 0 else "toward_non_success",
                "meaning": _feature_meaning(feature_names[index]),
            }
            for index, value in enumerate(patient_contrib)
        ]
        positive = sorted(
            [item for item in contributions if item["contribution"] > 0],
            key=lambda item: abs(item["contribution"]),
            reverse=True,
        )[:top_n]
        negative = sorted(
            [item for item in contributions if item["contribution"] < 0],
            key=lambda item: abs(item["contribution"]),
            reverse=True,
        )[:top_n]
        prediction_row = _prediction_for_patient(predictions, patient_id)
        explanations[patient_id] = {
            "patient_id": patient_id,
            "method": "linear model contribution approximation on logistic regression pipeline",
            "target": "treatment_success_binary",
            "prediction": prediction_row,
            "positive_contributions": positive,
            "negative_contributions": negative,
            "interpretation_rules": {
                "positive_contribution": "Pushes the synthetic model toward treatment_success_binary=1.",
                "negative_contribution": "Pushes the synthetic model toward treatment_success_binary=0.",
                "safety": "Explains a model trained on synthetic data; not clinical causality.",
            },
        }

    global_importance = sorted(
        [
            {"feature": feature_names[index], "importance": round(float(abs(coefficients[index])), 6)}
            for index in range(len(feature_names))
        ],
        key=lambda item: item["importance"],
        reverse=True,
    )
    payload = {
        "patients": explanations,
        "global_importance": global_importance[:25],
        "warning": "Synthetic model explanation only. Not clinical evidence.",
    }
    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "patients_explained": len(explanations),
        "output_json_path": str(output_path),
        "top_global_features": global_importance[:10],
    }


def load_complete_synthetic_patient_prediction(
    patient_id,
    predictions_csv_path: str = "Data/complete_synthetic_training/complete_synthetic_model_predictions.csv",
    response_regression_predictions_csv_path: str = "Data/complete_synthetic_training/complete_synthetic_response_regression_predictions.csv",
    metrics_json_path: str = "Data/complete_synthetic_training/complete_synthetic_model_metrics.json",
):
    path = Path(predictions_csv_path)
    if not path.exists():
        return None
    predictions = pd.read_csv(path)
    row = predictions[predictions["patient_id"] == patient_id]
    if row.empty:
        return None
    payload = _jsonable(row.iloc[0].to_dict())
    metrics = _load_metrics(metrics_json_path)
    regression_prediction = _load_patient_regression_prediction(
        patient_id,
        response_regression_predictions_csv_path,
    )
    if regression_prediction:
        payload["response_regression_prediction"] = regression_prediction
    payload["hybrid_mle_signal"] = _build_hybrid_mle_signal(payload, regression_prediction, metrics)
    return payload


def load_complete_synthetic_patient_xai(patient_id, xai_json_path: str = DEFAULT_SYNTHETIC_XAI_PATH):
    path = Path(xai_json_path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("patients", {}).get(patient_id)


def _prediction_for_patient(predictions, patient_id):
    if predictions.empty:
        return None
    row = predictions[predictions["patient_id"] == patient_id]
    if row.empty:
        return None
    return _jsonable(row.iloc[0].to_dict())


def _load_metrics(metrics_json_path):
    path = Path(metrics_json_path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _load_patient_regression_prediction(patient_id, predictions_csv_path):
    path = Path(predictions_csv_path)
    if not path.exists():
        return None
    predictions = pd.read_csv(path)
    row = predictions[predictions["patient_id"] == patient_id]
    if row.empty:
        return None
    return _jsonable(row.iloc[0].to_dict())


def _build_hybrid_mle_signal(classification_prediction, regression_prediction, metrics):
    best_classifier = (metrics or {}).get("best_model_by_patient_level_roc_auc")
    probability, probability_source = _best_probability(classification_prediction, best_classifier)
    response_score, response_source = _best_response_score(regression_prediction, metrics)
    if probability is None and response_score is None:
        return None

    components = {}
    if probability is not None:
        components["classification_probability_score"] = round(float(probability) * 100, 1)
    if response_score is not None:
        components["response_regression_score_percent"] = round(float(response_score), 3)
        components["response_regression_normalized_score"] = round(_response_score_to_0_100(response_score), 1)

    if probability is not None and response_score is not None:
        hybrid_score = (0.65 * float(probability) * 100) + (0.35 * _response_score_to_0_100(response_score))
        method = "classification_65_percent_plus_regression_35_percent"
    elif probability is not None:
        hybrid_score = float(probability) * 100
        method = "classification_only"
    else:
        hybrid_score = _response_score_to_0_100(response_score)
        method = "regression_only"

    hybrid_score = round(max(0, min(100, hybrid_score)), 1)
    classification_band = _classification_band(probability)
    regression_band = _regression_band(response_score)
    return {
        "schema_version": "hybrid_mle_signal_v1",
        "hybrid_score": hybrid_score,
        "status": _hybrid_status(hybrid_score),
        "method": method,
        "classification_model": probability_source,
        "classification_probability": round(float(probability), 4) if probability is not None else None,
        "classification_band": classification_band,
        "regression_model": response_source,
        "response_score_percent": round(float(response_score), 3) if response_score is not None else None,
        "regression_band": regression_band,
        "agreement": _signal_agreement(classification_band, regression_band),
        "components": components,
        "interpretation": (
            "Hybrid synthetic MLE signal: classifier estimates likelihood of a favorable simulated journey; "
            "regressor estimates continuous MRI-size reduction signal. This is not clinical validation."
        ),
    }


def _best_probability(prediction, best_classifier):
    keys = []
    if best_classifier:
        keys.append(f"{best_classifier}_calibrated_probability")
        keys.append(f"{best_classifier}_probability")
    keys.extend([
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
    ])
    for key in keys:
        value = (prediction or {}).get(key)
        if value is not None:
            return float(value), key.replace("_calibrated_probability", "").replace("_probability", "")
    return None, None


def _best_response_score(regression_prediction, metrics):
    best_regressor = (
        (metrics or {}).get("best_response_regressor_by_patient_level_mae")
        or ((metrics or {}).get("response_regression") or {}).get("best_model_by_patient_level_mae")
    )
    keys = []
    if best_regressor:
        keys.append(f"{best_regressor}_response_score_percent")
    keys.extend([
        "robust_response_ensemble_response_score_percent",
        "random_forest_regressor_response_score_percent",
        "extra_trees_regressor_response_score_percent",
        "gradient_boosting_regressor_response_score_percent",
        "gradient_boosting_huber_regressor_response_score_percent",
        "huber_regressor_response_score_percent",
        "ridge_regression_response_score_percent",
        "svr_rbf_regressor_response_score_percent",
    ])
    for key in keys:
        value = (regression_prediction or {}).get(key)
        if value is not None:
            return float(value), key.replace("_response_score_percent", "")
    return None, None


def _response_score_to_0_100(response_score_percent):
    return max(0, min(100, 50 + float(response_score_percent)))


def _classification_band(probability):
    if probability is None:
        return None
    if probability >= 0.66:
        return "favorable"
    if probability <= 0.40:
        return "lower"
    return "mixed"


def _regression_band(response_score_percent):
    if response_score_percent is None:
        return None
    if response_score_percent >= 20:
        return "favorable"
    if response_score_percent <= -10:
        return "lower"
    return "mixed"


def _hybrid_status(hybrid_score):
    if hybrid_score >= 70:
        return "favorable_response_signal"
    if hybrid_score < 45:
        return "lower_response_signal_or_review_needed"
    return "mixed_response_signal"


def _signal_agreement(classification_band, regression_band):
    if not classification_band or not regression_band:
        return "single_signal_available"
    if classification_band == regression_band:
        return "aligned"
    if "mixed" in {classification_band, regression_band}:
        return "partially_aligned"
    return "conflicting"


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if pd.isna(value):
        return None
    return value


def _clean_feature_name(name):
    cleaned = name.replace("numeric__", "").replace("categorical__", "")
    return cleaned


def _feature_meaning(feature):
    meanings = {
        "mri_percent_change_from_baseline": "MRI tumor size change over treatment cycles.",
        "mri_tumor_size_cm": "Latest per-cycle synthetic MRI tumor size.",
        "intervention_count": "Number of support interventions in the cycle.",
        "max_symptom_severity": "Highest symptom severity logged in the cycle.",
        "dose_delayed": "Whether a treatment cycle was delayed.",
        "dose_reduced": "Whether a treatment cycle was dose-reduced.",
        "nadir_anc": "Lowest neutrophil count signal after treatment.",
        "nadir_wbc": "Lowest WBC signal after treatment.",
        "nadir_hemoglobin": "Lowest hemoglobin signal after treatment.",
        "nadir_platelets": "Lowest platelet signal after treatment.",
    }
    if feature.startswith("stage_"):
        return "Synthetic cancer stage category."
    if feature.startswith("molecular_subtype_"):
        return "Synthetic molecular subtype category."
    if feature.startswith("regimen_"):
        return "Synthetic treatment regimen category."
    return meanings.get(feature, "Synthetic model input feature.")
