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
):
    path = Path(predictions_csv_path)
    if not path.exists():
        return None
    predictions = pd.read_csv(path)
    row = predictions[predictions["patient_id"] == patient_id]
    if row.empty:
        return None
    return _jsonable(row.iloc[0].to_dict())


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
