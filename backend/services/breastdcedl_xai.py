import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.services.breastdcedl_baseline import (
    CATEGORICAL_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    _logistic_regression_pipeline,
)


XAI_OUTPUT_PATH = "Data/breastdcedl_spy1_shap_explanations.json"


def generate_breastdcedl_shap_explanations(
    features_csv_path: str = "Data/breastdcedl_spy1_features.csv",
    output_json_path: str = XAI_OUTPUT_PATH,
    top_n: int = 5,
):
    import shap

    features = pd.read_csv(features_csv_path)
    features = features[features["pcr_label"].notna()].copy()
    X = features[FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS]
    y = features["pcr_label"].astype(int)

    model = _logistic_regression_pipeline()
    model.fit(X, y)

    preprocessor = model.named_steps["preprocess"]
    classifier = model.named_steps["classifier"]
    transformed = preprocessor.transform(X)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    feature_names = [_clean_feature_name(name) for name in preprocessor.get_feature_names_out()]
    explainer = shap.LinearExplainer(classifier, transformed)
    shap_result = explainer(transformed)
    shap_values = np.asarray(shap_result.values)

    explanations = {}
    for row_index, row in features.reset_index(drop=True).iterrows():
        patient_id = row["patient_id"]
        values = shap_values[row_index]
        contributions = [
            {
                "feature": feature_names[col_index],
                "shap_value": round(float(values[col_index]), 6),
                "direction": "toward_pcr" if values[col_index] > 0 else "toward_non_pcr",
                "meaning": _feature_meaning(feature_names[col_index]),
            }
            for col_index in range(len(feature_names))
        ]
        positive = sorted(
            [item for item in contributions if item["shap_value"] > 0],
            key=lambda item: abs(item["shap_value"]),
            reverse=True,
        )[:top_n]
        negative = sorted(
            [item for item in contributions if item["shap_value"] < 0],
            key=lambda item: abs(item["shap_value"]),
            reverse=True,
        )[:top_n]

        explanations[patient_id] = {
            "method": "shap.LinearExplainer on final logistic regression baseline",
            "target": "pCR positive / complete pathologic response",
            "positive_contributions": positive,
            "negative_contributions": negative,
            "interpretation_rules": {
                "positive_shap": "Pushes the model probability toward pCR/favorable complete response.",
                "negative_shap": "Pushes the model probability away from pCR and toward non-pCR.",
                "magnitude": "Larger absolute SHAP value means stronger influence in this model.",
                "safety": "SHAP explains this model's behavior; it does not prove clinical causality.",
            },
        }

    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(explanations, indent=2), encoding="utf-8")

    return {
        "features_csv_path": features_csv_path,
        "output_json_path": str(output_path),
        "patients_explained": len(explanations),
        "method": "shap.LinearExplainer",
    }


def load_patient_shap_explanation(patient_id, explanation_json_path: str = XAI_OUTPUT_PATH):
    path = Path(explanation_json_path)
    if not path.exists():
        return None
    explanations = json.loads(path.read_text(encoding="utf-8"))
    return explanations.get(patient_id)


def _clean_feature_name(name):
    cleaned = name.replace("numeric__", "").replace("categorical__", "")
    cleaned = cleaned.replace("molecular_subtype_", "subtype=")
    return cleaned


def _feature_meaning(feature):
    meanings = {
        "age": "Patient age in the dataset metadata.",
        "baseline_longest_diameter_mm": "Baseline MRI longest tumor diameter.",
        "tumor_voxel_count": "Tumor mask volume proxy.",
        "acq0_mask_mean": "Mean tumor-region signal before/early acquisition.",
        "acq1_mask_mean": "Mean tumor-region signal in acquisition 1.",
        "acq2_mask_mean": "Mean tumor-region signal in acquisition 2.",
        "early_enhancement_mean": "Average early enhancement inside the tumor mask.",
        "delayed_enhancement_mean": "Average delayed enhancement inside the tumor mask.",
        "washout_mean": "Average change from acquisition 1 to acquisition 2.",
        "early_enhancement_p90": "High-end early enhancement inside the tumor mask.",
        "delayed_enhancement_p90": "High-end delayed enhancement inside the tumor mask.",
        "washout_p10": "Low-end washout behavior inside the tumor mask.",
    }
    if feature.startswith("subtype="):
        return "Molecular subtype category used by the model."
    return "Model input feature."
