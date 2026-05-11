"""
Feature-contribution explanations for the synthetic champion model.

Method: SHAP LinearExplainer (logistic regression) when SHAP is available;
        falls back to exact linear coefficient x transformed-feature value
        (mathematically equivalent for logistic regression without calibration).

All explanations and plots are labelled as "feature contribution / model
explanation" - NOT clinical causality or SHAP for a non-linear model.

CLAIM BOUNDARY: explains a model trained on synthetic data only.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backend.services.complete_synthetic_training import CATEGORICAL_FEATURES, NUMERIC_FEATURES

# Optional: SHAP for LinearExplainer
try:
    import shap as _shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

# Optional: matplotlib for PNG plot
try:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


DEFAULT_SYNTHETIC_XAI_PATH = "Data/complete_synthetic_training/synthetic_xai_explanations.json"
DEFAULT_XAI_PLOT_PATH = "Data/complete_synthetic_training/xai_global_feature_importance.png"
DEFAULT_XAI_PLOT_JSON_PATH = "Data/complete_synthetic_training/xai_global_feature_importance_chart.json"


def generate_complete_synthetic_xai(
    ml_csv_path: str = "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv",
    model_path: str = "Data/complete_synthetic_training/logistic_regression_treatment_success_binary.joblib",
    predictions_csv_path: str = "Data/complete_synthetic_training/complete_synthetic_model_predictions.csv",
    output_json_path: str = DEFAULT_SYNTHETIC_XAI_PATH,
    plot_path: str = DEFAULT_XAI_PLOT_PATH,
    plot_json_path: str = DEFAULT_XAI_PLOT_JSON_PATH,
    top_n: int = 6,
):
    model = joblib.load(model_path)
    rows = pd.read_csv(ml_csv_path)
    predictions = (
        pd.read_csv(predictions_csv_path)
        if Path(predictions_csv_path).exists()
        else pd.DataFrame()
    )

    preprocessor = model.named_steps["preprocess"]
    classifier = model.named_steps["classifier"]
    X_raw = rows[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    transformed = preprocessor.transform(X_raw)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed)
    feature_names = [_clean_feature_name(n) for n in preprocessor.get_feature_names_out()]

    if not hasattr(classifier, "coef_"):
        raise ValueError("Synthetic XAI currently expects a linear classifier with coef_.")

    # -- Explanation method ----------------------------------------------------
    if _SHAP_AVAILABLE:
        explainer = _shap.LinearExplainer(classifier, transformed, feature_perturbation="correlation_dependent")
        shap_values = explainer.shap_values(transformed)  # shape (N, F)
        row_contributions = shap_values
        method = "shap_linear_explainer_logistic_regression"
        method_label = "SHAP LinearExplainer (logistic regression)"
    else:
        coefficients = classifier.coef_[0]
        row_contributions = transformed * coefficients
        method = "linear_model_coefficient_contribution"
        method_label = "Linear coefficient x transformed feature (equivalent to SHAP for log-reg)"

    # -- Per-patient explanations ----------------------------------------------
    explanations: dict = {}
    for patient_id, group in rows.reset_index(drop=True).groupby("patient_id"):
        indices = group.index.to_numpy()
        patient_contrib = np.mean(row_contributions[indices], axis=0)
        contributions = [
            {
                "feature": feature_names[i],
                "contribution": round(float(v), 6),
                "direction": "toward_success" if v > 0 else "toward_non_success",
                "meaning": _feature_meaning(feature_names[i]),
            }
            for i, v in enumerate(patient_contrib)
        ]
        positive = sorted(
            [c for c in contributions if c["contribution"] > 0],
            key=lambda c: abs(c["contribution"]),
            reverse=True,
        )[:top_n]
        negative = sorted(
            [c for c in contributions if c["contribution"] < 0],
            key=lambda c: abs(c["contribution"]),
            reverse=True,
        )[:top_n]
        prediction_row = _prediction_for_patient(predictions, patient_id)
        explanations[patient_id] = {
            "patient_id": patient_id,
            "method": method,
            "method_label": method_label,
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

    # -- Global feature importance ---------------------------------------------
    global_mean_abs = np.mean(np.abs(row_contributions), axis=0)
    global_importance = sorted(
        [
            {
                "feature": feature_names[i],
                "importance": round(float(global_mean_abs[i]), 6),
                "meaning": _feature_meaning(feature_names[i]),
            }
            for i in range(len(feature_names))
        ],
        key=lambda x: x["importance"],
        reverse=True,
    )

    # -- Generate PNG plot -----------------------------------------------------
    plot_artifact_path = None
    plot_chart_path = None
    if _MPL_AVAILABLE:
        plot_artifact_path = _save_importance_plot(global_importance, plot_path, method_label)
    plot_chart_path = _save_importance_chart_json(global_importance, plot_json_path, method_label)

    payload = {
        "method": method,
        "method_label": method_label,
        "shap_available": _SHAP_AVAILABLE,
        "plot_png_path": plot_artifact_path,
        "plot_chart_json_path": plot_chart_path,
        "patients": explanations,
        "global_importance": global_importance[:25],
        "warning": (
            "Feature contribution explains a synthetic-data model only. "
            "This is not clinical evidence or causal reasoning."
        ),
    }
    output = Path(output_json_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {
        "patients_explained": len(explanations),
        "method": method,
        "method_label": method_label,
        "shap_available": _SHAP_AVAILABLE,
        "output_json_path": str(output),
        "plot_png_path": plot_artifact_path,
        "plot_chart_json_path": plot_chart_path,
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
        patient_id, response_regression_predictions_csv_path
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


def load_xai_global_importance(xai_json_path: str = DEFAULT_SYNTHETIC_XAI_PATH):
    """Return global feature importance list and plot paths from saved XAI payload."""
    path = Path(xai_json_path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "method": payload.get("method"),
        "method_label": payload.get("method_label"),
        "shap_available": payload.get("shap_available"),
        "plot_png_path": payload.get("plot_png_path"),
        "plot_chart_json_path": payload.get("plot_chart_json_path"),
        "global_importance": payload.get("global_importance") or [],
        "warning": payload.get("warning"),
    }


# -- Plot helpers --------------------------------------------------------------

def _save_importance_plot(global_importance, plot_path: str, method_label: str) -> str | None:
    try:
        top = global_importance[:15]
        features = [item["feature"] for item in reversed(top)]
        importances = [item["importance"] for item in reversed(top)]

        fig, ax = plt.subplots(figsize=(9, 6))
        bars = ax.barh(features, importances, color="#4C72B0", edgecolor="white", height=0.7)
        ax.set_xlabel("Mean |contribution| (feature explanation units)", fontsize=10)
        ax.set_title(
            f"Synthetic Champion Model - Global Feature Explanation\n({method_label})",
            fontsize=11,
            pad=12,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.text(
            0.5, -0.04,
            "Synthetic data only - not clinical evidence.",
            ha="center", fontsize=8, color="gray",
        )
        fig.tight_layout()
        out = Path(plot_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(out)
    except Exception:
        return None


def _save_importance_chart_json(global_importance, chart_path: str, method_label: str) -> str | None:
    """Save a chart-data JSON so the frontend can render it without matplotlib."""
    try:
        top = global_importance[:15]
        chart = {
            "chart_type": "horizontal_bar",
            "title": "Synthetic Champion Model - Global Feature Explanation",
            "subtitle": method_label,
            "disclaimer": "Synthetic data only - not clinical evidence.",
            "x_label": "Mean |contribution|",
            "data": [
                {"feature": item["feature"], "importance": item["importance"], "meaning": item["meaning"]}
                for item in top
            ],
        }
        out = Path(chart_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(chart, indent=2), encoding="utf-8")
        return str(out)
    except Exception:
        return None


# -- Prediction / hybrid signal helpers ---------------------------------------

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


# -- Utilities -----------------------------------------------------------------

def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if pd.isna(value):
        return None
    return value


def _clean_feature_name(name: str) -> str:
    return name.replace("numeric__", "").replace("categorical__", "")


def _feature_meaning(feature: str) -> str:
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
