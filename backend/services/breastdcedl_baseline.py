import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


FEATURE_COLUMNS = [
    "age",
    "baseline_longest_diameter_mm",
    "tumor_voxel_count",
    "acq0_mask_mean",
    "acq1_mask_mean",
    "acq2_mask_mean",
    "early_enhancement_mean",
    "delayed_enhancement_mean",
    "washout_mean",
    "early_enhancement_p90",
    "delayed_enhancement_p90",
    "washout_p10",
]
CATEGORICAL_FEATURE_COLUMNS = ["molecular_subtype"]


def run_breastdcedl_baseline(
    manifest_csv_path: str = "Data/breastdcedl_spy1_manifest.csv",
    features_csv_path: str = "Data/breastdcedl_spy1_features.csv",
    metrics_json_path: str = "Data/breastdcedl_spy1_baseline_metrics.json",
    predictions_csv_path: str = "Data/breastdcedl_spy1_model_predictions.csv",
    max_patients: int | None = None,
):
    features = extract_breastdcedl_features(
        manifest_csv_path=manifest_csv_path,
        output_csv_path=features_csv_path,
        max_patients=max_patients,
    )
    metrics = train_pcr_baseline_models(
        features_csv_path=features_csv_path,
        metrics_json_path=metrics_json_path,
        predictions_csv_path=predictions_csv_path,
    )

    return {
        "features": features,
        "metrics": metrics,
        "note": "This is a lightweight PoC baseline, not a validated clinical model.",
    }


def extract_breastdcedl_features(
    manifest_csv_path: str = "Data/breastdcedl_spy1_manifest.csv",
    output_csv_path: str = "Data/breastdcedl_spy1_features.csv",
    max_patients: int | None = None,
):
    manifest = pd.read_csv(manifest_csv_path)
    eligible = manifest[
        (manifest["has_core_dce_inputs"] == True)
        & (manifest["has_mask"] == True)
        & (manifest["pcr_label"].notna())
    ].copy()
    if max_patients:
        eligible = eligible.head(max_patients)

    rows = []
    errors = []
    for _, row in eligible.iterrows():
        try:
            rows.append(_extract_patient_features(row))
        except Exception as exc:
            errors.append({"patient_id": row["patient_id"], "error": str(exc)})

    features = pd.DataFrame(rows)
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)

    return {
        "manifest_csv_path": manifest_csv_path,
        "eligible_patients": len(eligible),
        "features_created": len(features),
        "errors": errors,
        "output_csv_path": str(output_path),
    }


def train_pcr_baseline_models(
    features_csv_path: str = "Data/breastdcedl_spy1_features.csv",
    metrics_json_path: str = "Data/breastdcedl_spy1_baseline_metrics.json",
    predictions_csv_path: str = "Data/breastdcedl_spy1_model_predictions.csv",
):
    features = pd.read_csv(features_csv_path)
    features = features[features["pcr_label"].notna()].copy()
    if len(features) < 20:
        raise ValueError("Need at least 20 feature rows for baseline cross-validation")

    X = features[FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS]
    y = features["pcr_label"].astype(int)
    folds = min(5, int(y.value_counts().min()))
    if folds < 2:
        raise ValueError("Need at least two examples in each pCR class")

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    models = {
        "logistic_regression": _logistic_regression_pipeline(),
        "random_forest": _random_forest_pipeline(),
    }

    model_metrics = {}
    prediction_rows = features[["patient_id", "pcr_label", "molecular_subtype"]].copy()
    for name, model in models.items():
        probabilities = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        model_metrics[name] = {
            "accuracy": round(float(accuracy_score(y, predictions)), 3),
            "balanced_accuracy": round(float(balanced_accuracy_score(y, predictions)), 3),
            "roc_auc": round(float(roc_auc_score(y, probabilities)), 3),
        }
        prediction_rows[f"{name}_pcr_probability"] = np.round(probabilities, 6)
        prediction_rows[f"{name}_predicted_label"] = predictions

    best_model = max(model_metrics, key=lambda key: model_metrics[key]["roc_auc"])
    prediction_rows["best_model"] = best_model
    prediction_rows["best_model_pcr_probability"] = prediction_rows[f"{best_model}_pcr_probability"]
    prediction_rows["best_model_predicted_label"] = prediction_rows[f"{best_model}_predicted_label"]

    metrics = {
        "rows": int(len(features)),
        "positive_pcr": int(y.sum()),
        "negative_pcr": int((y == 0).sum()),
        "cv_folds": int(folds),
        "models": model_metrics,
        "best_model_by_roc_auc": best_model,
        "features_csv_path": features_csv_path,
        "predictions_csv_path": predictions_csv_path,
        "model_type": "cross_validated_tabular_baselines",
        "warning": "Exploratory PoC only. Not clinically validated.",
    }

    output_path = Path(metrics_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    prediction_output_path = Path(predictions_csv_path)
    prediction_output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_rows.to_csv(prediction_output_path, index=False)
    return metrics


def _logistic_regression_pipeline():
    return Pipeline([
        ("preprocess", _preprocessor(scale_numeric=True)),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=2000)),
    ])


def _random_forest_pipeline():
    return Pipeline([
        ("preprocess", _preprocessor(scale_numeric=False)),
        ("classifier", RandomForestClassifier(
            n_estimators=250,
            max_depth=5,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
        )),
    ])


def _preprocessor(scale_numeric):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    from sklearn.pipeline import Pipeline as SklearnPipeline

    return ColumnTransformer([
        ("numeric", SklearnPipeline(numeric_steps), FEATURE_COLUMNS),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURE_COLUMNS),
    ])


def _extract_patient_features(row):
    acq0 = _load_volume(row["acq0_path"])
    acq1 = _load_volume(row["acq1_path"])
    acq2 = _load_volume(row["acq2_path"])
    mask = _load_volume(row["mask_path"]) > 0
    if not np.any(mask):
        raise ValueError("Mask has no positive voxels")

    early = _relative_change(acq1, acq0)
    delayed = _relative_change(acq2, acq0)
    washout = _relative_change(acq2, acq1)

    return {
        "patient_id": row["patient_id"],
        "pcr_label": row["pcr_label"],
        "age": row["age"],
        "baseline_longest_diameter_mm": row["baseline_longest_diameter_mm"],
        "molecular_subtype": row["molecular_subtype"],
        "tumor_voxel_count": int(np.sum(mask)),
        "acq0_mask_mean": _masked_mean(acq0, mask),
        "acq1_mask_mean": _masked_mean(acq1, mask),
        "acq2_mask_mean": _masked_mean(acq2, mask),
        "early_enhancement_mean": _masked_mean(early, mask),
        "delayed_enhancement_mean": _masked_mean(delayed, mask),
        "washout_mean": _masked_mean(washout, mask),
        "early_enhancement_p90": _masked_percentile(early, mask, 90),
        "delayed_enhancement_p90": _masked_percentile(delayed, mask, 90),
        "washout_p10": _masked_percentile(washout, mask, 10),
    }


def _load_volume(path):
    return np.asanyarray(nib.load(str(path)).dataobj).astype(np.float32)


def _relative_change(later, earlier):
    return (later - earlier) / (np.abs(earlier) + 1e-3)


def _masked_mean(volume, mask):
    return round(float(np.mean(volume[mask])), 6)


def _masked_percentile(volume, mask, percentile):
    return round(float(np.percentile(volume[mask], percentile)), 6)
