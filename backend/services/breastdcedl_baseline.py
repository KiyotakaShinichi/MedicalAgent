import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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


def run_breastdcedl_baseline(
    manifest_csv_path: str = "Data/breastdcedl_spy1_manifest.csv",
    features_csv_path: str = "Data/breastdcedl_spy1_features.csv",
    metrics_json_path: str = "Data/breastdcedl_spy1_baseline_metrics.json",
    max_patients: int | None = None,
):
    features = extract_breastdcedl_features(
        manifest_csv_path=manifest_csv_path,
        output_csv_path=features_csv_path,
        max_patients=max_patients,
    )
    metrics = train_pcr_baseline(
        features_csv_path=features_csv_path,
        metrics_json_path=metrics_json_path,
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


def train_pcr_baseline(
    features_csv_path: str = "Data/breastdcedl_spy1_features.csv",
    metrics_json_path: str = "Data/breastdcedl_spy1_baseline_metrics.json",
):
    features = pd.read_csv(features_csv_path)
    features = features[features["pcr_label"].notna()].copy()
    if len(features) < 20:
        raise ValueError("Need at least 20 feature rows for baseline cross-validation")

    X = features[FEATURE_COLUMNS]
    y = features["pcr_label"].astype(int)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=2000)),
    ])
    folds = min(5, int(y.value_counts().min()))
    if folds < 2:
        raise ValueError("Need at least two examples in each pCR class")

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    probabilities = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "rows": int(len(features)),
        "positive_pcr": int(y.sum()),
        "negative_pcr": int((y == 0).sum()),
        "cv_folds": int(folds),
        "accuracy": round(float(accuracy_score(y, predictions)), 3),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, predictions)), 3),
        "roc_auc": round(float(roc_auc_score(y, probabilities)), 3),
        "features_csv_path": features_csv_path,
        "model_type": "logistic_regression_cross_validated",
        "warning": "Exploratory PoC only. Not clinically validated.",
    }

    output_path = Path(metrics_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


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
