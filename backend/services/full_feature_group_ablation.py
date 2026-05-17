from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from backend.services.artifact_manifest import build_artifact_manifest
from backend.services.biomarker_feature_benchmark import (
    BIOMARKER_NUMERIC_FEATURES,
    DEFAULT_SOURCE_CSV,
    FORBIDDEN_LEAKAGE_COLUMNS,
    REGRESSION_TARGET,
    TARGET,
    _add_synthetic_biomarker_features,
)


DEFAULT_OUTPUT_PATH = "Data/evals/models/latest_full_feature_group_ablation.json"

DEMOGRAPHIC_FEATURES = ["age"]
CONTEXT_CATEGORICAL = ["stage", "regimen", "molecular_subtype"]
TREATMENT_NUMERIC = ["cycle", "intervention_count", "dose_delayed", "dose_reduced"]
LAB_FEATURES = [
    "pre_wbc",
    "pre_anc",
    "pre_hemoglobin",
    "pre_platelets",
    "nadir_wbc",
    "nadir_anc",
    "nadir_hemoglobin",
    "nadir_platelets",
    "recovery_wbc",
    "recovery_hemoglobin",
    "recovery_platelets",
]
SYMPTOM_FEATURES = ["max_symptom_severity", "symptom_count"]
IMAGING_FEATURES = ["mri_tumor_size_cm", "mri_percent_change_from_baseline"]
BIOMARKER_FEATURES = [
    "er_positive_synthetic",
    "pr_positive_synthetic",
    "her2_positive_synthetic",
    "triple_negative_synthetic",
    "ki67_percent_synthetic",
    "pathology_high_grade_synthetic",
]
GENETIC_READINESS_FEATURES = ["germline_high_risk_gene_flag_synthetic"]
TUMOR_MARKER_FEATURES = [
    "ca15_3_value_synthetic",
    "ca27_29_value_synthetic",
    "cea_value_synthetic",
    "tumor_marker_rising_synthetic",
    "tumor_marker_missing_indicator",
]


def run_full_feature_group_ablation(
    source_csv: str = DEFAULT_SOURCE_CSV,
    output_path: str = DEFAULT_OUTPUT_PATH,
    seed: int = 42,
) -> dict[str, Any]:
    rows = pd.read_csv(source_csv)
    _validate_rows(rows)
    rows = _add_synthetic_biomarker_features(rows, seed=seed)
    train_patients, test_patients = _patient_split(rows, seed=seed)
    train_rows = rows[rows["patient_id"].isin(train_patients)].copy()
    test_rows = rows[rows["patient_id"].isin(test_patients)].copy()

    results: dict[str, Any] = {}
    for name, spec in _feature_groups().items():
        results[name] = _evaluate_group(
            train_rows=train_rows,
            test_rows=test_rows,
            numeric_features=spec["numeric"],
            categorical_features=spec["categorical"],
            seed=seed,
        )
        results[name]["purpose"] = spec["purpose"]
        results[name]["modalities"] = spec["modalities"]
        results[name]["features"] = {
            "numeric": spec["numeric"],
            "categorical": spec["categorical"],
        }

    baseline = results["clinical_timeline_only"]["classification"]
    full = results["clinical_labs_imaging_biomarkers_genetics_tumor_markers"]["classification"]
    recommended = _recommendation(results)
    report = {
        **build_artifact_manifest(seed=seed, dataset_paths={"source_rows": source_csv}),
        "schema_version": "full_feature_group_ablation_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": recommended["status"],
        "claim_boundary": (
            "Synthetic feature-group ablation only. Results test engineering behavior of feature groups, "
            "missingness masks, leakage controls, calibration, and monitor-only promotion logic. They do not "
            "validate clinical utility or prove that biomarkers, genetics-readiness, tumor markers, labs, or "
            "imaging predict real treatment response."
        ),
        "source_csv": source_csv,
        "rows": int(len(rows)),
        "patients": int(rows["patient_id"].nunique()),
        "train_patients": int(len(train_patients)),
        "test_patients": int(len(test_patients)),
        "feature_groups": results,
        "deltas": {
            "full_vs_clinical_auroc_delta": _delta(full.get("patient_level_auroc"), baseline.get("patient_level_auroc")),
            "full_vs_clinical_brier_delta": _delta(full.get("brier"), baseline.get("brier")),
            "full_vs_clinical_ece_delta": _delta(full.get("ece"), baseline.get("ece")),
            "full_vs_clinical_regression_mae_delta": _delta(
                results["clinical_labs_imaging_biomarkers_genetics_tumor_markers"]["regression"].get("mae"),
                results["clinical_timeline_only"]["regression"].get("mae"),
            ),
        },
        "leakage_audit": _leakage_audit(_feature_groups()),
        "recommendation": recommended,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_full_feature_group_ablation(output_path: str = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    path = Path(output_path)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "schema_version": "full_feature_group_ablation_v1",
        "status": "missing",
        "message": "Run scripts/run_full_feature_group_ablation.py to generate this artifact.",
    }


def _feature_groups() -> dict[str, dict[str, Any]]:
    return {
        "clinical_timeline_only": {
            "numeric": DEMOGRAPHIC_FEATURES + TREATMENT_NUMERIC,
            "categorical": CONTEXT_CATEGORICAL,
            "modalities": ["demographics", "treatment_timeline"],
            "purpose": "Age, stage/subtype/regimen, cycle, and intervention context only.",
        },
        "labs_only": {
            "numeric": DEMOGRAPHIC_FEATURES + LAB_FEATURES,
            "categorical": [],
            "modalities": ["demographics", "labs"],
            "purpose": "CBC trend signal only, plus age for minimal demographic context.",
        },
        "symptoms_only": {
            "numeric": DEMOGRAPHIC_FEATURES + SYMPTOM_FEATURES,
            "categorical": [],
            "modalities": ["demographics", "symptoms"],
            "purpose": "Symptom burden only, plus age.",
        },
        "imaging_only": {
            "numeric": DEMOGRAPHIC_FEATURES + IMAGING_FEATURES,
            "categorical": [],
            "modalities": ["demographics", "imaging"],
            "purpose": "MRI/CT/ultrasound-style response signal only, plus age.",
        },
        "clinical_plus_labs": {
            "numeric": DEMOGRAPHIC_FEATURES + TREATMENT_NUMERIC + LAB_FEATURES,
            "categorical": CONTEXT_CATEGORICAL,
            "modalities": ["demographics", "treatment_timeline", "labs"],
            "purpose": "Clinical/treatment context plus CBC trends.",
        },
        "clinical_labs_imaging": {
            "numeric": DEMOGRAPHIC_FEATURES + TREATMENT_NUMERIC + LAB_FEATURES + SYMPTOM_FEATURES + IMAGING_FEATURES,
            "categorical": CONTEXT_CATEGORICAL,
            "modalities": ["demographics", "treatment_timeline", "labs", "symptoms", "imaging"],
            "purpose": "Core monitoring model without biomarkers, genetics-readiness, or tumor markers.",
        },
        "clinical_labs_imaging_biomarkers": {
            "numeric": DEMOGRAPHIC_FEATURES + TREATMENT_NUMERIC + LAB_FEATURES + SYMPTOM_FEATURES + IMAGING_FEATURES + BIOMARKER_FEATURES,
            "categorical": CONTEXT_CATEGORICAL,
            "modalities": ["demographics", "treatment_timeline", "labs", "symptoms", "imaging", "biomarkers"],
            "purpose": "Adds ER/PR/HER2/Ki-67/pathology-style context to the core monitoring set.",
        },
        "clinical_labs_imaging_biomarkers_genetics": {
            "numeric": DEMOGRAPHIC_FEATURES + TREATMENT_NUMERIC + LAB_FEATURES + SYMPTOM_FEATURES + IMAGING_FEATURES + BIOMARKER_FEATURES + GENETIC_READINESS_FEATURES,
            "categorical": CONTEXT_CATEGORICAL,
            "modalities": ["demographics", "treatment_timeline", "labs", "symptoms", "imaging", "biomarkers", "genetic_readiness"],
            "purpose": "Adds genetic-counseling readiness flag as workflow context, not diagnosis.",
        },
        "clinical_labs_imaging_biomarkers_genetics_tumor_markers": {
            "numeric": DEMOGRAPHIC_FEATURES + TREATMENT_NUMERIC + LAB_FEATURES + SYMPTOM_FEATURES + IMAGING_FEATURES + BIOMARKER_FEATURES + GENETIC_READINESS_FEATURES + TUMOR_MARKER_FEATURES,
            "categorical": CONTEXT_CATEGORICAL,
            "modalities": ["demographics", "treatment_timeline", "labs", "symptoms", "imaging", "biomarkers", "genetic_readiness", "tumor_markers"],
            "purpose": "Full candidate feature set with tumor-marker trends as context only.",
        },
    }


def _evaluate_group(
    *,
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    seed: int,
) -> dict[str, Any]:
    x_train = train_rows[numeric_features + categorical_features]
    x_test = test_rows[numeric_features + categorical_features]
    classifier = Pipeline([
        ("preprocessor", _preprocessor(numeric_features, categorical_features)),
        ("model", GradientBoostingClassifier(random_state=seed)),
    ])
    classifier.fit(x_train, train_rows[TARGET].astype(int))
    probabilities = classifier.predict_proba(x_test)[:, 1]
    classification = _classification_metrics(test_rows, probabilities)

    regression_train = train_rows.dropna(subset=[REGRESSION_TARGET])
    regression_test = test_rows.dropna(subset=[REGRESSION_TARGET])
    regressor = Pipeline([
        ("preprocessor", _preprocessor(numeric_features, categorical_features)),
        ("model", GradientBoostingRegressor(random_state=seed)),
    ])
    regressor.fit(regression_train[numeric_features + categorical_features], regression_train[REGRESSION_TARGET])
    predictions = regressor.predict(regression_test[numeric_features + categorical_features])
    regression = {
        "mae": round(float(mean_absolute_error(regression_test[REGRESSION_TARGET], predictions)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(regression_test[REGRESSION_TARGET], predictions))), 4),
        "r2": round(float(r2_score(regression_test[REGRESSION_TARGET], predictions)), 4),
    }
    return {"classification": classification, "regression": regression}


def _preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    transformers: list[tuple[str, Any, list[str]]] = [
        ("numeric", SimpleImputer(strategy="median"), numeric_features),
    ]
    if categorical_features:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_features,
        ))
    return ColumnTransformer(transformers, sparse_threshold=0)


def _classification_metrics(test_rows: pd.DataFrame, probabilities: np.ndarray) -> dict[str, Any]:
    row_metrics = _binary_metrics(test_rows[TARGET].astype(int).to_numpy(), probabilities)
    frame = test_rows[["patient_id", TARGET, "molecular_subtype", "stage"]].copy()
    frame["probability"] = probabilities
    grouped = (
        frame.groupby("patient_id")
        .agg(
            actual_label=(TARGET, "max"),
            probability=("probability", "mean"),
            molecular_subtype=("molecular_subtype", "first"),
            stage=("stage", "first"),
        )
        .reset_index()
    )
    patient_metrics = _binary_metrics(grouped["actual_label"].astype(int).to_numpy(), grouped["probability"].to_numpy())
    subgroup_ece = _subgroup_ece(grouped)
    return {
        **row_metrics,
        **{f"patient_level_{key}": value for key, value in patient_metrics.items()},
        "subgroup_ece": subgroup_ece,
        "subgroup_ece_max": max((v["ece"] for v in subgroup_ece.values() if v["ece"] is not None), default=None),
        "abstention_rate": 0.0,
        "uncertainty_calibration": {
            "method": "ece_proxy_from_patient_level_probabilities",
            "ece": patient_metrics.get("ece"),
            "note": "Formal abstention is evaluated in latest_evidence_abstention_eval.json; this ablation force-scores all rows.",
        },
    }


def _binary_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predicted = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "auroc": _safe_auc(y_true, probabilities),
        "auprc": _safe_auprc(y_true, probabilities),
        "brier": round(float(brier_score_loss(y_true, probabilities)), 4),
        "ece": _ece(y_true, probabilities),
        "sensitivity": round(float(tp / (tp + fn)), 4) if (tp + fn) else None,
        "specificity": round(float(tn / (tn + fp)), 4) if (tn + fp) else None,
        "precision": round(float(tp / (tp + fp)), 4) if (tp + fp) else None,
        "false_negative_count": int(fn),
        "threshold": 0.5,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _subgroup_ece(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for column in ["molecular_subtype", "stage"]:
        for value, group in frame.groupby(column):
            if len(group) < 10:
                continue
            output[f"{column}:{value}"] = {
                "n": int(len(group)),
                "ece": _ece(group["actual_label"].astype(int).to_numpy(), group["probability"].to_numpy()),
            }
    return output


def _ece(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float | None:
    if len(y_true) == 0:
        return None
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= lower) & (probabilities < upper if upper < 1 else probabilities <= upper)
        if not mask.any():
            continue
        ece += (mask.sum() / total) * abs(float(probabilities[mask].mean()) - float(y_true[mask].mean()))
    return round(float(ece), 4)


def _safe_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float | None:
    if len(set(y_true.tolist())) < 2:
        return None
    return round(float(roc_auc_score(y_true, probabilities)), 4)


def _safe_auprc(y_true: np.ndarray, probabilities: np.ndarray) -> float | None:
    if len(set(y_true.tolist())) < 2:
        return None
    return round(float(average_precision_score(y_true, probabilities)), 4)


def _patient_split(rows: pd.DataFrame, seed: int) -> tuple[set[str], set[str]]:
    labels = rows.groupby("patient_id", as_index=False)[TARGET].max().sort_values("patient_id")
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, test_idx = next(splitter.split(labels[["patient_id"]], labels[TARGET]))
    return set(labels.iloc[train_idx]["patient_id"]), set(labels.iloc[test_idx]["patient_id"])


def _validate_rows(rows: pd.DataFrame) -> None:
    required = {"patient_id", TARGET, REGRESSION_TARGET, "molecular_subtype", "stage", "regimen"}
    all_features = {
        *DEMOGRAPHIC_FEATURES,
        *TREATMENT_NUMERIC,
        *LAB_FEATURES,
        *SYMPTOM_FEATURES,
        *IMAGING_FEATURES,
        *CONTEXT_CATEGORICAL,
    }
    missing = sorted((required | all_features) - set(rows.columns))
    if missing:
        raise ValueError(f"Missing required columns for full feature ablation: {missing}")


def _leakage_audit(feature_groups: dict[str, dict[str, Any]]) -> dict[str, Any]:
    violations = []
    for name, spec in feature_groups.items():
        used = set(spec["numeric"]) | set(spec["categorical"])
        bad = sorted(used & FORBIDDEN_LEAKAGE_COLUMNS)
        if bad:
            violations.append({"feature_group": name, "forbidden_columns": bad})
    return {
        "status": "passed" if not violations else "failed",
        "forbidden_columns": sorted(FORBIDDEN_LEAKAGE_COLUMNS),
        "violations": violations,
    }


def _recommendation(results: dict[str, Any]) -> dict[str, Any]:
    clinical = results["clinical_timeline_only"]["classification"]
    full = results["clinical_labs_imaging_biomarkers_genetics_tumor_markers"]["classification"]
    auroc_delta = _delta(full.get("patient_level_auroc"), clinical.get("patient_level_auroc")) or 0.0
    brier_delta = _delta(full.get("brier"), clinical.get("brier")) or 0.0
    ece_delta = _delta(full.get("ece"), clinical.get("ece")) or 0.0
    promotes = auroc_delta >= 0.01 and brier_delta <= 0 and ece_delta <= 0.02
    return {
        "status": "strong" if promotes else "acceptable",
        "promote_feature_set": bool(promotes),
        "recommended_use": "candidate_for_external_validation" if promotes else "monitor_only",
        "reason": (
            "Full feature set improved AUROC without worsening Brier/ECE beyond guardrails."
            if promotes else
            "Do not promote solely from synthetic ablation. Keep biomarker/genetic/tumor-marker features monitor-only until temporal and external validation support them."
        ),
        "guardrails": {
            "min_patient_level_auroc_delta": 0.01,
            "max_brier_delta": 0.0,
            "max_ece_delta": 0.02,
        },
    }


def _delta(new: float | None, old: float | None) -> float | None:
    if new is None or old is None:
        return None
    return round(float(new - old), 4)


__all__ = ["run_full_feature_group_ablation", "load_full_feature_group_ablation", "DEFAULT_OUTPUT_PATH"]
