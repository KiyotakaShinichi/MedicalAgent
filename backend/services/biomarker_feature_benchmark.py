from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from backend.services.artifact_manifest import build_artifact_manifest


DEFAULT_SOURCE_CSV = "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv"
DEFAULT_OUTPUT_PATH = "Data/mle_monitoring/biomarker_feature_benchmark.json"
DEFAULT_PREDICTIONS_CSV = "Data/mle_monitoring/biomarker_feature_benchmark_predictions.csv"
TARGET = "treatment_success_binary"
REGRESSION_TARGET = "response_score_percent"

BASE_NUMERIC_FEATURES = [
    "cycle",
    "age",
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
    "mri_tumor_size_cm",
    "mri_percent_change_from_baseline",
    "max_symptom_severity",
    "symptom_count",
    "intervention_count",
    "dose_delayed",
    "dose_reduced",
]

CLINICAL_NUMERIC_FEATURES = [
    feature for feature in BASE_NUMERIC_FEATURES
    if not feature.startswith("mri_")
]

IMAGING_NUMERIC_FEATURES = [
    "mri_tumor_size_cm",
    "mri_percent_change_from_baseline",
]

BASE_CATEGORICAL_FEATURES = ["stage", "regimen"]
CURRENT_DEFAULT_CATEGORICAL_FEATURES = ["stage", "regimen", "molecular_subtype"]

BIOMARKER_NUMERIC_FEATURES = [
    "er_positive_synthetic",
    "pr_positive_synthetic",
    "her2_positive_synthetic",
    "triple_negative_synthetic",
    "ki67_percent_synthetic",
    "pathology_high_grade_synthetic",
    "germline_high_risk_gene_flag_synthetic",
    "ca15_3_value_synthetic",
    "ca27_29_value_synthetic",
    "cea_value_synthetic",
    "tumor_marker_rising_synthetic",
    "tumor_marker_missing_indicator",
]

ENHANCED_NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + BIOMARKER_NUMERIC_FEATURES

FORBIDDEN_LEAKAGE_COLUMNS = {
    "latent_response_strength",
    "response_score_percent",
    "final_response_category",
    "final_cancer_status",
    "final_response_multiclass",
    "treatment_success_binary",
    "maintenance_needed",
    "toxicity_risk_binary",
    "support_intervention_needed",
    "urgent_intervention_needed",
    "cycle_response_trend_class",
}


def run_biomarker_feature_benchmark(
    source_csv: str = DEFAULT_SOURCE_CSV,
    output_path: str = DEFAULT_OUTPUT_PATH,
    predictions_csv_path: str = DEFAULT_PREDICTIONS_CSV,
    seed: int = 42,
) -> dict[str, Any]:
    rows = pd.read_csv(source_csv)
    _validate_source_rows(rows)
    rows = _add_synthetic_biomarker_features(rows, seed=seed)

    train_patients, test_patients = _patient_split(rows, seed=seed)
    train_rows = rows[rows["patient_id"].isin(train_patients)].copy()
    test_rows = rows[rows["patient_id"].isin(test_patients)].copy()

    feature_sets = _feature_sets()
    classification_results: dict[str, Any] = {}
    regression_results: dict[str, Any] = {}
    prediction_frames: list[pd.DataFrame] = []

    for feature_set_name, spec in feature_sets.items():
        classification = _evaluate_classification_set(
            train_rows=train_rows,
            test_rows=test_rows,
            numeric_features=spec["numeric"],
            categorical_features=spec["categorical"],
            seed=seed,
        )
        regression = _evaluate_regression_set(
            train_rows=train_rows,
            test_rows=test_rows,
            numeric_features=spec["numeric"],
            categorical_features=spec["categorical"],
            seed=seed,
        )
        classification_results[feature_set_name] = classification["metrics"]
        regression_results[feature_set_name] = regression["metrics"]
        prediction_frames.append(classification["predictions"].assign(feature_set=feature_set_name))

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions_path = Path(predictions_csv_path)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_path, index=False)

    baseline = classification_results["clinical_timeline_only"]["best_model"]
    biomarker_only = classification_results["clinical_plus_biomarkers"]["best_model"]
    enhanced = classification_results["clinical_plus_biomarkers_plus_imaging"]["best_model"]
    default = classification_results["current_default_with_subtype"]["best_model"]
    deltas = {
        "biomarker_vs_clinical_auroc_delta": _delta(biomarker_only.get("patient_level_auroc"), baseline.get("patient_level_auroc")),
        "biomarker_imaging_vs_clinical_auroc_delta": _delta(enhanced.get("patient_level_auroc"), baseline.get("patient_level_auroc")),
        "enhanced_vs_current_default_auroc_delta": _delta(enhanced.get("patient_level_auroc"), default.get("patient_level_auroc")),
        "biomarker_vs_clinical_brier_delta": _delta(biomarker_only.get("brier"), baseline.get("brier")),
        "biomarker_imaging_vs_clinical_brier_delta": _delta(enhanced.get("brier"), baseline.get("brier")),
        "enhanced_vs_current_default_brier_delta": _delta(enhanced.get("brier"), default.get("brier")),
    }

    report = {
        **build_artifact_manifest(seed=seed, dataset_paths={"biomarker_feature_source_rows": source_csv}),
        "schema_version": "biomarker_feature_benchmark_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": _status_from_deltas(deltas),
        "claim_boundary": (
            "Synthetic feature-ablation benchmark only. Biomarker, tumor-marker, and genetics-style features are "
            "simulated from existing synthetic rows to test engineering readiness, missingness handling, leakage checks, "
            "and retraining workflow. This is not clinical validation and not evidence that these markers predict outcomes in real patients."
        ),
        "source_csv": source_csv,
        "rows": int(len(rows)),
        "patients": int(rows["patient_id"].nunique()),
        "train_patients": int(len(train_patients)),
        "test_patients": int(len(test_patients)),
        "feature_sets": {
            name: {
                "numeric": spec["numeric"],
                "categorical": spec["categorical"],
                "purpose": spec["purpose"],
            }
            for name, spec in feature_sets.items()
        },
        "feature_lineage": _feature_lineage(),
        "source_alignment": _source_alignment(),
        "tumor_marker_policy": _tumor_marker_policy(),
        "missingness_report": _missingness_report(rows, ENHANCED_NUMERIC_FEATURES),
        "leakage_audit": _leakage_audit(feature_sets),
        "classification": classification_results,
        "response_regression": regression_results,
        "deltas": deltas,
        "recommendation": _recommendation(deltas),
        "artifacts": {
            "predictions_csv": str(predictions_path),
            "report_json": output_path,
        },
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_biomarker_feature_benchmark(output_path: str = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    saved = Path(output_path)
    if saved.exists():
        try:
            return json.loads(saved.read_text(encoding="utf-8"))
        except Exception:
            pass
    return run_biomarker_feature_benchmark(output_path=output_path)


def _validate_source_rows(rows: pd.DataFrame) -> None:
    required = {"patient_id", TARGET, REGRESSION_TARGET, "molecular_subtype", "stage", "regimen"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"Missing required source columns: {missing}")
    if rows["patient_id"].nunique() < 20:
        raise ValueError("Biomarker feature benchmark needs at least 20 patients for a patient-level split.")
    if rows.groupby("patient_id")[TARGET].max().nunique() < 2:
        raise ValueError(f"Target {TARGET} needs at least two classes.")


def _add_synthetic_biomarker_features(rows: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = rows.copy()
    subtype = rows["molecular_subtype"].fillna("").str.lower()
    stage = rows["stage"].fillna("").str.upper()
    cycle = pd.to_numeric(rows["cycle"], errors="coerce").fillna(1)
    tumor_size = pd.to_numeric(rows.get("mri_tumor_size_cm"), errors="coerce").fillna(3.0)
    symptoms = pd.to_numeric(rows.get("max_symptom_severity"), errors="coerce").fillna(0)

    rows["er_positive_synthetic"] = subtype.str.contains("hr\\+|hr-positive|er\\+").astype(int)
    rows["pr_positive_synthetic"] = rows["er_positive_synthetic"]
    rows["her2_positive_synthetic"] = subtype.str.contains("her2\\+|her2-positive").astype(int)
    rows["triple_negative_synthetic"] = subtype.str.contains("triple-negative|tnbc").astype(int)
    rows["pathology_high_grade_synthetic"] = (
        rows["triple_negative_synthetic"].eq(1)
        | rows["her2_positive_synthetic"].eq(1)
        | stage.isin({"IIIB", "IV"})
    ).astype(int)
    rows["germline_high_risk_gene_flag_synthetic"] = [
        1 if _stable_unit_interval(f"{pid}:{seed}:germline") < _germline_probability(st, sub) else 0
        for pid, st, sub in zip(rows["patient_id"], stage, subtype)
    ]

    ki67_base = (
        14
        + rows["triple_negative_synthetic"] * 24
        + rows["her2_positive_synthetic"] * 16
        + rows["pathology_high_grade_synthetic"] * 7
        + cycle.clip(0, 8) * 0.4
    )
    rows["ki67_percent_synthetic"] = (ki67_base + _stable_noise(rows, seed, "ki67", scale=3.5)).clip(2, 95).round(2)

    stage_load = stage.map({"I": 0.0, "IA": 0.0, "IB": 0.1, "IIA": 0.2, "IIB": 0.35, "IIIA": 0.55, "IIIB": 0.75, "IV": 1.15}).fillna(0.35)
    biology_load = (
        rows["triple_negative_synthetic"] * 0.35
        + rows["her2_positive_synthetic"] * 0.2
        + rows["pathology_high_grade_synthetic"] * 0.15
    )
    marker_base = 18 + stage_load * 18 + biology_load * 12 + tumor_size * 1.8 + symptoms * 0.7
    marker_trend = (cycle - 1) * (0.9 + stage_load * 0.8 + biology_load * 0.5)
    rows["ca15_3_value_synthetic"] = (marker_base + marker_trend + _stable_noise(rows, seed, "ca153", scale=4.0)).clip(4, 180).round(2)
    rows["ca27_29_value_synthetic"] = (marker_base * 1.12 + marker_trend + _stable_noise(rows, seed, "ca2729", scale=4.5)).clip(5, 220).round(2)
    rows["cea_value_synthetic"] = (2.0 + stage_load * 2.4 + biology_load * 1.2 + symptoms * 0.2 + _stable_noise(rows, seed, "cea", scale=0.8)).clip(0.2, 40).round(2)
    rows["tumor_marker_rising_synthetic"] = (marker_trend > 3.5).astype(int)

    marker_columns = ["ca15_3_value_synthetic", "ca27_29_value_synthetic", "cea_value_synthetic"]
    missing_any = np.zeros(len(rows), dtype=int)
    for column, rate in {"ca15_3_value_synthetic": 0.12, "ca27_29_value_synthetic": 0.16, "cea_value_synthetic": 0.1}.items():
        mask = np.array([
            _stable_unit_interval(f"{pid}:{cy}:{seed}:{column}:missing") < rate
            for pid, cy in zip(rows["patient_id"], cycle)
        ])
        missing_any = np.maximum(missing_any, mask.astype(int))
        rows.loc[mask, column] = np.nan
    rows["tumor_marker_missing_indicator"] = missing_any
    return rows


def _germline_probability(stage: str, subtype: str) -> float:
    probability = 0.06
    if "triple" in subtype or "tnbc" in subtype:
        probability += 0.09
    if "IV" in stage:
        probability += 0.04
    return min(probability, 0.24)


def _stable_noise(rows: pd.DataFrame, seed: int, label: str, scale: float) -> np.ndarray:
    values = [
        (_stable_unit_interval(f"{pid}:{cycle}:{seed}:{label}") - 0.5) * 2 * scale
        for pid, cycle in zip(rows["patient_id"], rows["cycle"])
    ]
    return np.array(values, dtype=float)


def _stable_unit_interval(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return int(digest, 16) / float(16**12 - 1)


def _feature_sets() -> dict[str, dict[str, Any]]:
    return {
        "clinical_timeline_only": {
            "numeric": CLINICAL_NUMERIC_FEATURES,
            "categorical": BASE_CATEGORICAL_FEATURES,
            "purpose": "Treatment-cycle, CBC, symptom, intervention, stage, and regimen features without receptor/genetic/tumor-marker or imaging response fields.",
        },
        "clinical_plus_biomarkers": {
            "numeric": CLINICAL_NUMERIC_FEATURES + BIOMARKER_NUMERIC_FEATURES,
            "categorical": BASE_CATEGORICAL_FEATURES,
            "purpose": "Clinical timeline plus structured receptor/Ki-67/germline-style and tumor-marker trend features, intentionally excluding imaging response fields.",
        },
        "clinical_plus_biomarkers_plus_imaging": {
            "numeric": CLINICAL_NUMERIC_FEATURES + BIOMARKER_NUMERIC_FEATURES + IMAGING_NUMERIC_FEATURES,
            "categorical": BASE_CATEGORICAL_FEATURES,
            "purpose": "Full candidate model: clinical timeline, biomarker/tumor-marker features, and MRI response features.",
        },
        "monitoring_without_biomarkers": {
            "numeric": BASE_NUMERIC_FEATURES,
            "categorical": BASE_CATEGORICAL_FEATURES,
            "purpose": "Legacy baseline: labs, imaging trend, symptoms, treatment-cycle context, stage, and regimen without receptor/genetic/tumor-marker fields.",
        },
        "current_default_with_subtype": {
            "numeric": BASE_NUMERIC_FEATURES,
            "categorical": CURRENT_DEFAULT_CATEGORICAL_FEATURES,
            "purpose": "Approximation of the current default tabular feature set, including molecular_subtype.",
        },
        "enhanced_biomarker_tumor_marker": {
            "numeric": ENHANCED_NUMERIC_FEATURES,
            "categorical": BASE_CATEGORICAL_FEATURES,
            "purpose": "Legacy enhanced set with structured ER/PR/HER2/Ki-67, synthetic germline-risk flag, tumor-marker trends, and imaging features.",
        },
    }


def _patient_split(rows: pd.DataFrame, seed: int) -> tuple[set[str], set[str]]:
    labels = (
        rows.groupby("patient_id", as_index=False)[TARGET]
        .max()
        .sort_values("patient_id")
        .reset_index(drop=True)
    )
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, test_idx = next(splitter.split(labels[["patient_id"]], labels[TARGET]))
    train = set(labels.iloc[train_idx]["patient_id"])
    test = set(labels.iloc[test_idx]["patient_id"])
    return train, test


def _evaluate_classification_set(
    *,
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    seed: int,
) -> dict[str, Any]:
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=180, min_samples_leaf=4, class_weight="balanced", random_state=seed),
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
    }
    metrics: dict[str, Any] = {}
    prediction_frames = []
    best_name = None
    best_auc = -np.inf

    for name, estimator in models.items():
        model = Pipeline([
            ("preprocessor", _preprocessor(numeric_features, categorical_features, scale_numeric=name == "logistic_regression")),
            ("model", estimator),
        ])
        model.fit(train_rows[numeric_features + categorical_features], train_rows[TARGET].astype(int))
        probabilities = model.predict_proba(test_rows[numeric_features + categorical_features])[:, 1]
        row_metrics = _binary_metrics(test_rows[TARGET].astype(int).to_numpy(), probabilities)
        patient_metrics, patient_predictions = _patient_level_classification(test_rows, probabilities)
        combined = {**row_metrics, **{f"patient_level_{k}": v for k, v in patient_metrics.items()}}
        metrics[name] = combined
        if (patient_metrics.get("auroc") or -np.inf) > best_auc:
            best_auc = patient_metrics.get("auroc") or -np.inf
            best_name = name
        prediction_frames.append(patient_predictions.assign(model=name))

    metrics["best_model_name"] = best_name
    metrics["best_model"] = metrics.get(best_name or "", {})
    return {"metrics": metrics, "predictions": pd.concat(prediction_frames, ignore_index=True)}


def _evaluate_regression_set(
    *,
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    seed: int,
) -> dict[str, Any]:
    train = train_rows.dropna(subset=[REGRESSION_TARGET]).copy()
    test = test_rows.dropna(subset=[REGRESSION_TARGET]).copy()
    models = {
        "random_forest_regressor": RandomForestRegressor(n_estimators=180, min_samples_leaf=4, random_state=seed),
        "gradient_boosting_regressor": GradientBoostingRegressor(random_state=seed),
    }
    metrics: dict[str, Any] = {}
    best_name = None
    best_mae = np.inf
    for name, estimator in models.items():
        model = Pipeline([
            ("preprocessor", _preprocessor(numeric_features, categorical_features, scale_numeric=False)),
            ("model", estimator),
        ])
        model.fit(train[numeric_features + categorical_features], train[REGRESSION_TARGET])
        predictions = model.predict(test[numeric_features + categorical_features])
        mae = float(mean_absolute_error(test[REGRESSION_TARGET], predictions))
        rmse = float(np.sqrt(mean_squared_error(test[REGRESSION_TARGET], predictions)))
        r2 = float(r2_score(test[REGRESSION_TARGET], predictions))
        metrics[name] = {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}
        if mae < best_mae:
            best_mae = mae
            best_name = name
    metrics["best_model_name"] = best_name
    metrics["best_model"] = metrics.get(best_name or "", {})
    return {"metrics": metrics}


def _preprocessor(numeric_features: list[str], categorical_features: list[str], scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric_features),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_features),
        ],
        sparse_threshold=0,
    )


def _binary_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float | None]:
    predicted = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "auroc": _safe_auc(y_true, probabilities),
        "auprc": _safe_auprc(y_true, probabilities),
        "brier": round(float(brier_score_loss(y_true, probabilities)), 4),
        "sensitivity": round(float(tp / (tp + fn)), 4) if (tp + fn) else None,
        "specificity": round(float(tn / (tn + fp)), 4) if (tn + fp) else None,
        "false_negative_rate": round(float(fn / (tp + fn)), 4) if (tp + fn) else None,
        "threshold": 0.5,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _patient_level_classification(test_rows: pd.DataFrame, probabilities: np.ndarray) -> tuple[dict[str, Any], pd.DataFrame]:
    frame = test_rows[["patient_id", TARGET]].copy()
    frame["probability"] = probabilities
    grouped = (
        frame.groupby("patient_id")
        .agg(actual_label=(TARGET, "max"), probability=("probability", "mean"))
        .reset_index()
    )
    metrics = _binary_metrics(grouped["actual_label"].astype(int).to_numpy(), grouped["probability"].to_numpy())
    grouped["predicted_label"] = (grouped["probability"] >= 0.5).astype(int)
    grouped["prediction_type"] = np.select(
        [
            (grouped["actual_label"].eq(1) & grouped["predicted_label"].eq(1)),
            (grouped["actual_label"].eq(0) & grouped["predicted_label"].eq(1)),
            (grouped["actual_label"].eq(0) & grouped["predicted_label"].eq(0)),
            (grouped["actual_label"].eq(1) & grouped["predicted_label"].eq(0)),
        ],
        ["TP", "FP", "TN", "FN"],
        default="unknown",
    )
    return metrics, grouped


def _safe_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float | None:
    if len(set(y_true.tolist())) < 2:
        return None
    return round(float(roc_auc_score(y_true, probabilities)), 4)


def _safe_auprc(y_true: np.ndarray, probabilities: np.ndarray) -> float | None:
    if len(set(y_true.tolist())) < 2:
        return None
    return round(float(average_precision_score(y_true, probabilities)), 4)


def _feature_lineage() -> dict[str, Any]:
    return {
        "er_positive_synthetic": {"source_columns": ["molecular_subtype"], "leakage_risk": "low", "note": "Synthetic receptor proxy derived from subtype text."},
        "pr_positive_synthetic": {"source_columns": ["molecular_subtype"], "leakage_risk": "low", "note": "Synthetic receptor proxy derived from subtype text."},
        "her2_positive_synthetic": {"source_columns": ["molecular_subtype"], "leakage_risk": "low", "note": "Synthetic HER2 proxy derived from subtype text."},
        "triple_negative_synthetic": {"source_columns": ["molecular_subtype"], "leakage_risk": "low", "note": "Synthetic subtype proxy."},
        "ki67_percent_synthetic": {"source_columns": ["molecular_subtype", "stage", "cycle"], "leakage_risk": "medium", "note": "Simulated proliferation marker. Uses cycle context, not future outcome labels."},
        "germline_high_risk_gene_flag_synthetic": {"source_columns": ["molecular_subtype", "stage", "patient_id_hash"], "leakage_risk": "medium", "note": "Synthetic genetics-readiness flag for workflow testing only."},
        "tumor_marker_values_synthetic": {"source_columns": ["stage", "molecular_subtype", "cycle", "mri_tumor_size_cm", "max_symptom_severity"], "leakage_risk": "medium", "note": "Simulated monitoring values. Not used as diagnosis; trend features require temporal audit before champion promotion."},
    }


def _source_alignment() -> dict[str, Any]:
    return {
        "breastdcedl": {
            "status": "mapped_locally",
            "local_artifact": "Data/breastdcedl_spy1_features.csv",
            "usable_predictors": ["age", "molecular_subtype", "HR/HER2-derived subtype proxy", "DCE-MRI enhancement features"],
            "target": "pcr_label",
            "role": "first external benchmark for HR/HER2/pCR plus imaging-response features.",
        },
        "tcga_brca_cbioportal": {
            "status": "schema_candidate",
            "usable_predictors": ["clinical subtype", "ER/PR/HER2-derived fields when available", "stage", "genomics"],
            "target": "survival/progression fields, not direct treatment-cycle response",
            "role": "external subtype/genomic distribution check before any real predictor claim.",
        },
        "metabric_cbioportal": {
            "status": "schema_candidate",
            "usable_predictors": ["ER", "PR", "HER2", "PAM50/subtype", "grade", "stage", "expression/copy-number"],
            "target": "survival/outcome fields where available",
            "role": "published-cohort context for biomarker/subtype priors.",
        },
        "aacr_genie_bpc_brca": {
            "status": "future_access_candidate",
            "usable_predictors": ["clinical-grade NGS", "ER/PR/HER2", "Oncotype DX", "multigene signatures", "selected biomarkers"],
            "target": "real-world treatment response, PFS, OS",
            "role": "highest-value future real-world biomarker benchmark; requires access workflow.",
        },
        "nci_edrn_breast_reference_set": {
            "status": "monitoring_context_only",
            "usable_predictors": ["CA15-3", "CEA-family/CEACAM5", "CA125", "CRP", "EGFR", "ERBB2"],
            "target": "reference-set case/control labels, not treatment response",
            "role": "tumor-marker realism and cautionary education; not a standalone predictor source.",
        },
    }


def _tumor_marker_policy() -> dict[str, Any]:
    return {
        "role": "monitoring_context_only",
        "allowed_uses": [
            "trend feature in clinician-reviewed research/evaluation model",
            "missingness and data-quality signal",
            "patient education about limitations and care-team review",
        ],
        "disallowed_uses": [
            "standalone cancer recurrence/progression diagnosis",
            "patient-facing treatment recommendation",
            "replacement for imaging, pathology, or oncology review",
        ],
        "modeling_note": (
            "Tumor markers are intentionally evaluated as auxiliary monitoring-context features. "
            "They should not drive a champion model without temporal validation and external cohort support."
        ),
    }


def _missingness_report(rows: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    rates = {
        feature: round(float(rows[feature].isna().mean()), 4)
        for feature in features
        if feature in rows.columns
    }
    high_missing = [feature for feature, rate in rates.items() if rate >= 0.25]
    return {
        "rates": rates,
        "high_missing_features": high_missing,
        "status": "needs_review" if high_missing else "passed",
        "handling": "Median numeric imputation plus missingness indicator for tumor marker records.",
    }


def _leakage_audit(feature_sets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    used = {
        feature
        for spec in feature_sets.values()
        for feature in [*spec["numeric"], *spec["categorical"]]
    }
    forbidden_used = sorted(used & FORBIDDEN_LEAKAGE_COLUMNS)
    caveats = [
        "This benchmark predicts final synthetic treatment_success_binary from cycle-level monitoring rows; later-cycle imaging/lab signals are post-baseline monitoring signals.",
        "Synthetic tumor-marker features are generated from allowed current-row context, not from target/final outcome columns.",
        "Do not promote enhanced features into a patient-facing model without a temporal split and external-data validation.",
    ]
    return {
        "status": "passed_with_caveats" if not forbidden_used else "failed",
        "forbidden_target_columns_used": forbidden_used,
        "forbidden_columns": sorted(FORBIDDEN_LEAKAGE_COLUMNS),
        "caveats": caveats,
    }


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return round(float(a - b), 4)


def _status_from_deltas(deltas: dict[str, float | None]) -> str:
    if deltas.get("biomarker_imaging_vs_clinical_auroc_delta") is None:
        return "needs_attention"
    if (deltas.get("biomarker_imaging_vs_clinical_auroc_delta") or 0) < -0.02:
        return "needs_attention"
    return "passed"


def _recommendation(deltas: dict[str, float | None]) -> dict[str, str]:
    auroc_delta = deltas.get("enhanced_vs_current_default_auroc_delta")
    if auroc_delta is not None and auroc_delta > 0.01:
        decision = "candidate_promising"
        note = "Enhanced biomarker/tumor-marker features improved synthetic patient-level AUROC versus the current default."
    elif auroc_delta is not None and auroc_delta < -0.01:
        decision = "do_not_promote"
        note = "Enhanced features did not beat the current default on this synthetic benchmark."
    else:
        decision = "monitor_only"
        note = "Enhanced features are roughly comparable to current defaults; keep as monitoring/research features until external validation exists."
    return {
        "decision": decision,
        "note": note,
        "next_step": "Run a true temporal split and external/public-data benchmark before using these features in a champion model.",
    }
