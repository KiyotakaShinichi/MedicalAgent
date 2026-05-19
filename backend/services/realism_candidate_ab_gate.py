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
from sklearn.metrics import accuracy_score, brier_score_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from backend.services.artifact_manifest import build_artifact_manifest
from backend.services.complete_synthetic_training import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from backend.services.oncology_canonical_schema import ROOT_DIR
from backend.services.public_distribution_realism_candidate import (
    DEFAULT_CANDIDATE_CSV,
    DEFAULT_OUTPUT_PATH as DEFAULT_REALISM_CANDIDATE_ARTIFACT,
)


DEFAULT_CURRENT_CSV = "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv"
DEFAULT_OUTPUT_PATH = "Data/evals/models/latest_realism_candidate_ab_gate.json"
LEGACY_OUTPUT_PATH = "Data/mle_monitoring/current_vs_realism_candidate.json"

LABEL_DENYLIST = {
    "latent_response_strength",
    "response_score_percent",
    "treatment_success_binary",
    "toxicity_risk_binary",
    "urgent_intervention_needed",
    "support_intervention_needed",
    "cycle_response_trend_class",
    "final_response_category",
    "final_cancer_status",
    "maintenance_needed",
    "final_response_multiclass",
}

CLAIM_BOUNDARY = (
    "Current-vs-realism-candidate A/B is a synthetic engineering gate. It compares the current synthetic rows "
    "with a public-distribution-tuned candidate under controlled tests, but neither dataset contains real "
    "clinician-reviewed OncoTrack labels. Passing this gate is not clinical validation and does not allow "
    "patient-facing prediction or treatment claims."
)


def run_realism_candidate_ab_gate(
    *,
    current_csv: str = DEFAULT_CURRENT_CSV,
    candidate_csv: str = DEFAULT_CANDIDATE_CSV,
    candidate_artifact_path: str = DEFAULT_REALISM_CANDIDATE_ARTIFACT,
    output_path: str = DEFAULT_OUTPUT_PATH,
    legacy_output_path: str | None = LEGACY_OUTPUT_PATH,
) -> dict[str, Any]:
    current = _read_frame(current_csv)
    candidate = _read_frame(candidate_csv)
    candidate_artifact = _read_json(candidate_artifact_path)

    current_eval = _evaluate_dataset("current", current)
    candidate_eval = _evaluate_dataset("public_distribution_candidate", candidate)
    deltas = _metric_deltas(current_eval, candidate_eval)
    critical = _critical_checks(current_eval, candidate_eval, candidate_artifact, deltas)
    status = "candidate" if critical["passed"] else "needs_attention"

    payload = {
        **build_artifact_manifest(dataset_paths={
            "current_csv": current_csv,
            "candidate_csv": candidate_csv,
            "candidate_artifact": candidate_artifact_path,
        }),
        "schema_version": "realism_candidate_ab_gate_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "current": current_eval,
        "candidate": candidate_eval,
        "deltas": deltas,
        "candidate_realism_summary": {
            "status": candidate_artifact.get("status"),
            "before_after_gaps": candidate_artifact.get("before_after_gaps"),
            "production_replacement_allowed": (
                (candidate_artifact.get("realism_candidate_decision") or {}).get("production_replacement_allowed")
            ),
        },
        "critical_checks": critical,
        "recommendation": {
            "decision": "keep_current_default",
            "candidate_use": "ab_test_only",
            "may_train_experimental_models": bool(critical["passed"]),
            "production_replacement_allowed": False,
            "reason": (
                "The candidate is useful for stress-testing public-distribution realism, but it changes synthetic "
                "feature distributions without new clinician-reviewed labels. Keep current as the default and use "
                "the candidate only for controlled A/B experiments."
            ),
        },
        "what_would_unlock_reconsideration": [
            "Exact-label external temporal validation",
            "Clinician review of generator assumptions and thresholds",
            "No degradation on leakage, shortcut, calibration, counterfactual, and subgroup gates",
            "Evidence that candidate shifts improve realism without distorting response/toxicity label semantics",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _write_json(_resolve(output_path), payload)
    if legacy_output_path:
        _write_json(_resolve(legacy_output_path), _legacy_shape(payload))
    return payload


def _evaluate_dataset(name: str, frame: pd.DataFrame) -> dict[str, Any]:
    features = [column for column in NUMERIC_FEATURES + CATEGORICAL_FEATURES if column in frame.columns and column not in LABEL_DENYLIST]
    train, test = _patient_split(frame)
    leakage = {
        "status": "passed" if not (set(features) & LABEL_DENYLIST) and _patient_overlap(train, test) == 0 else "failed",
        "feature_count": len(features),
        "blocked_feature_overlap": sorted(set(features) & LABEL_DENYLIST),
        "patient_id_overlap": _patient_overlap(train, test),
    }

    classifier = _classifier(features).fit(train[features], train["treatment_success_binary"].astype(int))
    class_prob = classifier.predict_proba(test[features])[:, 1]
    class_metrics = _classification_metrics(test, class_prob)

    regressor = _regressor(features).fit(train[features], train["response_score_percent"].astype(float))
    reg_pred = regressor.predict(test[features])
    regression_metrics = _regression_metrics(test, reg_pred)

    return {
        "name": name,
        "rows": int(len(frame)),
        "patients": int(frame["patient_id"].nunique()) if "patient_id" in frame else None,
        "train_patients": int(train["patient_id"].nunique()) if "patient_id" in train else None,
        "test_patients": int(test["patient_id"].nunique()) if "patient_id" in test else None,
        "feature_set": features,
        "leakage": leakage,
        "classification": class_metrics,
        "regression": regression_metrics,
        "shortcut_audit": _shortcut_audit(train, test, features),
        "counterfactual_stability": _counterfactual_stability(classifier, test, features),
    }


def _classification_metrics(test: pd.DataFrame, probabilities: np.ndarray) -> dict[str, Any]:
    patient = test[["patient_id", "treatment_success_binary"]].copy()
    patient["probability"] = probabilities
    grouped = patient.groupby("patient_id", as_index=False).agg(
        actual=("treatment_success_binary", "max"),
        probability=("probability", "mean"),
    )
    y = grouped["actual"].astype(int).to_numpy()
    p = grouped["probability"].astype(float).to_numpy()
    pred = (p >= 0.5).astype(int)
    return {
        "patient_level_auroc": _safe_auc(y, p),
        "patient_level_brier": round(float(brier_score_loss(y, p)), 4),
        "patient_level_accuracy": round(float(accuracy_score(y, pred)), 4),
        "patient_level_ece": _ece(y, p),
        "test_patients": int(len(grouped)),
    }


def _regression_metrics(test: pd.DataFrame, predictions: np.ndarray) -> dict[str, Any]:
    rows = test[["patient_id", "cycle", "response_score_percent"]].copy()
    rows["prediction"] = predictions
    final = rows.sort_values(["patient_id", "cycle"]).groupby("patient_id", as_index=False).tail(1)
    y = final["response_score_percent"].astype(float).to_numpy()
    pred = final["prediction"].astype(float).to_numpy()
    residual = np.abs(pred - y)
    return {
        "patient_final_mae": round(float(mean_absolute_error(y, pred)), 4),
        "patient_final_rmse": round(float(mean_squared_error(y, pred) ** 0.5), 4),
        "residual_q80": round(float(np.quantile(residual, 0.80)), 4) if len(residual) else None,
        "test_patients": int(len(final)),
    }


def _shortcut_audit(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    high_risk_groups = {
        "without_mri_response_proxy": [feature for feature in features if "mri_percent_change" not in feature],
        "without_all_mri": [feature for feature in features if not feature.startswith("mri_")],
        "without_nadir_cbc": [feature for feature in features if not feature.startswith("nadir_")],
    }
    full_model = _classifier(features).fit(train[features], train["treatment_success_binary"].astype(int))
    full_prob = full_model.predict_proba(test[features])[:, 1]
    y = test["treatment_success_binary"].astype(int).to_numpy()
    full_auc = _safe_auc(y, full_prob)
    removals: dict[str, Any] = {}
    for name, subset in high_risk_groups.items():
        if not subset:
            removals[name] = {"status": "not_computed", "reason": "no features remain"}
            continue
        model = _classifier(subset).fit(train[subset], train["treatment_success_binary"].astype(int))
        prob = model.predict_proba(test[subset])[:, 1]
        auc = _safe_auc(y, prob)
        removals[name] = {
            "auroc": auc,
            "auroc_delta_vs_full": round(float(auc - full_auc), 4) if auc is not None and full_auc is not None else None,
        }
    single_feature = {
        feature: _single_feature_auc(train, test, feature)
        for feature in ("mri_percent_change_from_baseline", "mri_tumor_size_cm", "nadir_anc", "nadir_wbc", "max_symptom_severity")
        if feature in features
    }
    return {
        "full_auroc": full_auc,
        "feature_removal": removals,
        "single_feature_auc": single_feature,
        "warning": "High single-feature or low feature-removal drop can indicate simulator shortcuts, not clinical intelligence.",
    }


def _single_feature_auc(train: pd.DataFrame, test: pd.DataFrame, feature: str) -> float | None:
    model = _classifier([feature]).fit(train[[feature]], train["treatment_success_binary"].astype(int))
    probabilities = model.predict_proba(test[[feature]])[:, 1]
    return _safe_auc(test["treatment_success_binary"].astype(int).to_numpy(), probabilities)


def _counterfactual_stability(model: Pipeline, test: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    sample = test.head(min(len(test), 250)).copy()
    if sample.empty:
        return {"status": "not_computed", "reason": "empty test set"}
    base = model.predict_proba(sample[features])[:, 1]
    scenarios = {
        "age_plus_one": ("age", 1.0),
        "pre_wbc_slight_increase": ("pre_wbc", 0.05),
        "nadir_wbc_slight_increase": ("nadir_wbc", 0.05),
        "symptom_severity_plus_one": ("max_symptom_severity", 1.0),
    }
    rows = []
    for scenario, (feature, delta) in scenarios.items():
        if feature not in features:
            continue
        mutated = sample.copy()
        mutated[feature] = pd.to_numeric(mutated[feature], errors="coerce") + delta
        if feature == "max_symptom_severity":
            mutated[feature] = mutated[feature].clip(0, 10)
        prob = model.predict_proba(mutated[features])[:, 1]
        deltas = np.abs(prob - base)
        rows.append({
            "scenario": scenario,
            "mean_probability_delta": round(float(deltas.mean()), 4),
            "max_probability_delta": round(float(deltas.max()), 4),
            "flip_rate": round(float(((prob >= 0.5) != (base >= 0.5)).mean()), 4),
        })
    unacceptable = [row for row in rows if row["max_probability_delta"] > 0.35 or row["flip_rate"] > 0.10]
    return {
        "status": "strong" if not unacceptable else "needs_attention",
        "scenario_count": len(rows),
        "unacceptable_flip_count": len(unacceptable),
        "scenarios": rows,
    }


def _metric_deltas(current: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "classification_auroc_delta": _delta(current, candidate, ["classification", "patient_level_auroc"]),
        "classification_brier_delta": _delta(candidate, current, ["classification", "patient_level_brier"]),
        "classification_ece_delta": _delta(candidate, current, ["classification", "patient_level_ece"]),
        "regression_mae_delta": _delta(candidate, current, ["regression", "patient_final_mae"]),
        "regression_rmse_delta": _delta(candidate, current, ["regression", "patient_final_rmse"]),
        "counterfactual_unacceptable_flip_delta": _delta(candidate, current, ["counterfactual_stability", "unacceptable_flip_count"]),
    }


def _critical_checks(current: dict[str, Any], candidate: dict[str, Any], candidate_artifact: dict[str, Any], deltas: dict[str, Any]) -> dict[str, Any]:
    checks = [
        {
            "name": "current_leakage_passed",
            "passed": current.get("leakage", {}).get("status") == "passed",
        },
        {
            "name": "candidate_leakage_passed",
            "passed": candidate.get("leakage", {}).get("status") == "passed",
        },
        {
            "name": "candidate_not_production_replacement",
            "passed": (candidate_artifact.get("realism_candidate_decision") or {}).get("production_replacement_allowed") is False,
        },
        {
            "name": "classification_no_large_auc_regression",
            "passed": (deltas.get("classification_auroc_delta") is None) or (deltas["classification_auroc_delta"] >= -0.03),
        },
        {
            "name": "regression_no_large_mae_regression",
            "passed": (deltas.get("regression_mae_delta") is None) or (deltas["regression_mae_delta"] <= 5.0),
        },
        {
            "name": "candidate_counterfactual_stability",
            "passed": candidate.get("counterfactual_stability", {}).get("unacceptable_flip_count", 1) == 0,
        },
    ]
    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "policy": (
            "Even if this gate passes, the candidate remains A/B-only until exact-label external temporal validation "
            "and clinician review are available."
        ),
    }


def _patient_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = sorted(frame["patient_id"].astype(str).unique())
    cutoff = int(len(ids) * 0.75)
    train_ids = set(ids[:cutoff])
    return frame[frame["patient_id"].astype(str).isin(train_ids)].copy(), frame[~frame["patient_id"].astype(str).isin(train_ids)].copy()


def _classifier(features: list[str]) -> Pipeline:
    return Pipeline([("pre", _preprocessor(features)), ("model", GradientBoostingClassifier(random_state=42))])


def _regressor(features: list[str]) -> Pipeline:
    return Pipeline([("pre", _preprocessor(features)), ("model", GradientBoostingRegressor(random_state=42))])


def _preprocessor(features: list[str]) -> ColumnTransformer:
    cat = [c for c in features if c in CATEGORICAL_FEATURES]
    num = [c for c in features if c not in cat]
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])


def _safe_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float | None:
    if len(set(np.asarray(y_true).tolist())) < 2:
        return None
    return round(float(roc_auc_score(y_true, probabilities)), 4)


def _ece(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    total = max(len(p), 1)
    score = 0.0
    for low, high in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        mask = (p >= low) & (p < high if high < 1 else p <= high)
        if mask.any():
            score += (mask.sum() / total) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return round(float(score), 4)


def _delta(left: dict[str, Any], right: dict[str, Any], path: list[str]) -> float | None:
    left_value = _dig(left, path)
    right_value = _dig(right, path)
    if left_value is None or right_value is None:
        return None
    return round(float(left_value) - float(right_value), 4)


def _dig(payload: dict[str, Any], path: list[str]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _patient_overlap(train: pd.DataFrame, test: pd.DataFrame) -> int:
    return len(set(train["patient_id"].astype(str)) & set(test["patient_id"].astype(str)))


def _legacy_shape(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "current_vs_realism_candidate_v2",
        "generated_at": report["generated_at"],
        "status": report["status"],
        "current": {
            "patient_level_roc_auc": report["current"]["classification"]["patient_level_auroc"],
            "patient_level_brier_score": report["current"]["classification"]["patient_level_brier"],
            "patient_final_regression_mae": report["current"]["regression"]["patient_final_mae"],
        },
        "candidate": {
            "patient_level_roc_auc": report["candidate"]["classification"]["patient_level_auroc"],
            "patient_level_brier_score": report["candidate"]["classification"]["patient_level_brier"],
            "patient_final_regression_mae": report["candidate"]["regression"]["patient_final_mae"],
            "realism_status": report["candidate_realism_summary"]["status"],
        },
        "recommendation": {
            "decision": report["recommendation"]["decision"],
            "candidate_use": report["recommendation"]["candidate_use"],
            "production_replacement_allowed": report["recommendation"]["production_replacement_allowed"],
            "classification_auroc_delta": report["deltas"]["classification_auroc_delta"],
            "regression_mae_delta": report["deltas"]["regression_mae_delta"],
        },
        "claim_boundary": report["claim_boundary"],
    }


def _read_frame(path: str | Path) -> pd.DataFrame:
    resolved = _resolve(path)
    if not resolved.exists() or resolved.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(resolved)


def _read_json(path: str | Path) -> dict[str, Any]:
    resolved = _resolve(path)
    if not resolved.exists():
        return {}
    return json.loads(resolved.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT_DIR / candidate


__all__ = ["DEFAULT_OUTPUT_PATH", "run_realism_candidate_ab_gate"]
