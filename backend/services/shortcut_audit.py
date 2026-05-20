from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from backend.services.artifact_manifest import build_artifact_manifest
from backend.services.complete_synthetic_training import CATEGORICAL_FEATURES, DEFAULT_ML_CSV_PATH, NUMERIC_FEATURES


DEFAULT_OUTPUT_PATH = "Data/evals/models/latest_shortcut_audit.json"
BASE_DENYLIST = {
    "patient_id", "treatment_date", "latent_response_strength", "response_score_percent",
    "treatment_success_binary", "toxicity_risk_binary", "urgent_intervention_needed",
    "support_intervention_needed", "cycle_response_trend_class", "final_response_category",
    "final_cancer_status", "maintenance_needed", "final_response_multiclass",
}
NADIR_FEATURES = {"nadir_wbc", "nadir_anc", "nadir_hemoglobin", "nadir_platelets"}
MRI_PERCENT_FEATURES = {"mri_percent_change_from_baseline"}


def run_shortcut_audit(source_csv: str = DEFAULT_ML_CSV_PATH, output_path: str = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    frame = pd.read_csv(source_csv)
    features = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in frame.columns and c not in BASE_DENYLIST]
    train, test = _patient_split(frame)

    tox_full = _fit_classifier(train, features, "toxicity_risk_binary")
    tox_no_nadir_features = [f for f in features if f not in NADIR_FEATURES]
    tox_no_nadir = _fit_classifier(train, tox_no_nadir_features, "toxicity_risk_binary")
    reg_full = _fit_regressor(train, features, "response_score_percent")
    reg_no_mri_features = [f for f in features if f not in MRI_PERCENT_FEATURES]
    reg_no_mri = _fit_regressor(train, reg_no_mri_features, "response_score_percent")

    tox_prob = tox_full.predict_proba(test[features])[:, 1]
    tox_no_nadir_prob = tox_no_nadir.predict_proba(test[tox_no_nadir_features])[:, 1]
    reg_pred = reg_full.predict(test[features])
    reg_no_mri_pred = reg_no_mri.predict(test[reg_no_mri_features])
    top_tox_features = _top_permutation(tox_full, test[features], test["toxicity_risk_binary"].astype(int), features, scoring="roc_auc")
    top_reg_features = _top_permutation(reg_full, test[features], test["response_score_percent"].astype(float), features, scoring="neg_mean_absolute_error")

    tox_auc = _safe_auc(test["toxicity_risk_binary"], tox_prob)
    tox_no_nadir_auc = _safe_auc(test["toxicity_risk_binary"], tox_no_nadir_prob)
    reg_mae = float(mean_absolute_error(test["response_score_percent"], reg_pred))
    reg_no_mri_mae = float(mean_absolute_error(test["response_score_percent"], reg_no_mri_pred))
    dominant = _dominant_features(top_tox_features + top_reg_features)
    mitigations = _mitigation_evidence(frame, dominant)

    payload = {
        **build_artifact_manifest(dataset_paths={"source_csv": source_csv}),
        "schema_version": "shortcut_audit_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": _status(dominant, mitigations),
        "toxicity_audit": {
            "full_auc": tox_auc,
            "no_nadir_cbc_auc": tox_no_nadir_auc,
            "auc_drop_without_nadir": round(float(tox_auc - tox_no_nadir_auc), 4),
            "near_label_proxy_risk": "present" if tox_auc >= 0.95 else "not_detected",
            "review_hint_only": True,
            "top_permutation_features": top_tox_features,
            "interpretation": (
                "Legacy toxicity AUC is not a headline performance metric. "
                "This audit treats it as shortcut-risk evidence for a review-only hint."
            ),
        },
        "regression_audit": {
            "full_mae": round(reg_mae, 4),
            "no_mri_percent_change_mae": round(reg_no_mri_mae, 4),
            "mae_increase_without_mri_percent_change": round(float(reg_no_mri_mae - reg_mae), 4),
            "top_permutation_features": top_reg_features,
            "interpretation": "Large MAE increase means response regression is strongly tied to MRI percent-change.",
        },
        "dominant_shortcut_features": dominant,
        "mitigations": mitigations,
        "recommendation": (
            "Treat very high synthetic metrics as generator-fit evidence. The legacy toxicity "
            "label should be described as shortcut-prone; prefer the softer toxicity-review "
            "target for synthetic experiments. Response regression is allowed to use direct "
            "imaging change only as a monitor-only imaging-supported signal, with abstention "
            "when imaging evidence is absent."
        ),
        "claim_boundary": "Shortcut audit is synthetic-only. It finds generator shortcuts, not clinical truth.",
        "display_policy": {
            "hide_toxicity_auc_as_headline": True,
            "visible_labels": ["shortcut_audit_status", "review_hint_only", "near_label_proxy_risk", "not_a_clinical_toxicity_predictor"],
            "do_not_claim": "Do not quote legacy toxicity AUC as learned clinical toxicity prediction.",
        },
    }
    _write_json(output_path, payload)
    return payload


def load_shortcut_audit(path: str = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"status": "missing"}


def _patient_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = sorted(frame["patient_id"].astype(str).unique())
    train_ids = set(ids[: int(len(ids) * 0.75)])
    return frame[frame["patient_id"].astype(str).isin(train_ids)].copy(), frame[~frame["patient_id"].astype(str).isin(train_ids)].copy()


def _preprocessor(features: list[str]) -> ColumnTransformer:
    cat = [c for c in features if c in CATEGORICAL_FEATURES]
    num = [c for c in features if c not in cat]
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])


def _fit_classifier(train: pd.DataFrame, features: list[str], target: str) -> Pipeline:
    pipe = Pipeline([("pre", _preprocessor(features)), ("model", RandomForestClassifier(n_estimators=120, random_state=42, min_samples_leaf=3))])
    return pipe.fit(train[features], train[target].astype(int))


def _fit_regressor(train: pd.DataFrame, features: list[str], target: str) -> Pipeline:
    pipe = Pipeline([("pre", _preprocessor(features)), ("model", RandomForestRegressor(n_estimators=120, random_state=42, min_samples_leaf=3))])
    return pipe.fit(train[features], train[target].astype(float))


def _top_permutation(model: Pipeline, X: pd.DataFrame, y, features: list[str], scoring: str) -> list[dict[str, Any]]:
    result = permutation_importance(model, X, y, n_repeats=5, random_state=42, scoring=scoring)
    order = np.argsort(result.importances_mean)[::-1][:8]
    return [
        {"feature": features[i], "importance_mean": round(float(result.importances_mean[i]), 4)}
        for i in order
    ]


def _dominant_features(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["feature"] in NADIR_FEATURES | MRI_PERCENT_FEATURES and row["importance_mean"] > 0.05]


def _mitigation_evidence(frame: pd.DataFrame, dominant: list[dict[str, Any]]) -> dict[str, Any]:
    has_mri_shortcut = any(row.get("feature") in MRI_PERCENT_FEATURES for row in dominant)
    has_toxicity_shortcut = any(row.get("feature") in NADIR_FEATURES for row in dominant)
    soft_status = "not_needed"
    soft_near_proxy_count = None
    if has_toxicity_shortcut:
        try:
            from backend.services.soft_toxicity_target_benchmark import TARGET, _make_soft_label
            from backend.services.toxicity_feature_audit import NEAR_LABEL_IDENTITY_GAP, _per_feature_label_separation

            soft_frame = frame.copy()
            soft_frame[TARGET], _ = _make_soft_label(soft_frame)
            near = [
                fr.feature for fr in _per_feature_label_separation(soft_frame, TARGET)
                if fr.label_separation_gap is not None and fr.label_separation_gap >= NEAR_LABEL_IDENTITY_GAP
            ]
            soft_near_proxy_count = len(near)
            soft_status = "candidate" if not near else "needs_attention"
        except Exception as exc:
            soft_status = f"unavailable:{exc}"

    return {
        "toxicity_soft_target": {
            "status": soft_status,
            "near_label_proxy_count": soft_near_proxy_count,
            "production_policy": (
                "Legacy toxicity AUC is not promoted as learned clinical skill. "
                "Use review-only wording; prefer the softer synthetic target for experiments."
            ),
        },
        "response_regression_direct_imaging": {
            "status": "bounded" if has_mri_shortcut else "not_triggered",
            "production_policy": (
                "MRI percent change is a direct response-monitoring signal, not hidden intelligence. "
                "The regression head must be described as imaging-supported and monitor-only; "
                "evidence-aware inference should abstain or lower confidence when imaging is absent."
            ),
        },
    }


def _status(dominant: list[dict[str, Any]], mitigations: dict[str, Any]) -> str:
    if not dominant:
        return "strong"
    toxicity_ok = mitigations.get("toxicity_soft_target", {}).get("status") in {"not_needed", "candidate"}
    response_ok = mitigations.get("response_regression_direct_imaging", {}).get("status") in {"not_triggered", "bounded"}
    return "acceptable" if toxicity_ok and response_ok else "needs_attention"


def _safe_auc(y_true, prob) -> float:
    values = np.asarray(y_true)
    if len(set(values.tolist())) < 2:
        return 0.0
    return round(float(roc_auc_score(y_true, prob)), 4)


def _write_json(path: str, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


__all__ = ["DEFAULT_OUTPUT_PATH", "run_shortcut_audit", "load_shortcut_audit"]
