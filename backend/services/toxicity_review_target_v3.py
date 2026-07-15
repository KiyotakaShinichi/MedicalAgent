from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from backend.services.biomarker_feature_benchmark import DEFAULT_SOURCE_CSV
from backend.services.complete_synthetic_training import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from backend.services.oncology_canonical_schema import ROOT_DIR
from backend.services.toxicity_shortcut_audit import _rule_score


DEFAULT_OUTPUT_PATH = "Data/evals/models/latest_toxicity_review_target_v3.json"
TARGET = "toxicity_review_priority_v3"
DENY = {
    "patient_id",
    "treatment_date",
    "toxicity_risk_binary",
    "urgent_intervention_needed",
    "support_intervention_needed",
    "toxicity_review_priority_v2",
    TARGET,
}

CLAIM_BOUNDARY = (
    "Toxicity review target v3 is a simulator-built review-priority experiment. "
    "It is not CTCAE grading, not toxicity diagnosis, not patient-facing advice, "
    "not clinical validation, and not evidence of real adverse-event detection."
)


def run_toxicity_review_target_v3(
    *,
    source_csv: str = DEFAULT_SOURCE_CSV,
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    rows = pd.read_csv(_resolve(source_csv)).copy()
    rows[TARGET], score, components = _make_target(rows)
    train, test = _patient_split(rows)
    features = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in rows.columns and c not in DENY]

    model = _pipeline(features)
    model.fit(train[features], train[TARGET])
    probs = model.predict_proba(test[features])[:, 1]
    labels = test[TARGET].astype(int).to_numpy()

    old_rule_scores = _rule_score(test)
    old_rule_labels = (old_rule_scores >= 1.0).astype(int)
    legacy_accuracy = round(float(accuracy_score(labels, old_rule_labels)), 4)
    legacy_auc = _safe_auc(labels, old_rule_scores)
    feature_corrs = _feature_correlations(score, components)
    residual_warning = bool(
        legacy_auc >= 0.90
        or max((abs(row["correlation_with_v3_score"]) for row in feature_corrs), default=0.0) >= 0.85
    )

    report = {
        "schema_version": "toxicity_review_target_v3_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "candidate_needs_review" if residual_warning else "candidate",
        "clinical_validation": False,
        "synthetic_only": True,
        "healthcare_production_ready": False,
        "source_csv": source_csv,
        "target": TARGET,
        "label_design": {
            "positive_rate": round(float(rows[TARGET].mean()), 4),
            "risk_score_threshold_quantile": 0.72,
            "design_intent": (
                "Reduce direct dependence on the legacy nadir-CBC rule by combining symptom "
                "severity, symptom persistence, intervention/dose context, pre-cycle "
                "vulnerability, limited nadir CBC contribution, recovery failure, and noise."
            ),
            "factors": [
                "current symptom severity",
                "symptom count",
                "intervention and dose-change context",
                "pre-cycle vulnerability",
                "limited/capped nadir CBC contribution",
                "recovery failure",
                "multi-cycle symptom persistence",
                "small stochastic noise to reduce exact rule reconstruction",
            ],
        },
        "model": {
            "features": features,
            "auroc": _safe_auc(labels, probs),
            "brier": round(float(brier_score_loss(labels, probs)), 4),
            "accuracy": round(float(accuracy_score(labels, probs >= 0.5)), 4),
            "interpretation": (
                "Synthetic discrimination against a simulator-built target only. High scores can "
                "still reflect generator structure and must not be presented as clinical toxicity detection."
            ),
        },
        "shortcut_comparison": {
            "legacy_rule_accuracy_against_v3": legacy_accuracy,
            "legacy_rule_auroc_against_v3": legacy_auc,
            "legacy_rule_does_not_define_v3": bool(legacy_accuracy < 0.85 and legacy_auc < 0.90),
            "residual_shortcut_warning": residual_warning,
        },
        "feature_group_sensitivity": {
            "correlations_with_v3_score": feature_corrs,
            "interpretation": (
                "Correlation is a synthetic target-design diagnostic, not causal evidence. "
                "Large correlations are kept visible as shortcut-risk warnings."
            ),
        },
        "recommendation": {
            "current_use": "candidate_review_priority_experiment",
            "production_policy": "review_hint_only",
            "promotion_decision": "hold_synthetic_only",
            "not_supported": [
                "clinical toxicity prediction",
                "CTCAE grade assignment",
                "patient-facing treatment action",
                "real adverse-event detection",
            ],
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    output = _resolve(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _make_target(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, pd.Series]]:
    rng = np.random.default_rng(20260715)
    rows = rows.sort_values(["patient_id", "cycle"]).copy()
    symptom = _num(rows, "max_symptom_severity").clip(0, 10) / 10.0
    symptom_count = _num(rows, "symptom_count").clip(0, 6) / 6.0
    intervention = _num(rows, "intervention_count").clip(0, 3) / 3.0
    dose_delay = _num(rows, "dose_delayed").clip(0, 1)
    dose_reduce = _num(rows, "dose_reduced").clip(0, 1)
    pre_anc_low = (_num(rows, "pre_anc", default=3.0) < 2.0).astype(float)
    nadir_anc_low = (_num(rows, "nadir_anc", default=3.0) < 1.0).astype(float)
    nadir_platelets_low = (_num(rows, "nadir_platelets", default=250.0) < 90.0).astype(float)
    recovery_wbc_low = (_num(rows, "recovery_wbc", default=5.0) < 3.5).astype(float)
    recovery_platelets_low = (_num(rows, "recovery_platelets", default=250.0) < 120.0).astype(float)
    persistent_symptom = (
        rows.groupby("patient_id")["max_symptom_severity"]
        .transform(lambda s: pd.to_numeric(s, errors="coerce").fillna(0).rolling(3, min_periods=1).mean())
        .clip(0, 10)
        / 10.0
    )
    previous_symptom = (
        rows.groupby("patient_id")["max_symptom_severity"]
        .shift(1)
        .fillna(0)
        .clip(0, 10)
        / 10.0
    )
    components = {
        "symptom": symptom,
        "symptom_count": symptom_count,
        "intervention": intervention,
        "dose_delay": dose_delay,
        "dose_reduce": dose_reduce,
        "pre_anc_low": pre_anc_low,
        "nadir_anc_low": nadir_anc_low,
        "nadir_platelets_low": nadir_platelets_low,
        "recovery_wbc_low": recovery_wbc_low,
        "recovery_platelets_low": recovery_platelets_low,
        "persistent_symptom": persistent_symptom,
        "previous_symptom": previous_symptom,
    }
    score = (
        0.20 * symptom
        + 0.14 * symptom_count
        + 0.16 * intervention
        + 0.11 * dose_delay
        + 0.08 * dose_reduce
        + 0.07 * pre_anc_low
        + 0.04 * nadir_anc_low
        + 0.03 * nadir_platelets_low
        + 0.08 * recovery_wbc_low
        + 0.04 * recovery_platelets_low
        + 0.14 * persistent_symptom
        + 0.05 * previous_symptom
        + rng.normal(0.0, 0.08, len(rows))
    )
    threshold = np.quantile(score, 0.72)
    return (score >= threshold).astype(int), np.asarray(score), components


def _feature_correlations(score: np.ndarray, components: dict[str, pd.Series]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, values in components.items():
        arr = np.asarray(values, dtype=float)
        if np.std(arr) == 0:
            corr = 0.0
        else:
            corr = float(np.corrcoef(score, arr)[0, 1])
        rows.append({"feature_group": name, "correlation_with_v3_score": round(corr, 4)})
    return sorted(rows, key=lambda row: abs(row["correlation_with_v3_score"]), reverse=True)


def _num(rows: pd.DataFrame, column: str, *, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(rows.get(column, default), errors="coerce").fillna(default)


def _patient_split(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = sorted(rows["patient_id"].astype(str).unique())
    cut = int(len(ids) * 0.75)
    train_ids = set(ids[:cut])
    return rows[rows["patient_id"].astype(str).isin(train_ids)], rows[~rows["patient_id"].astype(str).isin(train_ids)]


def _pipeline(features: list[str]) -> Pipeline:
    numeric = [c for c in features if c in NUMERIC_FEATURES]
    categorical = [c for c in features if c in CATEGORICAL_FEATURES]
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    return Pipeline([("pre", pre), ("clf", GradientBoostingClassifier(random_state=20260715, max_depth=2))])


def _safe_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if len(set(np.asarray(labels).tolist())) < 2:
        return 0.0
    return round(float(roc_auc_score(labels, probabilities)), 4)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT_DIR / candidate


__all__ = ["DEFAULT_OUTPUT_PATH", "TARGET", "run_toxicity_review_target_v3"]
