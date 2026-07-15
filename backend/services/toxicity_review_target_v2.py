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


DEFAULT_OUTPUT_PATH = "Data/evals/models/latest_toxicity_review_target_v2.json"
TARGET = "toxicity_review_priority_v2"
DENY = {
    "patient_id",
    "treatment_date",
    "toxicity_risk_binary",
    "urgent_intervention_needed",
    "support_intervention_needed",
    TARGET,
}

CLAIM_BOUNDARY = (
    "Toxicity target v2 is a simulator-built clinician-review priority experiment. It is not a CTCAE grade, "
    "not a toxicity diagnosis, and not evidence that NLCare detects real clinical adverse events."
)


def run_toxicity_review_target_v2(
    *,
    source_csv: str = DEFAULT_SOURCE_CSV,
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    rows = pd.read_csv(_resolve(source_csv)).copy()
    rows[TARGET], score = _make_target(rows)
    train, test = _patient_split(rows)
    features = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in rows.columns and c not in DENY]
    model = _pipeline(features)
    model.fit(train[features], train[TARGET])
    probs = model.predict_proba(test[features])[:, 1]
    labels = test[TARGET].astype(int).to_numpy()
    old_rule_scores = _rule_score(test)
    old_rule_labels = (old_rule_scores >= 1.0).astype(int)
    report = {
        "schema_version": "toxicity_review_target_v2_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "candidate" if _safe_auc(labels, probs) >= 0.70 else "needs_attention",
        "source_csv": source_csv,
        "target": TARGET,
        "label_design": {
            "positive_rate": round(float(rows[TARGET].mean()), 4),
            "risk_score_threshold_quantile": 0.70,
            "factors": [
                "symptom severity and count",
                "intervention and dose-change context",
                "pre-cycle vulnerability",
                "nadir CBC contribution capped below legacy-rule dominance",
                "recovery failure",
                "cycle-to-cycle persistence",
                "small stochastic noise to avoid exact rule reconstruction",
            ],
        },
        "model": {
            "features": features,
            "auroc": _safe_auc(labels, probs),
            "brier": round(float(brier_score_loss(labels, probs)), 4),
            "accuracy": round(float(accuracy_score(labels, probs >= 0.5)), 4),
        },
        "shortcut_comparison": {
            "legacy_rule_accuracy_against_v2": round(float(accuracy_score(labels, old_rule_labels)), 4),
            "legacy_rule_auroc_against_v2": _safe_auc(labels, old_rule_scores),
            "legacy_rule_does_not_define_v2": bool(accuracy_score(labels, old_rule_labels) < 0.95),
        },
        "recommendation": {
            "current_use": "candidate_review_priority_experiment",
            "production_policy": "toxicity signal remains review-hint-only",
            "not_supported": [
                "clinical toxicity prediction",
                "CTCAE grade assignment",
                "patient-facing treatment action",
            ],
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    output = _resolve(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _make_target(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260519)
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
    persistent_symptom = (
        rows.groupby("patient_id")["max_symptom_severity"]
        .transform(lambda s: pd.to_numeric(s, errors="coerce").fillna(0).rolling(2, min_periods=1).mean())
        .clip(0, 10)
        / 10.0
    )
    score = (
        0.18 * symptom
        + 0.13 * symptom_count
        + 0.13 * intervention
        + 0.10 * dose_delay
        + 0.08 * dose_reduce
        + 0.10 * pre_anc_low
        + 0.10 * nadir_anc_low
        + 0.06 * nadir_platelets_low
        + 0.06 * recovery_wbc_low
        + 0.10 * persistent_symptom
        + rng.normal(0.0, 0.06, len(rows))
    )
    threshold = np.quantile(score, 0.70)
    return (score >= threshold).astype(int), np.asarray(score)


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
    return Pipeline([("pre", pre), ("clf", GradientBoostingClassifier(random_state=20260519))])


def _safe_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if len(set(np.asarray(labels).tolist())) < 2:
        return 0.0
    return round(float(roc_auc_score(labels, probabilities)), 4)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT_DIR / candidate
