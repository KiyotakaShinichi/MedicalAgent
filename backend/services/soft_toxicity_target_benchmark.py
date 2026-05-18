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

from backend.services.artifact_manifest import build_artifact_manifest
from backend.services.biomarker_feature_benchmark import DEFAULT_SOURCE_CSV
from backend.services.complete_synthetic_training import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from backend.services.toxicity_shortcut_audit import _rule_score


DEFAULT_OUTPUT_PATH = "Data/evals/models/latest_soft_toxicity_target_benchmark.json"
TARGET = "soft_toxicity_review_label"
DENY = {"patient_id", "treatment_date", "toxicity_risk_binary", "urgent_intervention_needed", "support_intervention_needed"}


def run_soft_toxicity_target_benchmark(
    source_csv: str = DEFAULT_SOURCE_CSV,
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    rows = pd.read_csv(source_csv).copy()
    rows[TARGET], risk_score = _make_soft_label(rows)
    train, test = _patient_split(rows)
    features = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in rows.columns and c not in DENY]
    model = _pipeline(features)
    model.fit(train[features], train[TARGET])
    probs = model.predict_proba(test[features])[:, 1]
    y = test[TARGET].astype(int).to_numpy()
    old_rule = (_rule_score(test) >= 1.0).astype(int)
    old_y = test["toxicity_risk_binary"].astype(int).to_numpy()
    report = {
        **build_artifact_manifest(dataset_paths={"source_csv": source_csv}),
        "schema_version": "soft_toxicity_target_benchmark_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "candidate" if _safe_auc(y, probs) >= 0.70 else "needs_attention",
        "claim_boundary": (
            "Synthetic softer toxicity target. It reduces one-rule reconstruction pressure but remains simulator-built "
            "and must not be claimed as clinical toxicity prediction."
        ),
        "label_design": {
            "positive_rate": round(float(rows[TARGET].mean()), 4),
            "risk_score_quantile_threshold": 0.65,
            "uses": [
                "symptom persistence/severity",
                "interventions and dose changes",
                "lagged/pre-cycle vulnerability",
                "noisy nadir contribution",
            ],
        },
        "soft_target_model": {
            "auroc": _safe_auc(y, probs),
            "brier": round(float(brier_score_loss(y, probs)), 4),
            "accuracy": round(float(accuracy_score(y, probs >= 0.5)), 4),
        },
        "shortcut_comparison": {
            "old_toxicity_rule_accuracy_against_old_label": round(float(accuracy_score(old_y, old_rule)), 4),
            "old_toxicity_rule_accuracy_against_soft_label": round(float(accuracy_score(y, old_rule)), 4),
            "old_toxicity_rule_auroc_against_soft_label": _safe_auc(y, _rule_score(test)),
        },
        "recommendation": (
            "Keep current toxicity head as review-only. Use the soft target as a better synthetic experiment, "
            "then require clinician-reviewed adverse-event labels before any stronger claim."
        ),
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_soft_toxicity_target_benchmark(path: str = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"schema_version": "soft_toxicity_target_benchmark_v1", "status": "missing"}


def _make_soft_label(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260517)
    score = (
        0.22 * (pd.to_numeric(rows["max_symptom_severity"], errors="coerce").fillna(0) / 10.0)
        + 0.16 * (pd.to_numeric(rows["symptom_count"], errors="coerce").fillna(0).clip(0, 5) / 5.0)
        + 0.16 * (pd.to_numeric(rows["intervention_count"], errors="coerce").fillna(0).clip(0, 3) / 3.0)
        + 0.12 * pd.to_numeric(rows["dose_delayed"], errors="coerce").fillna(0)
        + 0.10 * pd.to_numeric(rows["dose_reduced"], errors="coerce").fillna(0)
        + 0.10 * (pd.to_numeric(rows["pre_anc"], errors="coerce").fillna(3.0) < 2.0).astype(float)
        + 0.08 * (pd.to_numeric(rows["nadir_anc"], errors="coerce").fillna(3.0) < 1.1).astype(float)
        + 0.06 * (pd.to_numeric(rows["nadir_platelets"], errors="coerce").fillna(250.0) < 75.0).astype(float)
        + rng.normal(0.0, 0.05, len(rows))
    )
    threshold = np.quantile(score, 0.65)
    return (score >= threshold).astype(int), np.asarray(score)


def _patient_split(rows: pd.DataFrame):
    ids = sorted(rows["patient_id"].astype(str).unique())
    cut = int(len(ids) * 0.75)
    train_ids = set(ids[:cut])
    return rows[rows["patient_id"].astype(str).isin(train_ids)], rows[~rows["patient_id"].astype(str).isin(train_ids)]


def _pipeline(features: list[str]) -> Pipeline:
    num = [c for c in features if c in NUMERIC_FEATURES]
    cat = [c for c in features if c in CATEGORICAL_FEATURES]
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]), cat),
    ])
    return Pipeline([("pre", pre), ("clf", GradientBoostingClassifier(random_state=43))])


def _safe_auc(y_true, y_score) -> float:
    if len(set(np.asarray(y_true).tolist())) < 2:
        return 0.0
    return round(float(roc_auc_score(y_true, y_score)), 4)


__all__ = ["run_soft_toxicity_target_benchmark", "load_soft_toxicity_target_benchmark", "DEFAULT_OUTPUT_PATH"]
