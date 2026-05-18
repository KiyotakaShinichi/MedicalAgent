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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from backend.services.artifact_manifest import build_artifact_manifest
from backend.services.biomarker_feature_benchmark import DEFAULT_SOURCE_CSV
from backend.services.complete_synthetic_training import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from backend.services.evidence_abstention_eval import SCENARIOS, _strip_modalities
from backend.services.evidence_sufficiency import MODALITY_GROUPS, assess_evidence


DEFAULT_OUTPUT_PATH = "Data/evals/models/latest_learned_abstention_experiment.json"
TARGET = "treatment_success_binary"
DENYLIST = {
    "patient_id", "treatment_date", TARGET, "response_score_percent", "latent_response_strength",
    "toxicity_risk_binary", "urgent_intervention_needed", "support_intervention_needed",
    "cycle_response_trend_class", "final_response_category", "final_cancer_status",
    "maintenance_needed", "final_response_multiclass",
}


def run_learned_abstention_experiment(
    source_csv: str = DEFAULT_SOURCE_CSV,
    output_path: str = DEFAULT_OUTPUT_PATH,
    max_rows: int = 1800,
    correctness_threshold: float = 0.65,
) -> dict[str, Any]:
    frame = pd.read_csv(source_csv)
    frame = frame.head(max_rows).copy()
    patient_ids = sorted(frame["patient_id"].astype(str).unique())
    train_ids, meta_ids, test_ids = _split_ids(patient_ids)
    train = frame[frame["patient_id"].astype(str).isin(train_ids)]
    meta = frame[frame["patient_id"].astype(str).isin(meta_ids)]
    test = frame[frame["patient_id"].astype(str).isin(test_ids)]

    features = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in frame.columns and c not in DENYLIST]
    model = _pipeline(features)
    model.fit(train[features], train[TARGET].astype(int))

    meta_X, meta_y = _build_meta_rows(model, meta, features)
    test_X, test_y, production_rule = _build_meta_rows(model, test, features, include_rule=True)
    abstention_head = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    abstention_head.fit(meta_X, meta_y)
    prob_correct = abstention_head.predict_proba(test_X)[:, 1]
    learned_cover = prob_correct >= correctness_threshold
    rule_cover = np.asarray(production_rule, dtype=bool)

    report = {
        **build_artifact_manifest(dataset_paths={"source_csv": source_csv}),
        "schema_version": "learned_abstention_experiment_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong" if _safe_auc(test_y, prob_correct) >= 0.70 else "needs_attention",
        "claim_boundary": (
            "Synthetic abstention-head experiment. It estimates when the synthetic classifier is likely correct "
            "under modality dropout; it is not clinical validation and does not replace deterministic safety rules."
        ),
        "rows": {"train": int(len(train)), "meta_train": int(len(meta_X)), "test_meta": int(len(test_X))},
        "abstention_head": {
            "target": "base_model_was_correct_under_modality_dropout",
            "auroc": _safe_auc(test_y, prob_correct),
            "brier": round(float(brier_score_loss(test_y, prob_correct)), 4),
            "threshold_probability_correct": correctness_threshold,
        },
        "comparison": {
            "learned": _coverage_metrics(test_y, learned_cover),
            "rule_based": _coverage_metrics(test_y, rule_cover),
        },
        "recommendation": (
            "Use this as candidate evidence for a learned abstention layer only after external validation; "
            "keep deterministic minimum-evidence rules as the safety backstop."
        ),
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_learned_abstention_experiment(path: str = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"schema_version": "learned_abstention_experiment_v1", "status": "missing"}


def _split_ids(ids: list[str]) -> tuple[set[str], set[str], set[str]]:
    n = len(ids)
    return set(ids[: int(n * 0.6)]), set(ids[int(n * 0.6): int(n * 0.8)]), set(ids[int(n * 0.8):])


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
    return Pipeline([("pre", pre), ("clf", GradientBoostingClassifier(random_state=42))])


def _build_meta_rows(model, frame: pd.DataFrame, features: list[str], include_rule: bool = False):
    rows, labels, rule_cover = [], [], []
    for scenario_name, drop_modalities in SCENARIOS.items():
        masked = _strip_modalities(frame.copy(), drop_modalities)
        probs = model.predict_proba(masked[features])[:, 1]
        preds = (probs >= 0.5).astype(int)
        y = masked[TARGET].astype(int).to_numpy()
        for i, (_, row) in enumerate(masked.iterrows()):
            evidence = assess_evidence(row.to_dict(), question="response_classification")
            present = set(evidence.modalities_present)
            rows.append([
                probs[i],
                abs(probs[i] - 0.5),
                len(evidence.modalities_missing) / max(1, len(MODALITY_GROUPS)),
                float(evidence.sufficiency == "sufficient"),
                float(evidence.sufficiency == "partial"),
                float("imaging" in present),
                float("cbc_nadir" in present),
                float("cbc_recovery" in present),
                float("symptoms" in present),
            ])
            labels.append(int(preds[i] == y[i]))
            if include_rule:
                rule_cover.append(not evidence.abstain)
    if include_rule:
        return np.asarray(rows), np.asarray(labels), rule_cover
    return np.asarray(rows), np.asarray(labels)


def _coverage_metrics(correct: np.ndarray, cover: np.ndarray) -> dict[str, Any]:
    covered = correct[cover]
    return {
        "coverage_rate": round(float(cover.mean()), 4),
        "abstention_rate": round(float(1.0 - cover.mean()), 4),
        "covered_accuracy": round(float(covered.mean()), 4) if len(covered) else None,
    }


def _safe_auc(y_true, y_score) -> float:
    if len(set(np.asarray(y_true).tolist())) < 2:
        return 0.0
    return round(float(roc_auc_score(y_true, y_score)), 4)


__all__ = ["run_learned_abstention_experiment", "load_learned_abstention_experiment", "DEFAULT_OUTPUT_PATH"]
