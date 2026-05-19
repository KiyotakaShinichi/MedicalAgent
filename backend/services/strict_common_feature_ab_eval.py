from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.services.oncology_canonical_schema import ROOT_DIR


DEFAULT_SYNTHETIC_CSV = "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv"
DEFAULT_BREASTDCEDL_CSV = "Data/breastdcedl_spy1_features.csv"
DEFAULT_OUTPUT_PATH = "Data/evals/models/latest_strict_common_feature_ab_eval.json"

COMMON_FEATURES = ["age", "baseline_tumor_size_mm", "hr_positive", "her2_positive", "triple_negative"]

CLAIM_BOUNDARY = (
    "Strict common-feature A/B is an engineering comparability check. It uses only fields shared by the "
    "synthetic timeline and BreastDCEDL/I-SPY pCR snapshot, but the labels are not the same clinical task. "
    "It must not be described as clinical validation or treatment superiority."
)


def run_strict_common_feature_ab_eval(
    *,
    synthetic_csv: str = DEFAULT_SYNTHETIC_CSV,
    breastdcedl_csv: str = DEFAULT_BREASTDCEDL_CSV,
    output_path: str = DEFAULT_OUTPUT_PATH,
    seed: int = 20260519,
) -> dict[str, Any]:
    synthetic_raw = pd.read_csv(_resolve(synthetic_csv))
    external_raw = pd.read_csv(_resolve(breastdcedl_csv))
    synthetic = _synthetic_patient_level(synthetic_raw)
    external = _breastdcedl_rows(external_raw)

    synthetic_metrics = _evaluate_dataset(synthetic, label_col="label", seed=seed)
    external_metrics = _evaluate_dataset(external, label_col="label", seed=seed)
    distribution = _distribution_comparison(synthetic, external)

    payload = {
        "schema_version": "strict_common_feature_ab_eval_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong" if synthetic_metrics["status"] == "computed" and external_metrics["status"] == "computed" else "needs_attention",
        "feature_set": COMMON_FEATURES,
        "datasets": {
            "synthetic_patient_level": {
                "rows": int(len(synthetic)),
                "label": "treatment_success_binary",
                "metrics": synthetic_metrics,
            },
            "breastdcedl_spy1": {
                "rows": int(len(external)),
                "label": "pCR",
                "metrics": external_metrics,
            },
        },
        "distribution_comparison": distribution,
        "ab_decision": {
            "decision": "hold_monitor_only",
            "reason": (
                "The strict common fields are useful for A/B sanity checks, but synthetic treatment success "
                "and external pCR are not identical endpoints."
            ),
            "promotion_allowed": False,
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    output = _resolve(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _synthetic_patient_level(rows: pd.DataFrame) -> pd.DataFrame:
    ordered = rows.sort_values(["patient_id", "cycle"])
    final = ordered.groupby("patient_id", as_index=False).tail(1).copy()
    subtype = final["molecular_subtype"].astype(str)
    final["baseline_tumor_size_mm"] = pd.to_numeric(final["mri_tumor_size_cm"], errors="coerce") * 10.0
    final["hr_positive"] = subtype.str.contains("HR\\+", case=False, regex=True).astype(int)
    final["her2_positive"] = subtype.str.contains("HER2\\+", case=False, regex=True).astype(int)
    final["triple_negative"] = subtype.str.contains("triple", case=False, regex=False).astype(int)
    final["label"] = pd.to_numeric(final["treatment_success_binary"], errors="coerce").fillna(0).astype(int)
    return final[COMMON_FEATURES + ["label"]].dropna(subset=["label"])


def _breastdcedl_rows(rows: pd.DataFrame) -> pd.DataFrame:
    frame = rows.copy()
    subtype = frame["molecular_subtype"].astype(str)
    normalized = subtype.str.lower()
    frame["baseline_tumor_size_mm"] = pd.to_numeric(frame["baseline_longest_diameter_mm"], errors="coerce")
    frame["hr_positive"] = normalized.str.contains("hrpos", regex=False).astype(int)
    frame["her2_positive"] = normalized.str.contains("her2pos", regex=False).astype(int)
    frame["triple_negative"] = normalized.str.contains("tripleneg", regex=False).astype(int)
    frame["label"] = pd.to_numeric(frame["pcr_label"], errors="coerce").fillna(0).astype(int)
    frame["age"] = pd.to_numeric(frame["age"], errors="coerce")
    return frame[COMMON_FEATURES + ["label"]].dropna(subset=["label"])


def _evaluate_dataset(frame: pd.DataFrame, *, label_col: str, seed: int) -> dict[str, Any]:
    y = frame[label_col].astype(int)
    if len(frame) < 20 or y.nunique() < 2:
        return {"status": "not_computed", "reason": "too few rows or only one label class"}
    train, test = train_test_split(frame, test_size=0.30, random_state=seed, stratify=y)
    model = Pipeline([
        ("pre", ColumnTransformer([
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), COMMON_FEATURES),
        ])),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit(train[COMMON_FEATURES], train[label_col].astype(int))
    probs = model.predict_proba(test[COMMON_FEATURES])[:, 1]
    labels = test[label_col].astype(int).to_numpy()
    preds = (probs >= 0.5).astype(int)
    return {
        "status": "computed",
        "rows": int(len(frame)),
        "test_rows": int(len(test)),
        "positive_rate": round(float(y.mean()), 4),
        "roc_auc": _safe_auc(labels, probs),
        "brier": round(float(brier_score_loss(labels, probs)), 4),
        "accuracy": round(float(accuracy_score(labels, preds)), 4),
    }


def _distribution_comparison(synthetic: pd.DataFrame, external: pd.DataFrame) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for feature in COMMON_FEATURES:
        s = pd.to_numeric(synthetic[feature], errors="coerce").dropna()
        e = pd.to_numeric(external[feature], errors="coerce").dropna()
        rows[feature] = {
            "synthetic_mean": round(float(s.mean()), 4) if len(s) else None,
            "external_mean": round(float(e.mean()), 4) if len(e) else None,
            "absolute_mean_delta": round(abs(float(s.mean() - e.mean())), 4) if len(s) and len(e) else None,
            "synthetic_missing_rate": round(float(synthetic[feature].isna().mean()), 4),
            "external_missing_rate": round(float(external[feature].isna().mean()), 4),
        }
    return rows


def _safe_auc(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    if len(set(labels.tolist())) < 2:
        return None
    return round(float(roc_auc_score(labels, probabilities)), 4)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT_DIR / candidate
