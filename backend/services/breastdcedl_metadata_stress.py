from __future__ import annotations

import ast
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.services.oncology_canonical_schema import ROOT_DIR


DEFAULT_CANONICAL_CSV = "Data/external_bridge/canonical_breastdcedl_spy1.csv"
DEFAULT_OUTPUT_PATH = "Data/evals/models/latest_breastdcedl_metadata_stress.json"
DEFAULT_PREDICTIONS_PATH = "Data/evals/models/latest_breastdcedl_metadata_stress_predictions.csv"
DEFAULT_DOC_PATH = "docs/breastdcedl_metadata_stress.md"

FEATURE_SET = [
    "age",
    "baseline_longest_diameter_mm",
    "hr_positive",
    "her2_positive",
    "triple_negative",
]

CLAIM_BOUNDARY = (
    "BreastDCEDL metadata-only stress is an external benchmark probe over public imaging-response metadata. "
    "The pCR endpoint is not the same as NLCare synthetic response, toxicity, or monitoring heads. This artifact "
    "is not clinical validation, does not update live models, and must not be used for diagnosis, prognosis, "
    "treatment recommendation, medication decisions, genetic-risk interpretation, tumor-marker interpretation, "
    "or patient-facing prediction."
)


def run_breastdcedl_metadata_stress(
    *,
    canonical_csv: str | Path = DEFAULT_CANONICAL_CSV,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    predictions_path: str | Path = DEFAULT_PREDICTIONS_PATH,
    doc_path: str | Path = DEFAULT_DOC_PATH,
    seed: int = 20260706,
    n_bootstrap: int = 300,
) -> dict[str, Any]:
    canonical = _read_frame(canonical_csv)
    rows = _metadata_frame(canonical)

    if len(rows) < 30 or rows["pcr_label"].nunique(dropna=True) < 2:
        payload = _not_computed_payload(rows, canonical_csv)
        _write_json(_resolve(output_path), payload)
        _write_doc(_resolve(doc_path), payload)
        _write_predictions(_resolve(predictions_path), rows, probabilities=None)
        return payload

    labels = rows["pcr_label"].astype(int).to_numpy()
    folds = min(5, int(pd.Series(labels).value_counts().min()))
    if folds < 2:
        payload = _not_computed_payload(rows, canonical_csv, reason="too few examples in one pCR class")
        _write_json(_resolve(output_path), payload)
        _write_doc(_resolve(doc_path), payload)
        _write_predictions(_resolve(predictions_path), rows, probabilities=None)
        return payload

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    model = _model()
    probabilities = cross_val_predict(
        model,
        rows[FEATURE_SET],
        labels,
        cv=cv,
        method="predict_proba",
    )[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "status": "computed",
        "rows": int(len(rows)),
        "positive_pcr": int(labels.sum()),
        "negative_pcr": int((labels == 0).sum()),
        "positive_rate": round(float(labels.mean()), 4),
        "cv_folds": int(folds),
        "roc_auc": _safe_auc(labels, probabilities),
        "balanced_accuracy": round(float(balanced_accuracy_score(labels, predictions)), 4),
        "accuracy": round(float(accuracy_score(labels, predictions)), 4),
        "brier": round(float(brier_score_loss(labels, probabilities)), 4),
        "probability_summary": _probability_summary(probabilities),
        "bootstrap_ci95": _bootstrap_intervals(labels, probabilities, seed=seed, n_bootstrap=n_bootstrap),
        "clinical_interpretation_allowed": False,
    }
    payload = _base_payload(rows, canonical_csv)
    payload.update(
        {
            "status": "strong",
            "stress_result": metrics,
            "predictions_path": str(_relative_to_root(_resolve(predictions_path))),
            "integration_decision": {
                "recommendation": "use_as_external_metadata_stress_only",
                "model_promotion_allowed": False,
                "live_model_update_allowed": False,
                "reason": (
                    "The benchmark provides external endpoint-shift pressure on metadata features, "
                    "but pCR does not validate NLCare's synthetic monitoring heads."
                ),
            },
        }
    )
    _write_predictions(_resolve(predictions_path), rows, probabilities=probabilities)
    _write_json(_resolve(output_path), payload)
    _write_doc(_resolve(doc_path), payload)
    return payload


def _not_computed_payload(
    rows: pd.DataFrame,
    canonical_csv: str | Path,
    *,
    reason: str = "too few rows or only one pCR class",
) -> dict[str, Any]:
    payload = _base_payload(rows, canonical_csv)
    payload.update(
        {
            "status": "needs_attention",
            "stress_result": {"status": "not_computed", "reason": reason, "rows": int(len(rows))},
            "predictions_path": str(_relative_to_root(_resolve(DEFAULT_PREDICTIONS_PATH))),
            "integration_decision": {
                "recommendation": "complete_or_refresh_breastdcedl_canonical_export",
                "model_promotion_allowed": False,
                "live_model_update_allowed": False,
                "reason": "Metadata stress cannot run until enough labelled pCR rows are available.",
            },
        }
    )
    return payload


def _base_payload(rows: pd.DataFrame, canonical_csv: str | Path) -> dict[str, Any]:
    return {
        "schema_version": "breastdcedl_metadata_stress_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "clinical_validation": False,
        "production_training_allowed": False,
        "source_dataset": "BreastDCEDL / I-SPY public imaging-response metadata",
        "source_path": str(_relative_to_root(_resolve(canonical_csv))),
        "feature_set": FEATURE_SET,
        "metadata_only": True,
        "image_pixel_training": False,
        "target": {
            "external_label": "pathologic complete response (pCR)",
            "nlcare_label_equivalent": False,
            "target_mismatch": (
                "pCR is an external imaging-response endpoint, not NLCare's synthetic response-pattern, "
                "response-score, toxicity-review, or patient-monitoring target."
            ),
        },
        "cohort_summary": _cohort_summary(rows),
        "missingness": _missingness(rows, FEATURE_SET),
        "blocked_claims": [
            "clinical validation",
            "real patient response prediction",
            "treatment recommendation",
            "prognosis or survival prediction",
            "diagnosis",
            "tumor-marker interpretation",
            "genetic-risk interpretation",
            "model promotion to patient-facing route",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _metadata_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["patient_id", "source_record_id", *FEATURE_SET, "pcr_label"])
    out = frame.copy()
    out["patient_id"] = out.get("patient_id", pd.Series(index=out.index, dtype=str)).astype(str)
    out["source_record_id"] = out.get("source_record_id", out["patient_id"]).astype(str)
    out["age"] = pd.to_numeric(out.get("age"), errors="coerce")
    out["baseline_longest_diameter_mm"] = _baseline_size(out)

    subtype = _text(out, "molecular_subtype").str.lower()
    er_status = _text(out, "er_status").str.lower()
    pr_status = _text(out, "pr_status").str.lower()
    her2_status = _text(out, "her2_status").str.lower()

    out["hr_positive"] = (
        subtype.str.contains("hrpos|luminal|luma|lumb|hr\\+", regex=True)
        | (er_status == "positive")
        | (pr_status == "positive")
    ).astype(int)
    out["her2_positive"] = (
        subtype.str.contains("her2pos|her2\\+", regex=True)
        | (her2_status == "positive")
    ).astype(int)
    out["triple_negative"] = (
        subtype.str.contains("tripleneg|triple|basal", regex=True)
        | ((er_status == "negative") & (pr_status == "negative") & (her2_status == "negative"))
    ).astype(int)

    if "pcr_label" in out.columns:
        out["pcr_label"] = pd.to_numeric(out["pcr_label"], errors="coerce")
    elif "outcome_label_name" in out.columns and "outcome_label_value" in out.columns:
        is_pcr = out["outcome_label_name"].astype(str).str.lower() == "pcr"
        out["pcr_label"] = pd.to_numeric(out["outcome_label_value"].where(is_pcr), errors="coerce")
    else:
        out["pcr_label"] = np.nan

    return out[["patient_id", "source_record_id", *FEATURE_SET, "pcr_label"]].dropna(subset=["pcr_label"]).copy()


def _baseline_size(frame: pd.DataFrame) -> pd.Series:
    if "baseline_longest_diameter_mm" in frame.columns:
        return pd.to_numeric(frame["baseline_longest_diameter_mm"], errors="coerce")
    if "imaging_features" not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    return frame["imaging_features"].map(_extract_baseline_size)


def _extract_baseline_size(value: Any) -> float | None:
    parsed = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return None
    if not isinstance(parsed, dict):
        return None
    for key in ("baseline_longest_diameter_mm", "baseline_tumor_size_mm"):
        if key in parsed:
            try:
                return float(parsed[key])
            except (TypeError, ValueError):
                return None
    return None


def _model() -> Pipeline:
    return Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(
                    [
                        (
                            "metadata",
                            Pipeline(
                                [
                                    ("impute", SimpleImputer(strategy="median")),
                                    ("scale", StandardScaler()),
                                ]
                            ),
                            FEATURE_SET,
                        )
                    ]
                ),
            ),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=2000)),
        ]
    )


def _bootstrap_intervals(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    seed: int,
    n_bootstrap: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    auc_values: list[float] = []
    brier_values: list[float] = []
    indices = np.arange(len(labels))
    for _ in range(max(0, n_bootstrap)):
        sample = rng.choice(indices, size=len(indices), replace=True)
        y = labels[sample]
        p = probabilities[sample]
        if len(np.unique(y)) >= 2:
            auc_values.append(float(roc_auc_score(y, p)))
        brier_values.append(float(brier_score_loss(y, p)))
    return {
        "n_bootstrap": int(n_bootstrap),
        "roc_auc": _ci(auc_values),
        "brier": _ci(brier_values),
        "note": "Intervals are over external pCR probe predictions only; not clinical validation.",
    }


def _ci(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"low": None, "high": None}
    return {
        "low": round(float(np.quantile(values, 0.025)), 4),
        "high": round(float(np.quantile(values, 0.975)), 4),
    }


def _safe_auc(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    return round(float(roc_auc_score(labels, probabilities)), 4)


def _probability_summary(values: np.ndarray) -> dict[str, float | None]:
    if len(values) == 0:
        return {"mean": None, "p10": None, "p50": None, "p90": None, "min": None, "max": None}
    return {
        "mean": round(float(np.mean(values)), 4),
        "p10": round(float(np.quantile(values, 0.10)), 4),
        "p50": round(float(np.quantile(values, 0.50)), 4),
        "p90": round(float(np.quantile(values, 0.90)), 4),
        "min": round(float(np.min(values)), 4),
        "max": round(float(np.max(values)), 4),
    }


def _cohort_summary(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {"rows": 0, "pcr_positive": 0, "pcr_negative": 0, "pcr_positive_rate": None}
    labels = rows["pcr_label"].astype(int)
    return {
        "rows": int(len(rows)),
        "pcr_positive": int(labels.sum()),
        "pcr_negative": int((labels == 0).sum()),
        "pcr_positive_rate": round(float(labels.mean()), 4),
        "age_mean": _mean(rows["age"]),
        "baseline_size_mean_mm": _mean(rows["baseline_longest_diameter_mm"]),
        "hr_positive_rate": _mean(rows["hr_positive"]),
        "her2_positive_rate": _mean(rows["her2_positive"]),
        "triple_negative_rate": _mean(rows["triple_negative"]),
    }


def _missingness(rows: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    if rows.empty:
        return {column: 1.0 for column in columns}
    return {column: round(float(rows[column].isna().mean()), 4) for column in columns}


def _mean(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return round(float(numeric.mean()), 4)


def _text(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.get(column, pd.Series(index=frame.index, dtype=str)).fillna("").astype(str)


def _write_predictions(path: Path, rows: pd.DataFrame, *, probabilities: np.ndarray | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = rows[["patient_id", "source_record_id", "pcr_label", *FEATURE_SET]].copy()
    if probabilities is not None:
        out["metadata_probe_pcr_probability"] = np.round(probabilities, 6)
        out["metadata_probe_predicted_label"] = (probabilities >= 0.5).astype(int)
    else:
        out["metadata_probe_pcr_probability"] = np.nan
        out["metadata_probe_predicted_label"] = np.nan
    out["clinical_interpretation_allowed"] = False
    out["target_mismatch_note"] = "pCR is not equivalent to NLCare synthetic monitoring targets."
    out.to_csv(path, index=False)


def _write_doc(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    result = payload.get("stress_result", {})
    lines = [
        "# BreastDCEDL Metadata-Only Stress Benchmark",
        "",
        payload["claim_boundary"],
        "",
        "## Scope",
        "",
        "- Metadata-only probe over age, baseline tumor size, HR status, HER2 status, and triple-negative context.",
        "- No image-pixel model training in this artifact.",
        "- pCR is treated as an external stress endpoint, not an NLCare clinical target.",
        "",
        "## Result",
        "",
        f"- Status: `{payload['status']}`",
        f"- Rows: `{payload['cohort_summary']['rows']}`",
        f"- Stress result status: `{result.get('status')}`",
        f"- ROC AUC: `{result.get('roc_auc')}`",
        f"- Brier: `{result.get('brier')}`",
        f"- Balanced accuracy: `{result.get('balanced_accuracy')}`",
        "",
        "## Decision",
        "",
        f"- Recommendation: `{payload['integration_decision']['recommendation']}`",
        f"- Model promotion allowed: `{payload['integration_decision']['model_promotion_allowed']}`",
        f"- Live model update allowed: `{payload['integration_decision']['live_model_update_allowed']}`",
        "",
        "## Blocked Claims",
        "",
        *[f"- {claim}" for claim in payload["blocked_claims"]],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_frame(path: str | Path) -> pd.DataFrame:
    resolved = _resolve(path)
    if not resolved.exists():
        return pd.DataFrame()
    return pd.read_csv(resolved)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT_DIR / candidate


def _relative_to_root(path: Path) -> Path:
    try:
        return path.relative_to(ROOT_DIR)
    except ValueError:
        return path


__all__ = [
    "CLAIM_BOUNDARY",
    "DEFAULT_CANONICAL_CSV",
    "DEFAULT_DOC_PATH",
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_PREDICTIONS_PATH",
    "FEATURE_SET",
    "run_breastdcedl_metadata_stress",
]
