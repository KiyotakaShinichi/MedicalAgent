from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


DEFAULT_TRAINING_CSV = "Data/complete_synthetic_training/locked_holdout/development_rows.csv"
DEFAULT_EXTERNAL_FEATURES_PATH = "Data/breastdcedl_spy1_features.csv"
DEFAULT_OUTPUT_PATH = "Data/mle_monitoring/synthetic_realism_report.json"
DEFAULT_THRESHOLDS_PATH = "config/safety_thresholds.yaml"


def build_synthetic_realism_report(
    training_csv: str = DEFAULT_TRAINING_CSV,
    external_features_path: str = DEFAULT_EXTERNAL_FEATURES_PATH,
    thresholds_path: str = DEFAULT_THRESHOLDS_PATH,
    output_path: str | None = DEFAULT_OUTPUT_PATH,
):
    training = _load_csv(training_csv)
    external = _load_csv(external_features_path)
    thresholds = _load_thresholds(thresholds_path)

    if training.empty:
        report = {
            "schema_version": "synthetic_realism_report_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "unavailable",
            "message": "Training CSV is missing or empty; realism audit not available.",
            "training_csv": training_csv,
            "external_features_path": external_features_path,
            "claim_boundary": "Synthetic realism checks are engineering proxies only.",
        }
        if output_path:
            _write_json(output_path, report)
        return report

    patient_count = int(training["patient_id"].nunique()) if "patient_id" in training.columns else 0
    numeric_distributions = _numeric_distributions(training)
    threshold_coverage = _threshold_coverage(training, thresholds)
    sim_to_real = _sim_to_real_comparison(training, external)

    report = {
        "schema_version": "synthetic_realism_report_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": _rollup_status([
            threshold_coverage.get("status"),
            sim_to_real.get("status"),
        ]),
        "training_csv": training_csv,
        "external_features_path": external_features_path,
        "training_rows": int(len(training)),
        "training_patients": patient_count,
        "numeric_distributions": numeric_distributions,
        "threshold_coverage": threshold_coverage,
        "sim_to_real_comparison": sim_to_real,
        "limitations": [
            "External comparison uses a small MRI-feature dataset and does not validate clinical performance.",
            "Distribution checks are engineering heuristics, not clinical realism certification.",
        ],
        "claim_boundary": "Synthetic realism checks are engineering proxies only.",
    }

    if output_path:
        _write_json(output_path, report)
    return report


def _numeric_distributions(training: pd.DataFrame) -> dict:
    patient_age = _patient_level_series(training, "age")
    baseline_size = _baseline_tumor_size(training)
    return {
        "age": _numeric_summary(patient_age),
        "baseline_tumor_size_cm": _numeric_summary(baseline_size),
        "pre_wbc": _numeric_summary(_safe_series(training, "pre_wbc")),
        "nadir_wbc": _numeric_summary(_safe_series(training, "nadir_wbc")),
        "pre_hemoglobin": _numeric_summary(_safe_series(training, "pre_hemoglobin")),
        "nadir_hemoglobin": _numeric_summary(_safe_series(training, "nadir_hemoglobin")),
        "pre_platelets": _numeric_summary(_safe_series(training, "pre_platelets")),
        "nadir_platelets": _numeric_summary(_safe_series(training, "nadir_platelets")),
        "mri_percent_change_from_baseline": _numeric_summary(
            _safe_series(training, "mri_percent_change_from_baseline")
        ),
    }


def _threshold_coverage(training: pd.DataFrame, thresholds: dict) -> dict:
    lab_thresholds = (thresholds or {}).get("lab_thresholds") or {}
    coverage = {}
    for lab_key, config in lab_thresholds.items():
        watch = config.get("watch")
        urgent = config.get("urgent_review")
        for prefix in ("pre", "nadir"):
            column = f"{prefix}_{lab_key}"
            if column not in training.columns:
                continue
            series = pd.to_numeric(training[column], errors="coerce").dropna()
            watch_rate = _rate_below(series, watch)
            urgent_rate = _rate_below(series, urgent)
            coverage[column] = {
                "watch_threshold": watch,
                "urgent_threshold": urgent,
                "watch_rate": watch_rate,
                "urgent_rate": urgent_rate,
                "status": _rate_status(watch_rate, urgent_rate),
            }

    status = _rollup_status([item.get("status") for item in coverage.values()])
    return {
        "status": status,
        "lab_thresholds_version": thresholds.get("threshold_config_version"),
        "coverage": coverage,
        "interpretation": (
            "Coverage rates show how often synthetic labs cross watch/urgent thresholds. "
            "Extremely low or extremely high rates may indicate unrealistic simulator behavior."
        ),
    }


def _sim_to_real_comparison(training: pd.DataFrame, external: pd.DataFrame) -> dict:
    if external.empty:
        return {
            "status": "unavailable",
            "message": "External BreastDCEDL features are missing; sim-to-real comparison skipped.",
        }

    comparisons = {}
    training_age = _patient_level_series(training, "age")
    external_age = _safe_series(external, "age")
    comparisons["age_ks"] = _ks_comparison(training_age, external_age)

    training_size = _baseline_tumor_size(training)
    external_size = _safe_series(external, "baseline_longest_diameter_mm") / 10.0
    comparisons["baseline_size_ks"] = _ks_comparison(training_size, external_size)

    training_subtype = _patient_level_categorical(training, "molecular_subtype")
    external_subtype = _safe_series(external, "molecular_subtype")
    comparisons["molecular_subtype_js"] = _js_comparison(training_subtype, external_subtype)

    status = _rollup_status([item.get("status") for item in comparisons.values()])
    return {
        "status": status,
        "comparisons": comparisons,
        "interpretation": (
            "Sim-to-real metrics compare broad distribution shape only. "
            "Higher divergence flags areas to tune the synthetic generator."
        ),
    }


def _ks_comparison(training_series: pd.Series, external_series: pd.Series) -> dict:
    training_values = training_series.dropna().astype(float).to_numpy()
    external_values = external_series.dropna().astype(float).to_numpy()
    if len(training_values) < 5 or len(external_values) < 5:
        return {"status": "unavailable", "ks": None, "n_training": len(training_values), "n_external": len(external_values)}
    ks_value = _ks_statistic(training_values, external_values)
    return {
        "status": _ks_status(ks_value),
        "ks": round(float(ks_value), 3),
        "n_training": int(len(training_values)),
        "n_external": int(len(external_values)),
    }


def _js_comparison(training_series: pd.Series, external_series: pd.Series) -> dict:
    if training_series.empty or external_series.empty:
        return {"status": "unavailable", "js": None}
    training_dist = _category_distribution(training_series)
    external_dist = _category_distribution(external_series)
    js_value = _js_divergence(training_dist, external_dist)
    return {
        "status": _js_status(js_value),
        "js": round(float(js_value), 3),
        "training_distribution": training_dist,
        "external_distribution": external_dist,
    }


def _patient_level_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns or "patient_id" not in frame.columns:
        return pd.Series(dtype=float)
    subset = frame[["patient_id", column]].dropna().copy()
    if subset.empty:
        return pd.Series(dtype=float)
    subset = subset.drop_duplicates(subset=["patient_id"])
    return subset[column]


def _patient_level_categorical(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns or "patient_id" not in frame.columns:
        return pd.Series(dtype=str)
    subset = frame[["patient_id", column]].dropna().copy()
    if subset.empty:
        return pd.Series(dtype=str)
    subset = subset.drop_duplicates(subset=["patient_id"])
    return subset[column].astype(str)


def _baseline_tumor_size(frame: pd.DataFrame) -> pd.Series:
    if "mri_tumor_size_cm" not in frame.columns:
        return pd.Series(dtype=float)
    working = frame[["patient_id", "mri_tumor_size_cm"] + (["cycle"] if "cycle" in frame.columns else [])].dropna()
    if working.empty:
        return pd.Series(dtype=float)
    if "cycle" in working.columns:
        working = working.sort_values(["patient_id", "cycle"])
    else:
        working = working.sort_values(["patient_id"])
    baseline = working.groupby("patient_id", as_index=False).first()
    return baseline["mri_tumor_size_cm"].astype(float)


def _safe_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _numeric_summary(series: pd.Series) -> dict:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"count": 0}
    return {
        "count": int(values.shape[0]),
        "mean": round(float(values.mean()), 3),
        "std": round(float(values.std(ddof=0)), 3),
        "min": round(float(values.min()), 3),
        "p5": round(float(np.percentile(values, 5)), 3),
        "p50": round(float(np.percentile(values, 50)), 3),
        "p95": round(float(np.percentile(values, 95)), 3),
        "max": round(float(values.max()), 3),
    }


def _rate_below(series: pd.Series, threshold) -> float | None:
    if threshold is None:
        return None
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return round(float((values < float(threshold)).mean()), 4)


def _rate_status(watch_rate: float | None, urgent_rate: float | None) -> str:
    rates = [rate for rate in (watch_rate, urgent_rate) if rate is not None]
    if not rates:
        return "unavailable"
    if any(rate < 0.005 or rate > 0.60 for rate in rates):
        return "unideal"
    if any(rate < 0.01 or rate > 0.50 for rate in rates):
        return "acceptable"
    return "passed"


def _ks_status(value: float | None) -> str:
    if value is None:
        return "unavailable"
    if value <= 0.10:
        return "strong"
    if value <= 0.20:
        return "passed"
    if value <= 0.30:
        return "acceptable"
    return "unideal"


def _js_status(value: float | None) -> str:
    if value is None:
        return "unavailable"
    if value <= 0.05:
        return "strong"
    if value <= 0.10:
        return "passed"
    if value <= 0.20:
        return "acceptable"
    return "unideal"


def _rollup_status(statuses: list[str | None]) -> str:
    status_values = [status for status in statuses if status]
    if not status_values:
        return "unavailable"
    if any(status in {"failed", "unideal"} for status in status_values):
        return "unideal"
    if any(status == "acceptable" for status in status_values):
        return "acceptable"
    if all(status == "strong" for status in status_values):
        return "strong"
    return "passed"


def _category_distribution(series: pd.Series) -> dict:
    values = series.dropna().astype(str)
    if values.empty:
        return {}
    counts = values.value_counts(normalize=True)
    return {key: round(float(value), 4) for key, value in counts.to_dict().items()}


def _js_divergence(p: dict, q: dict) -> float:
    keys = sorted(set(p) | set(q))
    p_values = np.array([p.get(key, 0.0) for key in keys], dtype=float)
    q_values = np.array([q.get(key, 0.0) for key in keys], dtype=float)
    p_values = p_values / p_values.sum() if p_values.sum() else p_values
    q_values = q_values / q_values.sum() if q_values.sum() else q_values
    m_values = 0.5 * (p_values + q_values)
    return 0.5 * _kl_divergence(p_values, m_values) + 0.5 * _kl_divergence(q_values, m_values)


def _kl_divergence(p_values: np.ndarray, q_values: np.ndarray) -> float:
    mask = (p_values > 0) & (q_values > 0)
    if not mask.any():
        return 0.0
    return float(np.sum(p_values[mask] * np.log2(p_values[mask] / q_values[mask])))


def _ks_statistic(values_a: np.ndarray, values_b: np.ndarray) -> float:
    values_a = np.sort(values_a)
    values_b = np.sort(values_b)
    combined = np.sort(np.concatenate([values_a, values_b]))
    cdf_a = np.searchsorted(values_a, combined, side="right") / len(values_a)
    cdf_b = np.searchsorted(values_b, combined, side="right") / len(values_b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _load_thresholds(path: str) -> dict:
    if not path:
        return {}
    threshold_path = Path(path)
    if not threshold_path.exists():
        return {}
    try:
        return yaml.safe_load(threshold_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {}


def _load_csv(path: str) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def _write_json(path: str, payload: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
