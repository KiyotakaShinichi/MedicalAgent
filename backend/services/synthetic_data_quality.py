"""Synthetic data quality proxy metrics.

These are *generator-quality* metrics for the synthetic patient
journeys.  They do not measure clinical realism; they measure
internal consistency of the synthetic distribution against
hand-curated lab/imaging ranges and structural expectations.

The label "synthetic generator quality proxy" is intentional and
appears in the JSON output.  An external reviewer should not
mistake this for clinical realism.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


DEFAULT_ROWS_PATH = Path("Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv")
DEFAULT_OUTPUT_PATH = Path("Data/evals/realism/latest_synthetic_data_quality.json")

# Hand-curated plausibility ranges for selected lab + imaging features.
# These are not clinical-grade reference intervals; they are sanity
# windows used to flag "is this value remotely physiological?"
LAB_RANGES: dict[str, tuple[float, float]] = {
    # K/uL or 10^9/L
    "pre_wbc": (0.5, 30.0),
    "nadir_wbc": (0.0, 30.0),
    "recovery_wbc": (0.5, 30.0),
    "pre_anc": (0.0, 25.0),
    "nadir_anc": (0.0, 25.0),
    "pre_hemoglobin": (5.0, 18.0),
    "nadir_hemoglobin": (5.0, 18.0),
    "recovery_hemoglobin": (5.0, 18.0),
    "pre_platelets": (20.0, 600.0),
    "nadir_platelets": (10.0, 600.0),
    "recovery_platelets": (20.0, 600.0),
    "mri_tumor_size_cm": (0.0, 15.0),
    "mri_percent_change_from_baseline": (-100.0, 100.0),
    "max_symptom_severity": (0.0, 10.0),
    "age": (18, 100),
}

# Pairs we expect to be positively correlated in a well-generated set.
# Each entry: (feature_a, feature_b, expected_min_pearson)
EXPECTED_CORRELATIONS: tuple[tuple[str, str, float], ...] = (
    ("pre_wbc", "pre_anc", 0.30),
    ("nadir_wbc", "nadir_anc", 0.30),
    ("pre_hemoglobin", "recovery_hemoglobin", 0.10),
)


@dataclass
class FeatureSummary:
    feature: str
    n_observed: int
    n_missing: int
    missing_rate: float
    min: float
    max: float
    mean: float
    std: float
    n_out_of_range: int
    out_of_range_rate: float
    range_low: float
    range_high: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "n_observed": self.n_observed,
            "n_missing": self.n_missing,
            "missing_rate": round(self.missing_rate, 4),
            "min": float(self.min),
            "max": float(self.max),
            "mean": round(float(self.mean), 4),
            "std": round(float(self.std), 4),
            "n_out_of_range": self.n_out_of_range,
            "out_of_range_rate": round(self.out_of_range_rate, 4),
            "range_low": self.range_low,
            "range_high": self.range_high,
        }


def _summarize_feature(name: str, series: pd.Series, lo: float, hi: float) -> FeatureSummary:
    numeric = pd.to_numeric(series, errors="coerce")
    n_total = len(numeric)
    n_missing = int(numeric.isna().sum())
    observed = numeric.dropna()
    n_obs = len(observed)
    if n_obs == 0:
        return FeatureSummary(
            feature=name,
            n_observed=0,
            n_missing=n_missing,
            missing_rate=1.0 if n_total else 0.0,
            min=float("nan"),
            max=float("nan"),
            mean=float("nan"),
            std=float("nan"),
            n_out_of_range=0,
            out_of_range_rate=0.0,
            range_low=lo,
            range_high=hi,
        )
    oor_mask = (observed < lo) | (observed > hi)
    n_oor = int(oor_mask.sum())
    return FeatureSummary(
        feature=name,
        n_observed=n_obs,
        n_missing=n_missing,
        missing_rate=n_missing / n_total if n_total else 0.0,
        min=float(observed.min()),
        max=float(observed.max()),
        mean=float(observed.mean()),
        std=float(observed.std(ddof=0)),
        n_out_of_range=n_oor,
        out_of_range_rate=n_oor / n_obs,
        range_low=lo,
        range_high=hi,
    )


def _correlation(rows: pd.DataFrame, a: str, b: str) -> float | None:
    if a not in rows.columns or b not in rows.columns:
        return None
    s_a = pd.to_numeric(rows[a], errors="coerce")
    s_b = pd.to_numeric(rows[b], errors="coerce")
    mask = s_a.notna() & s_b.notna()
    if mask.sum() < 30:
        return None
    return float(np.corrcoef(s_a[mask], s_b[mask])[0, 1])


def build_quality_report(
    rows_path: Path = DEFAULT_ROWS_PATH,
    *,
    ranges: Mapping[str, tuple[float, float]] = LAB_RANGES,
    correlations: tuple[tuple[str, str, float], ...] = EXPECTED_CORRELATIONS,
) -> dict[str, Any]:
    rows = pd.read_csv(rows_path)
    n_rows = int(len(rows))
    n_patients = int(rows["patient_id"].nunique()) if "patient_id" in rows.columns else 0

    feature_reports: list[dict[str, Any]] = []
    n_features_with_oor: int = 0
    total_oor_rate: float = 0.0
    for name, (lo, hi) in ranges.items():
        if name not in rows.columns:
            continue
        summary = _summarize_feature(name, rows[name], lo, hi)
        feature_reports.append(summary.to_dict())
        if summary.n_out_of_range > 0:
            n_features_with_oor += 1
        total_oor_rate += summary.out_of_range_rate

    avg_oor_rate = total_oor_rate / len(feature_reports) if feature_reports else 0.0

    corr_reports: list[dict[str, Any]] = []
    for a, b, expected_min in correlations:
        observed = _correlation(rows, a, b)
        corr_reports.append({
            "feature_a": a,
            "feature_b": b,
            "expected_min_pearson": expected_min,
            "observed_pearson": None if observed is None else round(observed, 4),
            "passed": observed is not None and observed >= expected_min,
        })
    correlations_passed = sum(1 for c in corr_reports if c["passed"])

    return {
        "schema_version": "1.0",
        "status": "informational",
        "label": "synthetic_generator_quality_proxy",
        "disclaimer": (
            "These metrics measure internal consistency of the synthetic "
            "generator's output. They are NOT a measure of clinical "
            "realism and they do NOT establish that the data resembles "
            "any real patient population."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(rows_path),
        "n_rows": n_rows,
        "n_patients": n_patients,
        "features": feature_reports,
        "n_features_with_out_of_range": n_features_with_oor,
        "avg_out_of_range_rate": round(avg_oor_rate, 4),
        "correlations": corr_reports,
        "correlations_passed": correlations_passed,
        "correlations_total": len(corr_reports),
    }


def write_quality_report(
    rows_path: Path = DEFAULT_ROWS_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    report = build_quality_report(rows_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


__all__ = [
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_ROWS_PATH",
    "EXPECTED_CORRELATIONS",
    "LAB_RANGES",
    "build_quality_report",
    "write_quality_report",
]
