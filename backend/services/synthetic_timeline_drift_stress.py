"""Synthetic-only timeline drift stress scaffold.

Splits the synthetic temporal_ml_rows.csv into a *baseline* window
(earlier cycles) and a *recent* window (later cycles).  Applies one
of three deterministic shift mechanisms to the recent window and
checks whether two-sample distribution tests fire.

The runner is **review-only**: it reports detection rates and false
shift rates on the unmodified synthetic, and never claims clinical
deterioration detection.  ``review_only_boundary == True`` is
test-locked.

Output: ``Data/evals/models/latest_synthetic_timeline_drift_stress.json``
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROWS_PATH = Path("Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv")
OUTPUT_PATH = Path("Data/evals/models/latest_synthetic_timeline_drift_stress.json")


_LAB_COLUMNS: tuple[str, ...] = (
    "nadir_wbc", "nadir_anc", "nadir_hemoglobin", "nadir_platelets",
)
_SYMPTOM_COLUMNS: tuple[str, ...] = (
    "max_symptom_severity", "symptom_count",
)


def _ks_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    """Lightweight Kolmogorov-Smirnov two-sample p-value approximation.

    Avoids the scipy dependency.  Returns 1.0 when either window is
    empty.  Accuracy is fine for the qualitative "shift detected vs.
    not detected" tripwire this scaffold needs.
    """
    a = np.asarray([v for v in a if np.isfinite(v)])
    b = np.asarray([v for v in b if np.isfinite(v)])
    if len(a) == 0 or len(b) == 0:
        return 1.0
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    all_vals = np.concatenate([a_sorted, b_sorted])
    cdf_a = np.searchsorted(a_sorted, all_vals, side="right") / len(a_sorted)
    cdf_b = np.searchsorted(b_sorted, all_vals, side="right") / len(b_sorted)
    d = float(np.max(np.abs(cdf_a - cdf_b)))
    n_eff = len(a) * len(b) / (len(a) + len(b))
    # KS p-value approximation (Smirnov).
    arg = (np.sqrt(n_eff) + 0.12 + 0.11 / np.sqrt(n_eff)) * d
    if arg <= 0:
        return 1.0
    j = np.arange(1, 101)
    series = 2 * np.sum((-1) ** (j - 1) * np.exp(-2 * (j * arg) ** 2))
    return float(np.clip(series, 0.0, 1.0))


def _missingness_fraction(series: pd.Series) -> float:
    return float(series.isna().mean()) if len(series) else 0.0


def _split_baseline_recent(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "cycle" not in df.columns:
        half = len(df) // 2
        return df.iloc[:half].copy(), df.iloc[half:].copy()
    cycles = sorted(df["cycle"].dropna().unique())
    if len(cycles) < 4:
        half = len(df) // 2
        return df.iloc[:half].copy(), df.iloc[half:].copy()
    cutoff = cycles[len(cycles) // 2]
    baseline = df[df["cycle"] < cutoff].copy()
    recent = df[df["cycle"] >= cutoff].copy()
    return baseline, recent


def _apply_shift(recent: pd.DataFrame, kind: str, rng: np.random.Generator) -> pd.DataFrame:
    out = recent.copy()
    if kind == "lab_drop":
        for col in _LAB_COLUMNS:
            if col in out.columns:
                out[col] = out[col].astype(float) * 0.85  # 15% downward shift
    elif kind == "symptom_burst":
        if "max_symptom_severity" in out.columns:
            out["max_symptom_severity"] = (
                out["max_symptom_severity"].astype(float) + 2.0
            ).clip(0.0, 10.0)
        if "symptom_count" in out.columns:
            out["symptom_count"] = (out["symptom_count"].astype(float) + 1).clip(0, None)
    elif kind == "missingness_spike":
        for col in _LAB_COLUMNS:
            if col in out.columns:
                mask = rng.random(len(out)) < 0.30
                out.loc[mask, col] = np.nan
    return out


def _detect_shift(
    baseline: pd.DataFrame,
    recent: pd.DataFrame,
    columns: tuple[str, ...],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Return per-column KS p-values + the binary detection flag."""
    results: dict[str, float] = {}
    detected = False
    for col in columns:
        if col not in baseline.columns or col not in recent.columns:
            continue
        p = _ks_pvalue(
            baseline[col].astype(float).to_numpy(),
            recent[col].astype(float).to_numpy(),
        )
        results[col] = round(p, 4)
        if p < alpha:
            detected = True
    return {"per_column_pvalue": results, "detected_at_alpha_0_05": detected}


def _detect_missingness_shift(
    baseline: pd.DataFrame, recent: pd.DataFrame, alpha: float = 0.05
) -> dict[str, Any]:
    per_col: dict[str, dict[str, float]] = {}
    detected = False
    for col in _LAB_COLUMNS:
        if col not in baseline.columns or col not in recent.columns:
            continue
        base_miss = _missingness_fraction(baseline[col])
        rec_miss = _missingness_fraction(recent[col])
        # Simple delta tripwire — > 0.10 absolute increase counts.
        delta = rec_miss - base_miss
        flagged = delta > 0.10
        per_col[col] = {
            "baseline_missingness": round(base_miss, 4),
            "recent_missingness": round(rec_miss, 4),
            "delta": round(delta, 4),
            "flagged": bool(flagged),
        }
        if flagged:
            detected = True
    _ = alpha
    return {"per_column": per_col, "detected": detected}


def build_report() -> dict[str, Any]:
    started = time.perf_counter()
    if not ROWS_PATH.exists():
        return _missing_csv_report()

    df = pd.read_csv(ROWS_PATH)
    baseline, recent = _split_baseline_recent(df)
    rng = np.random.default_rng(seed=20260602)

    # Baseline ↔ recent unmodified: should NOT detect a shift in a
    # well-behaved synthetic distribution.  Any positive here is a
    # false-shift signal.
    base_recent_lab = _detect_shift(baseline, recent, _LAB_COLUMNS)
    base_recent_symptom = _detect_shift(baseline, recent, _SYMPTOM_COLUMNS)
    base_recent_missingness = _detect_missingness_shift(baseline, recent)

    # Apply three engineered shifts and check that the test fires.
    lab_dropped = _apply_shift(recent, "lab_drop", rng)
    lab_dropped_detection = _detect_shift(baseline, lab_dropped, _LAB_COLUMNS)
    symptom_burst = _apply_shift(recent, "symptom_burst", rng)
    symptom_burst_detection = _detect_shift(baseline, symptom_burst, _SYMPTOM_COLUMNS)
    missingness_spike = _apply_shift(recent, "missingness_spike", rng)
    missingness_spike_detection = _detect_missingness_shift(baseline, missingness_spike)

    n_engineered = 3
    n_detected = sum(
        1 for d in (lab_dropped_detection, symptom_burst_detection)
        if d["detected_at_alpha_0_05"]
    ) + (1 if missingness_spike_detection["detected"] else 0)

    n_baseline_runs = 3
    n_baseline_false_shifts = sum(
        1 for d in (base_recent_lab, base_recent_symptom)
        if d["detected_at_alpha_0_05"]
    ) + (1 if base_recent_missingness["detected"] else 0)

    metrics = {
        "distribution_shift_detection_rate": round(n_detected / n_engineered, 4),
        "false_shift_rate_on_baseline_synthetic": round(
            n_baseline_false_shifts / n_baseline_runs, 4
        ),
        "missingness_shift_detection": missingness_spike_detection["detected"],
        "lab_trend_shift_detection": lab_dropped_detection["detected_at_alpha_0_05"],
        "symptom_trend_shift_detection": symptom_burst_detection["detected_at_alpha_0_05"],
    }

    return {
        "schema_version": "synthetic_timeline_drift_stress_v1",
        "status": "informational",
        "label": "synthetic_timeline_drift_stress",
        "clinical_validation": False,
        "review_only_boundary": True,
        "claim_boundary": (
            "Synthetic-only timeline drift stress scaffold.  Splits the "
            "synthetic CSV into baseline / recent windows and exercises a "
            "two-sample KS proxy plus a missingness-delta tripwire.  This is "
            "NOT clinical deterioration detection, NOT real-world drift "
            "monitoring, and NOT clinical validation."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "wall_time_ms": round((time.perf_counter() - started) * 1000.0, 2),
        "source_csv": str(ROWS_PATH).replace("\\", "/"),
        "n_baseline_rows": int(len(baseline)),
        "n_recent_rows": int(len(recent)),
        "metrics": metrics,
        "baseline_vs_recent_unmodified": {
            "lab": base_recent_lab,
            "symptom": base_recent_symptom,
            "missingness": base_recent_missingness,
        },
        "engineered_shifts": {
            "lab_drop": lab_dropped_detection,
            "symptom_burst": symptom_burst_detection,
            "missingness_spike": missingness_spike_detection,
        },
        "contamination_note": (
            "Engineered shifts and seeds are fixed.  Promoting any metric "
            "here to a live monitor would require real-cohort distribution "
            "evidence under IRB."
        ),
    }


def _missing_csv_report() -> dict[str, Any]:
    return {
        "schema_version": "synthetic_timeline_drift_stress_v1",
        "status": "needs_attention",
        "label": "synthetic_timeline_drift_stress",
        "clinical_validation": False,
        "review_only_boundary": True,
        "claim_boundary": (
            "Source CSV missing; emitting placeholder.  Not clinical "
            "validation; not clinical deterioration detection."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def write_report(output_path: Path = OUTPUT_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(build_report(), indent=2), encoding="utf-8")
    return output_path


__all__ = ["OUTPUT_PATH", "build_report", "write_report"]
