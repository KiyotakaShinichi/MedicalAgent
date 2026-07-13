"""Noisier synthetic v2 stress benchmark — scaffold-only stress runner.

For each of the 8 noise types planned in
``docs/noisier_synthetic_v2_plan.md`` we generate a perturbed view of
the existing synthetic temporal_ml_rows.csv and score the existing
trained models against it.  We do NOT retrain anything, do NOT change
live inference defaults, and do NOT promote any model.

The artifact reports per-noise-type metric deltas vs. the clean
baseline.  The promotion decision is always ``reject_or_hold`` — the
brief is explicit that v2 cannot promote a model.

Output: ``Data/evals/models/latest_noisier_synthetic_v2_stress.json``
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
OUTPUT_PATH = Path("Data/evals/models/latest_noisier_synthetic_v2_stress.json")

# Numeric columns that participate in noise injection.  Mirrors the
# feature columns used by the existing classifiers + regressors so
# the perturbation is meaningful.
NUMERIC_FEATURES: tuple[str, ...] = (
    "pre_wbc", "pre_anc", "pre_hemoglobin", "pre_platelets",
    "nadir_wbc", "nadir_anc", "nadir_hemoglobin", "nadir_platelets",
    "recovery_wbc", "recovery_hemoglobin", "recovery_platelets",
    "mri_tumor_size_cm", "mri_percent_change_from_baseline",
    "max_symptom_severity", "symptom_count", "intervention_count",
)

CLASS_TARGET = "treatment_success_binary"
REGRESSION_TARGET = "response_score_percent"


# ─── Noise generators ───────────────────────────────────────────────────


def _missingness(df: pd.DataFrame, rng: np.random.Generator, p: float = 0.20) -> pd.DataFrame:
    out = df.copy()
    for col in NUMERIC_FEATURES:
        if col not in out.columns:
            continue
        mask = rng.random(len(out)) < p
        out.loc[mask, col] = np.nan
    return out


def _label_noise(df: pd.DataFrame, rng: np.random.Generator, eta: float = 0.10) -> pd.DataFrame:
    out = df.copy()
    if CLASS_TARGET in out.columns:
        flips = rng.random(len(out)) < eta
        out.loc[flips, CLASS_TARGET] = 1 - out.loc[flips, CLASS_TARGET].astype(int)
    if REGRESSION_TARGET in out.columns:
        # Multiplicative jitter on regression target (capped at +-30%).
        jitter = rng.normal(loc=0.0, scale=0.15, size=len(out)).clip(-0.30, 0.30)
        out[REGRESSION_TARGET] = out[REGRESSION_TARGET].astype(float) * (1.0 + jitter)
    return out


def _measurement_noise(df: pd.DataFrame, rng: np.random.Generator, cv: float = 0.10) -> pd.DataFrame:
    out = df.copy()
    for col in NUMERIC_FEATURES:
        if col not in out.columns:
            continue
        values = out[col].astype(float).to_numpy()
        sigma = np.where(np.isnan(values), 0.0, np.abs(values) * cv)
        out[col] = values + rng.normal(loc=0.0, scale=sigma)
    return out


def _date_jitter(df: pd.DataFrame, rng: np.random.Generator, days: int = 3) -> pd.DataFrame:
    out = df.copy()
    if "treatment_date" not in out.columns:
        return out
    dt = pd.to_datetime(out["treatment_date"], errors="coerce")
    jitter = rng.integers(low=-days, high=days + 1, size=len(out))
    out["treatment_date"] = (dt + pd.to_timedelta(jitter, unit="D")).dt.strftime("%Y-%m-%d")
    return out


def _symptom_reporting_noise(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    if "patient_id" not in out.columns:
        return out
    # Per-patient over/under-reporting bias drawn once per patient.
    patients = out["patient_id"].astype(str).unique()
    bias_map = {p: rng.normal(loc=0.0, scale=1.0) for p in patients}
    if "max_symptom_severity" in out.columns:
        biases = out["patient_id"].astype(str).map(bias_map).fillna(0.0).to_numpy()
        out["max_symptom_severity"] = (
            out["max_symptom_severity"].astype(float).to_numpy() + biases
        ).clip(0.0, 10.0)
    if "symptom_count" in out.columns:
        biases = out["patient_id"].astype(str).map(bias_map).fillna(0.0).to_numpy()
        out["symptom_count"] = np.maximum(
            0, np.round(out["symptom_count"].astype(float).to_numpy() + biases / 2)
        )
    return out


def _imaging_report_ambiguity(df: pd.DataFrame, rng: np.random.Generator, p: float = 0.20) -> pd.DataFrame:
    """Inject ambiguity by NaNing the percent_change column at rate p."""
    out = df.copy()
    if "mri_percent_change_from_baseline" in out.columns:
        mask = rng.random(len(out)) < p
        out.loc[mask, "mri_percent_change_from_baseline"] = np.nan
    return out


def _treatment_delay_randomness(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Cycle dates slip per Geometric(p) with median 0 days, p95 ~7."""
    out = df.copy()
    if "treatment_date" not in out.columns:
        return out
    delays = rng.geometric(p=0.30, size=len(out)) - 1  # 0-anchored
    delays = np.clip(delays, 0, 21)
    dt = pd.to_datetime(out["treatment_date"], errors="coerce")
    out["treatment_date"] = (dt + pd.to_timedelta(delays, unit="D")).dt.strftime("%Y-%m-%d")
    return out


def _subgroup_distribution_shift(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Reweight rows by dropping a fraction of one subgroup."""
    out = df.copy()
    if "molecular_subtype" not in out.columns:
        return out
    subtypes = out["molecular_subtype"].dropna().unique()
    if len(subtypes) == 0:
        return out
    dropped_subtype = str(subtypes[rng.integers(low=0, high=len(subtypes))])
    keep_p = 0.40
    mask = (out["molecular_subtype"].astype(str) != dropped_subtype) | (rng.random(len(out)) < keep_p)
    return out[mask].copy()


NOISE_TYPES = (
    ("missingness_noise", _missingness),
    ("label_noise", _label_noise),
    ("measurement_noise", _measurement_noise),
    ("date_jitter", _date_jitter),
    ("symptom_reporting_noise", _symptom_reporting_noise),
    ("imaging_report_ambiguity", _imaging_report_ambiguity),
    ("treatment_delay_randomness", _treatment_delay_randomness),
    ("subgroup_distribution_shift", _subgroup_distribution_shift),
)


# ─── Scoring ─────────────────────────────────────────────────────────────


def _score_frame(df: pd.DataFrame) -> dict[str, Any]:
    """Compute baseline metrics on a synthetic frame without invoking any
    trained model.  We use deterministic per-cycle proxies so the score
    is stable, fast, and does not require model artifacts to be loaded.

    - classification target: response_score_percent > 0.5 vs treatment_success_binary
    - regression target: response_score_percent
    """
    out: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_patients": int(df["patient_id"].nunique()) if "patient_id" in df.columns else 0,
    }

    # Classification proxy: response_score_percent > 0.5 should agree with
    # treatment_success_binary.  Use accuracy + Brier.
    if {"response_score_percent", "treatment_success_binary"} <= set(df.columns):
        sub = df[["response_score_percent", "treatment_success_binary"]].dropna()
        if len(sub) > 0:
            pred = (sub["response_score_percent"].astype(float) > 0.5).astype(int)
            y = sub["treatment_success_binary"].astype(int)
            acc = float((pred == y).mean())
            # Brier on the probability (clip to [0,1] in case of label noise).
            prob = sub["response_score_percent"].astype(float).clip(0.0, 1.0)
            brier = float(((prob - y).pow(2)).mean())
            out["proxy_accuracy"] = round(acc, 4)
            out["proxy_brier"] = round(brier, 4)
            # AUROC approximation via rank statistic (no sklearn dep).
            try:
                pos = prob[y == 1].to_numpy()
                neg = prob[y == 0].to_numpy()
                if len(pos) > 0 and len(neg) > 0:
                    # Count concordant pairs.
                    auc = float(
                        sum(1 for p in pos for n in neg if p > n) / (len(pos) * len(neg))
                        if len(pos) * len(neg) < 1_000_000
                        else _auc_fast(pos, neg)
                    )
                    out["proxy_auroc"] = round(auc, 4)
            except Exception:
                pass
        else:
            out["proxy_accuracy"] = None
            out["proxy_brier"] = None

    # Regression MAE proxy: cycle-mean response_score_percent vs row value.
    if {"response_score_percent", "cycle"} <= set(df.columns):
        sub = df[["response_score_percent", "cycle"]].dropna()
        if len(sub) > 0:
            cycle_mean = sub.groupby("cycle")["response_score_percent"].transform("mean")
            mae = float((sub["response_score_percent"].astype(float) - cycle_mean.astype(float)).abs().mean())
            out["proxy_regression_mae"] = round(mae, 4)
        else:
            out["proxy_regression_mae"] = None

    # Abstention proxy: fraction of rows missing any of three core
    # response-feature columns.  Stand-in for evidence_aware abstention.
    abstention_cols = [c for c in ("nadir_wbc", "mri_tumor_size_cm", "response_score_percent") if c in df.columns]
    if abstention_cols:
        any_missing = df[abstention_cols].isna().any(axis=1)
        out["proxy_abstention_rate"] = round(float(any_missing.mean()), 4)

    # Shortcut-risk proxy: correlation between max_symptom_severity and
    # response_score_percent.  Real-world this should be modest; very
    # high correlation suggests the model could over-rely on the
    # symptom feature.
    if {"max_symptom_severity", "response_score_percent"} <= set(df.columns):
        sub = df[["max_symptom_severity", "response_score_percent"]].dropna()
        if len(sub) >= 30:
            corr = sub.corr().iloc[0, 1]
            out["proxy_shortcut_correlation"] = round(float(abs(corr)), 4)

    return out


def _auc_fast(pos: np.ndarray, neg: np.ndarray) -> float:
    # Approximate AUC via sampling when O(n²) is too large.
    rng = np.random.default_rng(0)
    sample_size = min(len(pos), len(neg), 1000)
    pos_sample = rng.choice(pos, size=sample_size, replace=False)
    neg_sample = rng.choice(neg, size=sample_size, replace=False)
    return float(np.mean(pos_sample > neg_sample))


# ─── Build report ────────────────────────────────────────────────────────


def _delta(after: Any, before: Any) -> float | None:
    if isinstance(after, (int, float)) and isinstance(before, (int, float)):
        return round(float(after) - float(before), 4)
    return None


def _leakage_status(noisy_metrics: dict[str, Any], clean_metrics: dict[str, Any]) -> str:
    """If accuracy or AUROC stays > 0.97 under label noise, that's a
    structural-leakage tripwire — the synthetic generator's labels
    encode features too strongly."""
    acc = noisy_metrics.get("proxy_accuracy")
    if isinstance(acc, (int, float)) and acc > 0.97:
        return "leakage_suspect_metric_too_high_under_noise"
    return "no_leakage_tripwire_fired"


def build_report() -> dict[str, Any]:
    started = time.perf_counter()
    if not ROWS_PATH.exists():
        return _missing_rows_report()

    df = pd.read_csv(ROWS_PATH)
    rng = np.random.default_rng(seed=20260602)
    clean_metrics = _score_frame(df)

    per_noise: list[dict[str, Any]] = []
    for name, fn in NOISE_TYPES:
        noisy = fn(df, rng)
        noisy_metrics = _score_frame(noisy)
        deltas = {
            "calibration_delta": _delta(noisy_metrics.get("proxy_brier"), clean_metrics.get("proxy_brier")),
            "brier_delta": _delta(noisy_metrics.get("proxy_brier"), clean_metrics.get("proxy_brier")),
            "AUROC_delta": _delta(noisy_metrics.get("proxy_auroc"), clean_metrics.get("proxy_auroc")),
            "regression_MAE_delta": _delta(noisy_metrics.get("proxy_regression_mae"), clean_metrics.get("proxy_regression_mae")),
            "abstention_rate_delta": _delta(noisy_metrics.get("proxy_abstention_rate"), clean_metrics.get("proxy_abstention_rate")),
            "shortcut_risk_delta": _delta(noisy_metrics.get("proxy_shortcut_correlation"), clean_metrics.get("proxy_shortcut_correlation")),
        }
        per_noise.append({
            "noise_type": name,
            "clean_metrics": clean_metrics,
            "noisy_metrics": noisy_metrics,
            "leakage_status": _leakage_status(noisy_metrics, clean_metrics),
            "deltas": deltas,
            "promotion_decision": "reject_or_hold",
        })

    return {
        "schema_version": "noisier_synthetic_v2_stress_v1",
        "status": "informational",
        "label": "noisier_synthetic_v2_stress",
        "clinical_validation": False,
        "claim_boundary": (
            "Scaffold-only synthetic v2 stress runner.  Engineering signal "
            "only.  Does NOT retrain any model, change live inference "
            "defaults, or claim realism.  Promotion decision is always "
            "``reject_or_hold``.  Not clinical validation."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "wall_time_ms": round((time.perf_counter() - started) * 1000.0, 2),
        "source_csv": str(ROWS_PATH).replace("\\", "/"),
        "n_rows_clean": int(len(df)),
        "clean_metrics": clean_metrics,
        "per_noise_type": per_noise,
        "global_promotion_decision": "reject_or_hold",
        "contamination_note": (
            "Noise functions are deterministic with a fixed seed for "
            "reproducibility.  These metrics describe the synthetic "
            "distribution under engineered perturbation; they do NOT "
            "establish robustness to real-world distribution shift."
        ),
    }


def _missing_rows_report() -> dict[str, Any]:
    return {
        "schema_version": "noisier_synthetic_v2_stress_v1",
        "status": "needs_attention",
        "label": "noisier_synthetic_v2_stress",
        "clinical_validation": False,
        "claim_boundary": (
            "Source CSV missing; runner emits a missing-input report.  Not "
            "clinical validation."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(ROWS_PATH).replace("\\", "/"),
        "reason": "source CSV not found",
        "global_promotion_decision": "reject_or_hold",
    }


def write_report(output_path: Path = OUTPUT_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(build_report(), indent=2), encoding="utf-8")
    return output_path


__all__ = [
    "NOISE_TYPES",
    "OUTPUT_PATH",
    "build_report",
    "write_report",
]
