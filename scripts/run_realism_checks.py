import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_TRAINING_CSV = "Data/complete_synthetic_training/locked_holdout/development_rows.csv"
DEFAULT_CHECKS_PATH = "benchmarks/synthetic_realism_checks.json"
DEFAULT_OUTPUT_PATH = "Data/evals/realism/latest_realism_checks.json"


def _load_json(path: str) -> dict:
    json_path = Path(path)
    if not json_path.exists():
        return {}
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _rate_in_range(series: pd.Series, min_value: float, max_value: float) -> float | None:
    if series.empty:
        return None
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    within = values.between(min_value, max_value)
    return round(float(within.mean()), 3)


def _cycle_interval_check(frame: pd.DataFrame, min_days: int, max_days: int) -> dict:
    if "treatment_date" not in frame.columns or "patient_id" not in frame.columns:
        return {"status": "unavailable", "pass_rate": None}
    dates = frame.copy()
    dates["treatment_date"] = pd.to_datetime(dates["treatment_date"], errors="coerce")
    dates = dates.dropna(subset=["treatment_date"])
    if dates.empty:
        return {"status": "unavailable", "pass_rate": None}
    intervals = []
    for _, group in dates.sort_values(["patient_id", "treatment_date"]).groupby("patient_id"):
        diffs = group["treatment_date"].diff().dt.days.dropna()
        intervals.extend(diffs.tolist())
    if not intervals:
        return {"status": "unavailable", "pass_rate": None}
    intervals = np.array(intervals, dtype=float)
    pass_rate = round(float(np.mean((intervals >= min_days) & (intervals <= max_days))), 3)
    return {"status": "passed" if pass_rate >= 0.8 else "needs_attention", "pass_rate": pass_rate, "count": int(len(intervals))}


def _toxicity_pattern_check(frame: pd.DataFrame, wbc_threshold: float, anc_threshold: float, min_ratio: float) -> dict:
    if "toxicity_risk_binary" not in frame.columns:
        return {"status": "unavailable", "enrichment_ratio": None}
    risk = pd.to_numeric(frame["toxicity_risk_binary"], errors="coerce").fillna(0)
    nadir_wbc = pd.to_numeric(frame.get("nadir_wbc"), errors="coerce")
    nadir_anc = pd.to_numeric(frame.get("nadir_anc"), errors="coerce")
    low_counts = (nadir_wbc < wbc_threshold) | (nadir_anc < anc_threshold)
    risk_high = low_counts[risk == 1].mean() if (risk == 1).any() else 0.0
    risk_low = low_counts[risk == 0].mean() if (risk == 0).any() else 0.0
    ratio = (risk_high / risk_low) if risk_low else None
    status = "passed" if ratio is not None and ratio >= min_ratio else "needs_attention"
    return {"status": status, "enrichment_ratio": round(float(ratio), 3) if ratio is not None else None}


def _mri_trend_check(frame: pd.DataFrame, min_change: float, max_change: float, min_rate: float) -> dict:
    if "mri_percent_change_from_baseline" not in frame.columns:
        return {"status": "unavailable", "improving_rate": None}
    values = pd.to_numeric(frame["mri_percent_change_from_baseline"], errors="coerce")
    within_rate = _rate_in_range(values, min_change, max_change)
    improving = []
    for _, group in frame.groupby("patient_id"):
        series = pd.to_numeric(group["mri_percent_change_from_baseline"], errors="coerce").dropna()
        if len(series) < 2:
            continue
        improving.append(series.iloc[-1] <= series.iloc[0])
    improving_rate = round(float(np.mean(improving)), 3) if improving else None
    status = "passed" if improving_rate is not None and improving_rate >= min_rate else "needs_attention"
    return {
        "status": status,
        "improving_rate": improving_rate,
        "within_range_rate": within_rate,
    }


def _symptom_correlation(frame: pd.DataFrame, min_corr: float) -> dict:
    if "max_symptom_severity" not in frame.columns:
        return {"status": "unavailable", "correlation": None}
    target_column = "toxicity_risk_binary" if "toxicity_risk_binary" in frame.columns else "urgent_intervention_needed"
    if target_column not in frame.columns:
        return {"status": "unavailable", "correlation": None}
    symptoms = pd.to_numeric(frame["max_symptom_severity"], errors="coerce")
    target = pd.to_numeric(frame[target_column], errors="coerce")
    correlation = symptoms.corr(target)
    status = "passed" if correlation is not None and correlation >= min_corr else "needs_attention"
    return {"status": status, "correlation": round(float(correlation), 3) if correlation is not None else None}


def _subtype_distribution(frame: pd.DataFrame, min_categories: int, max_dominant_share: float) -> dict:
    if "molecular_subtype" not in frame.columns:
        return {"status": "unavailable", "categories": 0, "dominant_share": None}
    counts = frame["molecular_subtype"].dropna().value_counts()
    if counts.empty:
        return {"status": "unavailable", "categories": 0, "dominant_share": None}
    categories = int(len(counts))
    dominant_share = float(counts.iloc[0] / counts.sum())
    status = "passed" if categories >= min_categories and dominant_share <= max_dominant_share else "needs_attention"
    return {"status": status, "categories": categories, "dominant_share": round(dominant_share, 3)}


def main():
    parser = argparse.ArgumentParser(description="Run synthetic realism checks for benchmark ladder.")
    parser.add_argument("--training-csv", default=DEFAULT_TRAINING_CSV)
    parser.add_argument("--checks-path", default=DEFAULT_CHECKS_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    checks = _load_json(args.checks_path)
    training_path = Path(args.training_csv)
    if not training_path.exists():
        report = {
            "schema_version": "synthetic_realism_checks_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "unavailable",
            "message": "Training CSV is missing; realism checks not available.",
            "training_csv": args.training_csv,
        }
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"output_path": args.output_path, "status": report["status"]}, indent=2))
        return

    frame = pd.read_csv(training_path)

    range_results = {}
    for column, config in (checks.get("cbc_ranges") or {}).items():
        pass_rate = _rate_in_range(frame.get(column, pd.Series(dtype=float)), config.get("min"), config.get("max"))
        min_rate = config.get("min_pass_rate")
        status = "passed" if pass_rate is not None and min_rate is not None and pass_rate >= min_rate else "needs_attention"
        range_results[column] = {
            "status": status,
            "pass_rate": pass_rate,
            "min": config.get("min"),
            "max": config.get("max"),
            "min_pass_rate": min_rate,
        }

    interval_cfg = checks.get("cycle_interval_days") or {}
    interval_result = _cycle_interval_check(frame, interval_cfg.get("min", 7), interval_cfg.get("max", 28))

    toxicity_cfg = checks.get("toxicity_patterns") or {}
    toxicity_result = _toxicity_pattern_check(
        frame,
        toxicity_cfg.get("nadir_wbc_urgent_threshold", 1.0),
        toxicity_cfg.get("nadir_anc_urgent_threshold", 0.5),
        toxicity_cfg.get("min_enrichment_ratio", 1.5),
    )

    mri_cfg = checks.get("mri_response_trend") or {}
    mri_result = _mri_trend_check(
        frame,
        mri_cfg.get("min_change", -100.0),
        mri_cfg.get("max_change", 100.0),
        mri_cfg.get("min_improving_rate", 0.6),
    )

    symptom_cfg = checks.get("symptom_toxicity_correlation") or {}
    symptom_result = _symptom_correlation(frame, symptom_cfg.get("min_correlation", 0.2))

    subtype_cfg = checks.get("subtype_distribution") or {}
    subtype_result = _subtype_distribution(
        frame,
        subtype_cfg.get("min_categories", 3),
        subtype_cfg.get("max_dominant_share", 0.75),
    )

    status_values = [
        interval_result.get("status"),
        toxicity_result.get("status"),
        mri_result.get("status"),
        symptom_result.get("status"),
        subtype_result.get("status"),
    ] + [value.get("status") for value in range_results.values()]

    overall = "passed" if all(status == "passed" for status in status_values if status) else "needs_attention"

    report = {
        "schema_version": "synthetic_realism_checks_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": overall,
        "training_csv": args.training_csv,
        "cbc_ranges": range_results,
        "cycle_interval_days": interval_result,
        "toxicity_patterns": toxicity_result,
        "mri_response_trend": mri_result,
        "symptom_toxicity_correlation": symptom_result,
        "subtype_distribution": subtype_result,
        "claim_boundary": checks.get("claim_boundary") or "Engineering realism checks only.",
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({
        "output_path": args.output_path,
        "status": report.get("status"),
        "cbc_range_status": report.get("cbc_ranges"),
        "interval_status": interval_result.get("status"),
    }, indent=2))


if __name__ == "__main__":
    main()
