from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import pandas as pd


DEFAULT_OUTPUT_PATH = "Data/evals/drift/latest_drift_report.json"
DEFAULT_TRAINING_ROWS = "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv"
DEFAULT_PREDICTIONS = "Data/complete_synthetic_training/complete_synthetic_model_predictions.csv"
DEFAULT_MRI_REPORTS = "Data/complete_synthetic_breast_journeys/mri_reports.csv"
DEFAULT_METRICS = "Data/complete_synthetic_training/complete_synthetic_model_metrics.json"


def build_drift_report(output_path: str | None = DEFAULT_OUTPUT_PATH) -> dict:
    training_rows = _load_csv(DEFAULT_TRAINING_ROWS)
    predictions = _load_csv(DEFAULT_PREDICTIONS)
    mri_reports = _load_csv(DEFAULT_MRI_REPORTS)
    metrics = _load_json(DEFAULT_METRICS)

    if training_rows is None or training_rows.empty:
        payload = _unavailable_payload("Training rows unavailable; cannot compute drift metrics.")
        _write_json(output_path, payload)
        return payload

    split = _split_by_patient(training_rows)
    baseline = split["baseline"]
    current = split["current"]

    lab_features = [
        "pre_wbc",
        "pre_anc",
        "pre_hemoglobin",
        "pre_platelets",
        "nadir_wbc",
        "nadir_anc",
        "nadir_hemoglobin",
        "nadir_platelets",
        "recovery_wbc",
        "recovery_hemoglobin",
        "recovery_platelets",
    ]
    lab_shift = _feature_shift_report(baseline, current, lab_features, label="lab")
    symptom_shift = _feature_shift_report(baseline, current, ["max_symptom_severity", "symptom_count"], label="symptom")

    keyword_drift = _imaging_keyword_drift(mri_reports, split)
    confidence_drift = _confidence_drift(predictions, metrics, split)
    calibration_drift = _calibration_drift(predictions, metrics, split)
    subgroup_drift = _subgroup_drift(predictions, training_rows, split)

    completeness = _data_completeness_score(training_rows, lab_features)
    missing_cbc_rate = _missing_rate(training_rows, lab_features)

    payload = {
        "schema_version": "drift_report_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_source": "synthetic_demo" if _is_synthetic(training_rows) else "mixed_demo",
        "missing_cbc_rate": missing_cbc_rate,
        "data_completeness_score": completeness,
        "lab_distribution_shift": lab_shift,
        "symptom_frequency_shift": symptom_shift,
        "imaging_keyword_shift": keyword_drift,
        "model_confidence_drift": confidence_drift,
        "calibration_drift": calibration_drift,
        "subgroup_performance_drift": subgroup_drift,
        "limitations": [
            "Drift is computed on synthetic or demo data unless real monitoring data is present.",
            "Use this report for engineering monitoring only, not clinical validity claims.",
        ],
    }

    _write_json(output_path, payload)
    return payload


def _split_by_patient(training_rows: pd.DataFrame) -> dict:
    patient_ids = sorted(training_rows["patient_id"].dropna().unique())
    midpoint = max(1, len(patient_ids) // 2)
    baseline_ids = set(patient_ids[:midpoint])
    current_ids = set(patient_ids[midpoint:])
    return {
        "baseline": training_rows[training_rows["patient_id"].isin(baseline_ids)],
        "current": training_rows[training_rows["patient_id"].isin(current_ids)],
        "baseline_ids": baseline_ids,
        "current_ids": current_ids,
    }


def _feature_shift_report(baseline: pd.DataFrame, current: pd.DataFrame, features: list[str], label: str) -> dict:
    rows = []
    for feature in features:
        if feature not in baseline.columns or feature not in current.columns:
            continue
        base_mean = float(baseline[feature].dropna().mean())
        cur_mean = float(current[feature].dropna().mean())
        base_std = float(baseline[feature].dropna().std() or 1.0)
        standardized_shift = abs(cur_mean - base_mean) / max(base_std, 1e-6)
        rows.append({
            "feature": feature,
            "baseline_mean": round(base_mean, 3),
            "current_mean": round(cur_mean, 3),
            "standardized_shift": round(standardized_shift, 3),
            "status": _shift_status(standardized_shift),
        })

    return {
        "label": label,
        "status": _worst_status(row["status"] for row in rows),
        "feature_count": len(rows),
        "features": rows,
    }


def _imaging_keyword_drift(mri_reports: pd.DataFrame | None, split: dict) -> dict:
    if mri_reports is None or mri_reports.empty or "response_text" not in mri_reports.columns:
        return {"status": "unavailable", "keywords": []}

    baseline = mri_reports[mri_reports["patient_id"].isin(split["baseline_ids"])]
    current = mri_reports[mri_reports["patient_id"].isin(split["current_ids"])]
    keywords = ["decrease", "increase", "stable", "progression", "metastatic"]
    rows = []
    for keyword in keywords:
        base_rate = _keyword_rate(baseline, keyword)
        cur_rate = _keyword_rate(current, keyword)
        shift = abs(cur_rate - base_rate)
        rows.append({
            "keyword": keyword,
            "baseline_rate": round(base_rate, 3),
            "current_rate": round(cur_rate, 3),
            "shift": round(shift, 3),
            "status": _shift_status(shift / 0.15 if 0.15 else shift),
        })
    return {
        "status": _worst_status(row["status"] for row in rows),
        "keywords": rows,
    }


def _confidence_drift(predictions: pd.DataFrame | None, metrics: dict, split: dict) -> dict:
    if predictions is None or predictions.empty:
        return {"status": "unavailable", "message": "Prediction artifact not available."}

    probability_column = _best_probability_column(predictions, metrics)
    if probability_column not in predictions.columns:
        return {"status": "unavailable", "message": f"Missing {probability_column} column."}

    baseline = predictions[predictions["patient_id"].isin(split["baseline_ids"])]
    current = predictions[predictions["patient_id"].isin(split["current_ids"])]
    base_mean = float(baseline[probability_column].mean())
    cur_mean = float(current[probability_column].mean())
    base_std = float(baseline[probability_column].std() or 1.0)
    shift = abs(cur_mean - base_mean) / max(base_std, 1e-6)

    return {
        "status": _shift_status(shift),
        "probability_column": probability_column,
        "baseline_mean": round(base_mean, 3),
        "current_mean": round(cur_mean, 3),
        "standardized_shift": round(shift, 3),
    }


def _calibration_drift(predictions: pd.DataFrame | None, metrics: dict, split: dict) -> dict:
    if predictions is None or predictions.empty:
        return {"status": "unavailable", "message": "Prediction artifact not available."}
    probability_column = _best_probability_column(predictions, metrics)
    if probability_column not in predictions.columns or "actual_label" not in predictions.columns:
        return {"status": "unavailable", "message": "Missing calibration columns."}

    baseline = predictions[predictions["patient_id"].isin(split["baseline_ids"])]
    current = predictions[predictions["patient_id"].isin(split["current_ids"])]
    base_ece = _ece(baseline["actual_label"], baseline[probability_column])
    cur_ece = _ece(current["actual_label"], current[probability_column])
    delta = None if base_ece is None or cur_ece is None else round(cur_ece - base_ece, 3)
    return {
        "status": _shift_status(abs(delta or 0.0) / 0.08 if delta is not None else 0.0),
        "probability_column": probability_column,
        "baseline_ece": base_ece,
        "current_ece": cur_ece,
        "delta_ece": delta,
    }


def _subgroup_drift(predictions: pd.DataFrame | None, training_rows: pd.DataFrame, split: dict) -> dict:
    if predictions is None or predictions.empty:
        return {"status": "unavailable", "groups": []}

    probability_column = _best_probability_column(predictions, {})
    if probability_column not in predictions.columns:
        return {"status": "unavailable", "groups": []}

    context = training_rows.sort_values(["patient_id", "cycle"]).groupby("patient_id", as_index=False).agg(
        stage=("stage", "first"),
        molecular_subtype=("molecular_subtype", "first"),
    )
    merged = predictions.merge(context, on="patient_id", how="left")

    groups = []
    for group_field in ["stage", "molecular_subtype"]:
        if group_field not in merged.columns:
            continue
        for group_value, frame in merged.groupby(group_field):
            if frame.empty:
                continue
            baseline = frame[frame["patient_id"].isin(split["baseline_ids"])]
            current = frame[frame["patient_id"].isin(split["current_ids"])]
            if baseline.empty or current.empty:
                continue
            base_rate = float((baseline[probability_column] >= 0.5).mean())
            cur_rate = float((current[probability_column] >= 0.5).mean())
            shift = abs(cur_rate - base_rate)
            groups.append({
                "group": group_field,
                "value": str(group_value),
                "baseline_positive_rate": round(base_rate, 3),
                "current_positive_rate": round(cur_rate, 3),
                "shift": round(shift, 3),
                "status": _shift_status(shift / 0.2 if 0.2 else shift),
            })

    return {
        "status": _worst_status(row["status"] for row in groups),
        "groups": sorted(groups, key=lambda row: row["shift"], reverse=True)[:12],
    }


def _data_completeness_score(training_rows: pd.DataFrame, features: list[str]) -> float | None:
    if training_rows is None or training_rows.empty:
        return None
    missing_rates = [_missing_rate(training_rows, [feature]) for feature in features if feature in training_rows.columns]
    if not missing_rates:
        return None
    return round(max(0.0, 1 - mean(missing_rates)), 3)


def _missing_rate(frame: pd.DataFrame, columns: list[str]) -> float:
    if frame is None or frame.empty:
        return 0.0
    rates = []
    for column in columns:
        if column in frame.columns:
            rates.append(float(frame[column].isna().mean()))
    return round(mean(rates), 3) if rates else 0.0


def _best_probability_column(predictions: pd.DataFrame, metrics: dict) -> str:
    best = (metrics or {}).get("best_model_by_patient_level_roc_auc")
    if best:
        calibrated = f"{best}_calibrated_probability"
        if calibrated in predictions.columns:
            return calibrated
        raw = f"{best}_probability"
        if raw in predictions.columns:
            return raw
    for candidate in [
        "gradient_boosting_calibrated_probability",
        "gradient_boosting_probability",
        "random_forest_probability",
        "logistic_regression_probability",
    ]:
        if candidate in predictions.columns:
            return candidate
    return predictions.columns[2] if len(predictions.columns) > 2 else predictions.columns[-1]


def _ece(labels: pd.Series, probabilities: pd.Series, bins: int = 10) -> float | None:
    if labels is None or probabilities is None:
        return None
    labels = labels.dropna().astype(int)
    probabilities = probabilities.dropna().astype(float)
    if labels.empty or probabilities.empty:
        return None
    aligned = pd.DataFrame({"label": labels, "prob": probabilities}).dropna()
    if aligned.empty:
        return None
    total = len(aligned)
    expected_calibration_error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        if index == bins - 1:
            mask = (aligned["prob"] >= lower) & (aligned["prob"] <= upper)
        else:
            mask = (aligned["prob"] >= lower) & (aligned["prob"] < upper)
        count = int(mask.sum())
        if count:
            mean_probability = float(aligned.loc[mask, "prob"].mean())
            observed_rate = float(aligned.loc[mask, "label"].mean())
            gap = abs(observed_rate - mean_probability)
            expected_calibration_error += (count / total) * gap
    return round(expected_calibration_error, 3)


def _keyword_rate(frame: pd.DataFrame, keyword: str) -> float:
    if frame is None or frame.empty:
        return 0.0
    text = frame["response_text"].astype(str).str.lower()
    return float(text.str.contains(keyword, regex=False, na=False).mean())


def _shift_status(standardized_shift: float) -> str:
    if standardized_shift >= 2.0:
        return "failed"
    if standardized_shift >= 1.2:
        return "unideal"
    if standardized_shift >= 0.6:
        return "watch"
    return "passed"


def _worst_status(statuses) -> str:
    order = {"passed": 0, "watch": 1, "unideal": 2, "failed": 3}
    worst = "passed"
    for status in statuses:
        if order.get(status, 3) > order.get(worst, 0):
            worst = status
    return worst


def _is_synthetic(training_rows: pd.DataFrame) -> bool:
    if "patient_id" not in training_rows.columns:
        return True
    return any(str(pid).startswith("COMP") or str(pid).startswith("SYN") for pid in training_rows["patient_id"].unique())


def _load_csv(path: str) -> pd.DataFrame | None:
    csv_path = Path(path)
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def _load_json(path: str) -> dict:
    json_path = Path(path)
    if not json_path.exists():
        return {}
    return json.loads(json_path.read_text(encoding="utf-8"))


def _unavailable_payload(message: str) -> dict:
    return {
        "schema_version": "drift_report_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "unavailable",
        "message": message,
    }


def _write_json(path: str | None, payload: dict) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
