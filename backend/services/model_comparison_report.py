import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


DEFAULT_EVALUATION_ROOT = "Data/model_evaluation_reports"
DEFAULT_OUTPUT_DIR = "Data/model_comparison"


def build_model_comparison_report(
    evaluation_root=DEFAULT_EVALUATION_ROOT,
    output_dir=DEFAULT_OUTPUT_DIR,
    limit=6,
):
    reports = _load_reports(evaluation_root)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rows = [_report_row(report) for report in reports[-limit:]]
    deltas = _latest_delta(rows)
    report = {
        "schema_version": "model_comparison_report_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "available" if rows else "unavailable",
        "runs_compared": len(rows),
        "rows": rows,
        "latest_delta": deltas,
        "interpretation": (
            "Compares local synthetic evaluation reports across runs. Improvements here mean stronger engineering "
            "evidence on the simulator, not clinical validation."
        ),
        "claim_boundary": "Use this to discuss model iteration discipline, not real-world medical efficacy.",
    }
    files = {
        "model_comparison_json": str(output_path / "model_comparison_report.json"),
        "model_comparison_csv": str(output_path / "model_comparison_rows.csv"),
    }
    Path(files["model_comparison_json"]).write_text(json.dumps({**report, "files": files}, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(files["model_comparison_csv"], index=False)
    return {**report, "files": files}


def _load_reports(evaluation_root):
    root = Path(evaluation_root)
    reports = []
    for report_path in sorted(root.glob("*/evaluation_report.json")):
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            payload["_run_id"] = report_path.parent.name
            reports.append(payload)
        except (OSError, json.JSONDecodeError):
            continue
    return reports


def _report_row(report):
    advanced = report.get("advanced_model_evaluation") or {}
    performance = ((report.get("model_performance") or {}).get("synthetic_longitudinal_response") or {})
    calibration = advanced.get("calibration") or {}
    cost = advanced.get("cost_sensitive_thresholds") or {}
    subgroup = advanced.get("subgroup_performance") or {}
    false_negative = advanced.get("false_negative_review") or {}
    best_policy = (cost.get("policies") or [{}])[0]
    return {
        "run_id": report.get("_run_id"),
        "status": advanced.get("status"),
        "champion_model": advanced.get("champion_model") or performance.get("best_model"),
        "probability_source": advanced.get("probability_source"),
        "evaluated_patients": advanced.get("evaluated_patients"),
        "patient_level_roc_auc": performance.get("patient_level_roc_auc"),
        "patient_level_average_precision": performance.get("patient_level_average_precision"),
        "patient_level_brier_score": performance.get("patient_level_brier_score") or calibration.get("brier_score"),
        "expected_calibration_error": calibration.get("expected_calibration_error"),
        "subgroup_status": subgroup.get("status"),
        "low_support_group_count": subgroup.get("low_support_group_count"),
        "false_negative_count": false_negative.get("count", false_negative.get("case_count")),
        "cost_policy_threshold": best_policy.get("recommended_threshold"),
        "cost_policy_weighted_error": best_policy.get("weighted_error"),
    }


def _latest_delta(rows):
    if len(rows) < 2:
        return {}
    previous = rows[-2]
    current = rows[-1]
    metrics = [
        "patient_level_roc_auc",
        "patient_level_average_precision",
        "patient_level_brier_score",
        "expected_calibration_error",
        "false_negative_count",
        "cost_policy_weighted_error",
    ]
    output = {
        "previous_run_id": previous.get("run_id"),
        "current_run_id": current.get("run_id"),
    }
    for metric in metrics:
        output[f"{metric}_delta"] = _delta(current.get(metric), previous.get(metric))
    return output


def _delta(current, previous):
    if current is None or previous is None:
        return None
    try:
        return round(float(current) - float(previous), 4)
    except (TypeError, ValueError):
        return None
