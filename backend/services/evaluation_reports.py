import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backend.database import SessionLocal
from backend.services.admin_analytics import build_admin_analytics


DEFAULT_EVALUATION_REPORT_DIR = "Data/model_evaluation_reports"


def generate_versioned_evaluation_report(
    db=None,
    output_root: str = DEFAULT_EVALUATION_REPORT_DIR,
    run_id: str | None = None,
):
    owns_session = db is None
    if db is None:
        db = SessionLocal()
    try:
        analytics = build_admin_analytics(db)
    finally:
        if owns_session:
            db.close()

    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    advanced = analytics.get("advanced_model_evaluation") or {}
    guide = analytics.get("metric_interpretation_guide") or {}
    files = {}

    files["evaluation_report_json"] = _write_json(run_dir / "evaluation_report.json", analytics)
    files["metric_definitions_csv"] = _write_csv(
        run_dir / "metric_definitions.csv",
        guide.get("advanced_metric_definitions") or [],
    )
    files["calibration_bins_csv"] = _write_csv(
        run_dir / "calibration_bins.csv",
        (advanced.get("calibration") or {}).get("bins") or [],
    )
    files["threshold_operating_points_csv"] = _write_csv(
        run_dir / "threshold_operating_points.csv",
        (advanced.get("threshold_operating_points") or {}).get("rows") or [],
    )
    files["cost_sensitive_thresholds_csv"] = _write_csv(
        run_dir / "cost_sensitive_thresholds.csv",
        (advanced.get("cost_sensitive_thresholds") or {}).get("policies") or [],
    )
    files["false_negative_cases_csv"] = _write_csv(
        run_dir / "false_negative_cases.csv",
        (advanced.get("false_negative_review") or {}).get("cases") or [],
    )
    files["subgroup_metrics_csv"] = _write_csv(
        run_dir / "subgroup_metrics.csv",
        (advanced.get("subgroup_performance") or {}).get("rows") or [],
    )
    files["decision_impact_categories_csv"] = _write_csv(
        run_dir / "decision_impact_categories.csv",
        (advanced.get("decision_impact_simulation") or {}).get("categories") or [],
    )
    files["decision_impact_examples_csv"] = _write_csv(
        run_dir / "decision_impact_examples.csv",
        (advanced.get("decision_impact_simulation") or {}).get("examples") or [],
    )
    files["data_coverage_csv"] = _write_csv(
        run_dir / "data_coverage.csv",
        (analytics.get("data_coverage") or {}).get("items") or [],
    )

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(run_dir),
        "advanced_evaluation_status": advanced.get("status"),
        "champion_model": advanced.get("champion_model"),
        "files": files,
        "safety_note": "Evaluation reports are engineering artifacts. They are not clinical validation.",
    }
    files["manifest_json"] = _write_json(run_dir / "manifest.json", manifest)
    latest_path = Path(output_root) / "latest_manifest.json"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["files"] = files
    manifest["latest_manifest_path"] = str(latest_path)
    return manifest


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return str(path)


def _write_csv(path, rows):
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return str(path)

