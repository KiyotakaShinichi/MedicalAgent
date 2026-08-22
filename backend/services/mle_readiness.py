import json
from datetime import datetime, timezone
from pathlib import Path

from backend.services.agent_regression_eval import (
    DEFAULT_AGENT_REGRESSION_PATH,
    load_latest_agent_regression_report,
)
from backend.services.calibration_eval import run_calibration_eval
from backend.services.mle_readiness_checks import (
    NUMERIC_RANGES,
    REQUIRED_TEMPORAL_COLUMNS,
    advisory_gaps as _advisory_gaps,
    agent_quality_checks as _agent_quality_checks,
    artifact_checks as _artifact_checks,
    artifact_hashes as _artifact_hashes,
    category_statuses as _category_statuses,
    check as _check,
    ci_status as _ci_status,
    ci_widths as _ci_widths,
    data_contract_checks as _data_contract_checks,
    feature_store_checks as _feature_store_checks,
    higher_status as _higher_status,
    lifecycle_checks as _lifecycle_checks,
    lineage_leakage_holdout_checks as _lineage_leakage_holdout_checks,
    load_csv as _load_csv,
    load_json as _load_json,
    load_latest_evaluation_report as _load_latest_evaluation_report,
    lower_status as _lower_status,
    metric_check as _metric_check,
    model_artifact_path as _model_artifact_path,
    next_actions as _next_actions,
    operating_policy_false_negative_rate as _operating_policy_false_negative_rate,
    overall_status as _overall_status,
    performance_checks as _performance_checks,
    poc_demo_readiness as _poc_demo_readiness,
    range_violations as _range_violations,
    realism_checks as _realism_checks,
    release_recommendation as _release_recommendation,
    robustness_checks as _robustness_checks,
    rounded as _round,
    same_path as _same_path,
    sha256 as _sha256,
    worst_status as _worst_status,
)
from backend.services.mle_readiness_statistics import (
    hybrid_weight_ablation as _hybrid_weight_ablation,
    temporal_generalization_eval as _temporal_generalization_eval,
)
from backend.services.noise_eval import run_noise_eval
from backend.services.synthetic_realism_report import (
    DEFAULT_OUTPUT_PATH as DEFAULT_REALISM_REPORT_PATH,
    build_synthetic_realism_report,
)
from backend.services.temporal_eval import run_temporal_eval


__all__ = [
    "DEFAULT_AGENT_REGRESSION_PATH",
    "DEFAULT_EVALUATION_MANIFEST_PATH",
    "DEFAULT_LEAKAGE_AUDIT_PATH",
    "DEFAULT_LINEAGE_PATH",
    "DEFAULT_LOCKED_HOLDOUT_PATH",
    "DEFAULT_METRICS_PATH",
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_PREDICTIONS_PATH",
    "DEFAULT_REALISM_REPORT_PATH",
    "DEFAULT_TRAINING_CSV",
    "NUMERIC_RANGES",
    "REQUIRED_TEMPORAL_COLUMNS",
    "_advisory_gaps",
    "_agent_quality_checks",
    "_artifact_checks",
    "_artifact_hashes",
    "_category_statuses",
    "_check",
    "_ci_status",
    "_ci_widths",
    "_data_contract_checks",
    "_feature_store_checks",
    "_higher_status",
    "_hybrid_weight_ablation",
    "_lifecycle_checks",
    "_lineage_leakage_holdout_checks",
    "_load_csv",
    "_load_json",
    "_load_latest_evaluation_report",
    "_lower_status",
    "_metric_check",
    "_model_artifact_path",
    "_next_actions",
    "_operating_policy_false_negative_rate",
    "_overall_status",
    "_performance_checks",
    "_poc_demo_readiness",
    "_range_violations",
    "_realism_checks",
    "_release_recommendation",
    "_robustness_checks",
    "_round",
    "_same_path",
    "_sha256",
    "_temporal_generalization_eval",
    "_worst_status",
    "build_mle_readiness_summary",
    "load_latest_mle_readiness",
]


DEFAULT_TRAINING_CSV = "Data/complete_synthetic_training/locked_holdout/development_rows.csv"
if Path("Data/complete_synthetic_breast_journeys_realism_v2/temporal_ml_rows.csv").exists():
    DEFAULT_TRAINING_CSV = "Data/complete_synthetic_breast_journeys_realism_v2/temporal_ml_rows.csv"
DEFAULT_METRICS_PATH = (
    "Data/complete_synthetic_training_realism_v2/complete_synthetic_model_metrics.json"
    if Path(
        "Data/complete_synthetic_training_realism_v2/complete_synthetic_model_metrics.json"
    ).exists()
    else "Data/complete_synthetic_training/complete_synthetic_model_metrics.json"
)
DEFAULT_PREDICTIONS_PATH = (
    "Data/complete_synthetic_training_realism_v2/complete_synthetic_model_predictions.csv"
    if Path(
        "Data/complete_synthetic_training_realism_v2/complete_synthetic_model_predictions.csv"
    ).exists()
    else "Data/complete_synthetic_training/complete_synthetic_model_predictions.csv"
)
DEFAULT_EVALUATION_MANIFEST_PATH = (
    "Data/model_evaluation_reports_realism_v2/latest_manifest.json"
    if Path("Data/model_evaluation_reports_realism_v2/latest_manifest.json").exists()
    else "Data/model_evaluation_reports/latest_manifest.json"
)
DEFAULT_OUTPUT_PATH = "Data/mle_monitoring/latest_mle_readiness.json"
DEFAULT_LINEAGE_PATH = (
    "Data/complete_synthetic_training_realism_v2/dataset_lineage.json"
    if Path("Data/complete_synthetic_training_realism_v2/dataset_lineage.json").exists()
    else "Data/lineage/complete_synthetic_lineage.json"
)
DEFAULT_LEAKAGE_AUDIT_PATH = (
    "Data/complete_synthetic_training_realism_v2/leakage_audit/temporal_leakage_audit.json"
    if Path(
        "Data/complete_synthetic_training_realism_v2/leakage_audit/temporal_leakage_audit.json"
    ).exists()
    else "Data/complete_synthetic_training/leakage_audit/temporal_leakage_audit.json"
)
DEFAULT_LOCKED_HOLDOUT_PATH = (
    "Data/complete_synthetic_training_realism_v2/locked_holdout/locked_holdout_manifest.json"
    if Path(
        "Data/complete_synthetic_training_realism_v2/locked_holdout/locked_holdout_manifest.json"
    ).exists()
    else "Data/complete_synthetic_training/locked_holdout/locked_holdout_manifest.json"
)


def build_mle_readiness_summary(
    db=None,
    training_csv=DEFAULT_TRAINING_CSV,
    metrics_path=DEFAULT_METRICS_PATH,
    predictions_path=DEFAULT_PREDICTIONS_PATH,
    evaluation_manifest_path=DEFAULT_EVALUATION_MANIFEST_PATH,
    agent_regression_path=DEFAULT_AGENT_REGRESSION_PATH,
    output_path=None,
    lineage_path=DEFAULT_LINEAGE_PATH,
    leakage_audit_path=DEFAULT_LEAKAGE_AUDIT_PATH,
    locked_holdout_path=DEFAULT_LOCKED_HOLDOUT_PATH,
):
    training_rows = _load_csv(training_csv)
    metrics = _load_json(metrics_path)
    predictions = _load_csv(predictions_path)
    evaluation_report = _load_latest_evaluation_report(evaluation_manifest_path)
    agent_regression = load_latest_agent_regression_report(agent_regression_path)
    lineage = _load_json(lineage_path)
    leakage_audit = _load_json(leakage_audit_path)
    locked_holdout = _load_json(locked_holdout_path)

    checks = []
    checks.extend(
        _artifact_checks(metrics, metrics_path, predictions_path, training_csv, evaluation_report)
    )
    checks.extend(_data_contract_checks(training_rows))
    checks.extend(_feature_store_checks(training_csv))
    checks.extend(_lineage_leakage_holdout_checks(lineage, leakage_audit, locked_holdout))
    checks.extend(_performance_checks(metrics, evaluation_report))
    checks.extend(_lifecycle_checks(db, metrics))
    checks.extend(_agent_quality_checks(agent_regression))

    ablation = _hybrid_weight_ablation(predictions)
    temporal_gen = _temporal_generalization_eval(training_rows, predictions)

    temporal_eval_report = run_temporal_eval()
    noise_eval_report = run_noise_eval()
    calibration_report = run_calibration_eval()
    realism_report = build_synthetic_realism_report(
        training_csv=training_csv,
        output_path=DEFAULT_REALISM_REPORT_PATH,
    )

    checks.extend(_robustness_checks(temporal_eval_report, noise_eval_report, calibration_report))
    checks.extend(_realism_checks(realism_report))

    category_statuses = _category_statuses(checks)
    hard_failures = [
        check for check in checks if check["hard_gate"] and check["status"] == "failed"
    ]
    summary = {
        "schema_version": "mle_readiness_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Model lifecycle readiness gates for the breast cancer monitoring PoC: data contract, "
            "artifact availability, model quality, calibration, safety regression, registry, and audit readiness."
        ),
        "status": _overall_status(checks),
        "release_recommendation": _release_recommendation(checks),
        "hard_gate_status": "failed" if hard_failures else "passed",
        "hard_gate_failures": hard_failures,
        "poc_demo_readiness": _poc_demo_readiness(checks, category_statuses, hard_failures),
        "category_statuses": category_statuses,
        "checks": checks,
        "artifact_hashes": _artifact_hashes(
            [training_csv, metrics_path, predictions_path, evaluation_manifest_path]
        ),
        "next_actions": _next_actions(checks),
        "hybrid_weight_ablation": ablation,
        "temporal_generalization_eval": temporal_gen,
        "temporal_eval_report": temporal_eval_report,
        "noise_eval_report": noise_eval_report,
        "calibration_eval_report": calibration_report,
        "synthetic_realism_report": realism_report,
        "claim_boundary": (
            "These gates make the engineering workflow more production-like. They do not convert synthetic-data "
            "results into clinical validation."
        ),
    }

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_latest_mle_readiness(path=DEFAULT_OUTPUT_PATH):
    report_path = Path(path)
    if not report_path.exists():
        return {
            "status": "unavailable",
            "message": "No MLE readiness report has been generated yet.",
            "path": str(report_path),
        }
    return json.loads(report_path.read_text(encoding="utf-8"))
