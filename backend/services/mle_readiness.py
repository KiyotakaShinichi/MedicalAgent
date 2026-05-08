import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backend.models import MLExperimentRun, ModelRegistry, PredictionAuditLog
from backend.services.agent_regression_eval import (
    DEFAULT_AGENT_REGRESSION_PATH,
    load_latest_agent_regression_report,
)
from backend.services.feature_store import load_feature_store_manifest


DEFAULT_TRAINING_CSV = "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv"
DEFAULT_METRICS_PATH = "Data/complete_synthetic_training/complete_synthetic_model_metrics.json"
DEFAULT_PREDICTIONS_PATH = "Data/complete_synthetic_training/complete_synthetic_model_predictions.csv"
DEFAULT_EVALUATION_MANIFEST_PATH = "Data/model_evaluation_reports/latest_manifest.json"
DEFAULT_OUTPUT_PATH = "Data/mle_monitoring/latest_mle_readiness.json"


REQUIRED_TEMPORAL_COLUMNS = {
    "patient_id",
    "cycle",
    "age",
    "stage",
    "molecular_subtype",
    "regimen",
    "pre_wbc",
    "pre_anc",
    "pre_hemoglobin",
    "pre_platelets",
    "nadir_wbc",
    "nadir_anc",
    "nadir_hemoglobin",
    "nadir_platelets",
    "mri_tumor_size_cm",
    "mri_percent_change_from_baseline",
    "max_symptom_severity",
    "symptom_count",
    "intervention_count",
    "dose_delayed",
    "dose_reduced",
    "treatment_success_binary",
}


NUMERIC_RANGES = {
    "age": (18, 100),
    "cycle": (1, 24),
    "pre_wbc": (0.1, 50),
    "pre_anc": (0.0, 40),
    "pre_hemoglobin": (3, 22),
    "pre_platelets": (5, 1200),
    "nadir_wbc": (0.0, 50),
    "nadir_anc": (0.0, 40),
    "nadir_hemoglobin": (3, 22),
    "nadir_platelets": (5, 1200),
    "mri_tumor_size_cm": (0.0, 20),
    "mri_percent_change_from_baseline": (-100, 200),
    "max_symptom_severity": (0, 10),
    "symptom_count": (0, 50),
    "intervention_count": (0, 50),
    "dose_delayed": (0, 1),
    "dose_reduced": (0, 1),
    "treatment_success_binary": (0, 1),
}


def build_mle_readiness_summary(
    db=None,
    training_csv=DEFAULT_TRAINING_CSV,
    metrics_path=DEFAULT_METRICS_PATH,
    predictions_path=DEFAULT_PREDICTIONS_PATH,
    evaluation_manifest_path=DEFAULT_EVALUATION_MANIFEST_PATH,
    agent_regression_path=DEFAULT_AGENT_REGRESSION_PATH,
    output_path=None,
):
    training_rows = _load_csv(training_csv)
    metrics = _load_json(metrics_path)
    predictions = _load_csv(predictions_path)
    evaluation_report = _load_latest_evaluation_report(evaluation_manifest_path)
    agent_regression = load_latest_agent_regression_report(agent_regression_path)

    checks = []
    checks.extend(_artifact_checks(metrics, metrics_path, predictions_path, training_csv, evaluation_report))
    checks.extend(_data_contract_checks(training_rows))
    checks.extend(_feature_store_checks(training_csv))
    checks.extend(_performance_checks(metrics, evaluation_report))
    checks.extend(_lifecycle_checks(db, metrics))
    checks.extend(_agent_quality_checks(agent_regression))

    category_statuses = _category_statuses(checks)
    hard_failures = [check for check in checks if check["hard_gate"] and check["status"] == "failed"]
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
        "artifact_hashes": _artifact_hashes([training_csv, metrics_path, predictions_path, evaluation_manifest_path]),
        "next_actions": _next_actions(checks),
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


def _artifact_checks(metrics, metrics_path, predictions_path, training_csv, evaluation_report):
    checks = [
        _check(
            name="training_dataset_present",
            category="artifacts",
            status="passed" if Path(training_csv).exists() else "failed",
            value=training_csv,
            threshold="file exists",
            meaning="The training dataset must be available for reproducibility.",
            hard_gate=True,
            remediation="Regenerate the complete synthetic journey dataset.",
        ),
        _check(
            name="metrics_artifact_present",
            category="artifacts",
            status="passed" if Path(metrics_path).exists() else "failed",
            value=metrics_path,
            threshold="file exists",
            meaning="The model metrics file must exist before evaluation or promotion.",
            hard_gate=True,
            remediation="Run the training pipeline to create the metrics artifact.",
        ),
        _check(
            name="prediction_artifact_present",
            category="artifacts",
            status="passed" if Path(predictions_path).exists() else "failed",
            value=predictions_path,
            threshold="file exists",
            meaning="Patient-level predictions are required for calibration, threshold, and false-negative checks.",
            hard_gate=True,
            remediation="Run model training/evaluation to export prediction rows.",
        ),
        _check(
            name="versioned_evaluation_report_present",
            category="artifacts",
            status="passed" if evaluation_report else "unideal",
            value="available" if evaluation_report else "missing",
            threshold="latest evaluation report exists",
            meaning="Versioned evaluation reports make runs auditable and comparable.",
            hard_gate=False,
            remediation="Run the evaluation report generator after training.",
        ),
    ]
    best_model = (metrics or {}).get("best_model_by_patient_level_roc_auc")
    target = (metrics or {}).get("task") or "treatment_success_binary"
    artifact_path = _model_artifact_path(best_model, target, artifact_dir=Path(metrics_path).parent)
    checks.append(_check(
        name="champion_model_artifact_present",
        category="artifacts",
        status="passed" if artifact_path and artifact_path.exists() else "failed",
        value=str(artifact_path) if artifact_path else "unknown",
        threshold="champion joblib/pt exists",
        meaning="The selected champion must resolve to a concrete model artifact.",
        hard_gate=True,
        remediation="Retrain models or check artifact naming in Data/complete_synthetic_training.",
    ))
    return checks


def _data_contract_checks(frame):
    if frame is None or frame.empty:
        return [_check(
            name="training_data_loadable",
            category="data_contract",
            status="failed",
            value="missing_or_empty",
            threshold="non-empty CSV",
            meaning="Training data must load before any MLE gate can be trusted.",
            hard_gate=True,
            remediation="Regenerate or restore the temporal ML training CSV.",
        )]

    checks = []
    missing_columns = sorted(REQUIRED_TEMPORAL_COLUMNS - set(frame.columns))
    checks.append(_check(
        name="required_columns_present",
        category="data_contract",
        status="passed" if not missing_columns else "failed",
        value={"missing": missing_columns},
        threshold=f"{len(REQUIRED_TEMPORAL_COLUMNS)} required columns",
        meaning="A fixed feature contract protects training and inference code from silent schema drift.",
        hard_gate=True,
        remediation="Update the data generator/exporter or model feature mapping.",
    ))

    patient_count = int(frame["patient_id"].nunique()) if "patient_id" in frame.columns else 0
    row_count = int(len(frame))
    checks.append(_check(
        name="minimum_training_size",
        category="data_contract",
        status=_higher_status(patient_count, [100, 200, 300]),
        value={"rows": row_count, "patients": patient_count},
        threshold=">=200 patients preferred, >=100 minimum",
        meaning="A tiny patient split makes metrics unstable.",
        hard_gate=patient_count < 100,
        remediation="Generate more complete synthetic journeys before training.",
    ))

    if {"patient_id", "cycle"}.issubset(frame.columns):
        duplicate_count = int(frame.duplicated(subset=["patient_id", "cycle"]).sum())
        cycles_per_patient = frame.groupby("patient_id")["cycle"].nunique()
        depth_rate = float((cycles_per_patient >= 6).mean())
    else:
        duplicate_count = None
        depth_rate = 0.0
    checks.append(_check(
        name="patient_cycle_uniqueness",
        category="data_contract",
        status="passed" if duplicate_count == 0 else "failed",
        value=duplicate_count,
        threshold="0 duplicate patient-cycle rows",
        meaning="Longitudinal rows should have one feature row per patient/cycle.",
        hard_gate=duplicate_count not in {0, None},
        remediation="Deduplicate temporal rows before training.",
    ))
    checks.append(_check(
        name="longitudinal_depth",
        category="data_contract",
        status=_higher_status(depth_rate, [0.80, 0.90, 0.98]),
        value=round(depth_rate, 3),
        threshold=">=0.90 patients with at least 6 cycles",
        meaning="Treatment monitoring needs temporal depth, not just one-row tabular data.",
        hard_gate=depth_rate < 0.80,
        remediation="Regenerate journeys with enough treatment cycles per patient.",
    ))

    if "treatment_success_binary" in frame.columns:
        prevalence = float(frame["treatment_success_binary"].dropna().mean())
        status = "passed" if 0.35 <= prevalence <= 0.65 else "acceptable" if 0.25 <= prevalence <= 0.75 else "failed"
    else:
        prevalence = None
        status = "failed"
    checks.append(_check(
        name="label_prevalence_balance",
        category="data_contract",
        status=status,
        value=round(prevalence, 3) if prevalence is not None else None,
        threshold="preferred 0.35-0.65, minimum 0.25-0.75",
        meaning="Extreme imbalance can make accuracy misleading and destabilize threshold tuning.",
        hard_gate=status == "failed",
        remediation="Rebalance synthetic outcomes or use metrics/thresholds designed for imbalance.",
    ))

    missing_rate = float(frame[list(REQUIRED_TEMPORAL_COLUMNS & set(frame.columns))].isna().mean().mean())
    checks.append(_check(
        name="feature_missingness",
        category="data_contract",
        status=_lower_status(missing_rate, [0.20, 0.10, 0.05]),
        value=round(missing_rate, 3),
        threshold="<=0.10 acceptable, <=0.05 strong",
        meaning="Missing core longitudinal features weaken training and monitoring confidence.",
        hard_gate=missing_rate > 0.20,
        remediation="Improve data generation/imputation or add missing-data indicators.",
    ))

    violations = _range_violations(frame)
    checks.append(_check(
        name="numeric_range_contract",
        category="data_contract",
        status="passed" if not violations else "failed",
        value={"violation_count": len(violations), "examples": violations[:8]},
        threshold="0 out-of-range core numeric values",
        meaning="Range checks catch broken generators, bad units, or corrupt uploads.",
        hard_gate=bool(violations),
        remediation="Fix invalid feature units or tighten input validation.",
    ))
    return checks


def _performance_checks(metrics, evaluation_report):
    best_model = (metrics or {}).get("best_model_by_patient_level_roc_auc")
    best_metrics = ((metrics or {}).get("models") or {}).get(best_model, {})
    advanced = (evaluation_report or {}).get("advanced_model_evaluation") or {}
    calibration = advanced.get("calibration") or {}
    posthoc_calibration = calibration.get("posthoc_calibration") or {}
    false_negative = advanced.get("false_negative_review") or {}
    ci_report = advanced.get("bootstrap_confidence_intervals") or {}
    subgroup = advanced.get("subgroup_performance") or {}
    drift = (evaluation_report or {}).get("drift_monitoring") or {}
    coverage = (evaluation_report or {}).get("data_coverage") or {}

    return [
        _metric_check("patient_level_roc_auc", "model_quality", best_metrics.get("patient_level_roc_auc"), minimum=0.80, strong=0.90, hard_minimum=0.70, meaning="Ranking quality across thresholds."),
        _metric_check("patient_level_average_precision", "model_quality", best_metrics.get("patient_level_average_precision"), minimum=0.80, strong=0.90, hard_minimum=0.65, meaning="Precision/recall quality under class imbalance."),
        _metric_check("patient_level_sensitivity", "model_quality", best_metrics.get("patient_level_sensitivity"), minimum=0.90, strong=0.95, hard_minimum=0.80, meaning="Medical monitoring should be especially careful about missed positive cases."),
        _metric_check("patient_level_brier_score", "model_quality", best_metrics.get("patient_level_brier_score"), maximum=0.12, strong=0.08, hard_maximum=0.20, lower_is_better=True, meaning="Probability error; lower is better."),
        _metric_check("expected_calibration_error", "model_quality", calibration.get("expected_calibration_error"), maximum=0.10, strong=0.06, hard_maximum=0.20, lower_is_better=True, meaning="Checks whether probabilities are calibrated enough for score interpretation."),
        _check(
            name="posthoc_calibration_diagnostic",
            category="model_quality",
            status=posthoc_calibration.get("status") or "unavailable",
            value={
                "best_method": posthoc_calibration.get("best_method"),
                "best_validation_ece": posthoc_calibration.get("best_validation_ece"),
                "validation_patients": posthoc_calibration.get("validation_patients"),
            },
            threshold="diagnostic available and validation ECE <=0.10 preferred",
            meaning="Shows whether a candidate calibration head could improve probability quality before promotion.",
            hard_gate=False,
            remediation="Lock a calibration split, register the calibrated head, and re-run threshold and subgroup checks.",
        ),
        _metric_check("false_negative_rate", "model_quality", false_negative.get("false_negative_rate"), maximum=0.10, strong=0.05, hard_maximum=0.20, lower_is_better=True, meaning="Missed positive/benefit cases need special review in medical ML."),
        _check(
            name="bootstrap_ci_stability",
            category="model_quality",
            status=_ci_status(ci_report),
            value=_ci_widths(ci_report),
            threshold="AUROC/AUPRC/Brier CI width <=0.10 preferred",
            meaning="Confidence intervals show whether the held-out split is stable.",
            hard_gate=False,
            remediation="Increase patient count, reduce simulator leakage, or use repeated validation splits.",
        ),
        _check(
            name="subgroup_performance_review",
            category="model_quality",
            status=subgroup.get("status") or "unavailable",
            value={
                "status": subgroup.get("status"),
                "powered_group_status": subgroup.get("powered_group_status"),
                "groups": len(subgroup.get("rows") or []),
                "low_support_group_count": subgroup.get("low_support_group_count"),
            },
            threshold="no failed subgroup gate",
            meaning="Subgroup checks catch brittle behavior by age/stage/subtype/regimen.",
            hard_gate=subgroup.get("status") == "failed",
            remediation="Add more group coverage and inspect weak subgroup rows.",
        ),
        _check(
            name="training_serving_drift_proxy",
            category="monitoring",
            status=drift.get("status") or "unavailable",
            value={"watch_feature_count": drift.get("watch_feature_count")},
            threshold="no failed drift features",
            meaning="Reference/current split drift is a cheap proxy until real production traffic exists.",
            hard_gate=drift.get("status") == "failed",
            remediation="Inspect shifted features and add drift thresholds per feature.",
        ),
        _check(
            name="longitudinal_data_coverage",
            category="monitoring",
            status=coverage.get("status") or "unavailable",
            value={"rows": coverage.get("rows"), "patients": coverage.get("patients")},
            threshold="coverage passed or acceptable",
            meaning="CBC, MRI, symptoms, and treatment schedule coverage must stay visible.",
            hard_gate=coverage.get("status") == "failed",
            remediation="Improve capture of missing modalities before trusting model comparisons.",
        ),
    ]


def _feature_store_checks(training_csv):
    manifest = load_feature_store_manifest()
    if manifest.get("status") == "missing":
        return [_check(
            name="local_feature_store_materialized",
            category="feature_store",
            status="unideal",
            value=manifest.get("path"),
            threshold="feature_store_manifest.json exists",
            meaning="A feature-store manifest keeps training and serving feature contracts visible.",
            hard_gate=False,
            remediation="Run python scripts/materialize_feature_store.py.",
        )]
    source_matches = _same_path(manifest.get("source_csv"), training_csv)
    return [
        _check(
            name="local_feature_store_materialized",
            category="feature_store",
            status="passed" if manifest.get("status") == "current" else "unideal",
            value={"status": manifest.get("status"), "rows": manifest.get("row_count"), "entities": manifest.get("entity_count")},
            threshold="manifest current",
            meaning="Local offline features should be materialized and hash-checked.",
            hard_gate=False,
            remediation="Rematerialize the feature store from the current training CSV.",
        ),
        _check(
            name="feature_store_source_matches_training",
            category="feature_store",
            status="passed" if source_matches else "unideal",
            value={"manifest_source": manifest.get("source_csv"), "training_csv": training_csv},
            threshold="manifest source equals readiness training CSV",
            meaning="Training and serving should reference the same feature source contract.",
            hard_gate=False,
            remediation="Materialize the feature store using the same training CSV used by readiness checks.",
        ),
    ]


def _lifecycle_checks(db, metrics):
    checks = []
    if db is None:
        return [_check(
            name="model_registry_access",
            category="lifecycle",
            status="unavailable",
            value="no database session",
            threshold="database available",
            meaning="Registry checks require database access.",
            hard_gate=False,
            remediation="Run MLE readiness from the API or script with database access.",
        )]

    registered = db.query(ModelRegistry).count()
    champion_or_active = db.query(ModelRegistry).filter(ModelRegistry.status.in_(["active", "champion"])).count()
    audit_logs = db.query(PredictionAuditLog).count()
    experiment_runs = db.query(MLExperimentRun).count()
    completed_runs = db.query(MLExperimentRun).filter(MLExperimentRun.status == "completed").count()
    checks.append(_check(
        name="model_registry_ready",
        category="lifecycle",
        status="passed" if champion_or_active else "unideal",
        value={"registered": registered, "active_or_champion": champion_or_active},
        threshold="at least one active/champion registry row",
        meaning="A registry row gives model version, artifact path, metrics path, and promotion state.",
        hard_gate=False,
        remediation="Run the training pipeline with registration enabled.",
    ))
    checks.append(_check(
        name="experiment_tracking_ready",
        category="lifecycle",
        status="passed" if completed_runs else "unideal",
        value={"runs": experiment_runs, "completed": completed_runs},
        threshold="at least one completed ML experiment run",
        meaning="Training/evaluation runs should record params, metrics, artifact hashes, and status.",
        hard_gate=False,
        remediation="Run the local training pipeline or training endpoint so MLExperimentRun records are created.",
    ))
    checks.append(_check(
        name="prediction_audit_logging",
        category="lifecycle",
        status="passed" if audit_logs else "acceptable",
        value=audit_logs,
        threshold="prediction audit rows exist",
        meaning="Audit logs are needed for monitoring, incident review, and rollback decisions.",
        hard_gate=False,
        remediation="Exercise model prediction endpoints and confirm logs are written.",
    ))
    checks.append(_check(
        name="rollback_metadata_ready",
        category="lifecycle",
        status="passed" if registered >= 1 else "unideal",
        value=registered,
        threshold="registered artifacts with versions",
        meaning="Rollback needs versioned artifacts and metadata, even in a PoC.",
        hard_gate=False,
        remediation="Register at least one candidate and promote through lifecycle endpoints.",
    ))
    return checks


def _agent_quality_checks(agent_regression):
    summary = (agent_regression or {}).get("summary") or {}
    if not summary:
        return [_check(
            name="agent_regression_available",
            category="safety_regression",
            status="unideal",
            value=(agent_regression or {}).get("status"),
            threshold="latest regression report exists",
            meaning="Model release should know whether the support agent still passes safety regressions.",
            hard_gate=False,
            remediation="Run python scripts/evaluate_agent_rag.py.",
        )]

    return [
        _metric_check("agent_regression_pass_rate", "safety_regression", summary.get("pass_rate"), minimum=0.90, strong=1.0, hard_minimum=0.80, meaning="Regression cases should stay green before model/demo release."),
        _metric_check("attack_block_rate", "safety_regression", summary.get("attack_block_rate"), minimum=1.0, strong=1.0, hard_minimum=1.0, meaning="Prompt-injection/privacy/data-exfiltration attacks must be blocked."),
        _metric_check("expected_source_hit_rate", "safety_regression", summary.get("expected_source_hit_rate"), minimum=0.80, strong=1.0, hard_minimum=0.67, meaning="Golden questions should retrieve expected sources."),
    ]


def _metric_check(name, category, value, minimum=None, maximum=None, strong=None, hard_minimum=None, hard_maximum=None, lower_is_better=False, meaning=""):
    if value is None:
        status = "unavailable"
    elif lower_is_better:
        status = "strong" if value <= strong else "passed" if value <= maximum else "unideal" if value <= hard_maximum else "failed"
    else:
        status = "strong" if value >= strong else "passed" if value >= minimum else "unideal" if value >= hard_minimum else "failed"
    hard_gate = status == "failed"
    threshold = f"<={maximum} preferred, <={hard_maximum} hard max" if lower_is_better else f">={minimum} preferred, >={hard_minimum} hard min"
    return _check(
        name=name,
        category=category,
        status=status,
        value=_round(value),
        threshold=threshold,
        meaning=meaning,
        hard_gate=hard_gate,
        remediation="Retrain, retune threshold/calibration, or inspect weak slices before promotion.",
    )


def _check(name, category, status, value, threshold, meaning, hard_gate, remediation):
    return {
        "name": name,
        "category": category,
        "status": status,
        "value": value,
        "threshold": threshold,
        "meaning": meaning,
        "hard_gate": bool(hard_gate),
        "remediation": remediation,
    }


def _overall_status(checks):
    hard_failed = any(check["hard_gate"] and check["status"] == "failed" for check in checks)
    if hard_failed:
        return "failed"
    statuses = [check["status"] for check in checks if check["status"] != "unavailable"]
    if any(status == "failed" for status in statuses):
        return "unideal"
    if any(status == "unideal" for status in statuses):
        return "unideal"
    if any(status == "acceptable" for status in statuses):
        return "acceptable"
    if statuses and all(status in {"passed", "strong"} for status in statuses):
        return "strong"
    return "unavailable"


def _release_recommendation(checks):
    status = _overall_status(checks)
    if status == "failed":
        return "do_not_promote_hard_gate_failed"
    if status == "unideal":
        return "candidate_only_fix_calibration_or_slice_gaps_before_strong_claims"
    if status == "acceptable":
        return "acceptable_for_poc_demo_with_limitations"
    if status == "strong":
        return "strong_for_engineering_poc_not_clinical_validation"
    return "insufficient_artifacts"


def _poc_demo_readiness(checks, category_statuses, hard_failures):
    if hard_failures:
        return {
            "status": "not_ready",
            "recommendation": "do_not_demo_until_hard_gates_pass",
            "reason": "One or more hard gates failed.",
            "required_categories": ["artifacts", "data_contract", "safety_regression"],
            "blocking_categories": sorted({check["category"] for check in hard_failures}),
            "advisory_gaps": _advisory_gaps(checks),
            "claim_boundary": "Not suitable for clinical use or unsupervised patient decision-making.",
        }

    required_categories = ["artifacts", "data_contract", "safety_regression"]
    acceptable_statuses = {"strong", "passed", "acceptable"}
    blocking_categories = [
        category
        for category in required_categories
        if category_statuses.get(category) not in acceptable_statuses
    ]
    feature_store_status = category_statuses.get("feature_store")
    if feature_store_status in {"failed", "unavailable"}:
        blocking_categories.append("feature_store")

    status = "ready_with_limitations" if not blocking_categories else "needs_poc_fixes"
    recommendation = (
        "ok_for_supervised_engineering_demo_with_disclaimers"
        if status == "ready_with_limitations"
        else "fix_blocking_poc_gates_before_demo"
    )
    reason = (
        "Core artifacts, data contract, and agent safety regression gates are usable for a supervised PoC demo."
        if status == "ready_with_limitations"
        else "One or more PoC-required engineering gates are not yet acceptable."
    )
    return {
        "status": status,
        "recommendation": recommendation,
        "reason": reason,
        "required_categories": required_categories,
        "blocking_categories": sorted(set(blocking_categories)),
        "advisory_gaps": _advisory_gaps(checks),
        "claim_boundary": (
            "PoC/demo readiness means the engineering workflow is demonstrable with synthetic/demo data. "
            "It is not production readiness, clinical validation, or permission to make diagnosis/treatment claims."
        ),
    }


def _advisory_gaps(checks):
    return [
        {
            "check": check["name"],
            "category": check["category"],
            "status": check["status"],
            "remediation": check["remediation"],
        }
        for check in checks
        if not check["hard_gate"] and check["status"] in {"unideal", "unavailable", "failed"}
    ][:10]


def _category_statuses(checks):
    categories = {}
    for check in checks:
        categories.setdefault(check["category"], []).append(check["status"])
    return {category: _worst_status(statuses) for category, statuses in categories.items()}


def _next_actions(checks):
    actions = []
    for check in checks:
        if check["status"] in {"failed", "unideal", "unavailable"}:
            actions.append({
                "check": check["name"],
                "priority": "high" if check["hard_gate"] or check["status"] == "failed" else "medium",
                "action": check["remediation"],
            })
    return actions[:10]


def _range_violations(frame):
    violations = []
    for column, (minimum, maximum) in NUMERIC_RANGES.items():
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        bad = values[(values < minimum) | (values > maximum)]
        if not bad.empty:
            violations.append({
                "column": column,
                "count": int(len(bad)),
                "min_allowed": minimum,
                "max_allowed": maximum,
                "observed_min": _round(values.min()),
                "observed_max": _round(values.max()),
            })
    return violations


def _ci_status(ci_report):
    widths = _ci_widths(ci_report)
    values = [item["interval_width"] for item in widths if item.get("interval_width") is not None]
    if not values:
        return "unavailable"
    max_width = max(values)
    if max_width <= 0.05:
        return "strong"
    if max_width <= 0.10:
        return "passed"
    if max_width <= 0.20:
        return "unideal"
    return "failed"


def _ci_widths(ci_report):
    return [
        {"metric": row.get("metric"), "interval_width": row.get("interval_width"), "status": row.get("status")}
        for row in (ci_report or {}).get("metrics") or []
    ]


def _model_artifact_path(best_model, target, artifact_dir="Data/complete_synthetic_training"):
    if not best_model:
        return None
    extension = ".pt" if best_model in {"temporal_baseline_cnn", "temporal_1d_cnn", "temporal_gru"} else ".joblib"
    return Path(artifact_dir) / f"{best_model}_{target}{extension}"


def _load_latest_evaluation_report(manifest_path):
    manifest = _load_json(manifest_path)
    if not manifest:
        return None
    report_path = manifest.get("files", {}).get("evaluation_report") or manifest.get("evaluation_report")
    if not report_path:
        run_id = manifest.get("run_id")
        if run_id:
            report_path = str(Path(manifest_path).parent / run_id / "evaluation_report.json")
    return _load_json(report_path) if report_path else None


def _artifact_hashes(paths):
    hashes = []
    for path in paths:
        file_path = Path(path)
        hashes.append({
            "path": path,
            "exists": file_path.exists(),
            "sha256": _sha256(file_path) if file_path.exists() else None,
        })
    return hashes


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _higher_status(value, thresholds):
    if value is None:
        return "unavailable"
    if value < thresholds[0]:
        return "failed"
    if value < thresholds[1]:
        return "acceptable"
    if value < thresholds[2]:
        return "passed"
    return "strong"


def _lower_status(value, thresholds):
    if value is None:
        return "unavailable"
    if value > thresholds[0]:
        return "failed"
    if value > thresholds[1]:
        return "unideal"
    if value > thresholds[2]:
        return "acceptable"
    return "strong"


def _worst_status(statuses):
    priority = {"failed": 5, "unideal": 4, "acceptable": 3, "passed": 2, "strong": 1, "unavailable": 0}
    values = list(statuses)
    if not values:
        return "unavailable"
    available = [status for status in values if status != "unavailable"]
    if not available:
        return "unavailable"
    return max(available, key=lambda status: priority.get(status, 6))


def _load_csv(path):
    csv_path = Path(path)
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def _load_json(path):
    if not path:
        return None
    json_path = Path(path)
    if not json_path.exists():
        return None
    return json.loads(json_path.read_text(encoding="utf-8"))


def _same_path(left, right):
    if not left or not right:
        return False
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:
        return str(left) == str(right)


def _round(value, digits=3):
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value
