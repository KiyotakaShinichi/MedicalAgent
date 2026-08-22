from pathlib import Path

from backend.services.feature_store import load_feature_store_manifest
from backend.services.mle_readiness_checks.core import check, load_json


def artifact_checks(metrics, metrics_path, predictions_path, training_csv, evaluation_report):
    checks = [
        check(
            name="training_dataset_present",
            category="artifacts",
            status="passed" if Path(training_csv).exists() else "failed",
            value=training_csv,
            threshold="file exists",
            meaning="The training dataset must be available for reproducibility.",
            hard_gate=True,
            remediation="Regenerate the complete synthetic journey dataset.",
        ),
        check(
            name="metrics_artifact_present",
            category="artifacts",
            status="passed" if Path(metrics_path).exists() else "failed",
            value=metrics_path,
            threshold="file exists",
            meaning="The model metrics file must exist before evaluation or promotion.",
            hard_gate=True,
            remediation="Run the training pipeline to create the metrics artifact.",
        ),
        check(
            name="prediction_artifact_present",
            category="artifacts",
            status="passed" if Path(predictions_path).exists() else "failed",
            value=predictions_path,
            threshold="file exists",
            meaning="Patient-level predictions are required for calibration, threshold, and false-negative checks.",
            hard_gate=True,
            remediation="Run model training/evaluation to export prediction rows.",
        ),
        check(
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
    artifact_path = model_artifact_path(best_model, target, artifact_dir=Path(metrics_path).parent)
    checks.append(
        check(
            name="champion_model_artifact_present",
            category="artifacts",
            status="passed" if artifact_path and artifact_path.exists() else "failed",
            value=str(artifact_path) if artifact_path else "unknown",
            threshold="champion joblib/pt exists",
            meaning="The selected champion must resolve to a concrete model artifact.",
            hard_gate=True,
            remediation="Retrain models or check artifact naming in Data/complete_synthetic_training.",
        )
    )
    return checks


def feature_store_checks(training_csv):
    manifest = load_feature_store_manifest()
    if manifest.get("status") == "missing":
        return [
            check(
                name="local_feature_store_materialized",
                category="feature_store",
                status="unideal",
                value=manifest.get("path"),
                threshold="feature_store_manifest.json exists",
                meaning="A feature-store manifest keeps training and serving feature contracts visible.",
                hard_gate=False,
                remediation="Run python scripts/materialize_feature_store.py.",
            )
        ]
    source_matches = same_path(manifest.get("source_csv"), training_csv)
    return [
        check(
            name="local_feature_store_materialized",
            category="feature_store",
            status="passed" if manifest.get("status") == "current" else "unideal",
            value={
                "status": manifest.get("status"),
                "rows": manifest.get("row_count"),
                "entities": manifest.get("entity_count"),
            },
            threshold="manifest current",
            meaning="Local offline features should be materialized and hash-checked.",
            hard_gate=False,
            remediation="Rematerialize the feature store from the current training CSV.",
        ),
        check(
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


def model_artifact_path(best_model, target, artifact_dir="Data/complete_synthetic_training"):
    if not best_model:
        return None
    extension = (
        ".pt"
        if best_model in {"temporal_baseline_cnn", "temporal_1d_cnn", "temporal_gru"}
        else ".joblib"
    )
    return Path(artifact_dir) / f"{best_model}_{target}{extension}"


def load_latest_evaluation_report(manifest_path):
    manifest = load_json(manifest_path)
    if not manifest:
        return None
    report_path = manifest.get("files", {}).get("evaluation_report") or manifest.get(
        "evaluation_report"
    )
    if not report_path:
        run_id = manifest.get("run_id")
        if run_id:
            report_path = str(Path(manifest_path).parent / run_id / "evaluation_report.json")
    return load_json(report_path) if report_path else None


def same_path(left, right):
    if not left or not right:
        return False
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:
        return str(left) == str(right)
