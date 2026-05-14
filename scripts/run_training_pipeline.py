import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.database import SessionLocal
from backend.services.complete_synthetic_training import train_complete_synthetic_models
from backend.services.dataset_lineage import build_complete_synthetic_lineage
from backend.services.detailed_training_report import generate_detailed_training_report
from backend.services.evaluation_reports import generate_versioned_evaluation_report
from backend.services.locked_holdout import create_locked_holdout_manifest
from backend.services.mlops_tracking import finish_experiment_run, start_experiment_run
from backend.services.model_artifacts import register_complete_synthetic_champion
from backend.services.temporal_leakage_audit import run_temporal_leakage_audit


def main():
    parser = argparse.ArgumentParser(description="Run the complete synthetic monitoring training pipeline.")
    parser.add_argument("--ml-csv-path", default="Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv")
    parser.add_argument("--training-output-dir", default="Data/complete_synthetic_training")
    parser.add_argument("--evaluation-output-root", default="Data/model_evaluation_reports")
    parser.add_argument("--target", default="treatment_success_binary")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cnn-epochs", type=int, default=20)
    parser.add_argument("--cnn-batch-size", type=int, default=16)
    parser.add_argument("--model-version", default="synthetic-v1")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-register", action="store_true")
    args = parser.parse_args()

    if not Path(args.ml_csv_path).exists():
        raise FileNotFoundError(f"Training CSV not found: {args.ml_csv_path}")

    training_metrics = None
    run = start_experiment_run(
        experiment_name="complete_synthetic_training_pipeline",
        run_name=args.model_version,
        params={
            "ml_csv_path": args.ml_csv_path,
            "training_output_dir": args.training_output_dir,
            "target": args.target,
            "test_size": args.test_size,
            "seed": args.seed,
            "cnn_epochs": args.cnn_epochs,
            "cnn_batch_size": args.cnn_batch_size,
            "skip_training": args.skip_training,
            "skip_register": args.skip_register,
        },
        tags={"pipeline": "local_account_free", "warning": "synthetic_data_only"},
    )
    if not args.skip_training:
        try:
            training_metrics = train_complete_synthetic_models(
                ml_csv_path=args.ml_csv_path,
                output_dir=args.training_output_dir,
                target=args.target,
                test_size=args.test_size,
                seed=args.seed,
                cnn_epochs=args.cnn_epochs,
                cnn_batch_size=args.cnn_batch_size,
            )
        except Exception as exc:
            finish_experiment_run(
                run_id=run["run_id"],
                status="failed",
                error_message=str(exc),
            )
            raise

    db = SessionLocal()
    try:
        registry_model = None
        if not args.skip_register:
            registry_model = register_complete_synthetic_champion(
                db=db,
                version=args.model_version,
                metrics_path=str(Path(args.training_output_dir) / "complete_synthetic_model_metrics.json"),
                training_data_path=args.ml_csv_path,
                artifact_dir=args.training_output_dir,
                promotion_status="candidate",
                promotion_reason="Pipeline-registered synthetic champion for engineering evaluation.",
            )
        detailed_report = generate_detailed_training_report(
            training_rows_path=args.ml_csv_path,
            metrics_path=str(Path(args.training_output_dir) / "complete_synthetic_model_metrics.json"),
            classification_predictions_path=str(Path(args.training_output_dir) / "complete_synthetic_model_predictions.csv"),
            regression_predictions_path=str(Path(args.training_output_dir) / "complete_synthetic_response_regression_predictions.csv"),
            output_dir=str(Path(args.training_output_dir) / "detailed_eval"),
        )
        lineage = build_complete_synthetic_lineage(
            dataset_dir=str(Path(args.ml_csv_path).parent),
            output_path=str(Path(args.training_output_dir) / "dataset_lineage.json"),
        )
        leakage_audit = run_temporal_leakage_audit(
            training_rows_path=args.ml_csv_path,
            output_path=str(Path(args.training_output_dir) / "leakage_audit" / "temporal_leakage_audit.json"),
        )
        locked_holdout = create_locked_holdout_manifest(
            training_rows_path=args.ml_csv_path,
            output_dir=str(Path(args.training_output_dir) / "locked_holdout"),
        )
        report = generate_versioned_evaluation_report(
            db=db,
            output_root=args.evaluation_output_root,
        )
        finish_experiment_run(
            db=db,
            run_id=run["run_id"],
            status="completed",
            metrics=(training_metrics or {}),
            artifacts={
                "metrics": str(Path(args.training_output_dir) / "complete_synthetic_model_metrics.json"),
                "predictions": str(Path(args.training_output_dir) / "complete_synthetic_model_predictions.csv"),
                "evaluation_report": (report.get("files") or {}).get("evaluation_report_json"),
                "detailed_training_report": (detailed_report.get("files") or {}).get("html_report"),
                "dataset_lineage": str(Path(args.training_output_dir) / "dataset_lineage.json"),
                "temporal_leakage_audit": str(Path(args.training_output_dir) / "leakage_audit" / "temporal_leakage_audit.json"),
                "locked_holdout_manifest": (locked_holdout.get("files") or {}).get("manifest_json"),
                "registered_artifact": (registry_model or {}).get("artifact_path"),
            },
            tags={
                "registered": bool(registry_model),
                "leakage_audit_status": leakage_audit.get("status"),
                "dataset_hash": lineage.get("dataset_hash"),
            },
        )
    finally:
        db.close()

    result = {
        "mlops_run_id": run["run_id"],
        "training_ran": not args.skip_training,
        "best_model": (training_metrics or {}).get("best_model_by_patient_level_roc_auc"),
        "best_regressor": (training_metrics or {}).get("best_response_regressor_by_patient_level_mae"),
        "registered_model": registry_model,
        "evaluation_report": report,
        "detailed_training_report": detailed_report,
        "temporal_leakage_audit": leakage_audit,
        "locked_holdout": locked_holdout,
        "warning": "Synthetic training pipeline is for engineering practice and reproducibility, not clinical validation.",
    }
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
