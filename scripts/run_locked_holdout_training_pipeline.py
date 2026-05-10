import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.database import SessionLocal  # noqa: E402
from backend.services.complete_synthetic_training import train_complete_synthetic_models  # noqa: E402
from backend.services.dataset_lineage import build_complete_synthetic_lineage  # noqa: E402
from backend.services.detailed_training_report import generate_detailed_training_report  # noqa: E402
from backend.services.evaluation_reports import generate_versioned_evaluation_report  # noqa: E402
from backend.services.locked_holdout_evaluation import evaluate_locked_holdout  # noqa: E402
from backend.services.mle_readiness import build_mle_readiness_summary  # noqa: E402
from backend.services.mlops_tracking import finish_experiment_run, start_experiment_run  # noqa: E402
from backend.services.model_artifacts import register_complete_synthetic_champion  # noqa: E402
from backend.services.temporal_leakage_audit import run_temporal_leakage_audit  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Train on development rows only, then evaluate once on the frozen locked holdout."
    )
    parser.add_argument("--development-rows-path", default="Data/complete_synthetic_training/locked_holdout/development_rows.csv")
    parser.add_argument("--locked-holdout-rows-path", default="Data/complete_synthetic_training/locked_holdout/locked_holdout_rows.csv")
    parser.add_argument("--training-output-dir", default="Data/complete_synthetic_training")
    parser.add_argument("--evaluation-output-root", default="Data/model_evaluation_reports")
    parser.add_argument("--model-version", default="synthetic-v6-dev-locked")
    parser.add_argument("--target", default="treatment_success_binary")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cnn-epochs", type=int, default=5)
    parser.add_argument("--cnn-batch-size", type=int, default=32)
    parser.add_argument("--skip-register", action="store_true")
    args = parser.parse_args()

    run = start_experiment_run(
        experiment_name="locked_holdout_training_pipeline",
        run_name=args.model_version,
        params=vars(args),
        tags={
            "pipeline": "locked_holdout",
            "warning": "synthetic_data_only",
            "holdout_policy": "do_not_tune_on_locked_holdout",
        },
    )
    try:
        metrics = train_complete_synthetic_models(
            ml_csv_path=args.development_rows_path,
            output_dir=args.training_output_dir,
            target=args.target,
            test_size=args.test_size,
            seed=args.seed,
            cnn_epochs=args.cnn_epochs,
            cnn_batch_size=args.cnn_batch_size,
        )
        detailed_report = generate_detailed_training_report(
            training_rows_path=args.development_rows_path,
            metrics_path=str(Path(args.training_output_dir) / "complete_synthetic_model_metrics.json"),
            classification_predictions_path=str(Path(args.training_output_dir) / "complete_synthetic_model_predictions.csv"),
            regression_predictions_path=str(Path(args.training_output_dir) / "complete_synthetic_response_regression_predictions.csv"),
            output_dir=str(Path(args.training_output_dir) / "detailed_eval"),
        )
        locked_eval = evaluate_locked_holdout(
            model_dir=args.training_output_dir,
            metrics_path=str(Path(args.training_output_dir) / "complete_synthetic_model_metrics.json"),
            locked_holdout_rows_path=args.locked_holdout_rows_path,
            output_dir=str(Path(args.training_output_dir) / "locked_holdout_eval"),
        )
        lineage = build_complete_synthetic_lineage()
        leakage_audit = run_temporal_leakage_audit(training_rows_path=args.development_rows_path)

        db = SessionLocal()
        try:
            registry_model = None
            if not args.skip_register:
                registry_model = register_complete_synthetic_champion(
                    db=db,
                    version=args.model_version,
                    metrics_path=str(Path(args.training_output_dir) / "complete_synthetic_model_metrics.json"),
                    training_data_path=args.development_rows_path,
                    artifact_dir=args.training_output_dir,
                    promotion_status="candidate",
                    promotion_reason="Development-split trained champion evaluated once on locked synthetic holdout.",
                )
            versioned_report = generate_versioned_evaluation_report(db=db, output_root=args.evaluation_output_root)
            readiness = build_mle_readiness_summary(
                db=db,
                training_csv=args.development_rows_path,
                output_path="Data/mle_monitoring/latest_mle_readiness.json",
            )
            finish_experiment_run(
                db=db,
                run_id=run["run_id"],
                status="completed",
                metrics=metrics,
                artifacts={
                    "metrics": str(Path(args.training_output_dir) / "complete_synthetic_model_metrics.json"),
                    "development_predictions": str(Path(args.training_output_dir) / "complete_synthetic_model_predictions.csv"),
                    "locked_holdout_summary": (locked_eval.get("files") or {}).get("locked_holdout_summary_json"),
                    "detailed_training_report": (detailed_report.get("files") or {}).get("html_report"),
                    "evaluation_report": (versioned_report.get("files") or {}).get("evaluation_report_json"),
                },
                tags={
                    "registered": bool(registry_model),
                    "leakage_audit_status": leakage_audit.get("status"),
                    "dataset_hash": lineage.get("dataset_hash"),
                    "locked_holdout_patients": locked_eval.get("patients"),
                },
            )
        finally:
            db.close()
    except Exception as exc:
        finish_experiment_run(run_id=run["run_id"], status="failed", error_message=str(exc))
        raise

    result = {
        "mlops_run_id": run["run_id"],
        "training_rows": args.development_rows_path,
        "locked_holdout_rows": args.locked_holdout_rows_path,
        "best_model": metrics.get("best_model_by_patient_level_roc_auc"),
        "best_regressor": metrics.get("best_response_regressor_by_patient_level_mae"),
        "locked_holdout": locked_eval,
        "detailed_training_report": detailed_report,
        "versioned_report": versioned_report,
        "mle_readiness": readiness,
        "warning": "Synthetic locked-holdout discipline only. Not clinical validation.",
    }
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
