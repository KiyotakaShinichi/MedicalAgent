import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.detailed_training_report import generate_detailed_training_report  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Generate detailed test-set, residual, slice, and hybrid-rules reports.")
    parser.add_argument("--training-rows-path", default="Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv")
    parser.add_argument("--classification-predictions-path", default="Data/complete_synthetic_training/complete_synthetic_model_predictions.csv")
    parser.add_argument("--regression-predictions-path", default="Data/complete_synthetic_training/complete_synthetic_response_regression_predictions.csv")
    parser.add_argument("--metrics-path", default="Data/complete_synthetic_training/complete_synthetic_model_metrics.json")
    parser.add_argument("--output-dir", default="Data/complete_synthetic_training/detailed_eval")
    args = parser.parse_args()

    report = generate_detailed_training_report(
        training_rows_path=args.training_rows_path,
        classification_predictions_path=args.classification_predictions_path,
        regression_predictions_path=args.regression_predictions_path,
        metrics_path=args.metrics_path,
        output_dir=args.output_dir,
    )
    print(json.dumps({
        "status": "generated",
        "test_patients": report["test_patients"],
        "best_classifier": report["best_classifier"],
        "best_regressor": report["best_regressor"],
        "files": report["files"],
    }, indent=2))


if __name__ == "__main__":
    main()
