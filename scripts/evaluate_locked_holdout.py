import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.locked_holdout_evaluation import evaluate_locked_holdout  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Evaluate the current synthetic champion on the frozen locked holdout.")
    parser.add_argument("--model-dir", default="Data/complete_synthetic_training")
    parser.add_argument("--metrics-path", default="Data/complete_synthetic_training/complete_synthetic_model_metrics.json")
    parser.add_argument("--locked-holdout-rows-path", default="Data/complete_synthetic_training/locked_holdout/locked_holdout_rows.csv")
    parser.add_argument("--output-dir", default="Data/complete_synthetic_training/locked_holdout_eval")
    args = parser.parse_args()
    result = evaluate_locked_holdout(
        model_dir=args.model_dir,
        metrics_path=args.metrics_path,
        locked_holdout_rows_path=args.locked_holdout_rows_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
