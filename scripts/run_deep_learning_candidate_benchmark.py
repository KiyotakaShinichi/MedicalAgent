from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.deep_learning_candidate_benchmark import (  # noqa: E402
    DEFAULT_OUTPUT_PATH,
    run_deep_learning_candidate_benchmark,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train synthetic temporal DL classification/regression candidates.")
    parser.add_argument("--source-csv", default="Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report = run_deep_learning_candidate_benchmark(
        source_csv=args.source_csv,
        output_path=args.output,
        epochs=args.epochs,
        seed=args.seed,
    )
    print(json.dumps({
        "status": report["status"],
        "best_model": report["best_model"],
        "best_models": report["best_models"],
        "genetic_context_decision": report["genetic_context_ablation"]["decision"],
        "best_classification_auroc_delta_from_genetics": report["genetic_context_ablation"]["best_classification_auroc_delta"],
        "best_regression_mae_delta_from_genetics": report["genetic_context_ablation"]["best_regression_mae_delta"],
        "output_path": args.output,
    }, indent=2))
    return 0 if report["status"] in {"strong", "acceptable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
