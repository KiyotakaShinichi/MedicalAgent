import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.dataset_lineage import build_complete_synthetic_lineage  # noqa: E402
from backend.services.locked_holdout import create_locked_holdout_manifest  # noqa: E402
from backend.services.temporal_leakage_audit import run_temporal_leakage_audit  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Generate senior-style MLE lineage, leakage, and holdout artifacts.")
    parser.add_argument("--dataset-dir", default="Data/complete_synthetic_breast_journeys")
    parser.add_argument("--training-rows-path", default="Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv")
    parser.add_argument("--lineage-output-path", default="Data/lineage/complete_synthetic_lineage.json")
    parser.add_argument("--leakage-output-path", default="Data/complete_synthetic_training/leakage_audit/temporal_leakage_audit.json")
    parser.add_argument("--locked-holdout-output-dir", default="Data/complete_synthetic_training/locked_holdout")
    parser.add_argument("--holdout-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=314159)
    args = parser.parse_args()

    lineage = build_complete_synthetic_lineage(
        dataset_dir=args.dataset_dir,
        output_path=args.lineage_output_path,
    )
    leakage = run_temporal_leakage_audit(
        training_rows_path=args.training_rows_path,
        output_path=args.leakage_output_path,
    )
    holdout = create_locked_holdout_manifest(
        training_rows_path=args.training_rows_path,
        output_dir=args.locked_holdout_output_dir,
        holdout_size=args.holdout_size,
        seed=args.seed,
    )
    print(json.dumps({
        "status": "generated",
        "dataset_hash": lineage.get("dataset_hash"),
        "leakage_status": leakage.get("status"),
        "locked_holdout_patients": holdout.get("locked_holdout_patients"),
        "files": {
            "lineage": args.lineage_output_path,
            "leakage": args.leakage_output_path,
            "locked_holdout_manifest": holdout.get("files", {}).get("manifest_json"),
        },
    }, indent=2))


if __name__ == "__main__":
    main()
