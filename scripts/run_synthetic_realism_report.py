import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.synthetic_realism_report import (  # noqa: E402
    DEFAULT_EXTERNAL_FEATURES_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_THRESHOLDS_PATH,
    DEFAULT_TRAINING_CSV,
    build_synthetic_realism_report,
)


def main():
    parser = argparse.ArgumentParser(description="Run synthetic realism + sim-to-real audit report.")
    parser.add_argument("--training-csv", default=DEFAULT_TRAINING_CSV)
    parser.add_argument("--external-features", default=DEFAULT_EXTERNAL_FEATURES_PATH)
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    report = build_synthetic_realism_report(
        training_csv=args.training_csv,
        external_features_path=args.external_features,
        thresholds_path=args.thresholds,
        output_path=args.output_path,
    )
    print(json.dumps({
        "output_path": args.output_path,
        "status": report.get("status"),
        "training_rows": report.get("training_rows"),
        "training_patients": report.get("training_patients"),
        "sim_to_real_status": (report.get("sim_to_real_comparison") or {}).get("status"),
    }, indent=2))


if __name__ == "__main__":
    main()
