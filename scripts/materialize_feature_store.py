import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.feature_store import DEFAULT_FEATURE_STORE_DIR, materialize_feature_store  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Materialize local account-free feature store artifacts.")
    parser.add_argument("--source-csv", default="Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv")
    parser.add_argument("--output-dir", default=DEFAULT_FEATURE_STORE_DIR)
    parser.add_argument("--entity-column", default="patient_id")
    args = parser.parse_args()

    result = materialize_feature_store(
        source_csv=args.source_csv,
        output_dir=args.output_dir,
        entity_column=args.entity_column,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
