"""Generate the synthetic-data generator card."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.synthetic_generator_card import build_synthetic_generator_card


if __name__ == "__main__":
    payload = build_synthetic_generator_card()
    print(json.dumps({
        "status": payload.get("status"),
        "dataset_schema_version": payload.get("dataset_schema_version"),
        "patients_created": payload.get("cohort", {}).get("patients_created"),
        "rows_fingerprint": payload.get("cohort", {}).get("rows_fingerprint"),
        "row_count": payload.get("feature_distribution_summary", {}).get("row_count"),
        "card_version_matches_dataset": payload.get("card_version_matches_dataset"),
    }, indent=2))
    sys.exit(0 if payload.get("status") in {"passed", "strong"} else 1)
