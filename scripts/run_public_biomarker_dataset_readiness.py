from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.public_biomarker_dataset_readiness import (  # noqa: E402
    DEFAULT_OUTPUT_PATH,
    build_public_biomarker_dataset_readiness,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build public biomarker/tumor-marker dataset readiness artifact.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--live-enrich", action="store_true", help="Query public APIs for study-count enrichment.")
    args = parser.parse_args()

    report = build_public_biomarker_dataset_readiness(
        output_path=args.output,
        live_enrich=args.live_enrich,
    )
    print(json.dumps({
        "status": report["status"],
        "dataset_count": report["summary"]["dataset_count"],
        "biomarker_external_candidate_count": report["summary"]["biomarker_external_candidate_count"],
        "tumor_marker_response_train_ready": report["summary"]["tumor_marker_response_train_ready"],
        "production_retrain_now": report["retraining_decision"]["production_retrain_now"],
        "candidate_training_recommended": report["retraining_decision"]["candidate_training_recommended"],
        "output_path": args.output,
    }, indent=2))
    return 0 if report["status"] in {"strong", "acceptable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
