from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.breastdcedl_metadata_stress import (  # noqa: E402
    DEFAULT_CANONICAL_CSV,
    DEFAULT_DOC_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_PREDICTIONS_PATH,
    run_breastdcedl_metadata_stress,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run BreastDCEDL metadata-only external stress benchmark.")
    parser.add_argument("--canonical-csv", default=DEFAULT_CANONICAL_CSV)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--doc", default=DEFAULT_DOC_PATH)
    parser.add_argument("--bootstrap", type=int, default=300)
    args = parser.parse_args()

    report = run_breastdcedl_metadata_stress(
        canonical_csv=args.canonical_csv,
        output_path=args.output,
        predictions_path=args.predictions,
        doc_path=args.doc,
        n_bootstrap=args.bootstrap,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "rows": report["cohort_summary"]["rows"],
                "clinical_validation": report["clinical_validation"],
                "production_training_allowed": report["production_training_allowed"],
                "stress_result": report["stress_result"].get("status"),
                "roc_auc": report["stress_result"].get("roc_auc"),
                "brier": report["stress_result"].get("brier"),
                "output_path": args.output,
                "predictions_path": args.predictions,
                "doc_path": args.doc,
            },
            indent=2,
        )
    )
    return 0 if report["status"] in {"strong", "acceptable", "needs_attention"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
