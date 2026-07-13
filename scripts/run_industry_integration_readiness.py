from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.industry_integration_readiness import (  # noqa: E402
    DEFAULT_DOC_PATH,
    DEFAULT_OUTPUT_PATH,
    build_industry_integration_readiness,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build n8n/Pinecone industry integration readiness artifact.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--doc", default=DEFAULT_DOC_PATH)
    args = parser.parse_args()

    report = build_industry_integration_readiness(output_path=args.output, doc_path=args.doc)
    print(
        json.dumps(
            {
                "status": report["status"],
                "clinical_validation": report["clinical_validation"],
                "healthcare_production_ready": report["healthcare_production_ready"],
                "live_patient_route_enabled": report["live_patient_route_enabled"],
                "phi_allowed": report["phi_allowed"],
                "output_path": args.output,
                "doc_path": args.doc,
            },
            indent=2,
        )
    )
    return 0 if report["status"] in {"ready_for_optional_scaffold", "acceptable", "strong"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
