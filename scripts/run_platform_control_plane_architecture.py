from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.platform_control_plane_architecture import (  # noqa: E402
    DEFAULT_DOC_PATH,
    DEFAULT_OUTPUT_PATH,
    build_platform_control_plane_architecture,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build platform control-plane architecture artifact.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--doc", default=DEFAULT_DOC_PATH)
    args = parser.parse_args()

    report = build_platform_control_plane_architecture(output_path=args.output, doc_path=args.doc)
    print(
        json.dumps(
            {
                "status": report["status"],
                "clinical_validation": report["clinical_validation"],
                "healthcare_production_ready": report["healthcare_production_ready"],
                "live_patient_authority_added": report["live_patient_authority_added"],
                "output_path": args.output,
                "doc_path": args.doc,
            },
            indent=2,
        )
    )
    return 0 if report["status"] in {"strong", "acceptable", "needs_attention"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
