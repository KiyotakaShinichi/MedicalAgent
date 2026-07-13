from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.pinecone_shadow_retrieval import (  # noqa: E402
    DEFAULT_BASELINE_PATH,
    DEFAULT_DOC_PATH,
    DEFAULT_OUTPUT_PATH,
    build_pinecone_shadow_retrieval_comparison,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Pinecone shadow retrieval comparison readiness artifact.")
    parser.add_argument("--baseline", default=DEFAULT_BASELINE_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--doc", default=DEFAULT_DOC_PATH)
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args()

    report = build_pinecone_shadow_retrieval_comparison(
        baseline_path=args.baseline,
        output_path=args.output,
        doc_path=args.doc,
        allow_network=args.allow_network,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "pinecone_configured": report["pinecone_config"]["configured"],
                "comparison_completed": report["comparison_completed"],
                "clinical_validation": report["clinical_validation"],
                "phi_allowed": report["phi_allowed"],
                "output_path": args.output,
                "doc_path": args.doc,
            },
            indent=2,
        )
    )
    return 0 if report["status"] in {
        "ready_for_shadow_mode_not_configured",
        "configured_dry_run_only",
        "configured_ready_for_manual_shadow_run",
        "acceptable",
        "strong",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
