from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.container_security_scan import build_container_security_scan  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", action="store_true", help="execute Trivy instead of parsing the current raw result")
    parser.add_argument("--image", default="nlcare-synthetic-staging-backend:latest")
    parser.add_argument(
        "--sbom-output",
        type=Path,
        default=Path("Data/evals/ops/nlcare_backend_sbom.cdx.json"),
    )
    args = parser.parse_args()
    report = build_container_security_scan(
        image=args.image,
        execute_scan=args.scan,
        sbom_path=args.sbom_output,
    )
    print(json.dumps({
        "status": report["status"],
        "image": report["image"].get("reference"),
        "high_or_critical": report["summary"].get("high_or_critical_count"),
        "fixable_high_or_critical": report["summary"].get("fixable_high_or_critical_count"),
        "base_images_digest_pinned": report["summary"].get("base_images_digest_pinned"),
        "sbom_available": report["summary"].get("sbom_available"),
        "decision": report["deployment_decision"],
    }, indent=2))
