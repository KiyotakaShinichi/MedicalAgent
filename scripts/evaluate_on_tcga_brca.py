from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.tcga_brca_external_eval import build_tcga_brca_external_eval  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download/report TCGA-BRCA external clinical applicability. "
            "Optionally evaluate a mapped predictions CSV with actual_label and predicted_probability."
        )
    )
    parser.add_argument("--output-dir", default="Data/external_validation/tcga_brca")
    parser.add_argument("--mapped-predictions", default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--size", type=int, default=2000)
    args = parser.parse_args()

    report = build_tcga_brca_external_eval(
        output_dir=args.output_dir,
        mapped_predictions_csv=args.mapped_predictions,
        download=not args.no_download,
        size=args.size,
    )
    print(json.dumps({
        "status": report["status"],
        "rows": report["rows"],
        "report_json": report["files"]["report_json"],
        "claim_boundary": report["claim_boundary"],
    }, indent=2))


if __name__ == "__main__":
    main()
