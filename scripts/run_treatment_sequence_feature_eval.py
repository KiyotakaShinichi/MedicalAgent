from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.treatment_sequence_features import (  # noqa: E402
    DEFAULT_OUTPUT_PATH,
    DEFAULT_SEQUENCE_CSV,
    build_treatment_sequence_feature_eval,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build synthetic treatment-sequence feature artifact.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--sequence-csv", default=DEFAULT_SEQUENCE_CSV)
    args = parser.parse_args()

    report = build_treatment_sequence_feature_eval(
        output_path=args.output,
        sequence_csv=args.sequence_csv,
    )
    print(json.dumps({
        "status": report["status"],
        "patient_count": report["patient_count"],
        "pattern_count": report["pattern_count"],
        "output_path": args.output,
    }, indent=2))
    return 0 if report["status"] in {"strong", "acceptable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
