from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.public_treatment_dataset_readiness import (  # noqa: E402
    DEFAULT_OUTPUT_PATH,
    build_public_treatment_dataset_readiness,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build public breast-cancer treatment dataset readiness artifact.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    report = build_public_treatment_dataset_readiness(output_path=args.output)
    print(json.dumps({
        "status": report["status"],
        "dataset_count": report["summary"]["dataset_count"],
        "treatment_combination_candidate_count": report["summary"]["treatment_combination_candidate_count"],
        "immediate_full_treatment_combo_training_ready": report["summary"]["immediate_full_treatment_combo_training_ready"],
        "best_future_real_world_treatment_dataset": report["summary"]["best_future_real_world_treatment_dataset"],
        "output_path": args.output,
    }, indent=2))
    return 0 if report["status"] in {"strong", "acceptable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
