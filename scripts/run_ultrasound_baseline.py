from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.imaging_baseline_experiments import DEFAULT_ULTRASOUND_OUTPUT_PATH, run_ultrasound_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a public breast ultrasound image baseline if data is present.")
    parser.add_argument("--dataset-root", default="Datasets/BUSI")
    parser.add_argument("--output", default=DEFAULT_ULTRASOUND_OUTPUT_PATH)
    parser.add_argument("--max-images", type=int, default=600)
    args = parser.parse_args()

    result = run_ultrasound_baseline(
        dataset_root=args.dataset_root,
        output_path=args.output,
        max_images=args.max_images,
    )
    print(json.dumps({
        "status": result["status"],
        "task": result.get("task"),
        "image_count": result.get("image_count"),
        "best_model": result.get("best_model"),
        "output_path": args.output,
    }, indent=2))


if __name__ == "__main__":
    main()
