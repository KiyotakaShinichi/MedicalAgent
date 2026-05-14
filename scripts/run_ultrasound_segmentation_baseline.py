from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.ultrasound_segmentation_baseline import DEFAULT_OUTPUT_PATH, run_ultrasound_segmentation_baseline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run BUSI ultrasound segmentation baseline.")
    parser.add_argument("--dataset-root", default="Datasets/BUSI")
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-pairs", type=int, default=500)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()

    result = run_ultrasound_segmentation_baseline(
        dataset_root=args.dataset_root,
        output_path=args.output_path,
        max_pairs=args.max_pairs,
        image_size=args.image_size,
    )
    if args.print_summary:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
