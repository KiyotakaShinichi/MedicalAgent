from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.ultrasound_transfer_baseline import DEFAULT_OUTPUT_PATH, run_ultrasound_transfer_baseline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run BUSI ultrasound transfer-learning baseline.")
    parser.add_argument("--dataset-root", default="Datasets/BUSI")
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-images", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet weights if available/downloadable.")
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()

    result = run_ultrasound_transfer_baseline(
        dataset_root=args.dataset_root,
        output_path=args.output_path,
        max_images=args.max_images,
        epochs=args.epochs,
        batch_size=args.batch_size,
        pretrained=args.pretrained,
    )
    if args.print_summary:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
