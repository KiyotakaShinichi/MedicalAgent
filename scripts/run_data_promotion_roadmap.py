from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.data_promotion_roadmap import (  # noqa: E402
    DEFAULT_OUTPUT_PATH,
    build_data_promotion_roadmap,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build data-to-promotion roadmap artifact.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    report = build_data_promotion_roadmap(output_path=args.output)
    print(json.dumps({
        "status": report["status"],
        "model_head_count": len(report["model_heads"]),
        "blocker_count": len(report["cross_head_blockers"]),
        "current_global_policy": report["promotion_policy"]["current_global_policy"],
        "output_path": args.output,
    }, indent=2))
    return 0 if report["status"] in {"strong", "acceptable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
