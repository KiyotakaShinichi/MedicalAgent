from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.dep001_external_evaluation import run_official_external_evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the official DEP-001 external holdout exactly once.")
    parser.add_argument("holdout", type=Path)
    args = parser.parse_args()
    result = run_official_external_evaluation(args.holdout.resolve())
    metrics = result["metrics"]
    print(f"status={result['status']}")
    print(f"dep001={result['dep001_decision']}")
    print(f"cases={metrics['total_n']}")
    print(f"unsafe_released={metrics['unsafe_released_output_count']}")
    return 0 if result["dep001_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
