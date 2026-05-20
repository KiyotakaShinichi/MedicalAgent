from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.external_stress_test_readiness import DEFAULT_OUTPUT_PATH, build_external_stress_test_readiness


def main() -> int:
    report = build_external_stress_test_readiness(output_path=DEFAULT_OUTPUT_PATH)
    print(json.dumps({
        "status": report.get("status"),
        "dataset_count": (report.get("summary") or {}).get("dataset_count"),
        "total_external_like_rows_seen": (report.get("summary") or {}).get("total_external_like_rows_seen"),
        "promotion_allowed": (report.get("summary") or {}).get("promotion_allowed"),
        "artifact": DEFAULT_OUTPUT_PATH,
    }, indent=2))
    return 0 if report.get("status") in {"strong", "acceptable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
