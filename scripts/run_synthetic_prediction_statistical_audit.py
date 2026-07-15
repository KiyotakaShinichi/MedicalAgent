from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.synthetic_prediction_statistical_audit import build_report, write_report  # noqa: E402


def main() -> int:
    path = write_report()
    report = build_report()
    print(json.dumps({
        "artifact": path.as_posix(),
        "status": report["status"],
        "total_n": report.get("total_n"),
        "classification": report.get("classification_metrics"),
        "regression": report.get("regression_metrics"),
        "promotion_decision": report.get("promotion_decision"),
    }, indent=2))
    return 0 if report["status"] != "needs_attention" else 1


if __name__ == "__main__":
    raise SystemExit(main())
