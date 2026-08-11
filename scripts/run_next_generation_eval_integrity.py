from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.evaluation_dataset_integrity import write_integrity_report  # noqa: E402


if __name__ == "__main__":
    report = write_integrity_report()
    print(json.dumps({
        "status": report["status"],
        "dataset_count": report["dataset_count"],
        "integrity_failure_count": report["integrity_failure_count"],
        "external_review_status": report["external_review_status"],
    }, indent=2))
