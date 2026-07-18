from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.automation_reliability_dossier import build_automation_reliability_dossier  # noqa: E402


def main() -> int:
    report = build_automation_reliability_dossier()
    print(
        json.dumps(
            {
                "status": report["status"],
                "check_count": report["check_count"],
                "passed_count": report["passed_count"],
                "failed_required_count": report["failed_required_count"],
                "external_delivery_enabled_by_default": report["external_delivery_enabled_by_default"],
            },
            indent=2,
        )
    )
    return 0 if report["status"] in {"strong", "acceptable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
