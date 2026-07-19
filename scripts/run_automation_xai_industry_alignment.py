from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.automation_xai_industry_alignment import build_automation_xai_industry_alignment  # noqa: E402


def main() -> int:
    report = build_automation_xai_industry_alignment()
    print(
        "automation_xai_industry_alignment "
        f"status={report['status']} "
        f"automation_controls={report['automation_control_count']} "
        f"xai_controls={report['xai_control_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
