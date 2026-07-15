from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.confirmed_write_contract_eval import build_report, write_report  # noqa: E402


def main() -> int:
    path = write_report()
    report = build_report()
    print(json.dumps({"artifact": path.as_posix(), "status": report["status"], "case_count": report["case_count"], "pass_rate": report["pass_rate"]}, indent=2))
    return 0 if report["failed_n"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
