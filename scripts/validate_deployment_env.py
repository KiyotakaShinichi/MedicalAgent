from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.deployment_profile_validation import build_report, write_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate NLCare deployment environment without printing secrets")
    parser.add_argument("--strict", action="store_true", help="Fail unless the active staging/production profile passes")
    args = parser.parse_args()
    path = write_report()
    report = build_report()
    print(json.dumps({
        "artifact": path.as_posix(),
        "profile": report["profile"],
        "status": report["status"],
        "failed_checks": report["failed_checks"],
    }, indent=2))
    if report["status"] == "blocked":
        return 1
    if args.strict and not report["strict_profile"]:
        print("--strict requires ENVIRONMENT=staging or production", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
