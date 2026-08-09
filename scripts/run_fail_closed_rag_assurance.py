"""Execute and persist the fail-closed RAG release assurance suite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.fail_closed_rag_assurance import (
    DEFAULT_OUTPUT_PATH,
    run_fail_closed_assurance,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    args = parser.parse_args()
    report = run_fail_closed_assurance(
        output_path=args.output,
        timeout_seconds=args.timeout_seconds,
    )
    summary = report["summary"]
    print(
        "fail-closed RAG assurance: "
        f"{report['status']} ({summary['passed_tests']} passed, "
        f"{summary['failed_tests']} failed)"
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
