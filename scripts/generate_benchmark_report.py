from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.benchmark_registry import (  # noqa: E402
    DEFAULT_CSV_PATH,
    DEFAULT_JSON_PATH,
    DEFAULT_MD_PATH,
    build_benchmark_registry,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the consolidated MedicalAgent benchmark registry."
    )
    parser.add_argument("--output-path", default=DEFAULT_JSON_PATH)
    parser.add_argument("--report-path", default=DEFAULT_MD_PATH)
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH)
    parser.add_argument("--ttl-seconds", type=int, default=24 * 60 * 60)
    args = parser.parse_args()

    report = build_benchmark_registry(
        output_path=args.output_path,
        report_path=args.report_path,
        csv_path=args.csv_path,
        freshness_ttl_seconds=args.ttl_seconds,
    )
    print(json.dumps({
        "status": report.get("status"),
        "critical_status": report.get("critical_status"),
        "benchmark_count": len(report.get("benchmarks") or []),
        "issue_count": len(report.get("issues") or []),
        "output_path": args.output_path,
        "report_path": args.report_path,
        "csv_path": args.csv_path,
    }, indent=2))
    return 0 if report.get("status") != "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
