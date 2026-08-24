from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.research_paper_query_telemetry import (  # noqa: E402
    DEFAULT_FAILURES_PATH,
    DEFAULT_OUTPUT_PATH,
    run_research_paper_query_telemetry,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fixed research-paper agent query telemetry.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--failures-output", type=Path, default=DEFAULT_FAILURES_PATH)
    parser.add_argument(
        "--allow-provider",
        action="store_true",
        help="Allow configured LLM providers. Default is deterministic/offline telemetry.",
    )
    args = parser.parse_args()
    report = run_research_paper_query_telemetry(
        output_path=args.output,
        failures_path=args.failures_output,
        allow_provider=args.allow_provider,
    )
    if not report.get("evaluated", True):
        print(
            "Research-paper query telemetry: "
            f"{report['status'].upper()} - {report['reason']}"
        )
        print(
            "  no queries were issued; running this probe without the corpus "
            "would report retrieval misses as a measurement."
        )
        return 0

    print(json.dumps({
        "status": report["status"],
        "query_count": report["query_count"],
        "summary": report["summary"],
        "artifact": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
