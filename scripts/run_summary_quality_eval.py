import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.summary_quality_eval import (  # noqa: E402
    DEFAULT_OUTPUT_PATH,
    DEFAULT_RUBRIC_PATH,
    build_summary_quality_report,
)


def main():
    parser = argparse.ArgumentParser(description="Run summary quality evaluation using clinician review logs.")
    parser.add_argument("--rubric", default=DEFAULT_RUBRIC_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    report = build_summary_quality_report(rubric_path=args.rubric, output_path=args.output_path)
    print(json.dumps({
        "output_path": args.output_path,
        "status": report.get("status"),
        "review_count": report.get("review_count"),
        "summary_completeness_rate": report.get("summary_completeness_rate"),
        "unsafe_advice_rate": report.get("unsafe_advice_rate"),
    }, indent=2))


if __name__ == "__main__":
    main()
