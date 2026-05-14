import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.multilingual_refusal_eval import (  # noqa: E402
    DEFAULT_CASES_PATH,
    DEFAULT_OUTPUT_PATH,
    run_multilingual_refusal_eval,
)


def main():
    parser = argparse.ArgumentParser(description="Run multilingual refusal routing checks.")
    parser.add_argument("--cases-path", default=DEFAULT_CASES_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    report = run_multilingual_refusal_eval(args.cases_path, args.output_path)
    print(json.dumps({
        "output_path": args.output_path,
        "status": (report.get("summary") or {}).get("status"),
        "pass_rate": (report.get("summary") or {}).get("pass_rate"),
        "case_count": (report.get("summary") or {}).get("case_count"),
    }, indent=2))


if __name__ == "__main__":
    main()
