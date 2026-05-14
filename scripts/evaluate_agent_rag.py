import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.agent_regression_eval import (  # noqa: E402
    DEFAULT_AGENT_REGRESSION_PATH,
    load_eval_cases,
    run_agent_regression_suite,
)


def main():
    parser = argparse.ArgumentParser(description="Run offline patient-agent RAG/guardrail regression checks.")
    parser.add_argument("--output-path", default=DEFAULT_AGENT_REGRESSION_PATH)
    parser.add_argument("--max-cases", type=int, default=None, help="Run only the first N cases for CI smoke checks.")
    args = parser.parse_args()

    cases = load_eval_cases()
    if args.max_cases:
        cases = cases[: max(args.max_cases, 1)]
    report = run_agent_regression_suite(output_path=args.output_path, cases=cases)
    print(json.dumps({
        "output_path": args.output_path,
        "case_count": report["case_count"],
        "status": report["summary"]["status"],
        "pass_rate": report["summary"]["pass_rate"],
        "attack_block_rate": report["summary"]["attack_block_rate"],
        "expected_source_hit_rate": report["summary"]["expected_source_hit_rate"],
    }, indent=2))

    if report["summary"]["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
