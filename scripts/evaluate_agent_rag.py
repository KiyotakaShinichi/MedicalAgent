import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.agent_regression_eval import (  # noqa: E402
    DEFAULT_AGENT_REGRESSION_PATH,
    run_agent_regression_suite,
)


def main():
    parser = argparse.ArgumentParser(description="Run offline patient-agent RAG/guardrail regression checks.")
    parser.add_argument("--output-path", default=DEFAULT_AGENT_REGRESSION_PATH)
    args = parser.parse_args()

    report = run_agent_regression_suite(output_path=args.output_path)
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
