"""Run 300 isolated synthetic prompts through the real patient-agent pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.real_pipeline_scale_eval import run_real_pipeline_scale_eval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-provider", action="store_true")
    args = parser.parse_args()
    result = run_real_pipeline_scale_eval(allow_provider=args.allow_provider)
    summary = result["summary"]
    print(
        f"Real pipeline calls: {result['query_count']}; "
        f"completion={summary['pipeline_completion_rate']:.4f}; "
        f"contract={summary['contract_pass_rate']:.4f}; "
        f"warm p95={summary['warm_latency_p95_ms']:.2f} ms; "
        f"provider token coverage={summary['provider_usage_coverage_rate']:.4f}"
    )


if __name__ == "__main__":
    main()
