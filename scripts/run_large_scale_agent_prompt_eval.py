from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.large_scale_agent_prompt_eval import (  # noqa: E402
    DEFAULT_END_TO_END_SAMPLE_N,
    DEFAULT_SEED,
    DEFAULT_TARGET_N,
    evaluate_large_scale_agent_prompts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the internal large-scale agent prompt stress suite.")
    parser.add_argument("--target-n", type=int, default=DEFAULT_TARGET_N)
    parser.add_argument("--end-to-end-sample-n", type=int, default=DEFAULT_END_TO_END_SAMPLE_N)
    parser.add_argument("--multi-turn-variants", type=int, default=10)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    payload = evaluate_large_scale_agent_prompts(
        target_n=args.target_n,
        end_to_end_sample_n=args.end_to_end_sample_n,
        multi_turn_variants_per_scenario=args.multi_turn_variants,
        seed=args.seed,
    )
    print(json.dumps({
        "status": payload["status"],
        "prompt_bank_n": payload["prompt_bank_n"],
        "classifier_sweep": payload["classifier_sweep"],
        "bounded_agent_end_to_end_sample": payload["bounded_agent_end_to_end_sample"],
        "multi_turn_bounded_agent": payload["multi_turn_bounded_agent"],
        "failure_summary": payload["failure_summary"],
        "clinical_validation": payload["clinical_validation"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
