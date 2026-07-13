"""Run the multiturn adversarial agent safety eval.

Writes ``Data/evals/agentic_tool_use/latest_multiturn_adversarial_agent_eval.json``.
Does NOT invoke the live LLM (FAST_MODE forced) and does NOT change
live-agent behaviour.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.multiturn_adversarial_agent_eval import (  # noqa: E402
    OUTPUT_PATH,
    build_report,
    write_report,
)


def main() -> int:
    out = write_report(OUTPUT_PATH)
    report = build_report()
    m = report["metrics"]
    print(f"wrote: {out}")
    print(f"  status={report['status']}  n_cases={report['n_cases']}  turns={m['total_turns']}")
    print(
        f"  unsafe_leakage_rate={m['unsafe_leakage_rate']}  "
        f"refusal_consistency={m['refusal_consistency']}  "
        f"route_stability={m['route_stability']}"
    )
    print(
        f"  over_refusal_rate={m['over_refusal_rate']}  "
        f"safe_negative_control_pass_rate={m['safe_negative_control_pass_rate']}  "
        f"post_gen_validator_catch_rate={m['post_gen_validator_catch_rate']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
