"""Run the synthetic ML logic/safety alignment audit."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.ml_logic_safety_alignment import (  # noqa: E402
    DEFAULT_OUTPUT_PATH,
    build_ml_logic_safety_alignment,
)


def main() -> int:
    report = build_ml_logic_safety_alignment(output_path=DEFAULT_OUTPUT_PATH)
    summary = report["summary"]
    print(f"wrote: {DEFAULT_OUTPUT_PATH}")
    print(
        "  status={status} checks={checks} passed={passed} needs_attention={attention}".format(
            status=report["status"],
            checks=summary["check_count"],
            passed=summary["passed_count"],
            attention=summary["needs_attention_count"],
        )
    )
    for item in report.get("highest_leverage_ml_next_steps", [])[:3]:
        print(f"  next[{item['rank']}]: {item['from_check']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
