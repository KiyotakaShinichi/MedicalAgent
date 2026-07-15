from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.ml_coverage_risk_diagnostics import (  # noqa: E402
    DEFAULT_OUTPUT_PATH,
    build_ml_coverage_risk_diagnostics,
)


def main() -> int:
    report = build_ml_coverage_risk_diagnostics(output_path=DEFAULT_OUTPUT_PATH)
    print(f"wrote: {DEFAULT_OUTPUT_PATH}")
    print(json.dumps({
        "status": report["status"],
        "scenario_count": report["scenario_count"],
        "required_abstention_passed": report["required_abstention_scenarios"]["all_required_scenarios_passed"],
        "promotion_decision": report["promotion_decision"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
