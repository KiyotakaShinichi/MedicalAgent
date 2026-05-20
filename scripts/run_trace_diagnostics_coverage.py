from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.trace_diagnostics_coverage import DEFAULT_OUTPUT_PATH, build_trace_diagnostics_coverage


def main() -> int:
    report = build_trace_diagnostics_coverage(output_path=DEFAULT_OUTPUT_PATH)
    print(json.dumps({
        "status": report.get("status"),
        "rows_checked": (report.get("summary") or {}).get("rows_checked"),
        "rows_with_trace_diagnostics": (report.get("summary") or {}).get("rows_with_trace_diagnostics"),
        "artifact": DEFAULT_OUTPUT_PATH,
    }, indent=2))
    summary = report.get("summary") or {}
    schema_ok = summary.get("sample_trace_schema_valid") is True
    cot_blocked = summary.get("private_chain_of_thought_allowed") is False
    return 0 if schema_ok and cot_blocked else 1


if __name__ == "__main__":
    raise SystemExit(main())
