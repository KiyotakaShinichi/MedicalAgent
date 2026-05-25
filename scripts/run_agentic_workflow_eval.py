"""Evaluate the bounded agentic workflow planner."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.bounded_agentic_workflow import (  # noqa: E402
    DEFAULT_CASE_PATH,
    DEFAULT_OUTPUT_PATH,
    default_workflow_cases,
    evaluate_workflow_cases,
)


def _load_cases(path: Path) -> list[dict]:
    if not path.exists():
        return default_workflow_cases()
    cases: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(json.loads(line))
    return cases


def run_eval(
    case_path: Path = ROOT / DEFAULT_CASE_PATH,
    output_path: Path = ROOT / DEFAULT_OUTPUT_PATH,
) -> dict:
    cases = _load_cases(case_path)
    report = evaluate_workflow_cases(cases)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    report = run_eval()
    print(json.dumps({
        "status": report["status"],
        "total_n": report["total_n"],
        "pass_count": report["pass_count"],
        "fail_count": report["fail_count"],
        "output": str(DEFAULT_OUTPUT_PATH),
    }, indent=2))
    return 0 if report["status"] in {"strong", "acceptable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
