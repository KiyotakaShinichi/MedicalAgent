"""Materialize the bounded-agentic workflow case bank."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.bounded_agentic_workflow import DEFAULT_CASE_PATH, default_workflow_cases  # noqa: E402


def build_cases(path: Path = ROOT / DEFAULT_CASE_PATH) -> list[dict]:
    cases = default_workflow_cases()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(case, ensure_ascii=True) for case in cases) + "\n", encoding="utf-8")
    return cases


def main() -> int:
    cases = build_cases()
    print(json.dumps({"case_count": len(cases), "output": str(DEFAULT_CASE_PATH)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
