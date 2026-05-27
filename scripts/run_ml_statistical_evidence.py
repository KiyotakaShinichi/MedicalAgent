"""Build the synthetic-only ML statistical evidence dossier."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.ml_statistical_tests import DEFAULT_OUTPUT_PATH, build_ml_statistical_evidence  # noqa: E402


def main() -> int:
    report = build_ml_statistical_evidence(output_path=ROOT / DEFAULT_OUTPUT_PATH)
    print(json.dumps({
        "status": report["status"],
        "warning_count": report["warning_count"],
        "missing_artifacts": report["missing_artifacts"],
        "output": str(DEFAULT_OUTPUT_PATH),
    }, indent=2))
    return 0 if report["status"] in {"acceptable", "strong"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
