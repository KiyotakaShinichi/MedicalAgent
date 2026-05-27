from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.eval_credibility_audit import run_eval_credibility_audit  # noqa: E402


def main() -> int:
    payload = run_eval_credibility_audit()
    summary = payload["summary"]
    print(json.dumps({
        "status": payload["status"],
        "artifact_count": summary["artifact_count"],
        "external_authored_artifact_count": summary["external_authored_artifact_count"],
        "perfect_internal_score_count": summary["perfect_internal_score_count"],
        "contamination_disclosure_rate": summary["contamination_disclosure_rate"],
        "claim_boundary_rate": summary["claim_boundary_rate"],
        "high_risk_artifact_count": summary["high_risk_artifact_count"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
