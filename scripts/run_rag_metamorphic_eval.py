from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.rag_metamorphic_eval import run_rag_metamorphic_eval  # noqa: E402


def main() -> int:
    payload = run_rag_metamorphic_eval()
    print(json.dumps({
        "status": payload["status"],
        "total_n": payload["total_n"],
        "pass_rate": payload["pass_rate"],
        "unsafe_route_preservation_rate": payload["unsafe_route_preservation_rate"],
        "education_evidence_policy_rate": payload["education_evidence_policy_rate"],
    }, indent=2))
    return 0 if payload["status"] in {"strong", "acceptable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
