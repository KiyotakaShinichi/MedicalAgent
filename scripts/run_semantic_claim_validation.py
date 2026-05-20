from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.semantic_citation_verifier import run_semantic_claim_validation_eval


def main() -> int:
    payload = run_semantic_claim_validation_eval()
    print(json.dumps({
        "status": payload.get("status"),
        "summary": payload.get("summary"),
    }, indent=2))
    return 0 if payload.get("status") in {"strong", "acceptable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

