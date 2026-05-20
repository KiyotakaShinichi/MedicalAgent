from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.semantic_citation_verifier import run_semantic_citation_verification_eval


if __name__ == "__main__":
    report = run_semantic_citation_verification_eval()
    print(report["status"])
