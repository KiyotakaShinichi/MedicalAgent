"""Loader for the claim-level citation validation eval artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_PATH = "Data/evals/rag/latest_claim_level_citation_eval.json"


def load_claim_level_citation_eval(path: str = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {
            "schema_version": "claim_level_citation_eval_v1",
            "status": "missing",
            "message": "Run scripts/run_rag_claim_validation_eval.py to generate this artifact.",
            "summary": {},
            "cases": [],
        }
    return json.loads(file_path.read_text(encoding="utf-8"))


__all__ = ["DEFAULT_OUTPUT_PATH", "load_claim_level_citation_eval"]
