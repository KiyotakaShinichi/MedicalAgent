from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("RAG_FORCE_SPARSE", "true")

from backend.services.live_rag_eval import run_live_rag_eval


def main() -> int:
    payload = run_live_rag_eval()
    summary = payload.get("summary", {})
    print(json.dumps({
        "status": payload.get("status"),
        "case_count": summary.get("case_count"),
        "pass_rate": summary.get("pass_rate"),
        "claim_support_rate": summary.get("claim_support_rate"),
        "citation_precision": summary.get("citation_precision"),
        "source_tier_correctness": summary.get("source_tier_correctness"),
        "refusal_correctness": summary.get("refusal_correctness"),
        "escalation_correctness": summary.get("escalation_correctness"),
        "unsafe_answer_rate": summary.get("unsafe_answer_rate"),
        "taglish_safety_parity_rate": summary.get("taglish_safety_parity_rate"),
        "latency_p50_ms": summary.get("latency_p50_ms"),
    }, indent=2))
    return 0 if payload.get("status") in {"strong", "acceptable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
