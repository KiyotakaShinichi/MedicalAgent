"""Run the intent-aware RAG benchmark against the live agent."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.rag_intent_aware_eval import (
    DEFAULT_OUTPUT_PATH,
    run_intent_aware_eval,
)


def _live_agent_factory():
    """Build an agent callable that runs the production RAG pipeline.
    Lazy-imported so the script doesn't pull the full stack when only
    inspecting the eval contract."""
    from backend.services.agent_rag import run_agent_rag

    def _call(query: str) -> dict:
        try:
            return run_agent_rag(db=None, patient_id=None, query=query) or {}
        except Exception as exc:  # noqa: BLE001 — degraded result is fine
            return {"intent": None, "reply": "", "error": str(exc)}

    return _call


if __name__ == "__main__":
    payload = run_intent_aware_eval(
        agent=_live_agent_factory(),
        output_path=DEFAULT_OUTPUT_PATH,
        taglish_parity_path="Data/evals/safety/latest_taglish_safety_parity.json",
    )
    summary = payload.get("summary", {})
    print(json.dumps({
        "status": payload.get("status"),
        "pass_rate": summary.get("pass_rate"),
        "claim_support_rate": summary.get("claim_support_rate"),
        "citation_precision": summary.get("citation_precision"),
        "source_tier_correctness": summary.get("source_tier_correctness"),
        "refusal_correctness": summary.get("refusal_correctness"),
        "unsafe_answer_rate": summary.get("unsafe_answer_rate"),
        "taglish_safety_parity_rate": summary.get("taglish_safety_parity_rate"),
        "latency_p50_ms": summary.get("latency_p50_ms"),
    }, indent=2))
    sys.exit(0 if payload.get("status") in {"strong", "acceptable"} else 1)
