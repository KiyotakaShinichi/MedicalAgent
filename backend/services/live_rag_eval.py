"""Live-agent RAG evaluation.

Unlike the fast contract benchmark, this calls ``run_patient_agent_pipeline``
with an in-memory database so retrieval, source governance, claim validation,
post-generation validation, and finalization all execute together.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.services.rag_intent_aware_eval import (
    EVAL_CASES,
    load_canonical_cases,
    run_intent_aware_eval,
)


DEFAULT_OUTPUT_PATH = "Data/evals/rag/latest_live_rag_eval.json"


def run_live_rag_eval(
    *,
    output_path: str = DEFAULT_OUTPUT_PATH,
    case_limit: int = 0,
) -> dict[str, Any]:
    cases = load_canonical_cases() or EVAL_CASES
    if case_limit > 0:
        cases = cases[:case_limit]
    payload = run_intent_aware_eval(
        agent=_live_agent(),
        cases=cases,
        output_path=output_path,
        taglish_parity_path="Data/evals/safety/latest_taglish_safety_parity.json",
    )
    summary = payload.setdefault("summary", {})
    summary.setdefault("escalation_correctness", summary.get("refusal_correctness"))
    payload["schema_version"] = "live_rag_eval_v1"
    payload["eval_type"] = "live_patient_agent_pipeline"
    payload["case_source"] = "canonical_rag_eval_cases" if cases else "phase11_demo"
    payload["claim_boundary"] = (
        "Live-agent RAG evaluation is an engineering regression artifact. "
        "It exercises the real local pipeline but does not establish clinical "
        "medical correctness or patient safety in real care."
    )
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_live_rag_eval(path: str = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {
            "schema_version": "live_rag_eval_v1",
            "status": "missing",
            "message": "Run scripts/run_live_rag_eval.py to generate this artifact.",
            "summary": {},
            "cases": [],
        }
    return json.loads(file_path.read_text(encoding="utf-8"))


def _live_agent():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from backend.models import Base, Patient
    from backend.services.agent_rag import run_patient_agent_pipeline

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    patient_id = "LIVE-RAG-EVAL"
    db.merge(Patient(id=patient_id, name="Live RAG Eval"))
    db.commit()

    def _call(query: str) -> dict[str, Any]:
        try:
            return run_patient_agent_pipeline(
                db=db,
                patient_id=patient_id,
                query=query,
                patient_context={"patient_id": patient_id, "source": "live_rag_eval"},
                fallback_response=(
                    "I can provide monitoring support only. I cannot diagnose, "
                    "predict outcomes, or recommend treatment changes."
                ),
            ) or {}
        except Exception as exc:  # noqa: BLE001 - scored honestly by eval
            return {"intent": None, "reply": "", "error": str(exc)}

    return _call


__all__ = ["DEFAULT_OUTPUT_PATH", "load_live_rag_eval", "run_live_rag_eval"]
