import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["LLM_ADJUDICATION_ENABLED"] = "false"

from backend.database import SessionLocal  # noqa: E402
from backend.models import Patient  # noqa: E402
from backend.schema_migrations import ensure_schema  # noqa: E402
from backend.services.agent_rag import run_patient_agent_pipeline  # noqa: E402

DEFAULT_CASES_PATH = ROOT_DIR / "evals" / "rag_cases.json"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "Data" / "agent_eval" / "rag_eval_report.json"
EVAL_PATIENT_ID = "AGENT-EVAL-P001"


def ensure_eval_patient(db):
    row = db.query(Patient).filter(Patient.id == EVAL_PATIENT_ID).first()
    if row:
        return
    row = Patient(
        id=EVAL_PATIENT_ID,
        name="Agent Eval Patient",
        diagnosis="Breast cancer - doctor-confirmed",
    )
    db.add(row)
    db.commit()


def _contains_terms(reply, required_terms):
    reply_lower = (reply or "").lower()
    missing = [term for term in required_terms if term.lower() not in reply_lower]
    return missing


def run_case(db, case):
    query = case.get("input") or ""
    result = run_patient_agent_pipeline(
        db=db,
        patient_id=EVAL_PATIENT_ID,
        query=query,
        patient_context=case.get("patient_context") or {},
        fallback_response=case.get("fallback_response") or "I can help with monitoring questions.",
        actions=case.get("actions") or [],
        urgent_flags=case.get("urgent_flags") or [],
        preselected_intent=case.get("preselected_intent"),
    )

    citations = result.get("citations") or []
    citation_ids = [item.get("id") for item in citations if item.get("id")]
    expected_sources = case.get("expected_sources") or []
    missing_sources = [source for source in expected_sources if source not in citation_ids]

    required_terms = case.get("required_reply_terms") or []
    missing_terms = _contains_terms(result.get("reply"), required_terms)

    expected_intent = case.get("expected_intent")
    observed_intent = result.get("intent")

    cacheable_expected = case.get("cacheable")
    cacheable_observed = (result.get("cache") or {}).get("cacheable")
    cacheable_match = (cacheable_expected is None) or (cacheable_observed == cacheable_expected)

    passed = (
        observed_intent == expected_intent
        and not missing_sources
        and not missing_terms
        and cacheable_match
    )

    return {
        "id": case.get("id"),
        "category": case.get("category"),
        "query": query,
        "expected": {
            "intent": expected_intent,
            "sources": expected_sources,
            "required_terms": required_terms,
            "cacheable": cacheable_expected,
        },
        "observed": {
            "intent": observed_intent,
            "citation_ids": citation_ids,
            "cacheable": cacheable_observed,
            "cache_status": (result.get("cache") or {}).get("status"),
        },
        "missing": {
            "sources": missing_sources,
            "terms": missing_terms,
        },
        "passed": passed,
        "reply_excerpt": (result.get("reply") or "")[:180],
    }


def main():
    parser = argparse.ArgumentParser(description="Run RAG eval cases from evals/rag_cases.json.")
    parser.add_argument("--cases-path", default=str(DEFAULT_CASES_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()

    cases = json.loads(Path(args.cases_path).read_text(encoding="utf-8")).get("cases") or []
    ensure_schema()
    db = SessionLocal()
    try:
        ensure_eval_patient(db)
        results = [run_case(db, case) for case in cases]
    finally:
        db.close()

    status = "passed" if all(item["passed"] for item in results) else "failed"
    output = {
        "schema_version": "rag_eval_report_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "case_count": len(results),
        "cases": results,
        "claim_boundary": "RAG eval is engineering regression only, not clinical validation.",
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(json.dumps({
        "output_path": str(output_path.relative_to(ROOT_DIR)),
        "status": status,
        "failed_cases": [item["id"] for item in results if not item["passed"]],
    }, indent=2))

    if status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
