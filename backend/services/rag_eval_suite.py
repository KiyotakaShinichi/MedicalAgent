from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import Base
from backend.models import Patient
from backend.services.agent_rag import rewrite_and_decompose, run_patient_agent_pipeline


DEFAULT_OUTPUT_PATH = "Data/evals/rag/latest_rag_eval.json"
DEFAULT_CASES_PATH = "evals/rag_eval_cases.json"
EVAL_PATIENT_ID = "RAG-EVAL-P001"


def run_rag_eval_suite(output_path: str | None = DEFAULT_OUTPUT_PATH, cases: list[dict] | None = None) -> dict:
    cases = cases or load_rag_eval_cases()
    db, engine = _temp_db_session()
    try:
        _seed_eval_patient(db)
        results = [_run_case(db, index=index, case=case) for index, case in enumerate(cases, start=1)]
        summary = _summary(results)
        payload = {
            "schema_version": "rag_eval_suite_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "purpose": "RAG evaluation suite for citation coverage, grounding, retrieval quality, refusals, and rewrite quality.",
            "case_count": len(results),
            "summary": summary,
            "cases": results,
            "limitations": [
                "Metrics are heuristic proxies until labeled KB examples exist.",
                "Case catalog is synthetic and does not imply clinical validation.",
            ],
        }
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
    finally:
        db.close()
        engine.dispose()


def load_rag_eval_cases(path: str = DEFAULT_CASES_PATH) -> list[dict]:
    case_path = Path(path)
    if not case_path.exists():
        return []
    try:
        catalog = json.loads(case_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return catalog.get("cases") or []


def _run_case(db, index: int, case: dict) -> dict:
    query = case.get("input") or ""
    fallback = case.get("fallback_response") or "I can help with general monitoring and clinician review."
    result = run_patient_agent_pipeline(
        db=db,
        patient_id=EVAL_PATIENT_ID,
        query=query,
        patient_context={"patient_id": EVAL_PATIENT_ID},
        fallback_response=fallback,
        actions=[],
        urgent_flags=[],
    )

    citations = result.get("citations") or []
    citation_ids = [item.get("id") for item in citations if isinstance(item, dict)]
    retrieved = result.get("retrieval_context") or []
    retrieved_ids = [item.get("id") for item in retrieved if isinstance(item, dict)]

    expected_sources = case.get("expected_sources") or []
    expected_hit = _expected_source_hit(expected_sources, citation_ids, retrieved_ids)
    requires_citations = bool(case.get("requires_citations"))
    citation_present = bool(citations)
    citation_ok = (not requires_citations) or citation_present

    rag_eval = result.get("rag_evaluation") or {}
    grounding = (rag_eval.get("answer_grounding") or {}).get("score")
    hallucination = (rag_eval.get("hallucination") or {}).get("score")
    precision_at_3 = (rag_eval.get("retrieval_precision_at_3") or {}).get("value")

    expected_keywords = case.get("expected_context_keywords") or []
    domain_relevance = _domain_keyword_hit(expected_keywords, result, retrieved)

    requires_refusal = bool(case.get("requires_refusal"))
    refusal_ok = _refusal_correct(requires_refusal, result)

    intent = result.get("intent") or case.get("expected_intent")
    rewrite = rewrite_and_decompose(query, intent or "general_support")
    rewrite_terms = case.get("expected_rewrite_terms") or []
    rewrite_ok = _rewrite_term_hit(rewrite_terms, rewrite.get("expanded_query") or "")

    checks = [
        _check("citation_coverage", citation_ok, requires_citations, citation_present),
        _check("expected_source_hit", expected_hit, expected_sources, {"citations": citation_ids, "retrieved": retrieved_ids}),
        _check("domain_relevance", domain_relevance, expected_keywords, domain_relevance),
    ]
    if requires_refusal:
        checks.append(_check("refusal_correct", refusal_ok, True, refusal_ok))
    if rewrite_terms:
        checks.append(_check("rewrite_quality", rewrite_ok, rewrite_terms, rewrite.get("expanded_query")))

    passed = all(check["passed"] for check in checks)

    return {
        "case_id": case.get("id") or f"case_{index}",
        "input": query,
        "intent": intent,
        "expected_sources": expected_sources,
        "requires_citations": requires_citations,
        "requires_refusal": requires_refusal,
        "metrics": {
            "citation_present": citation_present,
            "expected_source_hit": expected_hit,
            "grounding_score": grounding,
            "hallucination_score": hallucination,
            "retrieval_precision_at_3": precision_at_3,
            "domain_relevance": domain_relevance,
            "rewrite_subquery_count": len(rewrite.get("subqueries") or []),
            "rewrite_term_hit": rewrite_ok if rewrite_terms else None,
        },
        "checks": checks,
        "pass": passed,
        "reply_preview": (result.get("reply") or "")[:220],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _summary(results: list[dict]) -> dict:
    citation_cases = [row for row in results if row.get("requires_citations")]
    refusal_cases = [row for row in results if row.get("requires_refusal")]
    rewrite_cases = [row for row in results if (row.get("metrics") or {}).get("rewrite_term_hit") is not None]

    def metric_values(key: str):
        return [row["metrics"].get(key) for row in results if row.get("metrics")]

    return {
        "status": _overall_status(results),
        "pass_rate": _rate(sum(1 for row in results if row.get("pass")), len(results)),
        "citation_coverage_rate": _rate(sum(1 for row in citation_cases if row["metrics"].get("citation_present")), len(citation_cases)),
        "expected_source_hit_rate": _rate(sum(1 for row in results if row["metrics"].get("expected_source_hit")), len(results)),
        "refusal_correct_rate": _rate(sum(1 for row in refusal_cases if _check_pass(row, "refusal_correct")), len(refusal_cases)),
        "domain_relevance_rate": _rate(sum(1 for row in results if row["metrics"].get("domain_relevance")), len(results)),
        "rewrite_term_hit_rate": _rate(sum(1 for row in rewrite_cases if row["metrics"].get("rewrite_term_hit")), len(rewrite_cases)),
        "average_grounding_score": _round_mean(metric_values("grounding_score")),
        "average_hallucination_score": _round_mean(metric_values("hallucination_score")),
        "average_retrieval_precision_at_3": _round_mean(metric_values("retrieval_precision_at_3")),
    }


def _expected_source_hit(expected_sources: list[str], citation_ids: list[str], retrieved_ids: list[str]) -> bool:
    if not expected_sources:
        return True
    expected = set(expected_sources)
    return bool(expected & set(citation_ids)) or bool(expected & set(retrieved_ids))


def _domain_keyword_hit(expected_keywords: list[str], result: dict, retrieved: list[dict]) -> bool:
    if not expected_keywords:
        return True
    reply = (result.get("reply") or "").lower()
    context = " ".join(item.get("text", "") for item in retrieved).lower()
    return any(keyword.lower() in reply or keyword.lower() in context for keyword in expected_keywords)


def _refusal_correct(requires_refusal: bool, result: dict) -> bool:
    if not requires_refusal:
        return True
    intent = result.get("intent") or ""
    reply = (result.get("reply") or "").lower()
    if intent in {"security_boundary", "safety_boundary", "treatment_decision_boundary"}:
        return True
    if any(term in reply for term in ["cannot", "not able", "clinician", "care team", "not a diagnosis"]):
        return True
    return False


def _rewrite_term_hit(expected_terms: list[str], expanded_query: str) -> bool:
    if not expected_terms:
        return True
    expanded = (expanded_query or "").lower()
    return all(term.lower() in expanded for term in expected_terms)


def _check(name, passed, expected, observed):
    return {
        "name": name,
        "passed": bool(passed),
        "expected": expected,
        "observed": observed,
    }


def _check_pass(row: dict, check_name: str) -> bool:
    for check in row.get("checks") or []:
        if check.get("name") == check_name:
            return bool(check.get("passed"))
    return False


def _overall_status(results: list[dict]) -> str:
    pass_rate = _rate(sum(1 for row in results if row.get("pass")), len(results)) or 0.0
    if pass_rate >= 0.95:
        return "strong"
    if pass_rate >= 0.85:
        return "acceptable"
    if pass_rate >= 0.7:
        return "unideal"
    return "failed"


def _round_mean(values):
    numeric = [float(value) for value in values if value is not None]
    return round(mean(numeric), 3) if numeric else None


def _rate(numerator: int, denominator: int) -> float | None:
    if not denominator:
        return None
    return round(numerator / denominator, 3)


def _temp_db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return Session(), engine


def _seed_eval_patient(db):
    db.add(Patient(
        id=EVAL_PATIENT_ID,
        name="RAG Evaluation Patient",
        diagnosis="Breast cancer - doctor-confirmed",
    ))
    db.commit()
