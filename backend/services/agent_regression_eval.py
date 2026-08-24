import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import Base
from backend.models import Patient
from backend.services.agent_rag import run_patient_agent_pipeline
from backend.services.agent_regression_eval_utils import (
    numeric as _numeric,
    rate as _rate,
    round_mean as _round_mean,
    source_ids as _source_ids,
    status_meaning as _status_meaning,
)
from backend.services.agent_regression_cases_core import CORE_AGENT_EVAL_CASES
from backend.services.agent_regression_cases_extended import EXTENDED_AGENT_EVAL_CASES

DEFAULT_AGENT_REGRESSION_PATH = "Data/agent_eval/latest_agent_regression.json"
EVALS_DIR = Path(__file__).resolve().parents[2] / "evals"
DEFAULT_RAG_CASES_PATH = str(EVALS_DIR / "rag_cases.json")
DEFAULT_SAFETY_CASES_PATH = str(EVALS_DIR / "safety_cases.json")
EVAL_PATIENT_ID = "AGENT-EVAL-P001"

DEFAULT_AGENT_EVAL_CASES = CORE_AGENT_EVAL_CASES + EXTENDED_AGENT_EVAL_CASES
del CORE_AGENT_EVAL_CASES, EXTENDED_AGENT_EVAL_CASES


def run_agent_regression_suite(output_path=DEFAULT_AGENT_REGRESSION_PATH, cases=None):
    cases = cases or load_eval_cases()
    db, engine = _temp_db_session()
    try:
        _seed_eval_patient(db)
        results = [
            _run_case(db, index=index, case=case)
            for index, case in enumerate(cases, start=1)
        ]
        payload = {
            "schema_version": "agent_regression_eval_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "purpose": (
                "Offline regression suite for patient-agent intent routing, retrieval expectations, "
                "guardrails, grounding, citations, and cost/latency proxies."
            ),
            "case_catalogs": {
                "rag_cases_path": DEFAULT_RAG_CASES_PATH,
                "safety_cases_path": DEFAULT_SAFETY_CASES_PATH,
            },
            "case_count": len(results),
            "summary": _summary(results),
            "quality_gates": _quality_gates(),
            "cases": results,
            "limitations": [
                "This is an engineering regression suite, not clinical validation.",
                "Expected-source checks use the current built-in/local KB snippets until a labeled research-paper KB exists.",
                "Grounding and hallucination scores are lightweight proxies; compare with RAGAS once the KB is populated.",
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


def load_latest_agent_regression_report(path=DEFAULT_AGENT_REGRESSION_PATH):
    report_path = Path(path)
    if not report_path.exists():
        return {
            "status": "unavailable",
            "message": "No agent regression report has been generated yet.",
            "path": str(report_path),
            "quality_gates": _quality_gates(),
        }
    return json.loads(report_path.read_text(encoding="utf-8"))


def load_eval_cases(
    rag_cases_path: str = DEFAULT_RAG_CASES_PATH,
    safety_cases_path: str = DEFAULT_SAFETY_CASES_PATH,
):
    rag_catalog = _load_case_catalog(rag_cases_path)
    safety_catalog = _load_case_catalog(safety_cases_path)
    if not rag_catalog and not safety_catalog:
        return DEFAULT_AGENT_EVAL_CASES

    defaults = {case["id"]: case for case in DEFAULT_AGENT_EVAL_CASES}
    cases = []
    cases.extend(_normalize_rag_cases(rag_catalog, defaults))
    cases.extend(_normalize_safety_cases(safety_catalog, defaults))
    cases = _dedupe_cases(cases)
    return cases or DEFAULT_AGENT_EVAL_CASES


def _load_case_catalog(path: str) -> dict:
    from backend.services.agent_regression_catalog import load_case_catalog

    return load_case_catalog(path, path_type=Path, json_module=json)


def _normalize_rag_cases(catalog: dict, defaults: dict) -> list[dict]:
    from backend.services.agent_regression_catalog import normalize_rag_cases

    return normalize_rag_cases(
        catalog,
        defaults,
        allow_no_citations_fn=_allow_no_citations,
        default_guardrail_status_fn=_default_guardrail_status,
        default_safety_level_fn=_default_safety_level,
    )


def _normalize_safety_cases(catalog: dict, defaults: dict) -> list[dict]:
    from backend.services.agent_regression_catalog import normalize_safety_cases

    return normalize_safety_cases(
        catalog,
        defaults,
        default_guardrail_status_fn=_default_guardrail_status,
        default_safety_level_fn=_default_safety_level,
    )


def _allow_no_citations(category: str | None, expected_intent: str | None) -> bool:
    from backend.services.agent_regression_catalog import allow_no_citations

    return allow_no_citations(category, expected_intent)


def _default_guardrail_status(expected_intent: str | None) -> str:
    from backend.services.agent_regression_catalog import default_guardrail_status

    return default_guardrail_status(expected_intent)


def _default_safety_level(expected_intent: str | None) -> str:
    from backend.services.agent_regression_catalog import default_safety_level

    return default_safety_level(expected_intent)


def _dedupe_cases(cases: list[dict]) -> list[dict]:
    from backend.services.agent_regression_catalog import dedupe_cases

    return dedupe_cases(cases)


def _run_case(db, index, case):
    result = run_patient_agent_pipeline(
        db=db,
        patient_id=EVAL_PATIENT_ID,
        query=case["query"],
        patient_context=case.get("patient_context") or {},
        fallback_response=case.get("fallback_response") or "I can help track this for review.",
        actions=case.get("actions") or [],
        urgent_flags=case.get("urgent_flags") or [],
    )
    evaluation = _evaluate_case(case, result)
    return {
        "index": index,
        "id": case["id"],
        "category": case["category"],
        "query": case["query"],
        "status": "passed" if evaluation["passed"] else "failed",
        "checks": evaluation["checks"],
        "observed": {
            "intent": result.get("intent"),
            "safety_level": (result.get("safety") or {}).get("level"),
            "input_guardrail": ((result.get("guardrails") or {}).get("input") or {}).get("status"),
            "output_guardrail": ((result.get("guardrails") or {}).get("output") or {}).get("status"),
            "cache_status": (result.get("cache") or {}).get("status"),
            "citation_ids": _source_ids(result.get("citations") or []),
            "retrieval_context_ids": _source_ids(result.get("retrieval_context") or []),
            "grounding_score": (((result.get("rag_evaluation") or {}).get("answer_grounding") or {}).get("score")),
            "hallucination_score": (((result.get("rag_evaluation") or {}).get("hallucination") or {}).get("score")),
            "hallucination_risk": (((result.get("rag_evaluation") or {}).get("hallucination") or {}).get("risk")),
            "latency_ms": (((result.get("rag_evaluation") or {}).get("cost_latency") or {}).get("latency_ms")),
            "estimated_total_tokens": (((result.get("rag_evaluation") or {}).get("cost_latency") or {}).get("estimated_total_tokens")),
        },
    }


def _evaluate_case(case, result):
    from backend.services.agent_regression_scoring import evaluate_case

    return evaluate_case(case, result, check_fn=_check, source_ids_fn=_source_ids)


def _summary(results):
    from backend.services.agent_regression_scoring import summary

    return summary(
        results,
        numeric_fn=_numeric,
        rate_fn=_rate,
        round_mean_fn=_round_mean,
        status_meaning_fn=_status_meaning,
        check_rate_fn=_check_rate,
        overall_status_fn=_overall_status,
    )


def _overall_status(metrics):
    from backend.services.agent_regression_scoring import overall_status

    return overall_status(metrics)


def _quality_gates():
    from backend.services.agent_regression_scoring import quality_gates

    return quality_gates()


def _seed_eval_patient(db):
    db.add(Patient(
        id=EVAL_PATIENT_ID,
        name="Agent Evaluation Patient",
        diagnosis="Breast cancer - doctor-confirmed",
    ))
    db.commit()


def _temp_db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return Session(), engine


def _check(name, passed, expected, observed):
    from backend.services.agent_regression_scoring import check

    return check(name, passed, expected, observed)


def _check_rate(results, check_name):
    from backend.services.agent_regression_scoring import check_rate

    return check_rate(results, check_name, rate_fn=_rate)
