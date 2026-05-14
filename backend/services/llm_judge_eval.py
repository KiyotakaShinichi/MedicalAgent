from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from backend.services.artifact_manifest import build_artifact_manifest
from backend.services.local_llm import configured_llm_providers, judge_rag_answer_with_local_llm


DEFAULT_RAG_EVAL_PATH = "Data/evals/rag/latest_rag_gold_eval.json"
DEFAULT_OUTPUT_PATH = "Data/evals/llm_judge/latest_llm_judge_eval.json"


def run_llm_judge_eval(
    rag_eval_path: str = DEFAULT_RAG_EVAL_PATH,
    output_path: str | None = DEFAULT_OUTPUT_PATH,
    max_cases: int = 30,
) -> dict:
    providers = configured_llm_providers()
    rag_eval = _load_json(rag_eval_path)
    cases = (rag_eval.get("cases") or [])[: max(1, max_cases)]

    if not providers:
        report = {
            **build_artifact_manifest(dataset_paths={"rag_eval": rag_eval_path}),
            "schema_version": "llm_judge_eval_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "unavailable",
            "message": "LLM adjudication is disabled or no Groq/Ollama provider is configured.",
            "provider": "none",
            "summary": {
                "case_count": len(cases),
                "judged_cases": 0,
                "pass_rate": None,
            },
            "cases": [],
            "claim_boundary": _claim_boundary(),
        }
        _write(output_path, report)
        return report

    rows = []
    for case in cases:
        answer = case.get("reply") or case.get("reply_preview") or ""
        result = judge_rag_answer_with_local_llm(
            case=case,
            answer=answer,
            citations=case.get("citations") or [],
            retrieved_context=case.get("retrieval_context") or [],
        )
        row = _normalize_judge_row(case, result)
        rows.append(row)

    judged = [row for row in rows if row.get("available")]
    pass_rate = _rate(sum(1 for row in judged if row.get("passes")), len(judged))
    coverage_rate = _rate(len(judged), len(cases))
    status = _judge_status(pass_rate=pass_rate, coverage_rate=coverage_rate)
    report = {
        **build_artifact_manifest(dataset_paths={"rag_eval": rag_eval_path}),
        "schema_version": "llm_judge_eval_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "provider": judged[0].get("provider") if judged else providers[0].get("provider"),
        "model": judged[0].get("model") if judged else providers[0].get("model"),
        "summary": {
            "case_count": len(cases),
            "judged_cases": len(judged),
            "coverage_rate": coverage_rate,
            "pass_rate": pass_rate,
            "average_groundedness_score": _avg(judged, "groundedness_score"),
            "average_citation_support_score": _avg(judged, "citation_support_score"),
            "average_refusal_quality_score": _avg(judged, "refusal_quality_score"),
            "unsafe_medical_advice_rate": _rate(sum(1 for row in judged if row.get("unsafe_medical_advice")), len(judged)),
            "failed_cases": [row["case_id"] for row in judged if not row.get("passes")],
        },
        "cases": rows,
        "limitations": [
            "LLM-as-judge is a heuristic quality signal, not ground truth.",
            "Scores can vary by judge model and prompt wording.",
            "This evaluates generated answers and source behavior; it is not clinical validation.",
        ],
        "claim_boundary": _claim_boundary(),
    }
    _write(output_path, report)
    return report


def load_llm_judge_eval(path: str = DEFAULT_OUTPUT_PATH) -> dict:
    report = _load_json(path)
    if report:
        return report
    return {
        "schema_version": "llm_judge_eval_v1",
        "status": "not_generated",
        "message": "No LLM-judge report has been generated yet.",
        "claim_boundary": _claim_boundary(),
    }


def _normalize_judge_row(case: dict, result: dict) -> dict:
    available = bool(result.get("available"))
    return {
        "case_id": case.get("case_id") or case.get("id"),
        "category": case.get("category"),
        "available": available,
        "provider": result.get("provider"),
        "model": result.get("model"),
        "groundedness_score": _float_or_none(result.get("groundedness_score")),
        "citation_support_score": _float_or_none(result.get("citation_support_score")),
        "refusal_quality_score": _float_or_none(result.get("refusal_quality_score")),
        "unsafe_medical_advice": bool(result.get("unsafe_medical_advice")) if available else None,
        "passes": bool(result.get("passes")) if available else False,
        "reason": result.get("reason"),
    }


def _judge_status(pass_rate: float | None, coverage_rate: float | None) -> str:
    if coverage_rate is None or coverage_rate == 0:
        return "unavailable"
    if coverage_rate < 0.8:
        return "partial"
    if pass_rate == 1.0:
        return "strong"
    if pass_rate and pass_rate >= 0.9:
        return "passed"
    return "needs_attention"


def _avg(rows: list[dict], key: str) -> float | None:
    values = [row.get(key) for row in rows if isinstance(row.get(key), (int, float))]
    if not values:
        return None
    return round(float(mean(values)), 4)


def _rate(numerator: int, denominator: int) -> float | None:
    if not denominator:
        return None
    return round(float(numerator / denominator), 4)


def _float_or_none(value) -> float | None:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _claim_boundary() -> str:
    return "Optional LLM-as-judge heuristic for engineering review only; not clinical truth or safety certification."


def _load_json(path: str) -> dict:
    artifact = Path(path)
    if not artifact.exists():
        return {}
    try:
        return json.loads(artifact.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write(path: str | None, payload: dict) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
