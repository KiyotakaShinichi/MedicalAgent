from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.database import SessionLocal
from backend.models import RAGEvaluationLog
from backend.services.agent_turn_trace import TURN_TRACE_TOP_LEVEL_KEYS, build_turn_trace, validate_trace_payload
from backend.services.oncology_canonical_schema import ROOT_DIR


DEFAULT_OUTPUT_PATH = "Data/evals/ops/latest_trace_diagnostics_coverage.json"

CLAIM_BOUNDARY = (
    "Trace diagnostics store discrete routing and validation decisions only. "
    "They do not store private chain-of-thought and do not establish clinical validation. "
    "Coverage applies to RAG rows written after the trace diagnostics migration; "
    "older historical rows may legitimately have empty trace fields."
)


def build_trace_diagnostics_coverage(
    *,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    db=None,
    limit: int = 100,
) -> dict[str, Any]:
    rows = _load_rows(db=db, limit=limit)
    coverage = [_row_coverage(row) for row in rows]
    rows_with_trace = sum(1 for item in coverage if item["has_trace_diagnostics"])
    rows_with_retrieval = sum(1 for item in coverage if item["has_retrieval_confidence"])
    sample_trace = build_turn_trace(
        model_used={"answer": "deterministic_local_or_untracked"},
        safety_scope={"level": "low_risk", "scope": "education_or_tracking"},
        intent={"deterministic_intent": "education", "route_chosen": "source_governed_rag"},
        retrieval_summary={"answerability_status": "answerable_with_citations", "retrieval_confidence": 0.9},
        post_gen_validator={"decision": "allowed"},
        refusal={"refused": False},
    ).to_dict()
    ok, problems = validate_trace_payload(sample_trace)
    if not ok:
        status = "needs_attention"
    elif rows and (rows_with_trace == 0 or rows_with_retrieval == 0):
        status = "needs_attention"
    else:
        status = "strong"
    payload = {
        "schema_version": "trace_diagnostics_coverage_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "summary": {
            "rows_checked": len(rows),
            "rows_with_trace_diagnostics": rows_with_trace,
            "rows_with_retrieval_confidence": rows_with_retrieval,
            "sample_trace_schema_valid": ok,
            "private_chain_of_thought_allowed": False,
        },
        "required_top_level_keys": sorted(TURN_TRACE_TOP_LEVEL_KEYS),
        "sample_trace_validation_errors": problems,
        "row_coverage": coverage[:50],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _write_json(output_path, payload)
    return payload


def _load_rows(*, db=None, limit: int) -> list[RAGEvaluationLog]:
    owns_session = db is None
    if db is None:
        db = SessionLocal()
    try:
        return (
            db.query(RAGEvaluationLog)
            .order_by(RAGEvaluationLog.created_at.desc(), RAGEvaluationLog.id.desc())
            .limit(max(1, min(limit, 500)))
            .all()
        )
    except Exception:
        return []
    finally:
        if owns_session:
            db.close()


def _row_coverage(row: RAGEvaluationLog) -> dict[str, Any]:
    trace = _loads(getattr(row, "trace_diagnostics_json", None))
    retrieval = _loads(getattr(row, "retrieval_confidence_json", None))
    return {
        "id": row.id,
        "created_at": str(row.created_at) if row.created_at else None,
        "has_trace_diagnostics": isinstance(trace, dict) and bool(trace),
        "has_retrieval_confidence": isinstance(retrieval, dict) and bool(retrieval),
        "trace_schema_valid": validate_trace_payload(trace)[0] if isinstance(trace, dict) and trace else None,
        "answerability_status": retrieval.get("answerability_status") if isinstance(retrieval, dict) else None,
    }


def _loads(value: Any) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except Exception:
        return None


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    candidate = Path(path)
    full_path = candidate if candidate.is_absolute() else ROOT_DIR / candidate
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


__all__ = ["DEFAULT_OUTPUT_PATH", "build_trace_diagnostics_coverage"]
