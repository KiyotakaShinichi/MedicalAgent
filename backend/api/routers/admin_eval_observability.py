"""Admin-only trace replay and system-health API surfaces."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.models import RAGEvaluationLog


def _loads(value: str | None, *, default: Any = None) -> Any:
    try:
        return json.loads(value) if value else default
    except (TypeError, ValueError):
        return default


def build_admin_observability_router(
    get_admin_access_context: Callable,
    get_db: Callable,
) -> APIRouter:
    """Build trace and runtime-observability routes with injected dependencies."""
    router = APIRouter(tags=["admin-evaluation"])

    @router.get("/admin/agent-trace-logs")
    def get_admin_agent_trace_logs_endpoint(
        limit: int = 50,
        context=Depends(get_admin_access_context),
        db: Session = Depends(get_db),
    ):
        """Return the most recent RAG/agent evaluation trace log entries."""
        safe_limit = max(1, min(limit, 200))
        rows = (
            db.query(RAGEvaluationLog)
            .order_by(RAGEvaluationLog.created_at.desc(), RAGEvaluationLog.id.desc())
            .limit(safe_limit)
            .all()
        )
        traces = [
            {
                "id": row.id,
                "patient_id": row.patient_id,
                "request_id": getattr(row, "request_id", None),
                "query_preview": row.query_preview or "(no preview)",
                "intent": row.intent,
                "safety_level": row.safety_level,
                "cache_status": row.cache_status,
                "terminal_step": row.terminal_step,
                "input_guardrail": row.input_guardrail_status,
                "output_guardrail": row.output_guardrail_status,
                "grounding_score": row.grounding_score,
                "hallucination_score": row.hallucination_score,
                "hallucination_risk": row.hallucination_risk,
                "latency_ms": row.latency_ms,
                "estimated_input_tokens": row.estimated_input_tokens,
                "estimated_output_tokens": row.estimated_output_tokens,
                "estimated_total_tokens": row.estimated_total_tokens,
                "estimated_cost_usd": row.estimated_llm_cost_usd,
                "model_used": getattr(row, "model_used", None),
                "stage_latency": _loads(getattr(row, "stage_latency_json", None)),
                "token_usage": _loads(getattr(row, "token_usage_json", None)),
                "retrieved_source_ids": _loads(row.retrieved_source_ids_json, default=[]),
                "cited_source_ids": _loads(row.cited_source_ids_json, default=[]),
                "compound_intent": _loads(getattr(row, "compound_intent_json", None)),
                "created_at": str(row.created_at),
            }
            for row in rows
        ]
        return {
            "count": len(traces),
            "traces": traces,
            "note": (
                "Each entry is one agent/RAG pipeline call. query_preview is truncated at 120 chars. "
                "Provider-reported tokens are identified in token_usage; all other token counts are estimates."
            ),
        }

    @router.get("/admin/rag-trace-replay")
    def get_admin_rag_trace_replay_endpoint(
        limit: int = 25,
        context=Depends(get_admin_access_context),
        db: Session = Depends(get_db),
    ):
        """Return recent RAG traces in the flattened replay-panel contract."""
        safe_limit = max(1, min(limit, 200))
        rows = (
            db.query(RAGEvaluationLog)
            .order_by(RAGEvaluationLog.created_at.desc(), RAGEvaluationLog.id.desc())
            .limit(safe_limit)
            .all()
        )
        traces = [
            {
                "id": row.id,
                "created_at": str(row.created_at) if row.created_at else None,
                "patient_id": row.patient_id,
                "query_preview": row.query_preview,
                "intent": row.intent,
                "safety_level": row.safety_level,
                "rag_mode": row.rag_mode,
                "rewritten_query": row.rewritten_query,
                "retrieved_source_ids": _loads(row.retrieved_source_ids_json, default=[]),
                "cited_source_ids": _loads(row.cited_source_ids_json, default=[]),
                "evidence_grade": _loads(row.evidence_grade_json),
                "claim_validation": _loads(row.claim_validation_json),
                "retrieval_confidence": _loads(
                    getattr(row, "retrieval_confidence_json", None)
                ),
                "trace_diagnostics": _loads(getattr(row, "trace_diagnostics_json", None)),
                "tier_filter": _loads(row.tier_filter_json),
                "post_gen_validator": _loads(row.post_gen_validator_json),
                "grounding_score": row.grounding_score,
                "hallucination_score": row.hallucination_score,
                "latency_ms": row.latency_ms,
                "input_guardrail": row.input_guardrail_status,
                "output_guardrail": row.output_guardrail_status,
            }
            for row in rows
        ]
        return {
            "count": len(traces),
            "traces": traces,
            "trace_coverage_note": (
                "Trace diagnostics and retrieval-confidence fields apply to new RAG rows "
                "written after the trace diagnostics migration; older rows may be blank."
            ),
        }

    @router.get("/admin/system-health")
    def get_admin_system_health_endpoint(
        context=Depends(get_admin_access_context),
        db: Session = Depends(get_db),
    ):
        """Return backend/frontend/artifact/dependency health for the engineering demo."""
        from backend.services.system_health import load_system_health_report

        return load_system_health_report(db=db)

    @router.post("/admin/system-health")
    def run_admin_system_health_endpoint(
        context=Depends(get_admin_access_context),
        db: Session = Depends(get_db),
    ):
        """Rebuild the system-health report."""
        from backend.services.system_health import build_system_health_report

        return {
            "message": "System health report generated.",
            "result": build_system_health_report(db=db),
        }

    return router

