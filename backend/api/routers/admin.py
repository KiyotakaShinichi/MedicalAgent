"""
Admin router — analytics, evaluation reports, task queue, MLOps, and registry endpoints.

All routes require admin role via get_admin_access_context.
"""

from __future__ import annotations

import time
import json

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.api.deps import get_admin_access_context, get_db

router = APIRouter(prefix="/admin", tags=["admin"])

_ADMIN_ANALYTICS_CACHE_TTL_SECONDS = 120
_ADMIN_ANALYTICS_CACHE: dict[str, object] = {"expires_at": 0.0, "payload_json": None}


# ─── Request models ───────────────────────────────────────────────────────────

class EvaluationReportRequest(BaseModel):
    output_root: str = "Data/model_evaluation_reports"
    run_id: str | None = None


class AsyncTaskRequest(BaseModel):
    task_type: str
    payload: dict | None = None


# ─── Analytics ────────────────────────────────────────────────────────────────

@router.get("/analytics")
def get_admin_analytics(
    context=Depends(get_admin_access_context),
    db: Session = Depends(get_db),
):
    from backend.services.admin_analytics import build_admin_analytics

    now = time.time()
    if _ADMIN_ANALYTICS_CACHE["payload_json"] is not None and float(_ADMIN_ANALYTICS_CACHE["expires_at"]) > now:
        return Response(
            content=str(_ADMIN_ANALYTICS_CACHE["payload_json"]),
            media_type="application/json",
            headers={"X-Analytics-Cache": "hit"},
        )

    payload = build_admin_analytics(db)
    payload_json = json.dumps(payload, default=str, separators=(",", ":"))
    _ADMIN_ANALYTICS_CACHE["payload_json"] = payload_json
    _ADMIN_ANALYTICS_CACHE["expires_at"] = now + _ADMIN_ANALYTICS_CACHE_TTL_SECONDS
    return Response(
        content=payload_json,
        media_type="application/json",
        headers={"X-Analytics-Cache": "miss"},
    )


# ─── Evaluation reports ───────────────────────────────────────────────────────

@router.post("/evaluation-report")
def generate_admin_evaluation_report(
    payload: EvaluationReportRequest,
    context=Depends(get_admin_access_context),
    db: Session = Depends(get_db),
):
    from backend.services.evaluation_reports import generate_versioned_evaluation_report

    return {
        "message": "Versioned evaluation report generated.",
        "result": generate_versioned_evaluation_report(
            db=db,
            output_root=payload.output_root,
            run_id=payload.run_id,
        ),
    }


@router.post("/training-evaluation-report")
def generate_admin_training_evaluation_report(
    context=Depends(get_admin_access_context),
):
    from backend.services.detailed_training_report import generate_detailed_training_report

    return {
        "message": "Detailed training evaluation report generated.",
        "result": generate_detailed_training_report(),
    }


@router.get("/training-evaluation-report")
def get_admin_training_evaluation_report(
    context=Depends(get_admin_access_context),
):
    from backend.services.detailed_training_report import generate_detailed_training_report

    return {
        "message": "Detailed training evaluation report loaded.",
        "result": generate_detailed_training_report(),
    }


@router.post("/locked-holdout-evaluation")
def generate_admin_locked_holdout_evaluation(
    context=Depends(get_admin_access_context),
):
    from backend.services.locked_holdout_evaluation import evaluate_locked_holdout

    return {
        "message": "Locked holdout evaluation generated.",
        "result": evaluate_locked_holdout(),
    }


@router.get("/locked-holdout-evaluation")
def get_admin_locked_holdout_evaluation(
    context=Depends(get_admin_access_context),
):
    from backend.services.locked_holdout_evaluation import evaluate_locked_holdout

    return {
        "message": "Locked holdout evaluation loaded.",
        "result": evaluate_locked_holdout(),
    }


@router.post("/external-validation")
def generate_admin_external_validation(
    context=Depends(get_admin_access_context),
):
    from backend.services.external_validation_report import build_external_validation_report

    return {
        "message": "External validation report generated.",
        "result": build_external_validation_report(),
    }


@router.get("/external-validation")
def get_admin_external_validation(
    context=Depends(get_admin_access_context),
):
    from backend.services.external_validation_report import build_external_validation_report

    return {
        "message": "External validation report loaded.",
        "result": build_external_validation_report(),
    }


@router.post("/model-comparison")
def generate_admin_model_comparison(
    context=Depends(get_admin_access_context),
):
    from backend.services.model_comparison_report import build_model_comparison_report

    return {
        "message": "Model comparison report generated.",
        "result": build_model_comparison_report(),
    }


@router.get("/model-comparison")
def get_admin_model_comparison(
    context=Depends(get_admin_access_context),
):
    from backend.services.model_comparison_report import build_model_comparison_report

    return {
        "message": "Model comparison report loaded.",
        "result": build_model_comparison_report(),
    }


# ─── MLOps ────────────────────────────────────────────────────────────────────

@router.get("/mlops-runs")
def get_admin_mlops_runs(
    limit: int = 50,
    context=Depends(get_admin_access_context),
    db: Session = Depends(get_db),
):
    from backend.services.mlops_tracking import list_experiment_runs

    safe_limit = max(1, min(limit, 200))
    return {
        "runs": list_experiment_runs(db=db, limit=safe_limit),
        "purpose": "Local account-free experiment tracking for params, metrics, artifacts, hashes, and run status.",
    }


# ─── Inference / LLM ──────────────────────────────────────────────────────────

@router.get("/inference-service")
def get_admin_inference_service(
    context=Depends(get_admin_access_context),
):
    from backend.services.inference_service import describe_inference_service

    return describe_inference_service()


@router.get("/llm-adjudication")
def get_admin_llm_adjudication(
    context=Depends(get_admin_access_context),
):
    from backend.services.local_llm import describe_llm_adjudication

    return describe_llm_adjudication()


# ─── Compound-intent live probe ──────────────────────────────────────────────


class IntentProbeRequest(BaseModel):
    """Body for the /admin/intent-classifier-probe endpoint.

    ``message`` is the raw user input.  ``use_llm`` defaults to True so
    the operator sees the merged deterministic + LLM verdict; setting
    it to False gives the hermetic deterministic-only result (useful
    for comparing the two).
    """
    message: str
    use_llm: bool = True


@router.post("/intent-classifier-probe")
def post_admin_intent_classifier_probe(
    payload: IntentProbeRequest,
    context=Depends(get_admin_access_context),
):
    """Live-probe the compound-intent router on an arbitrary message.

    Returns:
      - ``deterministic`` : envelope produced by the rule-based path
        (table + regex), tells you what the heuristic alone would do.
      - ``merged``        : envelope after merging with the LLM verdict
        (or identical to ``deterministic`` when LLM is unavailable).
      - ``llm``           : raw LLM verdict (language, confidence,
        provider, model) — or ``{"available": False, ...}``.

    This endpoint does NOT touch the chat database; it's a stateless
    probe.  Useful for debugging multilingual routing without sending
    a real chat.
    """
    from backend.services.compound_intent_router import (
        detect_compound_intents,
        detect_compound_intents_with_llm,
    )

    message = (payload.message or "").strip()
    if not message:
        return {
            "status": "empty",
            "deterministic": detect_compound_intents("").to_dict(),
            "merged": detect_compound_intents("").to_dict(),
            "llm": {"available": False, "reason": "empty_message"},
        }

    deterministic = detect_compound_intents(message)
    if payload.use_llm:
        merged, raw = detect_compound_intents_with_llm(message)
        llm_payload = (
            raw if raw is not None
            else {"available": False, "reason": "llm_unavailable_or_disabled"}
        )
    else:
        merged = deterministic
        llm_payload = {"available": False, "reason": "use_llm_false"}

    return {
        "status": "ok",
        "message": message,
        "deterministic": deterministic.to_dict(),
        "merged": merged.to_dict(),
        "llm": llm_payload,
    }


# ─── Fast-mode runtime toggle ────────────────────────────────────────────────


class FastModeToggleRequest(BaseModel):
    """Body for the FAST_MODE toggle endpoint.

    ``enabled``:
      - True   -> force fast mode ON  (skip LLM adjudication)
      - False  -> force fast mode OFF (re-enable LLM adjudication)
      - None / omitted -> CLEAR the runtime override, fall back to the
        ``ONCOTRACK_FAST_MODE`` env var.
    """
    enabled: bool | None = None


@router.get("/fast-mode")
def get_admin_fast_mode(
    context=Depends(get_admin_access_context),
):
    """Current FAST_MODE state.

    Returns the resolved boolean, plus the env-var source and any
    runtime override, so an operator can see WHY fast mode is in
    whatever state it's in.
    """
    from backend.services.local_llm import fast_mode_status

    return fast_mode_status()


@router.post("/fast-mode")
def post_admin_fast_mode(
    payload: FastModeToggleRequest,
    context=Depends(get_admin_access_context),
):
    """Flip the FAST_MODE runtime override.

    This is the emergency-degradation switch.  Use ONLY when the Groq
    cloud provider is degraded / rate-limiting / down, or when a local
    Ollama fallback is misconfigured and timing out.  The deterministic
    safety stack still enforces every refusal / boundary / claim
    contract when FAST_MODE is on; what you lose is the LLM "second
    opinion" on open-ended branches (general_support / education /
    security adjudication).
    """
    from backend.services.local_llm import fast_mode_status, set_fast_mode_override

    set_fast_mode_override(payload.enabled)
    return fast_mode_status()


# ─── Async task queue ─────────────────────────────────────────────────────────

@router.post("/tasks")
def enqueue_admin_task(
    payload: AsyncTaskRequest,
    context=Depends(get_admin_access_context),
    db: Session = Depends(get_db),
):
    from backend.services.task_queue import enqueue_task

    try:
        task = enqueue_task(db, task_type=payload.task_type, payload=payload.payload or {}, created_by=context.role)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"message": "Task queued.", "task": task}


@router.get("/tasks")
def list_admin_tasks(
    limit: int = 50,
    context=Depends(get_admin_access_context),
    db: Session = Depends(get_db),
):
    from backend.services.task_queue import list_tasks

    safe_limit = max(1, min(limit, 200))
    return {"tasks": list_tasks(db, limit=safe_limit)}


@router.post("/tasks/{task_id}/run")
def run_admin_task(
    task_id: int,
    context=Depends(get_admin_access_context),
    db: Session = Depends(get_db),
):
    from backend.services.task_queue import run_task

    try:
        return {"task": run_task(db, task_id)}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/tasks/run-next")
def run_next_admin_task(
    context=Depends(get_admin_access_context),
    db: Session = Depends(get_db),
):
    from backend.services.task_queue import run_next_queued_task

    task = run_next_queued_task(db)
    return {"task": task, "message": "No queued tasks." if task is None else "Task completed."}


# ─── RAG source registry ──────────────────────────────────────────────────────

@router.get("/rag-source-registry")
def get_admin_rag_source_registry(
    context=Depends(get_admin_access_context),
):
    from backend.services.rag_source_registry import build_rag_source_registry

    return build_rag_source_registry()
