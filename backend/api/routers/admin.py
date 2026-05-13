"""
Admin router — analytics, evaluation reports, task queue, MLOps, and registry endpoints.

All routes require admin role via get_admin_access_context.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.api.deps import get_admin_access_context, get_db

router = APIRouter(prefix="/admin", tags=["admin"])


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

    return build_admin_analytics(db)


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
