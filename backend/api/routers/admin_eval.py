from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session


def build_admin_eval_router(get_admin_access_context: Callable, get_db: Callable) -> APIRouter:
    router = APIRouter(tags=["admin-evaluation"])

    @router.post("/admin/agent-regression")
    def run_admin_agent_regression_endpoint(
        context=Depends(get_admin_access_context),
    ):
        from backend.services.agent_regression_eval import run_agent_regression_suite

        return {
            "message": "Agent regression suite completed.",
            "result": run_agent_regression_suite(),
        }

    @router.post("/admin/mle-readiness")
    def run_admin_mle_readiness_endpoint(
        context=Depends(get_admin_access_context),
        db: Session = Depends(get_db),
    ):
        from backend.services.mle_readiness import DEFAULT_OUTPUT_PATH, build_mle_readiness_summary

        return {
            "message": "MLE readiness checks completed.",
            "result": build_mle_readiness_summary(db=db, output_path=DEFAULT_OUTPUT_PATH),
        }

    @router.get("/admin/agent-trace-logs")
    def get_admin_agent_trace_logs_endpoint(
        limit: int = 50,
        context=Depends(get_admin_access_context),
        db: Session = Depends(get_db),
    ):
        """Return the most recent RAG/agent evaluation trace log entries."""
        import json as _json

        from backend.models import RAGEvaluationLog

        safe_limit = max(1, min(limit, 200))
        rows = (
            db.query(RAGEvaluationLog)
            .order_by(RAGEvaluationLog.created_at.desc(), RAGEvaluationLog.id.desc())
            .limit(safe_limit)
            .all()
        )

        def _loads(value):
            try:
                return _json.loads(value) if value else []
            except Exception:
                return []

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
                "estimated_total_tokens": row.estimated_total_tokens,
                "retrieved_source_ids": _loads(row.retrieved_source_ids_json),
                "cited_source_ids": _loads(row.cited_source_ids_json),
                "created_at": str(row.created_at),
            }
            for row in rows
        ]
        return {
            "count": len(traces),
            "traces": traces,
            "note": "Each entry is one agent/RAG pipeline call. query_preview is truncated at 120 chars.",
        }

    @router.get("/admin/noise-eval")
    def get_admin_noise_eval_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return noise robustness evaluation results from a saved artifact or computed fallback."""
        import json as _json
        from pathlib import Path

        from backend.services.noise_eval import DEFAULT_NOISE_EVAL_PATH, run_noise_eval

        saved = Path(DEFAULT_NOISE_EVAL_PATH)
        if saved.exists():
            try:
                return _json.loads(saved.read_text(encoding="utf-8"))
            except Exception:
                pass
        return run_noise_eval()

    @router.post("/admin/noise-eval")
    def run_admin_noise_eval_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Re-run noise robustness evaluation and persist the artifact."""
        from backend.services.noise_eval import DEFAULT_NOISE_EVAL_PATH, run_noise_eval

        return {"message": "Noise eval completed.", "result": run_noise_eval(output_path=DEFAULT_NOISE_EVAL_PATH)}

    @router.get("/admin/temporal-eval")
    def get_admin_temporal_eval_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return temporal generalization evaluation results."""
        import json as _json
        from pathlib import Path

        from backend.services.temporal_eval import DEFAULT_TEMPORAL_EVAL_PATH, run_temporal_eval

        saved = Path(DEFAULT_TEMPORAL_EVAL_PATH)
        if saved.exists():
            try:
                return _json.loads(saved.read_text(encoding="utf-8"))
            except Exception:
                pass
        return run_temporal_eval()

    @router.post("/admin/temporal-eval")
    def run_admin_temporal_eval_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Re-run temporal eval and persist."""
        from backend.services.temporal_eval import DEFAULT_TEMPORAL_EVAL_PATH, run_temporal_eval

        return {"message": "Temporal eval completed.", "result": run_temporal_eval(output_path=DEFAULT_TEMPORAL_EVAL_PATH)}

    @router.get("/admin/prediction-error-table")
    def get_admin_prediction_error_table_endpoint(
        limit: int = 100,
        context=Depends(get_admin_access_context),
    ):
        """Return per-prediction ML error table with TP/FP/TN/FN classification."""
        from backend.services.prediction_error_table import build_prediction_error_table

        safe_limit = max(10, min(limit, 120))
        return build_prediction_error_table(limit=safe_limit)

    @router.get("/admin/rag-ablation")
    def get_admin_rag_ablation_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return cached RAG ablation study or compute fresh."""
        import json as _json
        from pathlib import Path

        from backend.services.rag_ablation import ABLATION_OUTPUT_PATH, run_rag_ablation

        saved = Path(ABLATION_OUTPUT_PATH)
        if saved.exists():
            try:
                return _json.loads(saved.read_text(encoding="utf-8"))
            except Exception:
                pass
        return run_rag_ablation()

    @router.post("/admin/rag-ablation")
    def run_admin_rag_ablation_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Re-run RAG ablation study and persist artifact."""
        from backend.services.rag_ablation import ABLATION_OUTPUT_PATH, run_rag_ablation

        return {"message": "RAG ablation completed.", "result": run_rag_ablation(output_path=ABLATION_OUTPUT_PATH)}

    @router.get("/admin/summary-quality")
    def get_admin_summary_quality_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return cached summary quality evaluation or compute fresh."""
        import json as _json
        from pathlib import Path

        from backend.services.summary_quality_eval import DEFAULT_OUTPUT_PATH, build_summary_quality_report

        saved = Path(DEFAULT_OUTPUT_PATH)
        if saved.exists():
            try:
                return _json.loads(saved.read_text(encoding="utf-8"))
            except Exception:
                pass
        return build_summary_quality_report()

    @router.post("/admin/summary-quality")
    def run_admin_summary_quality_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Re-run summary quality evaluation and persist."""
        from backend.services.summary_quality_eval import DEFAULT_OUTPUT_PATH, build_summary_quality_report

        return {
            "message": "Summary quality evaluation completed.",
            "result": build_summary_quality_report(output_path=DEFAULT_OUTPUT_PATH),
        }

    @router.get("/admin/public-data-manifest")
    def get_admin_public_data_manifest_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return public-data feasibility, source lineage, and dataset-use limitations."""
        import json as _json
        from pathlib import Path

        from backend.services.public_data_manifest import DEFAULT_OUTPUT_PATH, build_public_data_manifest

        saved = Path(DEFAULT_OUTPUT_PATH)
        if saved.exists():
            try:
                return _json.loads(saved.read_text(encoding="utf-8"))
            except Exception:
                pass
        return build_public_data_manifest(output_path=DEFAULT_OUTPUT_PATH)

    @router.post("/admin/public-data-manifest")
    def run_admin_public_data_manifest_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Rebuild public-data feasibility and source lineage artifact."""
        from backend.services.public_data_manifest import DEFAULT_OUTPUT_PATH, build_public_data_manifest

        return {
            "message": "Public data manifest rebuilt.",
            "result": build_public_data_manifest(output_path=DEFAULT_OUTPUT_PATH),
        }

    @router.get("/admin/public-imaging-manifest")
    def get_admin_public_imaging_manifest_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return local public-imaging dataset availability and experiment readiness."""
        import json as _json
        from pathlib import Path

        from backend.services.public_imaging_datasets import DEFAULT_OUTPUT_PATH, build_public_imaging_manifest

        saved = Path(DEFAULT_OUTPUT_PATH)
        if saved.exists():
            try:
                return _json.loads(saved.read_text(encoding="utf-8"))
            except Exception:
                pass
        return build_public_imaging_manifest(output_path=DEFAULT_OUTPUT_PATH)

    @router.post("/admin/public-imaging-manifest")
    def run_admin_public_imaging_manifest_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Rebuild public-imaging dataset availability and readiness artifact."""
        from backend.services.public_imaging_datasets import DEFAULT_OUTPUT_PATH, build_public_imaging_manifest

        return {
            "message": "Public imaging manifest rebuilt.",
            "result": build_public_imaging_manifest(output_path=DEFAULT_OUTPUT_PATH),
        }

    @router.get("/admin/ultrasound-baseline")
    def get_admin_ultrasound_baseline_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return public ultrasound baseline metrics or an explicit unavailable artifact."""
        import json as _json
        from pathlib import Path

        from backend.services.imaging_baseline_experiments import DEFAULT_ULTRASOUND_OUTPUT_PATH, run_ultrasound_baseline

        saved = Path(DEFAULT_ULTRASOUND_OUTPUT_PATH)
        if saved.exists():
            try:
                return _json.loads(saved.read_text(encoding="utf-8"))
            except Exception:
                pass
        return run_ultrasound_baseline()

    @router.post("/admin/ultrasound-baseline")
    def run_admin_ultrasound_baseline_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Run public breast ultrasound baseline if dataset files are available."""
        from backend.services.imaging_baseline_experiments import DEFAULT_ULTRASOUND_OUTPUT_PATH, run_ultrasound_baseline

        return {
            "message": "Ultrasound baseline completed.",
            "result": run_ultrasound_baseline(output_path=DEFAULT_ULTRASOUND_OUTPUT_PATH),
        }

    @router.get("/admin/ct-lesion-workflow")
    def get_admin_ct_lesion_workflow_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return CT/PET-CT lesion workflow readiness report."""
        import json as _json
        from pathlib import Path

        from backend.services.imaging_baseline_experiments import DEFAULT_CT_WORKFLOW_PATH, build_ct_lesion_workflow_report

        saved = Path(DEFAULT_CT_WORKFLOW_PATH)
        if saved.exists():
            try:
                return _json.loads(saved.read_text(encoding="utf-8"))
            except Exception:
                pass
        return build_ct_lesion_workflow_report(output_path=DEFAULT_CT_WORKFLOW_PATH)

    @router.post("/admin/ct-lesion-workflow")
    def run_admin_ct_lesion_workflow_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Rebuild CT/PET-CT lesion workflow readiness report."""
        from backend.services.imaging_baseline_experiments import DEFAULT_CT_WORKFLOW_PATH, build_ct_lesion_workflow_report

        return {
            "message": "CT lesion workflow report completed.",
            "result": build_ct_lesion_workflow_report(output_path=DEFAULT_CT_WORKFLOW_PATH),
        }

    @router.get("/admin/sim-to-public-imaging")
    def get_admin_sim_to_public_imaging_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return synthetic-to-public imaging gap report."""
        import json as _json
        from pathlib import Path

        from backend.services.sim_to_public_imaging_report import DEFAULT_OUTPUT_PATH, build_sim_to_public_imaging_report

        saved = Path(DEFAULT_OUTPUT_PATH)
        if saved.exists():
            try:
                return _json.loads(saved.read_text(encoding="utf-8"))
            except Exception:
                pass
        return build_sim_to_public_imaging_report(output_path=DEFAULT_OUTPUT_PATH)

    @router.post("/admin/sim-to-public-imaging")
    def run_admin_sim_to_public_imaging_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Rebuild synthetic-to-public imaging gap report."""
        from backend.services.sim_to_public_imaging_report import DEFAULT_OUTPUT_PATH, build_sim_to_public_imaging_report

        return {
            "message": "Synthetic-to-public imaging gap report completed.",
            "result": build_sim_to_public_imaging_report(output_path=DEFAULT_OUTPUT_PATH),
        }

    return router
