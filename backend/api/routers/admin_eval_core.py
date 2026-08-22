"""Build core readiness, robustness, safety, and drift evaluation routes."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session


def build_admin_eval_core_router(
    get_admin_access_context: Callable,
    get_db: Callable,
) -> APIRouter:
    """Compose core readiness, robustness, safety, and drift evaluation routes."""
    router = APIRouter()

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

    @router.get("/admin/safety-center")
    def get_admin_safety_center_endpoint(
        context=Depends(get_admin_access_context),
        db: Session = Depends(get_db),
    ):
        """Return the unified safety/evaluation center artifact bundle."""
        from backend.services.safety_eval_center import build_safety_evaluation_center

        return build_safety_evaluation_center(db=db)

    @router.get("/admin/safety-red-team")
    def get_admin_safety_red_team_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return cached safety red-team artifact or compute fast offline fallback."""
        import json as _json
        from pathlib import Path

        from backend.services.safety_red_team import DEFAULT_OUTPUT_PATH, run_safety_red_team_suite

        saved = Path(DEFAULT_OUTPUT_PATH)
        if saved.exists():
            try:
                return _json.loads(saved.read_text(encoding="utf-8"))
            except Exception:
                pass
        return run_safety_red_team_suite(output_path=DEFAULT_OUTPUT_PATH, live_agent=False)

    @router.post("/admin/safety-red-team")
    def run_admin_safety_red_team_endpoint(
        live_agent: bool = False,
        context=Depends(get_admin_access_context),
    ):
        """Re-run safety red-team suite.

        live_agent=false is deterministic and fast for dashboards/CI.
        live_agent=true exercises the full patient-agent pipeline.
        """
        from backend.services.safety_red_team import DEFAULT_CSV_PATH, DEFAULT_OUTPUT_PATH, run_safety_red_team_suite

        return {
            "message": "Safety red-team suite completed.",
            "result": run_safety_red_team_suite(
                output_path=DEFAULT_OUTPUT_PATH,
                csv_path=DEFAULT_CSV_PATH,
                live_agent=live_agent,
            ),
        }

    @router.get("/admin/rag-eval")
    def get_admin_rag_eval_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return cached RAG eval artifact or compute fast offline fallback."""
        import json as _json
        from pathlib import Path

        from backend.services.rag_eval_suite import DEFAULT_OUTPUT_PATH, run_rag_eval_suite

        saved = Path(DEFAULT_OUTPUT_PATH)
        if saved.exists():
            try:
                return _json.loads(saved.read_text(encoding="utf-8"))
            except Exception:
                pass
        return run_rag_eval_suite(output_path=DEFAULT_OUTPUT_PATH, live_agent=False)

    @router.post("/admin/rag-eval")
    def run_admin_rag_eval_endpoint(
        live_agent: bool = False,
        context=Depends(get_admin_access_context),
    ):
        """Re-run RAG regression suite in fast offline or full live-agent mode."""
        from backend.services.rag_eval_suite import DEFAULT_CSV_PATH, DEFAULT_OUTPUT_PATH, run_rag_eval_suite

        return {
            "message": "RAG eval suite completed.",
            "result": run_rag_eval_suite(
                output_path=DEFAULT_OUTPUT_PATH,
                csv_path=DEFAULT_CSV_PATH,
                live_agent=live_agent,
            ),
        }

    @router.get("/admin/drift-report")
    def get_admin_drift_report_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return cached drift report or compute fallback."""
        import json as _json
        from pathlib import Path

        from backend.services.drift_monitoring import DEFAULT_OUTPUT_PATH, build_drift_report

        saved = Path(DEFAULT_OUTPUT_PATH)
        if saved.exists():
            try:
                return _json.loads(saved.read_text(encoding="utf-8"))
            except Exception:
                pass
        return build_drift_report(output_path=DEFAULT_OUTPUT_PATH)

    @router.post("/admin/drift-report")
    def run_admin_drift_report_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Re-run drift/data-quality proxy report."""
        from backend.services.drift_monitoring import DEFAULT_OUTPUT_PATH, build_drift_report

        return {"message": "Drift report completed.", "result": build_drift_report(output_path=DEFAULT_OUTPUT_PATH)}

    return router
