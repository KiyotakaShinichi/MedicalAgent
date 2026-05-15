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

    @router.get("/admin/ultrasound-transfer-baseline")
    def get_admin_ultrasound_transfer_baseline_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return BUSI transfer-learning baseline metrics or explicit unavailable artifact."""
        import json as _json
        from pathlib import Path

        from backend.services.ultrasound_transfer_baseline import DEFAULT_OUTPUT_PATH, run_ultrasound_transfer_baseline

        saved = Path(DEFAULT_OUTPUT_PATH)
        if saved.exists():
            try:
                return _json.loads(saved.read_text(encoding="utf-8"))
            except Exception:
                pass
        return run_ultrasound_transfer_baseline(output_path=DEFAULT_OUTPUT_PATH)

    @router.post("/admin/ultrasound-transfer-baseline")
    def run_admin_ultrasound_transfer_baseline_endpoint(
        pretrained: bool = False,
        context=Depends(get_admin_access_context),
    ):
        """Run hardware-friendly transfer-learning baseline if BUSI exists locally."""
        from backend.services.ultrasound_transfer_baseline import DEFAULT_OUTPUT_PATH, run_ultrasound_transfer_baseline

        return {
            "message": "Ultrasound transfer baseline completed.",
            "result": run_ultrasound_transfer_baseline(output_path=DEFAULT_OUTPUT_PATH, pretrained=pretrained),
        }

    @router.get("/admin/ultrasound-segmentation-baseline")
    def get_admin_ultrasound_segmentation_baseline_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return BUSI mask segmentation baseline metrics or explicit unavailable artifact."""
        import json as _json
        from pathlib import Path

        from backend.services.ultrasound_segmentation_baseline import DEFAULT_OUTPUT_PATH, run_ultrasound_segmentation_baseline

        saved = Path(DEFAULT_OUTPUT_PATH)
        if saved.exists():
            try:
                return _json.loads(saved.read_text(encoding="utf-8"))
            except Exception:
                pass
        return run_ultrasound_segmentation_baseline(output_path=DEFAULT_OUTPUT_PATH)

    @router.post("/admin/ultrasound-segmentation-baseline")
    def run_admin_ultrasound_segmentation_baseline_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Run classical BUSI segmentation baseline if masks exist locally."""
        from backend.services.ultrasound_segmentation_baseline import DEFAULT_OUTPUT_PATH, run_ultrasound_segmentation_baseline

        return {
            "message": "Ultrasound segmentation baseline completed.",
            "result": run_ultrasound_segmentation_baseline(output_path=DEFAULT_OUTPUT_PATH),
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

    @router.get("/admin/chat-latency-report")
    def get_admin_chat_latency_report_endpoint(
        context=Depends(get_admin_access_context),
        db: Session = Depends(get_db),
    ):
        """Return cached/derived support-agent latency observability report."""
        from backend.services.chat_latency_report import build_chat_latency_report

        return build_chat_latency_report(db=db)

    @router.post("/admin/chat-latency-report")
    def run_admin_chat_latency_report_endpoint(
        context=Depends(get_admin_access_context),
        db: Session = Depends(get_db),
    ):
        """Rebuild support-agent latency report from trace logs."""
        from backend.services.chat_latency_report import build_chat_latency_report

        return {"message": "Chat latency report completed.", "result": build_chat_latency_report(db=db)}

    @router.post("/admin/ai-ml-narrative-report")
    def run_admin_ai_ml_narrative_report_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Generate a human-readable AI/ML evaluation narrative artifact."""
        from backend.services.evaluation_narrative_report import build_ai_ml_narrative_report

        return {"message": "AI/ML narrative report generated.", "result": build_ai_ml_narrative_report()}

    @router.post("/admin/demo-storyline")
    def run_admin_demo_storyline_endpoint(
        patient_id: str = "P001",
        context=Depends(get_admin_access_context),
    ):
        """Generate a repeatable demo storyline for a patient journey."""
        from backend.services.demo_storyline import build_demo_storyline

        return {"message": "Demo storyline generated.", "result": build_demo_storyline(patient_id=patient_id)}

    @router.post("/admin/current-vs-realism-candidate")
    def run_admin_current_vs_realism_candidate_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Compare the current synthetic champion with the realism-calibrated candidate."""
        from backend.services.candidate_model_comparison import build_current_vs_candidate_report

        return {
            "message": "Current-vs-candidate comparison generated.",
            "result": build_current_vs_candidate_report(),
        }

    @router.get("/admin/current-vs-realism-candidate")
    def get_admin_current_vs_realism_candidate_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return cached current-vs-realism-candidate comparison if present."""
        import json as _json
        from pathlib import Path

        from backend.services.candidate_model_comparison import DEFAULT_OUTPUT_PATH, build_current_vs_candidate_report

        saved = Path(DEFAULT_OUTPUT_PATH)
        if saved.exists():
            try:
                return _json.loads(saved.read_text(encoding="utf-8"))
            except Exception:
                pass
        return build_current_vs_candidate_report(output_path=DEFAULT_OUTPUT_PATH)

    @router.get("/admin/multilingual-refusal-eval")
    def get_admin_multilingual_refusal_eval_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return cached multilingual refusal routing benchmark."""
        from backend.services.multilingual_refusal_eval import load_multilingual_refusal_eval

        return load_multilingual_refusal_eval()

    @router.post("/admin/multilingual-refusal-eval")
    def run_admin_multilingual_refusal_eval_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Run Tagalog/Taglish diagnosis/treatment/urgent routing checks."""
        from backend.services.multilingual_refusal_eval import run_multilingual_refusal_eval

        return {
            "message": "Multilingual refusal eval completed.",
            "result": run_multilingual_refusal_eval(),
        }

    @router.get("/admin/llm-judge-eval")
    def get_admin_llm_judge_eval_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return optional LLM-as-judge artifact if generated."""
        from backend.services.llm_judge_eval import load_llm_judge_eval

        return load_llm_judge_eval()

    @router.post("/admin/llm-judge-eval")
    def run_admin_llm_judge_eval_endpoint(
        max_cases: int = 30,
        context=Depends(get_admin_access_context),
    ):
        """Run optional LLM-as-judge eval. Returns unavailable when no provider is configured."""
        from backend.services.llm_judge_eval import run_llm_judge_eval

        return {
            "message": "LLM-judge eval completed.",
            "result": run_llm_judge_eval(max_cases=max(1, min(max_cases, 50))),
        }

    @router.get("/admin/benchmark-registry")
    def get_admin_benchmark_registry_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return the consolidated benchmark registry across safety, RAG, MLE, and imaging."""
        import json as _json
        from pathlib import Path

        from backend.services.benchmark_registry import DEFAULT_JSON_PATH, build_benchmark_registry

        saved = Path(DEFAULT_JSON_PATH)
        if saved.exists():
            try:
                return _json.loads(saved.read_text(encoding="utf-8"))
            except Exception:
                pass
        return build_benchmark_registry(output_path=DEFAULT_JSON_PATH)

    @router.post("/admin/benchmark-registry")
    def run_admin_benchmark_registry_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Rebuild the consolidated benchmark registry."""
        from backend.services.benchmark_registry import build_benchmark_registry

        return {
            "message": "Benchmark registry generated.",
            "result": build_benchmark_registry(),
        }

    return router
