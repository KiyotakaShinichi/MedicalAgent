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

        def _loads_or_none(value):
            """Like _loads but returns None when the value is absent —
            used for new structured JSON columns where empty list would
            misrepresent missing data."""
            try:
                return _json.loads(value) if value else None
            except Exception:
                return None

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
                "compound_intent": _loads_or_none(getattr(row, "compound_intent_json", None)),
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

    @router.get("/admin/benchmark-artifacts/{artifact_id}")
    def get_admin_normalized_benchmark_artifact_endpoint(
        artifact_id: str,
        context=Depends(get_admin_access_context),
    ):
        """Return any registered benchmark artifact in the normalized admin shape."""
        from backend.services.admin_benchmark_response import get_normalized_benchmark_artifact

        return get_normalized_benchmark_artifact(artifact_id)

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

    @router.get("/admin/genetic-counseling-eval")
    def get_admin_genetic_counseling_eval_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return cached Genetic Counseling Readiness safety benchmark."""
        import json as _json
        from pathlib import Path

        from backend.services.genetic_counseling_eval import DEFAULT_OUTPUT_PATH, run_genetic_counseling_eval

        saved = Path(DEFAULT_OUTPUT_PATH)
        if saved.exists():
            try:
                return _json.loads(saved.read_text(encoding="utf-8"))
            except Exception:
                pass
        return run_genetic_counseling_eval()

    @router.post("/admin/genetic-counseling-eval")
    def run_admin_genetic_counseling_eval_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Run genetic-counseling overclaim, VUS, privacy, and referral checks."""
        from backend.services.genetic_counseling_eval import run_genetic_counseling_eval

        return {
            "message": "Genetic counseling safety eval completed.",
            "result": run_genetic_counseling_eval(),
        }

    @router.get("/admin/biomarker-feature-benchmark")
    def get_admin_biomarker_feature_benchmark_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return cached biomarker/tumor-marker feature ablation if present."""
        from backend.services.biomarker_feature_benchmark import load_biomarker_feature_benchmark

        return load_biomarker_feature_benchmark()

    @router.post("/admin/biomarker-feature-benchmark")
    def run_admin_biomarker_feature_benchmark_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Run synthetic biomarker/tumor-marker feature benchmark with leakage checks."""
        from backend.services.biomarker_feature_benchmark import run_biomarker_feature_benchmark

        return {
            "message": "Biomarker/tumor-marker feature benchmark completed.",
            "result": run_biomarker_feature_benchmark(),
        }

    @router.get("/admin/full-feature-group-ablation")
    def get_admin_full_feature_group_ablation_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return the full modality-group ablation matrix if present."""
        from backend.services.full_feature_group_ablation import load_full_feature_group_ablation

        return load_full_feature_group_ablation()

    @router.post("/admin/full-feature-group-ablation")
    def run_admin_full_feature_group_ablation_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Run the full clinical/lab/symptom/imaging/biomarker/genetic/tumor-marker ablation."""
        from backend.services.full_feature_group_ablation import run_full_feature_group_ablation

        return {
            "message": "Full feature-group ablation completed.",
            "result": run_full_feature_group_ablation(),
        }

    @router.get("/admin/toxicity-shortcut-audit")
    def get_admin_toxicity_shortcut_audit_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return the toxicity-label shortcut audit artifact."""
        from backend.services.toxicity_shortcut_audit import load_toxicity_shortcut_audit

        return load_toxicity_shortcut_audit()

    @router.post("/admin/toxicity-shortcut-audit")
    def run_admin_toxicity_shortcut_audit_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Rerun the toxicity-label shortcut audit."""
        from backend.services.toxicity_shortcut_audit import run_toxicity_shortcut_audit

        return {
            "message": "Toxicity shortcut audit completed.",
            "result": run_toxicity_shortcut_audit(),
        }

    @router.get("/admin/leakage-audit")
    def get_admin_leakage_audit_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return the most recent training-data leakage audit artifact."""
        from backend.services.leakage_audit import load_leakage_audit

        return load_leakage_audit()

    @router.post("/admin/leakage-audit")
    def run_admin_leakage_audit_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Rerun the unified leakage audit and refresh the artifact on disk."""
        from backend.services.leakage_audit import run_leakage_audit

        return {
            "message": "Leakage audit completed.",
            "result": run_leakage_audit(),
        }

    @router.get("/admin/prediction-traces")
    def get_admin_prediction_traces_endpoint(
        limit: int = 50,
        patient_id: str | None = None,
        decision: str | None = None,
        abstained_only: bool = False,
        context=Depends(get_admin_access_context),
        db: Session = Depends(get_db),
    ):
        """Return recent prediction traces with optional filtering + a summary."""
        from backend.services.prediction_trace import (
            list_recent_traces,
            summarise_traces,
        )

        return {
            "traces": list_recent_traces(
                db,
                limit=limit,
                patient_id=patient_id,
                decision=decision,
                abstained_only=abstained_only,
            ),
            "summary": summarise_traces(db),
        }

    @router.get("/admin/evidence-abstention-eval")
    def get_admin_evidence_abstention_eval_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return the most recent evidence-aware abstention eval artifact."""
        from backend.services.evidence_abstention_eval import load_evidence_abstention_eval

        return load_evidence_abstention_eval()

    @router.get("/admin/modality-robustness-comparison")
    def get_admin_modality_robustness_comparison_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return the champion-vs-modality-robust comparison artifact."""
        from backend.services.modality_robustness_comparison import (
            load_modality_robustness_comparison,
        )

        return load_modality_robustness_comparison()

    @router.post("/admin/modality-robustness-comparison")
    def run_admin_modality_robustness_comparison_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Rerun the champion-vs-robust comparison sweep."""
        from backend.services.modality_robustness_comparison import (
            run_modality_robustness_comparison,
        )

        return {
            "message": "Modality-robustness comparison completed.",
            "result": run_modality_robustness_comparison(),
        }

    @router.get("/admin/kb-source-governance")
    def get_admin_kb_source_governance_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return the KB source-governance artifact (tier + allowed_use + staleness)."""
        from backend.services.kb_source_governance import load_kb_source_governance

        return load_kb_source_governance()

    @router.post("/admin/kb-source-governance")
    def run_admin_kb_source_governance_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Rebuild the KB source-governance artifact from the current KB chunks."""
        from backend.services.kb_source_governance import build_kb_source_governance

        return {
            "message": "KB source governance rebuilt.",
            "result": build_kb_source_governance(),
        }

    # ─── Phase 11: intent-aware RAG artifacts ──────────────────────────

    @router.get("/admin/rag-intent-modes")
    def get_admin_rag_intent_modes_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return the configured RAG mode registry (config-as-API)."""
        from backend.services.rag_intent_modes import INTENT_TO_MODE, list_modes
        modes = {
            name: {
                "mode": cfg.mode,
                "description": cfg.description,
                "audience": cfg.audience,
                "allowed_tiers": list(cfg.allowed_tiers),
                "allowed_use": list(cfg.allowed_use),
                "allow_citations": cfg.allow_citations,
                "insufficient_evidence_default": cfg.insufficient_evidence_default,
                "banned_claim_categories": list(cfg.banned_claim_categories),
                "max_retrieved_chunks": cfg.max_retrieved_chunks,
                "require_clinician_handoff_clause": cfg.require_clinician_handoff_clause,
            }
            for name, cfg in list_modes().items()
        }
        return {"modes": modes, "intent_to_mode": dict(INTENT_TO_MODE)}

    @router.get("/admin/rag-intent-aware-eval")
    def get_admin_rag_intent_aware_eval_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return the most recent intent-aware RAG benchmark artifact."""
        from backend.services.rag_intent_aware_eval import load_intent_aware_eval
        return load_intent_aware_eval()

    @router.get("/admin/live-rag-eval")
    def get_admin_live_rag_eval_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return the most recent live-agent RAG benchmark artifact."""
        from backend.services.live_rag_eval import load_live_rag_eval
        return load_live_rag_eval()

    @router.post("/admin/live-rag-eval")
    def run_admin_live_rag_eval_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Rerun the live-agent RAG benchmark."""
        from backend.services.live_rag_eval import run_live_rag_eval
        return {
            "message": "Live RAG eval completed.",
            "result": run_live_rag_eval(),
        }

    @router.get("/admin/claim-level-citation-eval")
    def get_admin_claim_level_citation_eval_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return the claim-level citation validation artifact."""
        from backend.services.claim_level_citation_eval import load_claim_level_citation_eval
        return load_claim_level_citation_eval()

    @router.get("/admin/rag-tier-ablation")
    def get_admin_rag_tier_ablation_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return the most recent source-tier ablation artifact."""
        from backend.services.rag_tier_ablation import load_tier_ablation
        return load_tier_ablation()

    @router.get("/admin/taglish-safety-parity")
    def get_admin_taglish_safety_parity_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return the Taglish ↔ English safety-route parity artifact."""
        from backend.services.taglish_safety_parity import load_taglish_safety_parity
        return load_taglish_safety_parity()

    @router.get("/admin/rag-trace-replay")
    def get_admin_rag_trace_replay_endpoint(
        limit: int = 25,
        context=Depends(get_admin_access_context),
        db: Session = Depends(get_db),
    ):
        """Return recent RAG evaluation log rows with Phase 11 fields
        (intent, rewrite, retrieved sources, source tiers, claim
        validation, post-gen validator, evidence grade, final answer)
        flattened into a single per-call shape ready for the trace
        replay panel."""
        import json as _json
        from backend.models import RAGEvaluationLog

        safe_limit = max(1, min(limit, 200))
        rows = (
            db.query(RAGEvaluationLog)
            .order_by(RAGEvaluationLog.created_at.desc(), RAGEvaluationLog.id.desc())
            .limit(safe_limit)
            .all()
        )

        def _safe_loads(value):
            try:
                return _json.loads(value) if value else None
            except Exception:
                return None

        # Phase 11 columns now live on the model (migration 0003) — direct
        # attribute access, no more silent None defaults via getattr.
        traces = []
        for row in rows:
            traces.append({
                "id": row.id,
                "created_at": str(row.created_at) if row.created_at else None,
                "patient_id": row.patient_id,
                "query_preview": row.query_preview,
                "intent": row.intent,
                "safety_level": row.safety_level,
                "rag_mode": row.rag_mode,
                "rewritten_query": row.rewritten_query,
                "retrieved_source_ids": _safe_loads(row.retrieved_source_ids_json) or [],
                "cited_source_ids": _safe_loads(row.cited_source_ids_json) or [],
                "evidence_grade": _safe_loads(row.evidence_grade_json),
                "claim_validation": _safe_loads(row.claim_validation_json),
                "tier_filter": _safe_loads(row.tier_filter_json),
                "post_gen_validator": _safe_loads(row.post_gen_validator_json),
                "grounding_score": row.grounding_score,
                "hallucination_score": row.hallucination_score,
                "latency_ms": row.latency_ms,
                "input_guardrail": row.input_guardrail_status,
                "output_guardrail": row.output_guardrail_status,
            })
        return {"count": len(traces), "traces": traces}

    @router.get("/admin/toxicity-feature-audit")
    def get_admin_toxicity_feature_audit_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return the latest toxicity feature-importance audit + no-proxy
        baseline.  Documents the synthetic generator's structural label
        leakage so the headline toxicity AUC isn't quoted in isolation."""
        from backend.services.toxicity_feature_audit import load_toxicity_feature_audit
        return load_toxicity_feature_audit()

    @router.post("/admin/toxicity-feature-audit")
    def run_admin_toxicity_feature_audit_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Rerun the toxicity audit + write a fresh artifact."""
        from backend.services.toxicity_feature_audit import run_toxicity_feature_audit
        return {
            "message": "Toxicity feature audit completed.",
            "result": run_toxicity_feature_audit(),
        }

    @router.get("/admin/synthetic-generator-card")
    def get_admin_synthetic_generator_card_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return the synthetic-data generator card artifact."""
        from backend.services.synthetic_generator_card import load_synthetic_generator_card

        return load_synthetic_generator_card()

    @router.post("/admin/synthetic-generator-card")
    def run_admin_synthetic_generator_card_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Rebuild the generator card from the current dataset summary + rows."""
        from backend.services.synthetic_generator_card import build_synthetic_generator_card

        return {
            "message": "Generator card refreshed.",
            "result": build_synthetic_generator_card(),
        }

    @router.get("/admin/failure-mode-registry")
    def get_admin_failure_mode_registry_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return the consolidated failure-mode registry artifact."""
        from backend.services.failure_mode_registry import load_failure_mode_registry

        return load_failure_mode_registry()

    @router.post("/admin/failure-mode-registry")
    def run_admin_failure_mode_registry_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Rebuild the failure-mode registry by re-aggregating its source artifacts."""
        from backend.services.failure_mode_registry import build_failure_mode_registry

        return {
            "message": "Failure-mode registry rebuilt.",
            "result": build_failure_mode_registry(),
        }

    @router.get("/admin/modality-robust-training")
    def get_admin_modality_robust_training_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return the modality-robust training metadata artifact."""
        from backend.services.modality_dropout_training import (
            load_modality_robust_training_metadata,
        )

        return load_modality_robust_training_metadata()

    @router.post("/admin/modality-robust-training")
    def run_admin_modality_robust_training_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Retrain the modality-robust classifier and refresh both the model
        artifact and the metadata report.  Long-running."""
        from backend.services.modality_dropout_training import (
            train_modality_robust_classifier,
        )

        return {
            "message": "Modality-robust classifier retrained.",
            "result": train_modality_robust_classifier(),
        }

    @router.get("/admin/quantile-regression-training")
    def get_admin_quantile_regression_training_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return the response-score quantile regression metadata artifact."""
        from backend.services.quantile_regression_training import (
            load_quantile_regression_training_metadata,
        )

        return load_quantile_regression_training_metadata()

    @router.post("/admin/quantile-regression-training")
    def run_admin_quantile_regression_training_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Retrain p10/p50/p90 response-score quantile heads."""
        from backend.services.quantile_regression_training import train_quantile_regression_heads

        return {
            "message": "Quantile regression heads retrained.",
            "result": train_quantile_regression_heads(),
        }

    @router.get("/admin/modality-robust-regression-training")
    def get_admin_modality_robust_regression_training_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return the modality-robust regression metadata artifact."""
        from backend.services.modality_dropout_regression_training import (
            load_modality_robust_regression_metadata,
        )

        return load_modality_robust_regression_metadata()

    @router.post("/admin/modality-robust-regression-training")
    def run_admin_modality_robust_regression_training_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Retrain the modality-robust response-score regressor."""
        from backend.services.modality_dropout_regression_training import (
            train_modality_robust_regressor,
        )

        return {
            "message": "Modality-robust response-score regressor retrained.",
            "result": train_modality_robust_regressor(),
        }

    @router.get("/admin/regression-robustness-comparison")
    def get_admin_regression_robustness_comparison_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return legacy-vs-modality-robust response-score comparison."""
        from backend.services.regression_robustness_comparison import (
            load_regression_robustness_comparison,
        )

        return load_regression_robustness_comparison()

    @router.post("/admin/regression-robustness-comparison")
    def run_admin_regression_robustness_comparison_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Rerun the response-score robustness comparison sweep."""
        from backend.services.regression_robustness_comparison import (
            run_regression_robustness_comparison,
        )

        return {
            "message": "Regression robustness comparison completed.",
            "result": run_regression_robustness_comparison(),
        }

    @router.get("/admin/modality-dropout-quantile-regression")
    def get_admin_modality_dropout_quantile_regression_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return modality-dropout quantile regression metadata."""
        from backend.services.modality_dropout_quantile_regression_training import (
            load_modality_dropout_quantile_regression_metadata,
        )

        return load_modality_dropout_quantile_regression_metadata()

    @router.post("/admin/modality-dropout-quantile-regression")
    def run_admin_modality_dropout_quantile_regression_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Train modality-dropout p10/p50/p90 response-score quantile heads."""
        from backend.services.modality_dropout_quantile_regression_training import (
            train_modality_dropout_quantile_regression_heads,
        )

        return {
            "message": "Modality-dropout quantile regression heads retrained.",
            "result": train_modality_dropout_quantile_regression_heads(),
        }

    @router.get("/admin/response-conformal-calibration")
    def get_admin_response_conformal_calibration_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return response-score conformal interval calibration."""
        from backend.services.response_conformal_calibration import load_response_conformal_calibration

        return load_response_conformal_calibration()

    @router.post("/admin/response-conformal-calibration")
    def run_admin_response_conformal_calibration_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Recompute response-score conformal interval calibration."""
        from backend.services.response_conformal_calibration import build_response_conformal_calibration

        return {
            "message": "Response-score conformal calibration completed.",
            "result": build_response_conformal_calibration(),
        }

    @router.get("/admin/robustness-stress")
    def get_admin_robustness_stress_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return synthetic robustness stress-suite artifact."""
        from backend.services.robustness_stress import load_robustness_stress_report

        return load_robustness_stress_report()

    @router.post("/admin/robustness-stress")
    def run_admin_robustness_stress_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Run missing/corrupt/conflicting-data stress cases."""
        from backend.services.robustness_stress import run_robustness_stress_suite

        return {
            "message": "Robustness stress suite completed.",
            "result": run_robustness_stress_suite(),
        }

    @router.get("/admin/medical-safety-contract")
    def get_admin_medical_safety_contract_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return clinical ontology, minimum evidence, and claim boundary contract."""
        from backend.services.medical_safety_contract import load_medical_safety_contract

        return load_medical_safety_contract()

    @router.post("/admin/medical-safety-contract")
    def run_admin_medical_safety_contract_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Regenerate the medical-safety contract artifact."""
        from backend.services.medical_safety_contract import build_medical_safety_contract

        return {
            "message": "Medical safety contract generated.",
            "result": build_medical_safety_contract(),
        }

    @router.post("/admin/evidence-abstention-eval")
    def run_admin_evidence_abstention_eval_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Rerun the abstention sweep across modality-dropout scenarios."""
        from backend.services.evidence_abstention_eval import run_evidence_abstention_eval

        return {
            "message": "Evidence abstention eval completed.",
            "result": run_evidence_abstention_eval(),
        }

    @router.get("/admin/public-biomarker-dataset-manifest")
    def get_admin_public_biomarker_dataset_manifest_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return public biomarker/tumor-marker predictor-source manifest."""
        from backend.services.public_biomarker_datasets import load_public_biomarker_dataset_manifest

        return load_public_biomarker_dataset_manifest()

    @router.post("/admin/public-biomarker-dataset-manifest")
    def run_admin_public_biomarker_dataset_manifest_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Rebuild public biomarker/tumor-marker predictor-source manifest."""
        from backend.services.public_biomarker_datasets import build_public_biomarker_dataset_manifest

        return {
            "message": "Public biomarker dataset manifest generated.",
            "result": build_public_biomarker_dataset_manifest(),
        }

    @router.get("/admin/public-biomarker-mapping-readiness")
    def get_admin_public_biomarker_mapping_readiness_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return source-to-feature mapping readiness for public biomarker sources."""
        from backend.services.public_biomarker_mapping import load_public_biomarker_mapping_readiness

        return load_public_biomarker_mapping_readiness()

    @router.post("/admin/public-biomarker-mapping-readiness")
    def run_admin_public_biomarker_mapping_readiness_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Rebuild source-to-feature mapping readiness for public biomarker sources."""
        from backend.services.public_biomarker_mapping import build_public_biomarker_mapping_readiness

        return {
            "message": "Public biomarker mapping readiness generated.",
            "result": build_public_biomarker_mapping_readiness(),
        }

    @router.get("/admin/cbioportal-biomarker-schema-mapping")
    def get_admin_cbioportal_biomarker_schema_mapping_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return TCGA-BRCA/METABRIC cBioPortal biomarker schema mapping readiness."""
        from backend.services.cbioportal_biomarker_mapper import load_cbioportal_biomarker_schema_mapping

        return load_cbioportal_biomarker_schema_mapping()

    @router.post("/admin/cbioportal-biomarker-schema-mapping")
    def run_admin_cbioportal_biomarker_schema_mapping_endpoint(
        live_fetch: bool = True,
        context=Depends(get_admin_access_context),
    ):
        """Rebuild cBioPortal biomarker schema mapping from public API when available."""
        from backend.services.cbioportal_biomarker_mapper import build_cbioportal_biomarker_schema_mapping

        return {
            "message": "cBioPortal biomarker schema mapping generated.",
            "result": build_cbioportal_biomarker_schema_mapping(live_fetch=live_fetch),
        }

    @router.get("/admin/clinical-safety-review-checklist")
    def get_admin_clinical_safety_review_checklist_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Return clinical safety review checklist artifact."""
        from backend.services.clinical_safety_checklist import load_clinical_safety_review_checklist

        return load_clinical_safety_review_checklist()

    @router.post("/admin/clinical-safety-review-checklist")
    def run_admin_clinical_safety_review_checklist_endpoint(
        context=Depends(get_admin_access_context),
    ):
        """Rebuild clinical safety review checklist artifact."""
        from backend.services.clinical_safety_checklist import build_clinical_safety_review_checklist

        return {
            "message": "Clinical safety review checklist generated.",
            "result": build_clinical_safety_review_checklist(),
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
        """Rebuild system health report."""
        from backend.services.system_health import build_system_health_report

        return {
            "message": "System health report generated.",
            "result": build_system_health_report(db=db),
        }

    return router
