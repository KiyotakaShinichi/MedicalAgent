"""Build evaluation reporting, registry, and multilingual/genetics routes."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session


def build_admin_eval_reporting_router(
    get_admin_access_context: Callable,
    get_db: Callable,
) -> APIRouter:
    """Compose evaluation reporting, registry, and multilingual/genetics routes."""
    router = APIRouter()

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


    return router
