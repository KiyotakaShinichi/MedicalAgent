"""Build ML feature, leakage, trace, abstention, and robustness routes."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session


def build_admin_eval_ml_router(
    get_admin_access_context: Callable,
    get_db: Callable,
) -> APIRouter:
    """Compose ML feature, leakage, trace, abstention, and robustness routes."""
    router = APIRouter()

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


    return router
