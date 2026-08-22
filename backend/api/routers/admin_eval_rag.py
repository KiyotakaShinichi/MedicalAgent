"""Build knowledge-base governance and intent-aware RAG evaluation routes."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, Depends


def build_admin_eval_rag_router(
    get_admin_access_context: Callable,
) -> APIRouter:
    """Compose knowledge-base governance and intent-aware RAG evaluation routes."""
    router = APIRouter()

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


    return router
