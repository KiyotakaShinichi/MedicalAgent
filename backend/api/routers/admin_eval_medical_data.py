"""Build medical-safety and public biomarker readiness routes."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, Depends


def build_admin_eval_medical_data_router(
    get_admin_access_context: Callable,
) -> APIRouter:
    """Compose medical-safety and public biomarker readiness routes."""
    router = APIRouter()

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

    return router
