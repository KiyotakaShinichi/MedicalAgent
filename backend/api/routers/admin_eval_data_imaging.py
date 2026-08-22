"""Build public-data and imaging evaluation routes."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, Depends


def build_admin_eval_data_imaging_router(
    get_admin_access_context: Callable,
) -> APIRouter:
    """Compose public-data and imaging evaluation routes."""
    router = APIRouter()

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

    return router
