"""Build model artifact, training, calibration, and robustness lifecycle routes."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, Depends


def build_admin_eval_lifecycle_router(
    get_admin_access_context: Callable,
) -> APIRouter:
    """Compose model artifact, training, calibration, and robustness lifecycle routes."""
    router = APIRouter()

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

    return router
