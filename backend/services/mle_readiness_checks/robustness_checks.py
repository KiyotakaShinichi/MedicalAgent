from backend.services.mle_readiness_checks.core import check


def robustness_checks(temporal_report, noise_report, calibration_report):
    """Build advisory, non-blocking checks from the robustness evaluations."""
    checks = []

    temporal_status = (temporal_report or {}).get("status", "unavailable")
    temporal_map = {
        "stable": "passed",
        "mild_drift": "unideal",
        "significant_drift": "failed",
        "unavailable": "unavailable",
    }
    checks.append(
        check(
            name="temporal_generalization",
            category="monitoring",
            status=temporal_map.get(temporal_status, "unavailable"),
            value=temporal_status,
            threshold="stable preferred (gap >= -0.03 AUROC)",
            meaning=(
                "Patient-timeline split eval: trains on earlier synthetic cohort, "
                "evaluates on later cohort. Stable means no significant distribution shift."
            ),
            hard_gate=False,
            remediation="Investigate simulator cohort effects or add online retraining.",
        )
    )

    noise_status = (noise_report or {}).get("status", "unavailable")
    noise_map = {
        "robust": "passed",
        "mild_degradation": "acceptable",
        "significant_degradation": "unideal",
        "unavailable": "unavailable",
    }
    max_drop = None
    summary = (noise_report or {}).get("summary") or []
    if summary:
        drops = [row.get("auroc_drop") for row in summary if row.get("auroc_drop") is not None]
        max_drop = round(max(drops), 4) if drops else None
    checks.append(
        check(
            name="noise_robustness",
            category="monitoring",
            status=noise_map.get(noise_status, "unavailable"),
            value=max_drop,
            threshold="max AUROC drop <= 0.05 robust, <= 0.12 mild, > 0.12 significant",
            meaning=(
                "Five EHR noise perturbations applied to test set. Measures model "
                "brittleness to lab missingness, jitter, unit errors, batch effects, "
                "and contradictory symptom records."
            ),
            hard_gate=False,
            remediation=(
                "Add missingness-robust imputation, outlier clipping, or feature "
                "normalisation to reduce noise sensitivity."
            ),
        )
    )

    calibration_status = (calibration_report or {}).get("status", "unavailable")
    best_ece = (calibration_report or {}).get("best_ece")
    checks.append(
        check(
            name="calibration_comparison",
            category="monitoring",
            status=calibration_status,
            value=best_ece,
            threshold="ECE <= 0.03 passed, <= 0.06 strong, <= 0.10 acceptable",
            meaning=(
                "Best ECE across raw, isotonic, Platt, and temperature scaling methods. "
                "Informs which probability source is safest to surface in the dashboard."
            ),
            hard_gate=False,
            remediation=(
                "Use the recommended_source from calibration_eval_report. "
                "Fit calibration on a locked split before patient-facing probability claims."
            ),
        )
    )
    return checks
