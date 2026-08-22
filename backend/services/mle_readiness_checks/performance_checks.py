from backend.services.mle_readiness_checks.core import check, rounded


def performance_checks(metrics, evaluation_report):
    best_model = (metrics or {}).get("best_model_by_patient_level_roc_auc")
    best_metrics = ((metrics or {}).get("models") or {}).get(best_model, {})
    advanced = (evaluation_report or {}).get("advanced_model_evaluation") or {}
    calibration = advanced.get("calibration") or {}
    posthoc_calibration = calibration.get("posthoc_calibration") or {}
    false_negative = advanced.get("false_negative_review") or {}
    operating_fnr = operating_policy_false_negative_rate(advanced)
    ci_report = advanced.get("bootstrap_confidence_intervals") or {}
    subgroup = advanced.get("subgroup_performance") or {}
    drift = (evaluation_report or {}).get("drift_monitoring") or {}
    coverage = (evaluation_report or {}).get("data_coverage") or {}

    return [
        metric_check(
            "patient_level_roc_auc",
            "model_quality",
            best_metrics.get("patient_level_roc_auc"),
            minimum=0.80,
            strong=0.90,
            hard_minimum=0.70,
            meaning="Ranking quality across thresholds.",
        ),
        metric_check(
            "patient_level_average_precision",
            "model_quality",
            best_metrics.get("patient_level_average_precision"),
            minimum=0.80,
            strong=0.90,
            hard_minimum=0.65,
            meaning="Precision/recall quality under class imbalance.",
        ),
        metric_check(
            "patient_level_sensitivity",
            "model_quality",
            best_metrics.get("patient_level_sensitivity"),
            minimum=0.90,
            strong=0.95,
            hard_minimum=0.80,
            meaning="Medical monitoring should be especially careful about missed positive cases.",
        ),
        metric_check(
            "patient_level_brier_score",
            "model_quality",
            best_metrics.get("patient_level_brier_score"),
            maximum=0.12,
            strong=0.08,
            hard_maximum=0.20,
            lower_is_better=True,
            meaning="Probability error; lower is better.",
        ),
        metric_check(
            "expected_calibration_error",
            "model_quality",
            calibration.get("expected_calibration_error"),
            maximum=0.10,
            strong=0.06,
            hard_maximum=0.20,
            lower_is_better=True,
            meaning="Checks whether probabilities are calibrated enough for score interpretation.",
        ),
        check(
            name="posthoc_calibration_diagnostic",
            category="model_quality",
            status=posthoc_calibration.get("status") or "unavailable",
            value={
                "best_method": posthoc_calibration.get("best_method"),
                "best_validation_ece": posthoc_calibration.get("best_validation_ece"),
                "validation_patients": posthoc_calibration.get("validation_patients"),
            },
            threshold="diagnostic available and validation ECE <=0.10 preferred",
            meaning="Shows whether a candidate calibration head could improve probability quality before promotion.",
            hard_gate=False,
            remediation="Lock a calibration split, register the calibrated head, and re-run threshold and subgroup checks.",
        ),
        metric_check(
            "operating_policy_false_negative_rate",
            "model_quality",
            operating_fnr.get("rate"),
            maximum=0.10,
            strong=0.05,
            hard_maximum=0.20,
            lower_is_better=True,
            meaning="False-negative rate under the declared cost-sensitive operating policy, not only the default 0.5 threshold.",
        ),
        check(
            name="default_threshold_false_negative_review",
            category="model_quality",
            status=false_negative.get("status") or "unavailable",
            value={
                "threshold": 0.5,
                "false_negative_rate": false_negative.get("false_negative_rate"),
                "false_negative_count": false_negative.get("count"),
                "positive_cases": false_negative.get("positive_cases"),
                "operating_policy": operating_fnr,
            },
            threshold="0.5-threshold review documented; operating policy may use a lower safety-first threshold",
            meaning="Keeps the default-threshold miss pattern visible while release gating uses the explicit safety-first policy.",
            hard_gate=False,
            remediation="Inspect false-negative cases and justify any lower threshold against review burden.",
        ),
        check(
            name="bootstrap_ci_stability",
            category="model_quality",
            status=ci_status(ci_report),
            value=ci_widths(ci_report),
            threshold="AUROC/AUPRC/Brier CI width <=0.10 preferred",
            meaning="Confidence intervals show whether the held-out split is stable.",
            hard_gate=False,
            remediation="Increase patient count, reduce simulator leakage, or use repeated validation splits.",
        ),
        check(
            name="subgroup_performance_review",
            category="model_quality",
            status=subgroup.get("status") or "unavailable",
            value={
                "status": subgroup.get("status"),
                "powered_group_status": subgroup.get("powered_group_status"),
                "groups": len(subgroup.get("rows") or []),
                "low_support_group_count": subgroup.get("low_support_group_count"),
            },
            threshold="no failed subgroup gate",
            meaning="Subgroup checks catch brittle behavior by age/stage/subtype/regimen.",
            hard_gate=subgroup.get("status") == "failed",
            remediation="Add more group coverage and inspect weak subgroup rows.",
        ),
        check(
            name="training_serving_drift_proxy",
            category="monitoring",
            status=drift.get("status") or "unavailable",
            value={"watch_feature_count": drift.get("watch_feature_count")},
            threshold="no failed drift features",
            meaning="Reference/current split drift is a cheap proxy until real production traffic exists.",
            hard_gate=drift.get("status") == "failed",
            remediation="Inspect shifted features and add drift thresholds per feature.",
        ),
        check(
            name="longitudinal_data_coverage",
            category="monitoring",
            status=coverage.get("status") or "unavailable",
            value={"rows": coverage.get("rows"), "patients": coverage.get("patients")},
            threshold="coverage passed or acceptable",
            meaning="CBC, MRI, symptoms, and treatment schedule coverage must stay visible.",
            hard_gate=coverage.get("status") == "failed",
            remediation="Improve capture of missing modalities before trusting model comparisons.",
        ),
    ]


def metric_check(
    name,
    category,
    value,
    minimum=None,
    maximum=None,
    strong=None,
    hard_minimum=None,
    hard_maximum=None,
    lower_is_better=False,
    meaning="",
):
    if value is None:
        status = "unavailable"
    elif lower_is_better:
        status = (
            "strong"
            if value <= strong
            else "passed"
            if value <= maximum
            else "unideal"
            if value <= hard_maximum
            else "failed"
        )
    else:
        status = (
            "strong"
            if value >= strong
            else "passed"
            if value >= minimum
            else "unideal"
            if value >= hard_minimum
            else "failed"
        )
    hard_gate = status == "failed"
    threshold = (
        f"<={maximum} preferred, <={hard_maximum} hard max"
        if lower_is_better
        else f">={minimum} preferred, >={hard_minimum} hard min"
    )
    return check(
        name=name,
        category=category,
        status=status,
        value=rounded(value),
        threshold=threshold,
        meaning=meaning,
        hard_gate=hard_gate,
        remediation="Retrain, retune threshold/calibration, or inspect weak slices before promotion.",
    )


def operating_policy_false_negative_rate(advanced):
    policies = ((advanced or {}).get("cost_sensitive_thresholds") or {}).get("policies") or []
    safety_first = next(
        (policy for policy in policies if policy.get("name") == "safety_first"), None
    )
    false_negative_review = (advanced or {}).get("false_negative_review") or {}
    positive_cases = false_negative_review.get("positive_cases")

    if safety_first and positive_cases:
        false_negative_count = safety_first.get("false_negative")
        if false_negative_count is not None:
            return {
                "source": "safety_first_cost_sensitive_policy",
                "threshold": safety_first.get("recommended_threshold"),
                "rate": round(float(false_negative_count) / float(positive_cases), 3),
                "false_negative_count": false_negative_count,
                "positive_cases": positive_cases,
                "sensitivity": safety_first.get("sensitivity"),
                "specificity": safety_first.get("specificity"),
                "status": safety_first.get("status"),
            }

    return {
        "source": "default_threshold_0_5",
        "threshold": 0.5,
        "rate": false_negative_review.get("false_negative_rate"),
        "false_negative_count": false_negative_review.get("count"),
        "positive_cases": positive_cases,
        "status": false_negative_review.get("status"),
    }


def ci_status(ci_report):
    widths = ci_widths(ci_report)
    values = [item["interval_width"] for item in widths if item.get("interval_width") is not None]
    if not values:
        return "unavailable"
    max_width = max(values)
    if max_width <= 0.05:
        return "strong"
    if max_width <= 0.10:
        return "passed"
    if max_width <= 0.20:
        return "unideal"
    return "failed"


def ci_widths(ci_report):
    return [
        {
            "metric": row.get("metric"),
            "interval_width": row.get("interval_width"),
            "status": row.get("status"),
        }
        for row in (ci_report or {}).get("metrics") or []
    ]
