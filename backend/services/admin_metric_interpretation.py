"""Metric interpretation and status helpers for admin analytics."""

from __future__ import annotations

import math

import pandas as pd


def _metric_interpretation_guide():
    return {
        "status_levels": [
            {
                "status": "failed",
                "meaning": "Below the minimum engineering gate. Do not present as a reliable model signal.",
            },
            {
                "status": "unideal",
                "meaning": "Works weakly or has important risk. Useful for debugging, not for headline claims.",
            },
            {
                "status": "acceptable",
                "meaning": "Reasonable for a PoC if limitations are explicit and humans stay in the loop.",
            },
            {
                "status": "strong",
                "meaning": "Good engineering result on the current validation setup.",
            },
            {
                "status": "passed",
                "meaning": "Passes this project gate. This is not clinical validation.",
            },
        ],
        "model_metric_bands": {
            "AUROC": "Failed <0.60, unideal 0.60-0.70, acceptable 0.70-0.80, strong 0.80-0.90, passed >=0.90.",
            "Average precision / AUPRC": "Interpreted relative to class prevalence. For this PoC: failed <0.50, unideal 0.50-0.65, acceptable 0.65-0.80, strong 0.80-0.90, passed >=0.90.",
            "Sensitivity": "False negatives matter in monitoring. Failed <0.75, unideal 0.75-0.85, acceptable 0.85-0.90, strong 0.90-0.95, passed >=0.95.",
            "Specificity": "Controls false alarms. Failed <0.60, unideal 0.60-0.70, acceptable 0.70-0.80, strong 0.80-0.90, passed >=0.90.",
            "Brier score": "Lower is better. Failed >0.25, unideal 0.18-0.25, acceptable 0.12-0.18, strong 0.08-0.12, passed <=0.08.",
            "Expected calibration error": "Lower is better. Failed >0.15, unideal 0.10-0.15, acceptable 0.06-0.10, strong 0.03-0.06, passed <=0.03.",
            "Bootstrap CI width": "Narrower is more stable. Failed >0.25, unideal 0.15-0.25, acceptable 0.10-0.15, strong 0.05-0.10, passed <=0.05.",
            "False-negative rate": "Lower is safer. Failed >0.35, unideal 0.20-0.35, acceptable 0.10-0.20, strong 0.05-0.10, passed <=0.05.",
            "Drift standardized mean shift": "Lower is better. Passed <0.20, acceptable 0.20-0.50, unideal 0.50-0.80, failed >=0.80.",
            "Missingness": "Lower is better. Passed <=5%, acceptable <=10%, unideal <=20%, failed >20%.",
            "Data coverage": "Higher is better. Failed <70%, unideal 70-85%, acceptable 85-95%, passed >=95%.",
            "Clinician accepted-review rate": "Higher means clinicians usually approve/edit/escalate instead of reject. Failed <30%, unideal 30-45%, acceptable 45-60%, strong 60-75%, passed >=75%.",
            "Clinician quality score": "Average 1-5 rating. Failed <2, unideal 2-3, acceptable 3-4, strong 4-4.5, passed >=4.5.",
        },
        "advanced_metric_definitions": [
            {
                "metric": "Expected calibration error",
                "for": "Checks whether a 0.80 probability behaves like about 80% observed positives.",
            },
            {
                "metric": "Bootstrap confidence interval",
                "for": "Shows metric uncertainty from the finite held-out patient set.",
            },
            {
                "metric": "Decision curve / net benefit",
                "for": "Checks whether flagging patients at a threshold is better than flagging everyone or no one.",
            },
            {
                "metric": "Threshold operating points",
                "for": "Shows sensitivity, specificity, precision, and false-negative tradeoffs at several score cutoffs.",
            },
            {
                "metric": "Cost-sensitive threshold",
                "for": "Chooses thresholds after assigning higher cost to missed cases or false alarms.",
            },
            {
                "metric": "Decision-impact simulation",
                "for": "Maps model and timeline signals to simulated clinician-review routing categories.",
            },
            {
                "metric": "False-negative review",
                "for": "Lists positive cases missed by the model; these are highest priority in medical monitoring.",
            },
            {
                "metric": "Subgroup performance",
                "for": "Checks whether performance changes by stage, subtype, age band, or regimen.",
            },
            {
                "metric": "Clinician-loop metrics",
                "for": "Tracks whether clinicians approve, edit, reject, or escalate AI summaries.",
            },
            {
                "metric": "Data coverage",
                "for": "Checks whether CBC, MRI, symptoms, treatment schedule, and longitudinal depth are complete enough.",
            },
            {
                "metric": "LLM summary quality",
                "for": "Uses clinician ratings and edit/reject behavior as a proxy for factuality and usefulness.",
            },
            {
                "metric": "MRI-derived feature summary",
                "for": "Documents the current imaging input as tabular MRI trend features, not raw-MRI diagnosis.",
            },
        ],
        "what_current_metrics_do_not_prove": [
            "They do not prove clinical safety.",
            "They do not prove generalization to real hospitals.",
            "They do not prove fairness across age, stage, subtype, or scanner/site groups.",
            "Synthetic-data metrics mostly prove that the model learned the simulator.",
        ],
        "recommended_next_metrics": [
            "Calibration curve and expected calibration error.",
            "Decision curve analysis / net benefit.",
            "False-negative case review table.",
            "Subgroup performance by stage, molecular subtype, age band, and data source.",
            "Confidence intervals via bootstrap resampling.",
            "Alert precision: how many clinician-review flags were accepted by clinicians.",
            "Time-to-review and override-rate metrics for clinician workflow.",
            "Data freshness and missing-baseline indicators.",
            "Scanner/site/protocol drift when real MRI metadata is available.",
            "LLM summary quality: factuality, completeness, safety, and clinician edit distance.",
        ],
        "next_steps": [
            "Move calibration, threshold policies, subgroup metrics, and false-negative review into saved training reports.",
            "Add BreastDCEDL real-dataset subgroup tables when enough metadata is mapped.",
            "Add visual calibration plots and decision-curve charts.",
            "Start logging clinician decisions and compare AI flags against accepted/rejected reviews.",
            "Separate synthetic simulator metrics from real-dataset metrics visually in the dashboard.",
        ],
    }


def _score_model_metric_set(metrics):
    return {
        "patient_level_roc_auc": _higher_is_better_status(metrics.get("patient_level_roc_auc"), [0.60, 0.70, 0.80, 0.90]),
        "patient_level_average_precision": _higher_is_better_status(metrics.get("patient_level_average_precision"), [0.50, 0.65, 0.80, 0.90]),
        "patient_level_sensitivity": _higher_is_better_status(metrics.get("patient_level_sensitivity"), [0.75, 0.85, 0.90, 0.95]),
        "patient_level_specificity": _higher_is_better_status(metrics.get("patient_level_specificity"), [0.60, 0.70, 0.80, 0.90]),
        "patient_level_brier_score": _lower_is_better_status(metrics.get("patient_level_brier_score"), [0.25, 0.18, 0.12, 0.08]),
    }


def _higher_is_better_status(value, thresholds):
    if value is None:
        return "unavailable"
    if value < thresholds[0]:
        return "failed"
    if value < thresholds[1]:
        return "unideal"
    if value < thresholds[2]:
        return "acceptable"
    if value < thresholds[3]:
        return "strong"
    return "passed"


def _lower_is_better_status(value, thresholds):
    if value is None:
        return "unavailable"
    if value > thresholds[0]:
        return "failed"
    if value > thresholds[1]:
        return "unideal"
    if value > thresholds[2]:
        return "acceptable"
    if value > thresholds[3]:
        return "strong"
    return "passed"


def _standardized_shift_status(value):
    if value < 0.20:
        return "passed"
    if value < 0.50:
        return "acceptable"
    if value < 0.80:
        return "unideal"
    return "failed"


def _missing_rate_status(value):
    if value <= 0.05:
        return "passed"
    if value <= 0.10:
        return "acceptable"
    if value <= 0.20:
        return "unideal"
    return "failed"


def _ece_status(value):
    if value <= 0.03:
        return "passed"
    if value <= 0.06:
        return "strong"
    if value <= 0.10:
        return "acceptable"
    if value <= 0.15:
        return "unideal"
    return "failed"


def _ci_width_status(value):
    if value <= 0.05:
        return "passed"
    if value <= 0.10:
        return "strong"
    if value <= 0.15:
        return "acceptable"
    if value <= 0.25:
        return "unideal"
    return "failed"


def _false_negative_status(value):
    if value <= 0.05:
        return "passed"
    if value <= 0.10:
        return "strong"
    if value <= 0.20:
        return "acceptable"
    if value <= 0.35:
        return "unideal"
    return "failed"


def _subgroup_status(metrics, sample_size):
    if sample_size < 8:
        return "low_support"
    if metrics["roc_auc"] is None:
        return "unavailable"
    return _higher_is_better_status(metrics["roc_auc"], [0.55, 0.65, 0.75, 0.85])


def _acceptance_rate_status(value):
    if value is None:
        return "unavailable"
    return _higher_is_better_status(value, [0.30, 0.45, 0.60, 0.75])


def _quality_score_status(value):
    if value is None:
        return "unavailable"
    if value < 2.0:
        return "failed"
    if value < 3.0:
        return "unideal"
    if value < 4.0:
        return "acceptable"
    if value < 4.5:
        return "strong"
    return "passed"


def _coverage_status(value):
    if value is None:
        return "unavailable"
    if value < 0.70:
        return "failed"
    if value < 0.85:
        return "unideal"
    if value < 0.95:
        return "acceptable"
    return "passed"


def _weighted_error_status(value):
    if value <= 0.05:
        return "passed"
    if value <= 0.10:
        return "strong"
    if value <= 0.18:
        return "acceptable"
    if value <= 0.30:
        return "unideal"
    return "failed"


def _decision_category_meaning(category):
    meanings = {
        "routine_monitoring": "No additional simulated review route from the current model/timeline signals.",
        "close_monitoring": "Model uncertainty would prompt closer monitoring or repeat data checks.",
        "response_concern_review": "Low response signal would prompt clinician response-trend review.",
        "toxicity_review": "CBC or symptom toxicity signal would prompt clinician review.",
        "discordant_response_toxicity_review": "Response looks favorable, but toxicity signals still need review.",
    }
    return meanings.get(category, "Unrecognized review-routing category.")


def _worst_status(statuses):
    priority = {
        "failed": 4,
        "unideal": 3,
        "acceptable": 2,
        "strong": 1,
        "passed": 0,
    }
    status_list = list(statuses)
    if not status_list:
        return "unavailable"
    available = [status for status in status_list if status != "unavailable"]
    if not available:
        return "unavailable"
    status_list = available
    return max(status_list, key=lambda status: priority.get(status, 5))


def _round(value, digits=3):
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return round(numeric, digits)


def _cell(row, column):
    if column not in row:
        return None
    value = row[column]
    if pd.isna(value):
        return None
    return value


def _status_meaning(status):
    meanings = {
        "failed": "Below gate; fix before relying on this signal.",
        "unideal": "Weak or risky; investigate before presenting as strong.",
        "acceptable": "Usable for PoC with clear caveats.",
        "strong": "Good current engineering signal.",
        "passed": "Meets this project gate, not clinical validation.",
        "low_support": "Too few examples for a reliable subgroup claim.",
        "unavailable": "Metric could not be computed.",
    }
    return meanings.get(status, "Status not recognized.")


__all__ = [
    "_metric_interpretation_guide",
    "_score_model_metric_set",
    "_higher_is_better_status",
    "_lower_is_better_status",
    "_standardized_shift_status",
    "_missing_rate_status",
    "_ece_status",
    "_ci_width_status",
    "_false_negative_status",
    "_subgroup_status",
    "_acceptance_rate_status",
    "_quality_score_status",
    "_coverage_status",
    "_weighted_error_status",
    "_decision_category_meaning",
    "_worst_status",
    "_round",
    "_cell",
    "_status_meaning",
]
