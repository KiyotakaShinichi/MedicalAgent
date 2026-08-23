"""Admin analytics dashboard payload.

Engineering monitoring for ML and retrieval behaviour on synthetic data. These
panels do not diagnose, do not decide treatment, and are not clinical evidence.

The implementation is split by responsibility:

* ``admin_analytics_readiness`` - artifact loading and the MLE readiness panel;
  the one place that decides what a missing input means;
* ``admin_analytics_panels`` - one function per dashboard panel, each built
  from already-loaded data;
* ``admin_analytics_summary`` - the statistical layer: intervals, thresholds,
  decision analysis, subgroup and false-negative review.

This module composes them and remains the public import surface, so
``from backend.services.admin_analytics import build_admin_analytics`` keeps
working for the admin router and the evaluation reports. The private panel
helpers are re-exported too, because they were module attributes here before
the split.
"""

from backend.services.app_logging import build_app_monitoring_summary
from backend.services.agent_feedback import build_agent_feedback_summary
from backend.services.agent_regression_eval import load_latest_agent_regression_report
from backend.services.rag_analytics import build_rag_evaluation_summary
from backend.services.rag_source_registry import build_rag_source_registry
from backend.services.summary_quality_eval import build_summary_quality_report
from backend.services.admin_dashboard_panels import (
    _calibration_panel,
    _confusion_matrix_panel,
    _domain_gap_analysis,
    _evaluation_report_links,
    _feature_explanation_panel,
)
from backend.services.admin_metric_interpretation import _metric_interpretation_guide
from backend.services.admin_analytics_readiness import (
    DEFAULT_BREASTDCEDL_METRICS_PATH,
    DEFAULT_MLE_READINESS_PATH,
    DEFAULT_SYNTHETIC_METRICS_PATH,
    DEFAULT_SYNTHETIC_MRI_REPORTS_CSV,
    DEFAULT_SYNTHETIC_PREDICTIONS_PATH,
    DEFAULT_SYNTHETIC_TRAINING_CSV,
    _load_csv,
    _load_json,
    _prefer_existing as _prefer_existing,
)
from backend.services.admin_analytics_panels import (
    _ab_testing,
    _audit_and_feedback,
    _clinician_loop_metrics,
    _coverage_item as _coverage_item,
    _data_coverage,
    _data_quality,
    _drift_monitoring,
    _evidence_separation,
    _frontend_guardrail_summary,
    _frontend_rag_summary,
    _model_performance,
    _mri_report_feature_pipeline as _mri_report_feature_pipeline,
)
from backend.services.admin_analytics_summary import (
    _advanced_model_evaluation,
    _binary_metric_summary as _binary_metric_summary,
    _bootstrap_confidence_intervals as _bootstrap_confidence_intervals,
    _cost_sensitive_thresholds as _cost_sensitive_thresholds,
    _decision_curve as _decision_curve,
    _decision_impact_simulation as _decision_impact_simulation,
    _false_negative_review as _false_negative_review,
    _metric_estimate as _metric_estimate,
    _mri_derived_feature_summary as _mri_derived_feature_summary,
    _patient_context as _patient_context,
    _subgroup_performance as _subgroup_performance,
    _threshold_operating_points as _threshold_operating_points,
)


def build_admin_analytics(db):
    synthetic_metrics = _load_json(DEFAULT_SYNTHETIC_METRICS_PATH)
    breastdcedl_metrics = _load_json(DEFAULT_BREASTDCEDL_METRICS_PATH)
    predictions = _load_csv(DEFAULT_SYNTHETIC_PREDICTIONS_PATH)
    training_rows = _load_csv(DEFAULT_SYNTHETIC_TRAINING_CSV)
    mri_reports = _load_csv(DEFAULT_SYNTHETIC_MRI_REPORTS_CSV)
    audit_and_feedback = _audit_and_feedback(db)

    advanced_eval = _advanced_model_evaluation(synthetic_metrics, predictions, training_rows, mri_reports)
    rag_summary = _frontend_rag_summary(build_rag_evaluation_summary(db))
    regression_report = load_latest_agent_regression_report()
    guardrails = _frontend_guardrail_summary(rag_summary, regression_report)
    readiness = _load_json(DEFAULT_MLE_READINESS_PATH) or {
        "status": "unavailable",
        "message": "No precomputed MLE readiness artifact is available.",
        "clinical_validation": False,
    }
    return {
        "roles": {
            "patient": "Personal portal, uploads, symptom/CBC/medication logging, support agent.",
            "clinician": "Patient list, timeline review, AI summary approval/edit/reject workflow.",
            "admin": "Model evaluation, drift monitoring, A/B comparison, audit and feedback analytics.",
        },
        "model_performance": _model_performance(synthetic_metrics, breastdcedl_metrics),
        "evidence_separation": _evidence_separation(synthetic_metrics, breastdcedl_metrics),
        "metric_interpretation_guide": _metric_interpretation_guide(),
        "advanced_model_evaluation": advanced_eval,
        "confusion_matrix": _confusion_matrix_panel(advanced_eval),
        "calibration_panel": _calibration_panel(advanced_eval),
        "feature_explanation": _feature_explanation_panel(),
        "evaluation_report_links": _evaluation_report_links(),
        "drift_monitoring": _drift_monitoring(training_rows),
        "ab_testing": _ab_testing(synthetic_metrics, predictions),
        "audit_and_feedback": audit_and_feedback,
        "app_monitoring": build_app_monitoring_summary(db),
        "rag_evaluation": rag_summary,
        "guardrails": guardrails,
        "rag_source_registry": build_rag_source_registry(),
        "agent_regression_evaluation": regression_report,
        "agent_feedback": build_agent_feedback_summary(db),
        "summary_quality_eval": build_summary_quality_report(db=db),
        "clinician_loop_metrics": _clinician_loop_metrics(audit_and_feedback["clinical_feedback"]),
        "data_quality": _data_quality(training_rows),
        "data_coverage": _data_coverage(training_rows),
        "mle_readiness": readiness,
        "api_cost": {"estimated_cost_usd": rag_summary.get("estimated_cost_usd")},
        "domain_gap_analysis": _domain_gap_analysis(breastdcedl_metrics),
        "safety_positioning": (
            "Admin analytics are for ML engineering monitoring only. They do not diagnose or make treatment decisions."
        ),
    }

# ── compatibility re-exports ────────────────────────────────────────────────
# These were module attributes of admin_analytics before the split, because the
# original module imported them at module scope. Callers and tests can bind to
# any of them - `_calibration_metrics` is monkeypatched by the breast-monitoring
# suite - so the facade stays a strict superset of the pre-split surface rather
# than only of the names it happens to define.
import json as json  # noqa: E402
from pathlib import Path as Path  # noqa: E402

import numpy as np  # noqa: E402,F401
import pandas as pd  # noqa: E402,F401
from sklearn.metrics import (  # noqa: E402
    average_precision_score as average_precision_score,
    brier_score_loss as brier_score_loss,
    confusion_matrix as confusion_matrix,
    roc_auc_score as roc_auc_score,
)

from backend.models import (  # noqa: E402
    ModelRegistry as ModelRegistry,
    PredictionAuditLog as PredictionAuditLog,
)
from backend.services.admin_calibration import (  # noqa: E402
    _calibration_metrics as _calibration_metrics,
)
from backend.services.clinician_feedback import (  # noqa: E402
    clinical_feedback_summary as clinical_feedback_summary,
)
from backend.services.mri_derived_features import (  # noqa: E402,F401
    build_mri_derived_feature_summary as build_mri_derived_feature_summary_service,
)
from backend.services.admin_metric_interpretation import (  # noqa: E402
    _acceptance_rate_status as _acceptance_rate_status,
    _cell as _cell,
    _ci_width_status as _ci_width_status,
    _coverage_status as _coverage_status,
    _decision_category_meaning as _decision_category_meaning,
    _false_negative_status as _false_negative_status,
    _missing_rate_status as _missing_rate_status,
    _quality_score_status as _quality_score_status,
    _round as _round,
    _score_model_metric_set as _score_model_metric_set,
    _standardized_shift_status as _standardized_shift_status,
    _status_meaning as _status_meaning,
    _subgroup_status as _subgroup_status,
    _weighted_error_status as _weighted_error_status,
    _worst_status as _worst_status,
)

__all__ = ["build_admin_analytics"]
