"""Dashboard panels built directly from the loaded analytics inputs.

One function per panel of the admin dashboard: model performance, evidence
separation, drift, A/B comparison, data quality and coverage, RAG and guardrail
summaries, and the clinician review loop. Each takes already-loaded data and
returns the panel's payload, so a panel can be read, tested, and changed
without loading anything.

These are engineering-monitoring panels. They describe model and pipeline
behaviour on synthetic data; none of them diagnoses or decides treatment.
"""

import pandas as pd

from backend.models import ModelRegistry, PredictionAuditLog

from backend.services.clinician_feedback import clinical_feedback_summary

from backend.services.admin_metric_interpretation import (
    _acceptance_rate_status,
    _coverage_status,
    _missing_rate_status,
    _quality_score_status,
    _round,
    _score_model_metric_set,
    _standardized_shift_status,
    _status_meaning,
    _worst_status,
)


def _frontend_rag_summary(summary):
    """Keep the React dashboard stable across old and new analytics schemas."""
    summary = dict(summary or {})
    summary.setdefault("evaluations", summary.get("call_count", 0))
    summary.setdefault("grounding_score", summary.get("average_grounding_score"))
    summary.setdefault("hallucination_score", summary.get("average_hallucination_score"))
    summary.setdefault("precision_at_3", summary.get("average_retrieval_precision_at_3"))
    summary.setdefault("estimated_cost_usd", summary.get("estimated_llm_cost_usd"))
    api_costs = summary.get("api_costs") or {}
    summary.setdefault("input_tokens", api_costs.get("estimated_input_tokens"))
    summary.setdefault("output_tokens", api_costs.get("estimated_output_tokens"))
    return summary


def _frontend_guardrail_summary(rag_summary, regression_report):
    regression_summary = (regression_report or {}).get("summary") or {}
    input_counts = (rag_summary or {}).get("input_guardrail_counts") or {}
    output_counts = (rag_summary or {}).get("output_guardrail_counts") or {}
    input_blocks = sum(
        int(count or 0)
        for status, count in input_counts.items()
        if status not in {"passed", "ok", "unknown"}
    )
    output_blocks = sum(
        int(count or 0)
        for status, count in output_counts.items()
        if status not in {"passed", "ok", "unknown"}
    )
    return {
        "input_blocks": input_blocks,
        "output_blocks": output_blocks,
        "attack_block_rate": regression_summary.get("attack_block_rate"),
        "pass_rate": regression_summary.get("pass_rate"),
    }


def _model_performance(synthetic_metrics, breastdcedl_metrics):
    synthetic_models = (synthetic_metrics or {}).get("models") or {}
    best = (synthetic_metrics or {}).get("best_model_by_patient_level_roc_auc")
    best_metrics = synthetic_models.get(best, {}) if best else {}
    synthetic_payload = {
        "task": (synthetic_metrics or {}).get("task"),
        "best_model": best,
        "patient_level_roc_auc": best_metrics.get("patient_level_roc_auc"),
        "patient_level_average_precision": best_metrics.get("patient_level_average_precision"),
        "patient_level_brier_score": best_metrics.get("patient_level_brier_score"),
        "patient_level_sensitivity": best_metrics.get("patient_level_sensitivity"),
        "patient_level_specificity": best_metrics.get("patient_level_specificity"),
        "warning": (synthetic_metrics or {}).get("warning"),
    }
    synthetic_payload["metric_statuses"] = _score_model_metric_set(synthetic_payload)

    return {
        "synthetic_longitudinal_response": {
            **synthetic_payload,
        },
        "real_breastdcedl_baseline": breastdcedl_metrics or {
            "status": "not_available",
            "message": "BreastDCEDL baseline metrics file was not found.",
        },
    }


def _evidence_separation(synthetic_metrics, breastdcedl_metrics):
    synthetic_models = (synthetic_metrics or {}).get("models") or {}
    synthetic_best = (synthetic_metrics or {}).get("best_model_by_patient_level_roc_auc")
    synthetic_best_metrics = synthetic_models.get(synthetic_best, {}) if synthetic_best else {}
    real_models = (breastdcedl_metrics or {}).get("models") or {}
    real_best = (breastdcedl_metrics or {}).get("best_model_by_roc_auc")
    real_best_metrics = real_models.get(real_best, {}) if real_best else {}
    return {
        "purpose": "Separates simulator-learning evidence from real-dataset exploratory evidence so project claims stay honest.",
        "sections": [
            {
                "name": "Synthetic longitudinal simulator",
                "status": "engineering_evidence",
                "rows": (synthetic_metrics or {}).get("rows"),
                "patients": (synthetic_metrics or {}).get("patients"),
                "best_model": synthetic_best,
                "primary_metric": "patient_level_roc_auc",
                "primary_metric_value": synthetic_best_metrics.get("patient_level_roc_auc"),
                "claim_boundary": "Useful for workflow, MLOps, and model-practice evidence. It does not prove real clinical performance.",
            },
            {
                "name": "BreastDCEDL / I-SPY1 MRI-derived baseline",
                "status": "exploratory_real_dataset_baseline" if breastdcedl_metrics else "not_available",
                "rows": (breastdcedl_metrics or {}).get("rows"),
                "patients": (breastdcedl_metrics or {}).get("rows"),
                "best_model": real_best,
                "primary_metric": "roc_auc",
                "primary_metric_value": real_best_metrics.get("roc_auc"),
                "claim_boundary": "Small real-data MRI-derived tabular baseline. Not longitudinal clinical validation.",
            },
            {
                "name": "Raw MRI computer vision",
                "status": "planned_integration",
                "rows": None,
                "patients": None,
                "best_model": None,
                "primary_metric": None,
                "primary_metric_value": None,
                "claim_boundary": "Planned multimodal work. Current longitudinal response models use MRI-derived tabular trend features.",
            },
        ],
    }


def _drift_monitoring(training_rows):
    if training_rows is None or training_rows.empty or "patient_id" not in training_rows.columns:
        return {"status": "unavailable", "features": []}

    patient_order = sorted(training_rows["patient_id"].dropna().unique())
    if len(patient_order) < 4:
        return {"status": "insufficient_data", "features": []}

    midpoint = len(patient_order) // 2
    reference_ids = set(patient_order[:midpoint])
    current_ids = set(patient_order[midpoint:])
    reference = training_rows[training_rows["patient_id"].isin(reference_ids)]
    current = training_rows[training_rows["patient_id"].isin(current_ids)]

    feature_rows = []
    for feature in ["age", "nadir_wbc", "nadir_hemoglobin", "nadir_platelets", "mri_percent_change_from_baseline", "max_symptom_severity"]:
        if feature not in training_rows.columns:
            continue
        ref_mean = float(reference[feature].dropna().mean())
        cur_mean = float(current[feature].dropna().mean())
        ref_std = float(reference[feature].dropna().std() or 1.0)
        standardized_mean_shift = abs(cur_mean - ref_mean) / max(ref_std, 1e-6)
        status = _standardized_shift_status(standardized_mean_shift)
        feature_rows.append({
            "feature": feature,
            "reference_mean": round(ref_mean, 3),
            "current_mean": round(cur_mean, 3),
            "standardized_mean_shift": round(standardized_mean_shift, 3),
            "status": status,
            "meaning": _status_meaning(status),
        })

    watch_count = sum(1 for row in feature_rows if row["status"] in {"unideal", "failed"})
    return {
        "status": _worst_status(row["status"] for row in feature_rows),
        "method": "reference/current split by synthetic patient id; standardized mean shift.",
        "watch_feature_count": watch_count,
        "features": feature_rows,
    }


def _ab_testing(synthetic_metrics, predictions):
    models = (synthetic_metrics or {}).get("models") or {}
    candidates = []
    for name, metrics in models.items():
        candidate = {
            "model": name,
            "patient_level_roc_auc": metrics.get("patient_level_roc_auc"),
            "patient_level_average_precision": metrics.get("patient_level_average_precision"),
            "patient_level_brier_score": metrics.get("patient_level_brier_score"),
            "patient_level_sensitivity": metrics.get("patient_level_sensitivity"),
            "patient_level_specificity": metrics.get("patient_level_specificity"),
        }
        candidate["metric_statuses"] = _score_model_metric_set(candidate)
        candidates.append(candidate)
    candidates = sorted(
        candidates,
        key=lambda row: row.get("patient_level_roc_auc") if row.get("patient_level_roc_auc") is not None else -1,
        reverse=True,
    )

    disagreement = None
    if predictions is not None and not predictions.empty:
        probability_columns = [column for column in predictions.columns if column.endswith("_probability")]
        if len(probability_columns) >= 2:
            label_frame = predictions[probability_columns].apply(lambda column: column >= 0.5)
            disagreement = round(float(label_frame.nunique(axis=1).gt(1).mean()), 3)

    return {
        "champion": candidates[0] if candidates else None,
        "challengers": candidates[1:4],
        "prediction_disagreement_rate": disagreement,
        "recommendation": (
            "Use champion/challenger evaluation offline until clinician-feedback and real-world monitoring data are strong enough."
        ),
    }


def _audit_and_feedback(db):
    return {
        "registered_model_count": db.query(ModelRegistry).count(),
        "prediction_audit_count": db.query(PredictionAuditLog).count(),
        "clinical_feedback": clinical_feedback_summary(db),
    }


def _data_quality(training_rows):
    if training_rows is None or training_rows.empty:
        return {"status": "unavailable", "missingness": []}

    missingness = []
    for column in training_rows.columns:
        rate = float(training_rows[column].isna().mean())
        if rate:
            status = _missing_rate_status(rate)
            missingness.append({
                "column": column,
                "missing_rate": round(rate, 3),
                "status": status,
                "meaning": _status_meaning(status),
            })
    return {
        "status": _worst_status(row["status"] for row in missingness) if missingness else "passed",
        "rows": int(len(training_rows)),
        "patients": int(training_rows["patient_id"].nunique()) if "patient_id" in training_rows.columns else None,
        "missingness": sorted(missingness, key=lambda row: row["missing_rate"], reverse=True)[:20],
    }


def _clinician_loop_metrics(feedback):
    review_count = int((feedback or {}).get("review_count") or 0)
    decisions = (feedback or {}).get("decision_counts") or {}
    if not review_count:
        return {
            "status": "unavailable",
            "purpose": "Measures whether clinicians accept, edit, reject, or escalate AI-generated summaries.",
            "message": "No clinician reviews have been logged yet.",
        }

    accepted = sum(int(decisions.get(name, 0)) for name in ["approved", "edited", "needs_followup"])
    rejected = int(decisions.get("rejected", 0))
    edited = int(decisions.get("edited", 0))
    needs_followup = int(decisions.get("needs_followup", 0))
    acceptance_rate = accepted / review_count
    rejection_rate = rejected / review_count
    edit_rate = edited / review_count
    followup_rate = needs_followup / review_count
    explanation_quality = (feedback or {}).get("average_explanation_quality_score")
    usefulness = (feedback or {}).get("average_model_usefulness_score")

    return {
        "status": _worst_status([
            _acceptance_rate_status(acceptance_rate),
            _quality_score_status(explanation_quality),
            _quality_score_status(usefulness),
        ]),
        "purpose": "A clinician-in-the-loop proxy for alert precision and summary usefulness. It is workflow evidence, not ground-truth clinical accuracy.",
        "review_count": review_count,
        "accepted_review_rate": _round(acceptance_rate),
        "rejected_review_rate": _round(rejection_rate),
        "edited_review_rate": _round(edit_rate),
        "needs_followup_rate": _round(followup_rate),
        "average_explanation_quality_score": explanation_quality,
        "average_model_usefulness_score": usefulness,
        "accepted_review_status": _acceptance_rate_status(acceptance_rate),
        "summary_quality_status": _quality_score_status(explanation_quality),
        "model_usefulness_status": _quality_score_status(usefulness),
    }


def _data_coverage(training_rows):
    if training_rows is None or training_rows.empty:
        return {"status": "unavailable", "items": []}

    items = []
    patient_count = training_rows["patient_id"].nunique() if "patient_id" in training_rows.columns else len(training_rows)
    if "cycle" in training_rows.columns and "patient_id" in training_rows.columns:
        cycles_per_patient = training_rows.groupby("patient_id")["cycle"].nunique()
        complete_rate = float((cycles_per_patient >= 6).mean())
        items.append(_coverage_item("Longitudinal depth", complete_rate, f"{int((cycles_per_patient >= 6).sum())}/{patient_count} patients have at least 6 cycles."))

    for name, columns in [
        ("CBC coverage", ["pre_wbc", "pre_anc", "pre_hemoglobin", "pre_platelets", "nadir_wbc", "nadir_anc", "nadir_hemoglobin", "nadir_platelets"]),
        ("MRI trend coverage", ["mri_tumor_size_cm", "mri_percent_change_from_baseline"]),
        ("Treatment schedule coverage", ["treatment_date", "cycle", "regimen"]),
        ("Symptom/toxicity coverage", ["max_symptom_severity", "symptom_count", "intervention_count"]),
    ]:
        available = [column for column in columns if column in training_rows.columns]
        if not available:
            items.append({"name": name, "coverage_rate": None, "status": "unavailable", "detail": "Required columns are unavailable."})
            continue
        coverage_rate = float(1.0 - training_rows[available].isna().mean().mean())
        items.append(_coverage_item(name, coverage_rate, f"{len(available)}/{len(columns)} expected columns present."))

    statuses = [item["status"] for item in items]
    return {
        "status": _worst_status(statuses),
        "purpose": "Shows whether the longitudinal dataset is complete enough to trust model and timeline metrics.",
        "rows": int(len(training_rows)),
        "patients": int(patient_count),
        "items": items,
    }


def _coverage_item(name, rate, detail):
    return {
        "name": name,
        "coverage_rate": _round(rate),
        "status": _coverage_status(rate),
        "detail": detail,
    }


def _mri_report_feature_pipeline(mri_reports):
    if mri_reports is None or mri_reports.empty or "patient_id" not in mri_reports.columns:
        return {
            "status": "unavailable",
            "purpose": "No MRI report table was available for derived-feature inventory.",
            "steps": [],
        }

    reports = mri_reports.copy()
    reports["date"] = pd.to_datetime(reports.get("date"), errors="coerce")
    sort_columns = [column for column in ["patient_id", "date", "cycle"] if column in reports.columns]
    reports = reports.sort_values(sort_columns)
    patient_count = int(reports["patient_id"].nunique())
    baseline = reports[reports.get("timepoint", "").astype(str).str.lower().eq("baseline")] if "timepoint" in reports.columns else pd.DataFrame()
    followup = reports[~reports.index.isin(baseline.index)] if not baseline.empty else reports

    patient_latest = reports.groupby("patient_id", as_index=False).tail(1)
    change_column = "percent_change_from_baseline"
    latest_changes = patient_latest[change_column].dropna().astype(float) if change_column in patient_latest.columns else pd.Series(dtype=float)
    size_values = reports["tumor_size_cm"].dropna().astype(float) if "tumor_size_cm" in reports.columns else pd.Series(dtype=float)
    coverage = float(len(latest_changes) / patient_count) if patient_count else 0.0
    status = _coverage_status(coverage)
    trend_buckets = {
        "strong_decrease": int((latest_changes <= -50).sum()),
        "partial_decrease": int(((latest_changes > -50) & (latest_changes <= -20)).sum()),
        "stable_or_weak_decrease": int(((latest_changes > -20) & (latest_changes <= 10)).sum()),
        "increase": int((latest_changes > 10).sum()),
    }

    return {
        "status": status,
        "purpose": "Inventory of the MRI-derived feature pipeline from synthetic MRI measurements.",
        "patients_with_mri": patient_count,
        "measurement_rows": int(len(reports)),
        "patients_with_baseline": int(baseline["patient_id"].nunique()) if not baseline.empty else 0,
        "patients_with_followup": int(followup["patient_id"].nunique()) if not followup.empty else 0,
        "latest_change_coverage": _round(coverage),
        "tumor_size_cm_mean": _round(size_values.mean()) if not size_values.empty else None,
        "latest_percent_change_mean": _round(latest_changes.mean()) if not latest_changes.empty else None,
        "response_trend_buckets": trend_buckets,
        "steps": [
            "Read one synthetic MRI measurement row per patient baseline and treatment-cycle follow-up.",
            "Sort measurements by patient and date.",
            "Use baseline tumor size as the reference measurement.",
            "Compute latest tumor size and percent change from baseline.",
            "Bucket MRI-derived trend as strong decrease, partial decrease, stable/weak decrease, or increase.",
            "Join MRI-derived trend features into the longitudinal treatment model table.",
        ],
    }
