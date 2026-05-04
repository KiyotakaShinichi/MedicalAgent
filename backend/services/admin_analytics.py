import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score

from backend.models import ModelRegistry, PredictionAuditLog
from backend.services.clinician_feedback import clinical_feedback_summary


DEFAULT_SYNTHETIC_METRICS_PATH = "Data/complete_synthetic_training/complete_synthetic_model_metrics.json"
DEFAULT_SYNTHETIC_PREDICTIONS_PATH = "Data/complete_synthetic_training/complete_synthetic_model_predictions.csv"
DEFAULT_SYNTHETIC_TRAINING_CSV = "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv"
DEFAULT_SYNTHETIC_MRI_REPORTS_CSV = "Data/complete_synthetic_breast_journeys/mri_reports.csv"
DEFAULT_BREASTDCEDL_METRICS_PATH = "Data/breastdcedl_spy1_baseline_metrics.json"


def build_admin_analytics(db):
    synthetic_metrics = _load_json(DEFAULT_SYNTHETIC_METRICS_PATH)
    breastdcedl_metrics = _load_json(DEFAULT_BREASTDCEDL_METRICS_PATH)
    predictions = _load_csv(DEFAULT_SYNTHETIC_PREDICTIONS_PATH)
    training_rows = _load_csv(DEFAULT_SYNTHETIC_TRAINING_CSV)
    mri_reports = _load_csv(DEFAULT_SYNTHETIC_MRI_REPORTS_CSV)
    audit_and_feedback = _audit_and_feedback(db)

    return {
        "roles": {
            "patient": "Personal portal, uploads, symptom/CBC/medication logging, support agent.",
            "clinician": "Patient list, timeline review, AI summary approval/edit/reject workflow.",
            "admin": "Model evaluation, drift monitoring, A/B comparison, audit and feedback analytics.",
        },
        "model_performance": _model_performance(synthetic_metrics, breastdcedl_metrics),
        "metric_interpretation_guide": _metric_interpretation_guide(),
        "advanced_model_evaluation": _advanced_model_evaluation(synthetic_metrics, predictions, training_rows, mri_reports),
        "drift_monitoring": _drift_monitoring(training_rows),
        "ab_testing": _ab_testing(synthetic_metrics, predictions),
        "audit_and_feedback": audit_and_feedback,
        "clinician_loop_metrics": _clinician_loop_metrics(audit_and_feedback["clinical_feedback"]),
        "data_quality": _data_quality(training_rows),
        "data_coverage": _data_coverage(training_rows),
        "safety_positioning": (
            "Admin analytics are for ML engineering monitoring only. They do not diagnose or make treatment decisions."
        ),
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


def _advanced_model_evaluation(synthetic_metrics, predictions, training_rows, mri_reports=None):
    best_model = (synthetic_metrics or {}).get("best_model_by_patient_level_roc_auc")
    if predictions is None or predictions.empty or not best_model:
        return {
            "status": "unavailable",
            "message": "Synthetic prediction artifacts are required for advanced evaluation.",
        }

    probability_column = f"{best_model}_probability"
    if probability_column not in predictions.columns or "actual_label" not in predictions.columns:
        return {
            "status": "unavailable",
            "message": f"Prediction column {probability_column} was not found.",
        }

    frame = predictions[["patient_id", "actual_label", probability_column]].copy()
    frame = frame.dropna(subset=["actual_label", probability_column])
    frame["actual_label"] = frame["actual_label"].astype(int)
    frame["probability"] = frame[probability_column].astype(float)
    frame["predicted_label"] = (frame["probability"] >= 0.5).astype(int)

    context = _patient_context(training_rows)
    if context is not None:
        frame = frame.merge(context, on="patient_id", how="left")

    labels = frame["actual_label"].to_numpy(dtype=int)
    probabilities = frame["probability"].to_numpy(dtype=float)
    calibration = _calibration_metrics(labels, probabilities)
    confidence_intervals = _bootstrap_confidence_intervals(labels, probabilities)
    false_negative_review = _false_negative_review(frame)
    subgroup_performance = _subgroup_performance(frame)
    threshold_metrics = _threshold_operating_points(labels, probabilities)
    cost_sensitive = _cost_sensitive_thresholds(labels, probabilities)
    decision_impact = _decision_impact_simulation(frame)
    mri_features = _mri_derived_feature_summary(frame, mri_reports)

    return {
        "status": _worst_status([
            calibration["status"],
            confidence_intervals["status"],
            false_negative_review["status"],
            subgroup_performance["status"],
            cost_sensitive["status"],
        ]),
        "champion_model": best_model,
        "threshold": 0.5,
        "evaluated_patients": int(len(frame)),
        "calibration": calibration,
        "bootstrap_confidence_intervals": confidence_intervals,
        "decision_curve": _decision_curve(labels, probabilities),
        "threshold_operating_points": threshold_metrics,
        "cost_sensitive_thresholds": cost_sensitive,
        "decision_impact_simulation": decision_impact,
        "false_negative_review": false_negative_review,
        "subgroup_performance": subgroup_performance,
        "mri_derived_features": mri_features,
    }


def _patient_context(training_rows):
    if training_rows is None or training_rows.empty or "patient_id" not in training_rows.columns:
        return None

    rows = training_rows.sort_values(["patient_id", "cycle"]).copy()
    aggregations = {}
    for output, column, func in [
        ("age", "age", "first"),
        ("stage", "stage", "first"),
        ("molecular_subtype", "molecular_subtype", "first"),
        ("regimen", "regimen", "first"),
        ("cycles_observed", "cycle", "max"),
        ("latest_mri_percent_change", "mri_percent_change_from_baseline", "last"),
        ("latest_mri_tumor_size_cm", "mri_tumor_size_cm", "last"),
        ("max_symptom_severity", "max_symptom_severity", "max"),
        ("nadir_wbc", "nadir_wbc", "min"),
        ("nadir_anc", "nadir_anc", "min"),
        ("nadir_hemoglobin", "nadir_hemoglobin", "min"),
        ("nadir_platelets", "nadir_platelets", "min"),
        ("intervention_count", "intervention_count", "sum"),
        ("dose_delay_count", "dose_delayed", "sum"),
        ("dose_reduction_count", "dose_reduced", "sum"),
        ("final_cancer_status", "final_cancer_status", "last"),
        ("final_response_category", "final_response_category", "last"),
    ]:
        if column in rows.columns:
            aggregations[output] = (column, func)
    context = rows.groupby("patient_id", as_index=False).agg(**aggregations)
    if "age" in context.columns:
        context["age_band"] = pd.cut(
            context["age"],
            bins=[0, 44, 54, 64, 74, 120],
            labels=["<45", "45-54", "55-64", "65-74", "75+"],
            include_lowest=True,
        ).astype(str)
    return context


def _calibration_metrics(labels, probabilities, bins=10):
    if len(labels) == 0:
        return {"status": "unavailable", "bins": []}

    bin_rows = []
    total = len(labels)
    expected_calibration_error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        if index == bins - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        count = int(mask.sum())
        if count:
            mean_probability = float(probabilities[mask].mean())
            observed_rate = float(labels[mask].mean())
            gap = abs(observed_rate - mean_probability)
            expected_calibration_error += (count / total) * gap
            bin_rows.append({
                "range": f"{lower:.1f}-{upper:.1f}",
                "count": count,
                "mean_probability": _round(mean_probability),
                "observed_positive_rate": _round(observed_rate),
                "gap": _round(gap),
            })

    ece = _round(expected_calibration_error)
    return {
        "status": _ece_status(expected_calibration_error),
        "purpose": "Checks whether predicted probabilities behave like real probabilities, not just rankings.",
        "expected_calibration_error": ece,
        "brier_score": _round(brier_score_loss(labels, probabilities)),
        "bins": bin_rows,
    }


def _bootstrap_confidence_intervals(labels, probabilities, resamples=300, seed=42):
    if len(labels) < 10:
        return {"status": "unavailable", "metrics": [], "resamples": 0}

    rng = np.random.default_rng(seed)
    metric_values = {"AUROC": [], "AUPRC": [], "Brier": []}
    n = len(labels)
    for _ in range(resamples):
        indices = rng.integers(0, n, n)
        sample_labels = labels[indices]
        sample_probabilities = probabilities[indices]
        if len(set(sample_labels.tolist())) > 1:
            metric_values["AUROC"].append(float(roc_auc_score(sample_labels, sample_probabilities)))
            metric_values["AUPRC"].append(float(average_precision_score(sample_labels, sample_probabilities)))
        metric_values["Brier"].append(float(brier_score_loss(sample_labels, sample_probabilities)))

    rows = []
    statuses = []
    for metric, values in metric_values.items():
        if not values:
            rows.append({"metric": metric, "status": "unavailable"})
            statuses.append("unavailable")
            continue
        low, high = np.quantile(values, [0.025, 0.975])
        estimate = _metric_estimate(metric, labels, probabilities)
        width = float(high - low)
        status = _ci_width_status(width)
        statuses.append(status)
        rows.append({
            "metric": metric,
            "estimate": _round(estimate),
            "ci_low": _round(low),
            "ci_high": _round(high),
            "interval_width": _round(width),
            "status": status,
        })

    return {
        "status": _worst_status(statuses),
        "purpose": "Shows how stable the metric is under resampling; wide intervals mean the validation set is too small or noisy.",
        "method": "Patient-level bootstrap with replacement.",
        "resamples": resamples,
        "metrics": rows,
    }


def _decision_curve(labels, probabilities, thresholds=None):
    thresholds = thresholds or [0.30, 0.50, 0.70]
    n = len(labels)
    prevalence = float(np.mean(labels)) if n else 0.0
    rows = []
    for threshold in thresholds:
        predictions = probabilities >= threshold
        tp = int(((predictions == 1) & (labels == 1)).sum())
        fp = int(((predictions == 1) & (labels == 0)).sum())
        net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold)) if n else None
        treat_all = prevalence - (1 - prevalence) * (threshold / (1 - threshold))
        rows.append({
            "threshold": threshold,
            "flagged_patients": int(predictions.sum()),
            "true_positive": tp,
            "false_positive": fp,
            "model_net_benefit": _round(net_benefit),
            "treat_all_net_benefit": _round(treat_all),
            "treat_none_net_benefit": 0.0,
            "status": "passed" if net_benefit is not None and net_benefit > max(treat_all, 0) else "unideal",
        })
    return {
        "purpose": "Estimates whether using the model at a threshold adds value versus flagging everyone or no one.",
        "rows": rows,
    }


def _threshold_operating_points(labels, probabilities, thresholds=None):
    thresholds = thresholds or [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    rows = []
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) else None
        specificity = tn / (tn + fp) if (tn + fp) else None
        precision = tp / (tp + fp) if (tp + fp) else None
        false_negative_rate = fn / (tp + fn) if (tp + fn) else None
        false_positive_rate = fp / (tn + fp) if (tn + fp) else None
        rows.append({
            "threshold": threshold,
            "flagged_positive_rate": _round(float(predictions.mean())),
            "sensitivity": _round(sensitivity),
            "specificity": _round(specificity),
            "precision": _round(precision),
            "false_negative_rate": _round(false_negative_rate),
            "false_positive_rate": _round(false_positive_rate),
            "true_positive": int(tp),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_negative": int(tn),
            "status": _false_negative_status(false_negative_rate or 0),
        })
    return {
        "purpose": "Shows the tradeoff between catching positive response cases and adding false alarms at different thresholds.",
        "rows": rows,
    }


def _cost_sensitive_thresholds(labels, probabilities, thresholds=None):
    thresholds = thresholds or [round(value / 100, 2) for value in range(10, 91, 5)]
    policies = [
        {
            "name": "safety_first",
            "false_negative_cost": 5,
            "false_positive_cost": 1,
            "purpose": "Prioritizes avoiding missed positive/benefit cases, accepting more review flags.",
        },
        {
            "name": "balanced",
            "false_negative_cost": 2,
            "false_positive_cost": 1,
            "purpose": "Keeps false negatives more expensive while limiting unnecessary review burden.",
        },
        {
            "name": "precision_first",
            "false_negative_cost": 1,
            "false_positive_cost": 2,
            "purpose": "Prioritizes fewer false alarms when review capacity is limited.",
        },
    ]
    rows = []
    statuses = []
    for policy in policies:
        candidates = []
        for threshold in thresholds:
            predictions = (probabilities >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
            weighted_error = (
                (policy["false_negative_cost"] * fn) + (policy["false_positive_cost"] * fp)
            ) / max(len(labels), 1)
            candidates.append({
                "threshold": threshold,
                "weighted_error": float(weighted_error),
                "true_positive": int(tp),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_negative": int(tn),
                "sensitivity": tp / (tp + fn) if (tp + fn) else None,
                "specificity": tn / (tn + fp) if (tn + fp) else None,
            })
        best = min(candidates, key=lambda row: row["weighted_error"])
        status = _weighted_error_status(best["weighted_error"])
        statuses.append(status)
        rows.append({
            **policy,
            "recommended_threshold": best["threshold"],
            "weighted_error": _round(best["weighted_error"]),
            "sensitivity": _round(best["sensitivity"]),
            "specificity": _round(best["specificity"]),
            "false_negative": best["false_negative"],
            "false_positive": best["false_positive"],
            "status": status,
        })
    return {
        "status": _worst_status(statuses),
        "purpose": "Treats errors differently so the threshold matches the clinical-review workflow being simulated.",
        "policies": rows,
    }


def _decision_impact_simulation(frame):
    rows = []
    for _, row in frame.iterrows():
        probability = float(row["probability"])
        latest_mri_change = _cell(row, "latest_mri_percent_change")
        max_symptom = _cell(row, "max_symptom_severity")
        nadir_wbc = _cell(row, "nadir_wbc")
        nadir_anc = _cell(row, "nadir_anc")
        interventions = _cell(row, "intervention_count") or 0

        toxicity_review = (
            (max_symptom is not None and float(max_symptom) >= 7)
            or (nadir_wbc is not None and float(nadir_wbc) < 2.0)
            or (nadir_anc is not None and float(nadir_anc) < 1.0)
        )
        mri_unfavorable = latest_mri_change is not None and float(latest_mri_change) > -20

        if toxicity_review and probability >= 0.70:
            category = "discordant_response_toxicity_review"
            action = "Favorable response signal, but toxicity signals would route to clinician review."
        elif toxicity_review:
            category = "toxicity_review"
            action = "CBC/symptom toxicity signals would route to clinician review."
        elif probability < 0.35 or (probability < 0.50 and mri_unfavorable):
            category = "response_concern_review"
            action = "Low response probability or weak MRI improvement would route to response-trend review."
        elif probability < 0.65:
            category = "close_monitoring"
            action = "Uncertain response signal would trigger closer monitoring and repeat data check."
        else:
            category = "routine_monitoring"
            action = "Favorable response signal would remain in routine monitoring, assuming no clinician concern."

        rows.append({
            "patient_id": row["patient_id"],
            "category": category,
            "action": action,
            "probability": _round(probability),
            "latest_mri_percent_change": _round(latest_mri_change),
            "max_symptom_severity": _round(max_symptom),
            "nadir_wbc": _round(nadir_wbc),
            "intervention_count": _round(interventions),
        })

    category_counts = {}
    for row in rows:
        category_counts[row["category"]] = category_counts.get(row["category"], 0) + 1
    total = max(len(rows), 1)
    summary = [
        {
            "category": category,
            "count": count,
            "rate": _round(count / total),
            "meaning": _decision_category_meaning(category),
        }
        for category, count in sorted(category_counts.items())
    ]
    return {
        "purpose": "Simulates what model/timeline signals would change in the clinician-review workflow. It does not recommend treatment changes.",
        "categories": summary,
        "examples": rows[:12],
        "safety_note": "These are review-routing categories, not chemotherapy recommendations.",
    }


def _mri_derived_feature_summary(frame, mri_reports=None):
    report_summary = _mri_report_feature_pipeline(mri_reports)
    if "latest_mri_percent_change" not in frame.columns:
        return {
            "status": report_summary.get("status", "unavailable"),
            "purpose": "MRI-derived features are expected, but no MRI trend columns were found.",
            "features": [],
            "report_pipeline": report_summary,
        }

    changes = frame["latest_mri_percent_change"].dropna().astype(float)
    sizes = frame["latest_mri_tumor_size_cm"].dropna().astype(float) if "latest_mri_tumor_size_cm" in frame.columns else pd.Series(dtype=float)
    if changes.empty:
        return {
            "status": "unavailable",
            "purpose": "MRI-derived features exist in schema, but values are missing for this evaluation split.",
            "features": [],
            "report_pipeline": report_summary,
        }

    categories = {
        "strong_decrease": int((changes <= -50).sum()),
        "partial_decrease": int(((changes > -50) & (changes <= -20)).sum()),
        "stable_or_weak_decrease": int(((changes > -20) & (changes <= 10)).sum()),
        "increase": int((changes > 10).sum()),
    }
    return {
        "status": "acceptable",
        "purpose": "Summarizes MRI-derived numeric trend features used by the longitudinal simulator; this is not raw MRI interpretation.",
        "features": [
            {
                "name": "latest_mri_percent_change",
                "meaning": "Percent tumor-size change from baseline MRI-derived measurement.",
                "mean": _round(changes.mean()),
                "min": _round(changes.min()),
                "max": _round(changes.max()),
            },
            {
                "name": "latest_mri_tumor_size_cm",
                "meaning": "Latest tumor-size measurement in centimeters when available.",
                "mean": _round(sizes.mean()) if not sizes.empty else None,
                "min": _round(sizes.min()) if not sizes.empty else None,
                "max": _round(sizes.max()) if not sizes.empty else None,
            },
        ],
        "response_trend_buckets": categories,
        "report_pipeline": report_summary,
        "safety_note": "Raw DICOM/NIfTI computer vision remains a planned integration path. Current longitudinal models use MRI-derived tabular features.",
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


def _false_negative_review(frame):
    positives = int(frame["actual_label"].sum())
    false_negative_mask = (frame["actual_label"] == 1) & (frame["predicted_label"] == 0)
    false_negatives = frame[false_negative_mask].copy()
    rate = float(len(false_negatives) / positives) if positives else 0.0
    cases = []
    for _, row in false_negatives.sort_values("probability", ascending=False).head(10).iterrows():
        cases.append({
            "patient_id": row["patient_id"],
            "probability": _round(row["probability"]),
            "stage": _cell(row, "stage"),
            "molecular_subtype": _cell(row, "molecular_subtype"),
            "latest_mri_percent_change": _round(_cell(row, "latest_mri_percent_change")),
            "max_symptom_severity": _round(_cell(row, "max_symptom_severity")),
            "nadir_wbc": _round(_cell(row, "nadir_wbc")),
            "final_cancer_status": _cell(row, "final_cancer_status"),
        })
    return {
        "status": _false_negative_status(rate),
        "purpose": "Finds positive/benefit cases the model missed at the current threshold. These are the cases to inspect first in medical ML.",
        "count": int(len(false_negatives)),
        "positive_cases": positives,
        "false_negative_rate": _round(rate),
        "cases": cases,
    }


def _subgroup_performance(frame):
    rows = []
    statuses = []
    for column, label in [
        ("stage", "Cancer stage"),
        ("molecular_subtype", "Molecular subtype"),
        ("age_band", "Age band"),
        ("regimen", "Treatment regimen"),
    ]:
        if column not in frame.columns:
            continue
        for value, group in frame.dropna(subset=[column]).groupby(column):
            if str(value) in {"nan", ""}:
                continue
            labels = group["actual_label"].to_numpy(dtype=int)
            probabilities = group["probability"].to_numpy(dtype=float)
            metrics = _binary_metric_summary(labels, probabilities)
            status = _subgroup_status(metrics, len(group))
            statuses.append(status)
            rows.append({
                "group": label,
                "value": str(value),
                "n": int(len(group)),
                "positive_rate": _round(float(labels.mean())) if len(labels) else None,
                "roc_auc": metrics["roc_auc"],
                "average_precision": metrics["average_precision"],
                "brier_score": metrics["brier_score"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "status": status,
            })

    rows = sorted(rows, key=lambda row: (row["group"], -row["n"], row["value"]))
    return {
        "status": _worst_status(statuses) if statuses else "unavailable",
        "purpose": "Checks whether the model behaves differently across clinically relevant groups.",
        "rows": rows[:40],
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


def _metric_estimate(metric, labels, probabilities):
    if metric == "AUROC":
        return roc_auc_score(labels, probabilities)
    if metric == "AUPRC":
        return average_precision_score(labels, probabilities)
    return brier_score_loss(labels, probabilities)


def _binary_metric_summary(labels, probabilities):
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "roc_auc": _round(roc_auc_score(labels, probabilities)) if len(set(labels.tolist())) > 1 else None,
        "average_precision": _round(average_precision_score(labels, probabilities)) if len(set(labels.tolist())) > 1 else None,
        "brier_score": _round(brier_score_loss(labels, probabilities)),
        "sensitivity": _round(tp / (tp + fn)) if (tp + fn) else None,
        "specificity": _round(tn / (tn + fp)) if (tn + fp) else None,
    }


def _load_json(path):
    json_path = Path(path)
    if not json_path.exists():
        return None
    return json.loads(json_path.read_text(encoding="utf-8"))


def _load_csv(path):
    csv_path = Path(path)
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


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
        return "unideal"
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
        "unavailable": "Metric could not be computed.",
    }
    return meanings.get(status, "Status not recognized.")
