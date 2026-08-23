"""Model-evaluation summary: thresholds, intervals, and decision analysis.

The statistical layer beneath the panels. `_advanced_model_evaluation`
composes it: bootstrap confidence intervals, the decision curve, threshold
operating points, cost-sensitive thresholds, decision-impact simulation,
subgroup performance, and false-negative review.

These are the numbers most easily over-read, so the helpers report intervals
and operating points rather than a single headline score - a point estimate
with no interval invites a confidence the synthetic data does not support.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

from backend.services.admin_calibration import _calibration_metrics
from backend.services.mri_derived_features import (
    build_mri_derived_feature_summary as build_mri_derived_feature_summary_service,
)

from backend.services.admin_metric_interpretation import (
    _cell,
    _ci_width_status,
    _decision_category_meaning,
    _false_negative_status,
    _round,
    _subgroup_status,
    _weighted_error_status,
    _worst_status,
)


def _advanced_model_evaluation(synthetic_metrics, predictions, training_rows, mri_reports=None):
    best_model = (synthetic_metrics or {}).get("best_model_by_patient_level_roc_auc")
    if predictions is None or predictions.empty or not best_model:
        return {
            "status": "unavailable",
            "message": "Synthetic prediction artifacts are required for advanced evaluation.",
        }

    calibrated_column = f"{best_model}_calibrated_probability"
    raw_probability_column = f"{best_model}_probability"
    probability_column = calibrated_column if calibrated_column in predictions.columns else raw_probability_column
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
        "probability_column": probability_column,
        "probability_source": "calibrated_champion" if probability_column == calibrated_column else "raw_champion",
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
    return build_mri_derived_feature_summary_service(
        evaluation_frame=frame,
        mri_reports=mri_reports,
    )


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
    gate_statuses = []
    powered_statuses = []
    low_support_groups = []
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
            gate_status = "acceptable" if status == "low_support" else status
            gate_statuses.append(gate_status)
            if status == "low_support":
                low_support_groups.append({"group": label, "value": str(value), "n": int(len(group))})
            else:
                powered_statuses.append(status)
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
        "status": _worst_status(gate_statuses) if gate_statuses else "unavailable",
        "purpose": "Checks whether the model behaves differently across clinically relevant groups.",
        "powered_group_status": _worst_status(powered_statuses),
        "low_support_group_count": len(low_support_groups),
        "low_support_groups": low_support_groups[:12],
        "interpretation": (
            "Low-support groups are validation coverage gaps, not proof of model failure. "
            "Adequately powered subgroup failures should block stronger claims."
        ),
        "rows": rows[:40],
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
