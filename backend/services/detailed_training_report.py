import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score


DEFAULT_TRAINING_ROWS_PATH = "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv"
DEFAULT_CLASSIFICATION_PREDICTIONS_PATH = "Data/complete_synthetic_training/complete_synthetic_model_predictions.csv"
DEFAULT_REGRESSION_PREDICTIONS_PATH = "Data/complete_synthetic_training/complete_synthetic_response_regression_predictions.csv"
DEFAULT_METRICS_PATH = "Data/complete_synthetic_training/complete_synthetic_model_metrics.json"
DEFAULT_OUTPUT_DIR = "Data/complete_synthetic_training/detailed_eval"


def generate_detailed_training_report(
    training_rows_path=DEFAULT_TRAINING_ROWS_PATH,
    classification_predictions_path=DEFAULT_CLASSIFICATION_PREDICTIONS_PATH,
    regression_predictions_path=DEFAULT_REGRESSION_PREDICTIONS_PATH,
    metrics_path=DEFAULT_METRICS_PATH,
    output_dir=DEFAULT_OUTPUT_DIR,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    locked_output_path = None

    training_rows = pd.read_csv(training_rows_path)
    classification = pd.read_csv(classification_predictions_path)
    regression = pd.read_csv(regression_predictions_path) if Path(regression_predictions_path).exists() else pd.DataFrame()
    metrics = _load_json(metrics_path)

    best_classifier = metrics.get("best_model_by_patient_level_roc_auc") or _infer_best_classifier(classification)
    best_regressor = (
        metrics.get("best_response_regressor_by_patient_level_mae")
        or (metrics.get("response_regression") or {}).get("best_model_by_patient_level_mae")
        or _infer_best_regressor(regression)
    )
    context = _patient_context(training_rows)
    detailed = classification.merge(context, on="patient_id", how="left")
    if not regression.empty:
        detailed = detailed.merge(regression, on="patient_id", how="left")

    detailed = _add_hybrid_columns(detailed, best_classifier, best_regressor)
    detailed = _add_rule_columns(detailed)
    detailed = detailed.sort_values(["hybrid_review_priority", "patient_id"]).reset_index(drop=True)

    regression_slices = _regression_slices(detailed, best_regressor)
    residual_review = _regression_residual_review(detailed, best_regressor)
    hybrid_policy = _hybrid_threshold_policy()
    hybrid_summary = _hybrid_summary(detailed)
    classification_review = _classification_review(detailed, best_classifier)
    error_taxonomy = _error_taxonomy(detailed, best_classifier, best_regressor)
    cost_sensitive = _cost_sensitive_evaluation(detailed)
    metrics_summary = _metrics_summary(metrics, detailed, best_classifier, best_regressor)

    files = {
        "test_set_predictions_detailed_csv": str(output_path / "test_set_predictions_detailed.csv"),
        "regression_slice_metrics_csv": str(output_path / "regression_slice_metrics.csv"),
        "regression_residual_review_csv": str(output_path / "regression_residual_review.csv"),
        "hybrid_threshold_policy_csv": str(output_path / "hybrid_threshold_policy.csv"),
        "hybrid_review_summary_csv": str(output_path / "hybrid_review_summary.csv"),
        "error_taxonomy_csv": str(output_path / "error_taxonomy.csv"),
        "cost_sensitive_evaluation_csv": str(output_path / "cost_sensitive_evaluation.csv"),
        "training_eval_summary_json": str(output_path / "training_eval_summary.json"),
        "markdown_report": str(output_path / "training_eval_report.md"),
        "html_report": str(output_path / "training_eval_report.html"),
    }
    try:
        _write_report_tables(files, detailed, regression_slices, residual_review, hybrid_policy, hybrid_summary, error_taxonomy, cost_sensitive)
    except PermissionError:
        locked_output_path = output_path
        output_path = output_path.parent / f"detailed_eval_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        output_path.mkdir(parents=True, exist_ok=True)
        files = {
            "test_set_predictions_detailed_csv": str(output_path / "test_set_predictions_detailed.csv"),
            "regression_slice_metrics_csv": str(output_path / "regression_slice_metrics.csv"),
            "regression_residual_review_csv": str(output_path / "regression_residual_review.csv"),
            "hybrid_threshold_policy_csv": str(output_path / "hybrid_threshold_policy.csv"),
            "hybrid_review_summary_csv": str(output_path / "hybrid_review_summary.csv"),
            "error_taxonomy_csv": str(output_path / "error_taxonomy.csv"),
            "cost_sensitive_evaluation_csv": str(output_path / "cost_sensitive_evaluation.csv"),
            "training_eval_summary_json": str(output_path / "training_eval_summary.json"),
            "markdown_report": str(output_path / "training_eval_report.md"),
            "html_report": str(output_path / "training_eval_report.html"),
        }
        _write_report_tables(files, detailed, regression_slices, residual_review, hybrid_policy, hybrid_summary, error_taxonomy, cost_sensitive)

    summary = {
        "schema_version": "detailed_training_eval_v2",
        "best_classifier": best_classifier,
        "best_regressor": best_regressor,
        "test_patients": int(len(detailed)),
        "classification_review": classification_review,
        "hybrid_review_summary": hybrid_summary.to_dict(orient="records"),
        "error_taxonomy": error_taxonomy.to_dict(orient="records"),
        "cost_sensitive_evaluation": cost_sensitive.to_dict(orient="records"),
        "regression_slice_count": int(len(regression_slices)),
        "residual_review_count": int(len(residual_review)),
        "metrics_summary": metrics_summary,
        "files": files,
        "locked_output_path": str(locked_output_path) if locked_output_path else None,
        "claim_boundary": "Synthetic-data engineering evaluation only. Not clinical validation.",
    }
    Path(files["training_eval_summary_json"]).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    markdown = _markdown_report(summary, detailed, regression_slices, residual_review, hybrid_policy, error_taxonomy, cost_sensitive)
    Path(files["markdown_report"]).write_text(markdown, encoding="utf-8")
    Path(files["html_report"]).write_text(_html_report(markdown), encoding="utf-8")
    latest_manifest = Path(DEFAULT_OUTPUT_DIR).parent / "latest_detailed_eval_manifest.json"
    latest_manifest.write_text(
        json.dumps({
            "schema_version": "latest_detailed_eval_manifest_v1",
            "summary_path": files["training_eval_summary_json"],
            "html_report": files["html_report"],
            "files": files,
        }, indent=2),
        encoding="utf-8",
    )
    return summary


def _write_report_tables(files, detailed, regression_slices, residual_review, hybrid_policy, hybrid_summary, error_taxonomy, cost_sensitive):
    detailed.to_csv(files["test_set_predictions_detailed_csv"], index=False)
    regression_slices.to_csv(files["regression_slice_metrics_csv"], index=False)
    residual_review.to_csv(files["regression_residual_review_csv"], index=False)
    hybrid_policy.to_csv(files["hybrid_threshold_policy_csv"], index=False)
    hybrid_summary.to_csv(files["hybrid_review_summary_csv"], index=False)
    error_taxonomy.to_csv(files["error_taxonomy_csv"], index=False)
    cost_sensitive.to_csv(files["cost_sensitive_evaluation_csv"], index=False)


def _load_json(path):
    if not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _infer_best_classifier(predictions):
    for column in predictions.columns:
        if column.endswith("_calibrated_probability"):
            return column.replace("_calibrated_probability", "")
    for column in predictions.columns:
        if column.endswith("_probability"):
            return column.replace("_probability", "")
    return None


def _infer_best_regressor(regression):
    for column in regression.columns:
        if column.endswith("_response_score_percent") and not column.startswith("actual_"):
            return column.replace("_response_score_percent", "")
    return None


def _patient_context(training_rows):
    rows = training_rows.sort_values(["patient_id", "cycle"]).copy()
    grouped = rows.groupby("patient_id", as_index=False).agg(
        age=("age", "first"),
        stage=("stage", "first"),
        molecular_subtype=("molecular_subtype", "first"),
        regimen=("regimen", "first"),
        cycles_observed=("cycle", "max"),
        latest_mri_percent_change=("mri_percent_change_from_baseline", "last"),
        latest_mri_tumor_size_cm=("mri_tumor_size_cm", "last"),
        max_symptom_severity=("max_symptom_severity", "max"),
        symptom_count=("symptom_count", "sum"),
        nadir_wbc=("nadir_wbc", "min"),
        nadir_anc=("nadir_anc", "min"),
        nadir_hemoglobin=("nadir_hemoglobin", "min"),
        nadir_platelets=("nadir_platelets", "min"),
        intervention_count=("intervention_count", "sum"),
        dose_delay_count=("dose_delayed", "sum"),
        dose_reduction_count=("dose_reduced", "sum"),
        final_cancer_status=("final_cancer_status", "last"),
        final_response_category=("final_response_category", "last"),
    )
    grouped["age_band"] = pd.cut(
        grouped["age"],
        bins=[0, 44, 54, 64, 74, 120],
        labels=["<45", "45-54", "55-64", "65-74", "75+"],
        include_lowest=True,
    ).astype(str)
    return grouped


def _add_hybrid_columns(frame, best_classifier, best_regressor):
    output = frame.copy()
    raw_col = f"{best_classifier}_probability"
    calibrated_col = f"{best_classifier}_calibrated_probability"
    probability_col = calibrated_col if calibrated_col in output.columns else raw_col
    response_col = f"{best_regressor}_response_score_percent"
    output["champion_probability"] = pd.to_numeric(output.get(probability_col), errors="coerce")
    output["champion_probability_source"] = probability_col
    output["champion_response_score_percent"] = pd.to_numeric(output.get(response_col), errors="coerce")
    output["champion_response_score_source"] = response_col
    output["classification_probability_score"] = output["champion_probability"] * 100
    output["regression_normalized_score"] = (50 + output["champion_response_score_percent"]).clip(0, 100)
    output["hybrid_score"] = (
        (0.65 * output["classification_probability_score"])
        + (0.35 * output["regression_normalized_score"])
    ).round(3)
    output["classification_band"] = output["champion_probability"].apply(_classification_band)
    output["regression_band"] = output["champion_response_score_percent"].apply(_regression_band)
    output["hybrid_band"] = output["hybrid_score"].apply(_hybrid_band)
    output["model_agreement"] = output.apply(
        lambda row: _agreement(row.get("classification_band"), row.get("regression_band")),
        axis=1,
    )
    return output


def _add_rule_columns(frame):
    output = frame.copy()
    max_symptom = pd.to_numeric(output["max_symptom_severity"], errors="coerce")
    nadir_wbc = pd.to_numeric(output["nadir_wbc"], errors="coerce")
    nadir_anc = pd.to_numeric(output["nadir_anc"], errors="coerce")
    nadir_platelets = pd.to_numeric(output["nadir_platelets"], errors="coerce")
    output["toxicity_rule_triggered"] = (
        ((max_symptom >= 8) & ((nadir_wbc < 1.0) | (nadir_anc < 0.5)))
        | (nadir_wbc < 0.5)
        | (nadir_anc < 0.2)
        | (nadir_platelets < 50)
    )
    output["response_review_rule_triggered"] = (
        (pd.to_numeric(output["hybrid_score"], errors="coerce") < 55)
        | (pd.to_numeric(output["latest_mri_percent_change"], errors="coerce") > -20)
    )
    output["discordant_signal_rule_triggered"] = output["model_agreement"].eq("conflicting")
    output["hybrid_review_category"] = output.apply(_review_category, axis=1)
    priority = {
        "toxicity_review": 1,
        "discordant_signal_review": 2,
        "response_trend_review": 3,
        "watch_closely": 4,
        "routine_monitoring": 5,
    }
    output["hybrid_review_priority"] = output["hybrid_review_category"].map(priority).fillna(9).astype(int)
    output["rule_explanation"] = output.apply(_rule_explanation, axis=1)
    return output


def _review_category(row):
    if bool(row.get("toxicity_rule_triggered")):
        return "toxicity_review"
    if bool(row.get("discordant_signal_rule_triggered")):
        return "discordant_signal_review"
    if bool(row.get("response_review_rule_triggered")):
        return "response_trend_review"
    if float(row.get("hybrid_score") or 0) < 75:
        return "watch_closely"
    return "routine_monitoring"


def _rule_explanation(row):
    reasons = []
    if bool(row.get("toxicity_rule_triggered")):
        reasons.append("CBC/symptom toxicity rule")
    if bool(row.get("discordant_signal_rule_triggered")):
        reasons.append("classifier/regressor conflict")
    if bool(row.get("response_review_rule_triggered")):
        reasons.append("low hybrid or weak MRI improvement")
    if not reasons:
        reasons.append("no major synthetic review rule")
    return "; ".join(reasons)


def _regression_slices(frame, best_regressor):
    pred_col = f"{best_regressor}_response_score_percent"
    if pred_col not in frame.columns or "actual_response_score_percent" not in frame.columns:
        return pd.DataFrame()
    rows = []
    for group_name in ["stage", "molecular_subtype", "regimen", "age_band", "hybrid_review_category"]:
        if group_name not in frame.columns:
            continue
        for value, group in frame.dropna(subset=[group_name, pred_col, "actual_response_score_percent"]).groupby(group_name):
            labels = group["actual_response_score_percent"].astype(float)
            predictions = group[pred_col].astype(float)
            rows.append({
                "slice": group_name,
                "value": value,
                "n": int(len(group)),
                "mae": round(float(mean_absolute_error(labels, predictions)), 3),
                "rmse": round(float(np.sqrt(mean_squared_error(labels, predictions))), 3),
                "r2": round(float(r2_score(labels, predictions)), 3) if len(group) >= 2 else None,
                "status": _slice_status(len(group), mean_absolute_error(labels, predictions)),
            })
    return pd.DataFrame(rows).sort_values(["status", "slice", "value"]) if rows else pd.DataFrame()


def _regression_residual_review(frame, best_regressor, limit=25):
    pred_col = f"{best_regressor}_response_score_percent"
    if pred_col not in frame.columns or "actual_response_score_percent" not in frame.columns:
        return pd.DataFrame()
    output = frame.dropna(subset=[pred_col, "actual_response_score_percent"]).copy()
    output["response_residual"] = output[pred_col].astype(float) - output["actual_response_score_percent"].astype(float)
    output["absolute_response_residual"] = output["response_residual"].abs()
    cols = [
        "patient_id",
        "actual_response_score_percent",
        pred_col,
        "response_residual",
        "absolute_response_residual",
        "stage",
        "molecular_subtype",
        "regimen",
        "hybrid_score",
        "hybrid_review_category",
        "rule_explanation",
        "latest_mri_percent_change",
        "max_symptom_severity",
        "nadir_wbc",
        "nadir_anc",
    ]
    return output.sort_values("absolute_response_residual", ascending=False)[cols].head(limit)


def _hybrid_threshold_policy():
    return pd.DataFrame([
        {
            "rule_order": 1,
            "category": "toxicity_review",
            "condition": "(max_symptom_severity >= 8 AND (nadir_wbc < 1.0 OR nadir_anc < 0.5)) OR nadir_wbc < 0.5 OR nadir_anc < 0.2 OR nadir_platelets < 50",
            "reason": "Only severe CBC/symptom patterns override favorable ML signal; routine chemo nadirs should not dominate every route.",
        },
        {
            "rule_order": 2,
            "category": "discordant_signal_review",
            "condition": "classification band conflicts with regression band",
            "reason": "Classifier and continuous response estimate disagree.",
        },
        {
            "rule_order": 3,
            "category": "response_trend_review",
            "condition": "hybrid_score < 55 OR latest_mri_percent_change > -20",
            "reason": "Lower or weak response signal needs review.",
        },
        {
            "rule_order": 4,
            "category": "watch_closely",
            "condition": "55 <= hybrid_score < 75",
            "reason": "Mixed signal; monitor closely.",
        },
        {
            "rule_order": 5,
            "category": "routine_monitoring",
            "condition": "hybrid_score >= 75 and no override rules",
            "reason": "Favorable synthetic monitoring signal with no major rule trigger.",
        },
    ])


def _hybrid_summary(frame):
    rows = (
        frame.groupby("hybrid_review_category", as_index=False)
        .agg(
            patients=("patient_id", "count"),
            mean_hybrid_score=("hybrid_score", "mean"),
            mean_probability=("champion_probability", "mean"),
            mean_response_score=("champion_response_score_percent", "mean"),
            toxicity_rule_rate=("toxicity_rule_triggered", "mean"),
            response_review_rule_rate=("response_review_rule_triggered", "mean"),
        )
    )
    total = max(len(frame), 1)
    rows["patient_rate"] = rows["patients"] / total
    for col in ["mean_hybrid_score", "mean_probability", "mean_response_score", "toxicity_rule_rate", "response_review_rule_rate", "patient_rate"]:
        rows[col] = rows[col].round(3)
    return rows.sort_values("patients", ascending=False)


def _classification_review(frame, best_classifier):
    prob_col = "champion_probability"
    labels = frame["actual_label"].astype(int)
    probabilities = frame[prob_col].astype(float)
    predicted = (probabilities >= 0.5).astype(int)
    false_negatives = frame[(labels == 1) & (predicted == 0)]
    false_positives = frame[(labels == 0) & (predicted == 1)]
    return {
        "model": best_classifier,
        "probability_column": frame["champion_probability_source"].dropna().iloc[0] if not frame.empty else None,
        "test_patients": int(len(frame)),
        "false_negatives": int(len(false_negatives)),
        "false_positives": int(len(false_positives)),
        "false_negative_examples": false_negatives["patient_id"].head(10).tolist(),
        "false_positive_examples": false_positives["patient_id"].head(10).tolist(),
    }


def _error_taxonomy(frame, best_classifier, best_regressor):
    rows = []
    prob = pd.to_numeric(frame.get("champion_probability"), errors="coerce")
    labels = pd.to_numeric(frame.get("actual_label"), errors="coerce")
    pred = (prob >= 0.5).astype(int)
    residual_col = f"{best_regressor}_response_score_percent"
    residual = None
    if residual_col in frame.columns and "actual_response_score_percent" in frame.columns:
        residual = (
            pd.to_numeric(frame[residual_col], errors="coerce")
            - pd.to_numeric(frame["actual_response_score_percent"], errors="coerce")
        )

    definitions = [
        (
            "false_negative_favorable_response",
            (labels.eq(1) & pred.eq(0)),
            "Classifier missed a synthetically favorable final outcome. In medicine this is reviewed carefully because false negatives can hide benefit signals.",
        ),
        (
            "false_positive_overoptimism",
            (labels.eq(0) & pred.eq(1)),
            "Classifier predicted favorable response for an unfavorable synthetic outcome. This can over-reassure a review workflow.",
        ),
        (
            "delayed_toxicity_detection",
            frame.get("toxicity_rule_triggered", pd.Series(False, index=frame.index)).astype(bool)
            & prob.ge(0.66),
            "Deterministic CBC/symptom toxicity rule triggers even though the response classifier is favorable.",
        ),
        (
            "subtype_confusion",
            frame.get("molecular_subtype", pd.Series("", index=frame.index)).astype(str).isin(["HR+/HER2+", "HER2+"])
            & frame.get("model_agreement", pd.Series("", index=frame.index)).eq("conflicting"),
            "HER2-related subgroup where classifier and response-regressor disagree.",
        ),
        (
            "sparse_history_instability",
            pd.to_numeric(frame.get("cycles_observed", pd.Series(0, index=frame.index)), errors="coerce").lt(4)
            | frame[["champion_probability", "champion_response_score_percent"]].isna().any(axis=1),
            "Limited temporal history or missing model signal makes the patient-level estimate less stable.",
        ),
        (
            "regimen_shift_uncertainty",
            frame.get("regimen", pd.Series("", index=frame.index)).astype(str).str.contains("TCHP then endocrine", case=False, na=False)
            & frame.get("hybrid_review_category", pd.Series("", index=frame.index)).isin(["discordant_signal_review", "response_trend_review"]),
            "Regimen-specific review gap for HR+/HER2+ TCHP followed by endocrine therapy.",
        ),
        (
            "calibration_boundary_case",
            prob.between(0.40, 0.60, inclusive="both"),
            "Probability is close to the operating threshold; threshold changes may flip routing.",
        ),
    ]
    if residual is not None:
        definitions.append((
            "response_regression_outlier",
            residual.abs().ge(20),
            "Continuous response estimate differs from the synthetic MRI-derived label by at least 20 percentage points.",
        ))

    for error_type, mask, meaning in definitions:
        examples = frame.loc[mask.fillna(False), "patient_id"].head(10).tolist()
        rows.append({
            "error_type": error_type,
            "count": int(mask.fillna(False).sum()),
            "rate": round(float(mask.fillna(False).mean()), 3),
            "example_patient_ids": "; ".join(examples),
            "meaning": meaning,
        })
    return pd.DataFrame(rows).sort_values(["count", "error_type"], ascending=[False, True])


def _cost_sensitive_evaluation(frame):
    labels = pd.to_numeric(frame["actual_label"], errors="coerce").astype(int)
    probabilities = pd.to_numeric(frame["champion_probability"], errors="coerce").astype(float)
    rows = []
    for fn_cost, fp_cost in [(5, 1), (10, 1), (3, 1)]:
        for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:
            predicted = (probabilities >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
            weighted_cost = (fn_cost * fn) + (fp_cost * fp)
            rows.append({
                "threshold": threshold,
                "fn_cost": fn_cost,
                "fp_cost": fp_cost,
                "weighted_cost": int(weighted_cost),
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
                "sensitivity": round(float(tp / max(tp + fn, 1)), 3),
                "specificity": round(float(tn / max(tn + fp, 1)), 3),
                "interpretation": "Lower weighted_cost is better for this synthetic threshold policy.",
            })
    return pd.DataFrame(rows).sort_values(["fn_cost", "weighted_cost", "threshold"], ascending=[False, True, True])


def _metrics_summary(metrics, frame, best_classifier, best_regressor):
    model_metrics = (metrics.get("models") or {}).get(best_classifier) or {}
    regression_metrics = ((metrics.get("response_regression") or {}).get("models") or {}).get(best_regressor) or {}
    calibrated = metrics.get("calibrated_champion") or {}
    return {
        "best_classifier": best_classifier,
        "classifier_patient_level_roc_auc": model_metrics.get("patient_level_roc_auc"),
        "classifier_patient_level_average_precision": model_metrics.get("patient_level_average_precision"),
        "classifier_patient_level_brier_score": model_metrics.get("patient_level_brier_score"),
        "classifier_patient_level_confusion_matrix": model_metrics.get("patient_level_confusion_matrix"),
        "calibrated_champion_status": calibrated.get("status"),
        "calibrated_validation_ece": (((calibrated.get("calibrated_validation") or {}).get("validation_calibration") or {}).get("before_temperature_scaling") or {}).get("ece"),
        "best_regressor": best_regressor,
        "regressor_patient_level_mae": regression_metrics.get("patient_level_mae"),
        "regressor_patient_level_rmse": regression_metrics.get("patient_level_rmse"),
        "regressor_patient_level_r2": regression_metrics.get("patient_level_r2"),
    }


def _classification_band(probability):
    if pd.isna(probability):
        return None
    if probability >= 0.66:
        return "favorable"
    if probability <= 0.40:
        return "lower"
    return "mixed"


def _regression_band(score):
    if pd.isna(score):
        return None
    if score >= 20:
        return "favorable"
    if score <= -10:
        return "lower"
    return "mixed"


def _hybrid_band(score):
    if pd.isna(score):
        return None
    if score >= 75:
        return "favorable"
    if score < 55:
        return "lower"
    return "mixed"


def _agreement(classification_band, regression_band):
    if not classification_band or not regression_band:
        return "single_signal_available"
    if classification_band == regression_band:
        return "aligned"
    if "mixed" in {classification_band, regression_band}:
        return "partially_aligned"
    return "conflicting"


def _slice_status(n, mae):
    if n < 5:
        return "low_support"
    if mae <= 3:
        return "strong"
    if mae <= 6:
        return "acceptable"
    return "needs_review"


def _markdown_table(frame, limit=12):
    if frame.empty:
        return "_No rows._"
    shown = frame.head(limit).copy()
    columns = shown.columns.tolist()
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in shown.iterrows():
        rows.append("| " + " | ".join(_md_cell(row[col]) for col in columns) + " |")
    return "\n".join([header, sep, *rows])


def _md_cell(value):
    if pd.isna(value):
        return ""
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")[:120]


def _markdown_report(summary, detailed, regression_slices, residual_review, hybrid_policy, error_taxonomy, cost_sensitive):
    metrics = summary["metrics_summary"]
    return f"""# Detailed Synthetic Training Evaluation Report

Generated from the current local synthetic training artifacts.

## Claim Boundary

Synthetic-data engineering evaluation only. This report helps visualize model behavior, rule routing, and error modes. It is not clinical validation.

## Headline Results

- Test patients: {summary["test_patients"]}
- Best classifier: `{summary["best_classifier"]}`
- Best regressor: `{summary["best_regressor"]}`
- Classifier patient-level AUROC: `{metrics.get("classifier_patient_level_roc_auc")}`
- Classifier patient-level AUPRC: `{metrics.get("classifier_patient_level_average_precision")}`
- Classifier patient-level Brier score: `{metrics.get("classifier_patient_level_brier_score")}`
- Calibrated champion status: `{metrics.get("calibrated_champion_status")}`
- Calibrated validation ECE: `{metrics.get("calibrated_validation_ece")}`
- Regressor patient-level MAE: `{metrics.get("regressor_patient_level_mae")}`
- Regressor patient-level RMSE: `{metrics.get("regressor_patient_level_rmse")}`
- Regressor patient-level R2: `{metrics.get("regressor_patient_level_r2")}`

## Hybrid Ruleset

{_markdown_table(hybrid_policy, limit=10)}

## Hybrid Routing Summary

{_markdown_table(pd.DataFrame(summary["hybrid_review_summary"]), limit=20)}

## Error Taxonomy

{_markdown_table(error_taxonomy, limit=20)}

## Cost-Sensitive Threshold Evaluation

{_markdown_table(cost_sensitive, limit=20)}

## Example Test-Set Predictions

{_markdown_table(detailed[[
    "patient_id",
    "actual_label",
    "champion_probability",
    "actual_response_score_percent",
    "champion_response_score_percent",
    "hybrid_score",
    "model_agreement",
    "hybrid_review_category",
    "rule_explanation",
]], limit=20)}

## Regression Slice Metrics

{_markdown_table(regression_slices, limit=30)}

## Largest Regression Residuals

{_markdown_table(residual_review, limit=25)}

## Output Files

{chr(10).join(f"- `{key}`: `{value}`" for key, value in summary["files"].items())}
"""


def _html_report(markdown):
    escaped = (
        markdown.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Detailed Synthetic Training Evaluation</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 32px; max-width: 1180px; color: #172033; line-height: 1.55; }}
    pre {{ white-space: pre-wrap; background: #f5f7fb; border: 1px solid #dfe5ee; padding: 18px; overflow-x: auto; }}
  </style>
</head>
<body><pre>{escaped}</pre></body>
</html>
"""
