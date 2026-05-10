import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)


DEFAULT_METRICS_PATH = "Data/breastdcedl_spy1_baseline_metrics.json"
DEFAULT_PREDICTIONS_PATH = "Data/breastdcedl_spy1_model_predictions.csv"
DEFAULT_CNN_METRICS_PATH = "Data/breastdcedl_spy1_cnn_metrics.json"
DEFAULT_OUTPUT_DIR = "Data/external_validation"


def build_external_validation_report(
    metrics_path=DEFAULT_METRICS_PATH,
    predictions_path=DEFAULT_PREDICTIONS_PATH,
    cnn_metrics_path=DEFAULT_CNN_METRICS_PATH,
    output_dir=DEFAULT_OUTPUT_DIR,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics = _load_json(metrics_path)
    cnn_metrics = _load_json(cnn_metrics_path)
    predictions = pd.read_csv(predictions_path) if Path(predictions_path).exists() else pd.DataFrame()

    if predictions.empty or not metrics:
        report = {
            "schema_version": "external_validation_report_v1",
            "status": "unavailable",
            "message": "BreastDCEDL/I-SPY1 baseline artifacts are missing.",
            "claim_boundary": "External validation is not available until real-data artifacts are generated.",
        }
    else:
        best_model = metrics.get("best_model_by_roc_auc") or _infer_best_model(predictions)
        probability_col = f"{best_model}_pcr_probability" if best_model else "best_model_pcr_probability"
        if probability_col not in predictions.columns:
            probability_col = "best_model_pcr_probability"
        report = _summarize_external_predictions(metrics, cnn_metrics, predictions, best_model, probability_col)

    files = {
        "external_validation_json": str(output_path / "external_validation_report.json"),
        "external_validation_subgroups_csv": str(output_path / "external_validation_subgroups.csv"),
    }
    Path(files["external_validation_json"]).write_text(json.dumps({**report, "files": files}, indent=2), encoding="utf-8")
    pd.DataFrame(report.get("subgroup_metrics") or []).to_csv(files["external_validation_subgroups_csv"], index=False)
    return {**report, "files": files}


def _summarize_external_predictions(metrics, cnn_metrics, predictions, best_model, probability_col):
    frame = predictions.dropna(subset=["pcr_label", probability_col]).copy()
    labels = frame["pcr_label"].astype(float).astype(int).to_numpy()
    probabilities = frame[probability_col].astype(float).to_numpy()
    predicted = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    model_metrics = {
        "accuracy": round(float(accuracy_score(labels, predicted)), 3),
        "balanced_accuracy": round(float(balanced_accuracy_score(labels, predicted)), 3),
        "precision": round(float(precision_score(labels, predicted, zero_division=0)), 3),
        "sensitivity": round(float(recall_score(labels, predicted, zero_division=0)), 3),
        "specificity": round(float(tn / max(tn + fp, 1)), 3),
        "brier_score": round(float(brier_score_loss(labels, probabilities)), 3),
        "roc_auc": round(float(roc_auc_score(labels, probabilities)), 3) if len(set(labels.tolist())) > 1 else None,
        "average_precision": round(float(average_precision_score(labels, probabilities)), 3) if len(set(labels.tolist())) > 1 else None,
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }
    return {
        "schema_version": "external_validation_report_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": _status(model_metrics),
        "dataset": "BreastDCEDL / I-SPY1 derived-feature baseline",
        "rows": int(len(frame)),
        "positive_pcr": int(labels.sum()),
        "negative_pcr": int((labels == 0).sum()),
        "best_model": best_model,
        "probability_column": probability_col,
        "model_metrics": model_metrics,
        "cross_validation_metrics": metrics.get("models", {}),
        "cnn_baseline": _cnn_summary(cnn_metrics),
        "subgroup_metrics": _subgroup_metrics(frame, probability_col),
        "interpretation": (
            "This is an exploratory real-data direction using MRI-derived tabular features and cross-validated "
            "predictions. It is useful as an external sanity check, not as clinical validation."
        ),
        "claim_boundary": "Do not compare synthetic metrics and BreastDCEDL metrics as apples-to-apples clinical performance.",
    }


def _subgroup_metrics(frame, probability_col):
    if "molecular_subtype" not in frame.columns:
        return []
    rows = []
    for value, group in frame.groupby("molecular_subtype"):
        labels = group["pcr_label"].astype(float).astype(int).to_numpy()
        probabilities = group[probability_col].astype(float).to_numpy()
        predicted = (probabilities >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
        rows.append({
            "group": "molecular_subtype",
            "value": value,
            "n": int(len(group)),
            "positive_rate": round(float(labels.mean()), 3),
            "roc_auc": round(float(roc_auc_score(labels, probabilities)), 3) if len(set(labels.tolist())) > 1 else None,
            "average_precision": round(float(average_precision_score(labels, probabilities)), 3) if len(set(labels.tolist())) > 1 else None,
            "sensitivity": round(float(tp / max(tp + fn, 1)), 3),
            "specificity": round(float(tn / max(tn + fp, 1)), 3),
            "status": "low_support" if len(group) < 15 else "review",
        })
    return rows


def _cnn_summary(cnn_metrics):
    if not cnn_metrics:
        return {"status": "unavailable"}
    final = cnn_metrics.get("final_validation") or {}
    return {
        "status": "exploratory_baseline",
        "model_type": cnn_metrics.get("model_type"),
        "rows": cnn_metrics.get("rows"),
        "validation_rows": cnn_metrics.get("validation_rows"),
        "val_accuracy": final.get("val_accuracy"),
        "val_balanced_accuracy": final.get("val_balanced_accuracy"),
        "val_roc_auc": final.get("val_roc_auc"),
        "warning": cnn_metrics.get("warning"),
    }


def _status(metrics):
    auc = metrics.get("roc_auc")
    if auc is None:
        return "unavailable"
    if auc >= 0.75:
        return "strong_exploratory"
    if auc >= 0.60:
        return "weak_external_signal"
    return "needs_review"


def _infer_best_model(predictions):
    for column in predictions.columns:
        if column.endswith("_pcr_probability") and column != "best_model_pcr_probability":
            return column.replace("_pcr_probability", "")
    return None


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}
