"""
Per-prediction ML error table.

Reads the saved predictions CSV, classifies each row into TP/FP/TN/FN,
computes absolute error, and attaches SHAP feature contributions where available.
All data is synthetic — clearly labelled in every response field.
"""

from __future__ import annotations

import json
from pathlib import Path

PREDICTIONS_CSV = "Data/complete_synthetic_training/complete_synthetic_model_predictions.csv"
SHAP_JSON = "Data/complete_synthetic_training/detailed_eval/complete_synthetic_shap_values.json"
PRIMARY_MODEL = "logistic_regression"
THRESHOLD = 0.40


def build_prediction_error_table(
    predictions_path: str = PREDICTIONS_CSV,
    shap_path: str = SHAP_JSON,
    model: str = PRIMARY_MODEL,
    threshold: float = THRESHOLD,
    limit: int = 100,
) -> dict:
    try:
        import pandas as pd
    except ImportError:
        return _unavailable("pandas not available")

    p = Path(predictions_path)
    if not p.exists():
        return _unavailable("Predictions CSV not found. Run training pipeline first.")

    df = pd.read_csv(p)
    prob_col = f"{model}_probability"
    pred_col = f"{model}_predicted_label"

    if prob_col not in df.columns or "actual_label" not in df.columns:
        return _unavailable(f"Required columns missing: {prob_col}, actual_label")

    shap_map = _load_shap(shap_path)

    rows = []
    for _, row in df.head(limit).iterrows():
        patient_id = str(row.get("patient_id", "—"))
        actual = int(row["actual_label"])
        prob = float(row[prob_col])
        pred = int(prob >= threshold)
        abs_error = round(abs(actual - prob), 4)

        confusion = _confusion_type(actual, pred)
        rows.append({
            "patient_id": patient_id,
            "actual_label": actual,
            "predicted_probability": round(prob, 4),
            "predicted_class": pred,
            "threshold_used": threshold,
            "absolute_error": abs_error,
            "confusion_type": confusion,
            "top_features": shap_map.get(patient_id) or [],
            "note": _case_note(confusion, prob, actual),
        })

    total = len(rows)
    tp = sum(1 for r in rows if r["confusion_type"] == "TP")
    fp = sum(1 for r in rows if r["confusion_type"] == "FP")
    tn = sum(1 for r in rows if r["confusion_type"] == "TN")
    fn = sum(1 for r in rows if r["confusion_type"] == "FN")
    mae = round(sum(r["absolute_error"] for r in rows) / total, 4) if total else None

    return {
        "schema_version": "prediction_error_table_v1",
        "model": model,
        "threshold": threshold,
        "total_predictions": total,
        "confusion_summary": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "mae": mae,
        "sensitivity": round(tp / (tp + fn), 4) if (tp + fn) > 0 else None,
        "specificity": round(tn / (tn + fp), 4) if (tn + fp) > 0 else None,
        "rows": rows,
        "claim_boundary": "All predictions are on synthetic data. These metrics do not represent clinical performance.",
        "shap_available": bool(shap_map),
    }


def _confusion_type(actual: int, pred: int) -> str:
    if actual == 1 and pred == 1:
        return "TP"
    if actual == 0 and pred == 1:
        return "FP"
    if actual == 0 and pred == 0:
        return "TN"
    return "FN"


def _case_note(confusion: str, prob: float, actual: int) -> str:
    if confusion == "FN":
        return f"Missed positive (prob={prob:.2f}). FN cost is high in cancer monitoring — review threshold."
    if confusion == "FP":
        return f"False alarm (prob={prob:.2f}). Increases clinician review burden but not directly harmful."
    if confusion == "TP":
        return f"Correct positive (prob={prob:.2f}). Monitoring flag correctly triggered."
    return f"Correct negative (prob={prob:.2f}). Low-risk case correctly identified."


def _load_shap(shap_path: str) -> dict[str, list]:
    p = Path(shap_path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "per_patient" in data:
            return {str(k): v for k, v in data["per_patient"].items()}
        return {}
    except Exception:
        return {}


def _unavailable(reason: str) -> dict:
    return {
        "status": "unavailable",
        "reason": reason,
        "rows": [],
        "claim_boundary": "All predictions are on synthetic data.",
    }
