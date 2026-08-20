"""Statistical diagnostics used by the synthetic-only MLE readiness report."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import roc_auc_score


def hybrid_weight_ablation(predictions: pd.DataFrame | None) -> dict[str, Any]:
    """Measure sensitivity to classifier/regressor blending weights."""
    if predictions is None or predictions.empty:
        return {
            "status": "unavailable",
            "message": "Prediction CSV required for ablation sweep.",
        }

    probability_column = next(
        (
            column
            for suffix in ("_calibrated_probability", "_probability")
            for column in predictions.columns
            if column.endswith(suffix) and "regression" not in column
        ),
        None,
    )
    regression_column = next(
        (column for column in predictions.columns if column.endswith("_response_score_percent")),
        None,
    )
    if "actual_label" not in predictions.columns or probability_column is None:
        return {
            "status": "unavailable",
            "message": (
                "Need actual_label and a probability column. "
                f"Found prob={probability_column}, reg={regression_column}."
            ),
        }

    frame = predictions[["patient_id", "actual_label", probability_column]].copy().dropna()
    if regression_column and regression_column in predictions.columns:
        frame = frame.join(predictions[[regression_column]], how="left")
        frame["reg_normalized"] = (
            (frame[regression_column].fillna(0) + 50).clip(0, 100)
        ) / 100.0
    else:
        frame["reg_normalized"] = None

    labels = frame["actual_label"].astype(int).to_numpy()
    classification_probabilities = frame[probability_column].astype(float).to_numpy()
    regression_available = bool(frame["reg_normalized"].notna().any())
    regression_scores = (
        frame["reg_normalized"].fillna(0.5).to_numpy()
        if regression_available
        else None
    )
    if len(set(labels.tolist())) < 2:
        return {"status": "unavailable", "message": "Need both classes in labels for AUROC."}

    rows: list[dict[str, Any]] = []
    for classification_weight in (
        0.0,
        0.10,
        0.20,
        0.30,
        0.35,
        0.40,
        0.50,
        0.60,
        0.65,
        0.70,
        0.80,
        0.90,
        1.0,
    ):
        regression_weight = round(1.0 - classification_weight, 2)
        hybrid_scores = (
            classification_weight * classification_probabilities
            + regression_weight * regression_scores
            if regression_scores is not None
            else classification_probabilities
        )
        try:
            auroc = round(float(roc_auc_score(labels, hybrid_scores)), 4)
        except Exception:  # noqa: BLE001 - diagnostic returns unavailable metric on estimator failure
            auroc = None
        rows.append(
            {
                "classification_weight": classification_weight,
                "regression_weight": regression_weight,
                "hybrid_auroc": auroc,
                "is_default": classification_weight == 0.65,
            }
        )

    best = max(
        (row for row in rows if row["hybrid_auroc"] is not None),
        key=lambda row: row["hybrid_auroc"],
        default=None,
    )
    default_row = next((row for row in rows if row["is_default"]), None)
    auroc_gap = None
    if best and default_row:
        best_auroc = best["hybrid_auroc"]
        default_auroc = default_row["hybrid_auroc"]
        if best_auroc is not None and default_auroc is not None:
            auroc_gap = round(best_auroc - default_auroc, 4)

    return {
        "status": "available" if rows else "unavailable",
        "purpose": (
            "Sensitivity analysis: how much does the 65/35 classifier/regressor weight choice matter? "
            "Sweep shows AUROC at each weight combination on synthetic data."
        ),
        "regression_available": regression_available,
        "probability_column": probability_column,
        "regression_column": regression_column,
        "default_weight": {"classification": 0.65, "regression": 0.35},
        "best_weight": best,
        "default_auroc": default_row["hybrid_auroc"] if default_row else None,
        "auroc_gap_from_optimal": auroc_gap,
        "sweep": rows,
        "interpretation": (
            "Small AUROC gap means the 65/35 default is near-optimal. "
            "Large gap suggests the weight deserves tuning via cross-validation on a dev set."
        ),
        "warning": "Synthetic data only - not clinical evidence.",
    }


def temporal_generalization_eval(
    training_rows: pd.DataFrame | None,
    predictions: pd.DataFrame | None,
) -> dict[str, Any]:
    """Compare prediction behavior between simulator-defined cycle cohorts."""
    if training_rows is None or training_rows.empty:
        return {"status": "unavailable", "message": "Training rows required for temporal eval."}
    if predictions is None or predictions.empty:
        return {"status": "unavailable", "message": "Prediction CSV required for temporal eval."}
    if not {"patient_id", "cycle"}.issubset(training_rows.columns):
        return {"status": "unavailable", "message": "patient_id and cycle columns required."}

    first_cycle = training_rows.groupby("patient_id")["cycle"].min().reset_index()
    first_cycle.columns = ["patient_id", "first_cycle"]
    median_first = float(first_cycle["first_cycle"].median())
    early_ids = set(first_cycle[first_cycle["first_cycle"] <= median_first]["patient_id"])
    late_ids = set(first_cycle[first_cycle["first_cycle"] > median_first]["patient_id"])
    if not early_ids or not late_ids:
        return {"status": "unavailable", "message": "Cannot split into early/late patient groups."}

    probability_column = next(
        (
            column
            for suffix in ("_calibrated_probability", "_probability")
            for column in predictions.columns
            if column.endswith(suffix) and "regression" not in column
        ),
        None,
    )
    if "actual_label" not in predictions.columns or probability_column is None:
        return {"status": "unavailable", "message": "Need actual_label and probability column."}

    def group_metrics(patient_ids: set[Any]) -> dict[str, Any] | None:
        subset = predictions[predictions["patient_id"].isin(patient_ids)][
            ["patient_id", "actual_label", probability_column]
        ].dropna()
        if len(subset) < 5:
            return None
        labels = subset["actual_label"].astype(int).to_numpy()
        probabilities = subset[probability_column].astype(float).to_numpy()
        positive_rate = round(float(labels.mean()), 3)
        if len(set(labels.tolist())) < 2:
            return {"n": len(subset), "auroc": None, "positive_rate": positive_rate}
        return {
            "n": len(subset),
            "auroc": round(float(roc_auc_score(labels, probabilities)), 4),
            "positive_rate": positive_rate,
        }

    early_metrics = group_metrics(early_ids)
    late_metrics = group_metrics(late_ids)
    auroc_delta = None
    if early_metrics and late_metrics:
        early_auroc = early_metrics.get("auroc")
        late_auroc = late_metrics.get("auroc")
        if early_auroc is not None and late_auroc is not None:
            auroc_delta = round(late_auroc - early_auroc, 4)

    status = "unavailable"
    if auroc_delta is not None:
        status = (
            "stable"
            if abs(auroc_delta) <= 0.05
            else "mild_drift"
            if abs(auroc_delta) <= 0.10
            else "significant_drift"
        )

    return {
        "status": status,
        "purpose": (
            "Proxy for temporal generalization: compares AUROC and outcome rate between patients "
            "with early vs late first observed cycles. Large delta suggests time-dependent performance decay."
        ),
        "median_first_cycle": median_first,
        "early_cohort": (
            {**early_metrics, "first_cycle_threshold": f"<= {median_first}"}
            if early_metrics
            else None
        ),
        "late_cohort": (
            {**late_metrics, "first_cycle_threshold": f"> {median_first}"}
            if late_metrics
            else None
        ),
        "auroc_delta_late_minus_early": auroc_delta,
        "interpretation": (
            "auroc_delta ≈ 0 -> stable across cycles. "
            "auroc_delta << 0 -> model degrades on later-cycle patients (possible distribution shift). "
            "Next step: true temporal train/eval split on a real longitudinal dataset."
        ),
        "warning": "Synthetic data only - temporal structure is simulator-generated, not real patient time.",
    }
