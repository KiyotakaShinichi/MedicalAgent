"""
Per-subgroup Expected Calibration Error (ECE).

Why this exists
---------------
The model benchmark already reports a single global ECE.  But a global number
hides the case where calibration is fine on average but materially worse on
one molecular subtype, one stage band, or one age bracket.  In a clinical
context that gap matters — it is one half of the standard fairness audit
(the other half is performance disparity).

This module computes ECE separately within each level of each grouping
column, returns a structured report, and surfaces the **max disparity** —
the largest absolute ECE difference between any two groups within a column —
so the benchmark JSON can carry a single number a reviewer can read at a
glance.

Inputs
~~~~~~
- ``predictions``: pandas DataFrame with columns ``patient_id``,
  ``<model>_probability``, ``<model>_predicted_label``, ``actual_label``.
- ``subgroup_lookup``: pandas DataFrame keyed by ``patient_id`` carrying the
  subgroup columns we want to slice on (``stage``, ``molecular_subtype``,
  ``age_bucket`` if synthesised).

Returned shape
~~~~~~~~~~~~~~
::

    {
      "status": "passed" | "needs_attention" | "not_computed",
      "model": "<model_key>",
      "global_ece": float,
      "subgroups": {
        "molecular_subtype": {
          "Luminal A": {"n": 23, "ece": 0.041},
          "Luminal B": {"n": 19, "ece": 0.067},
          ...
        },
        "stage": { ... },
      },
      "max_disparity_by_column": {
         "molecular_subtype": 0.052,
         "stage":             0.018,
      },
      "max_overall_disparity": 0.052,
      "claim_boundary": "Synthetic-data subgroups only..."
    }

Caveat
~~~~~~
On a 60-row holdout, several subgroup buckets will have <5 rows.  The
function returns ``ece=None`` and ``status="too_few_samples"`` for those.
This is honest reporting — small-N buckets do not get a fake calibration
number.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, confusion_matrix, precision_score, roc_auc_score

# Buckets below this size produce unreliable ECE; we skip them rather than
# emit a noisy point estimate.
MIN_BUCKET_SIZE = 10


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float | None:
    """Standard binary-classifier ECE.  Returns None if input is empty."""
    if probs.size == 0:
        return None
    probs = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    labels = np.asarray(labels, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = probs.size
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        bin_conf = probs[mask].mean()
        bin_acc = labels[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return round(float(ece), 4)


def _bucket_age(age: Any) -> str:
    try:
        age_f = float(age)
    except (TypeError, ValueError):
        return "unknown"
    if age_f < 40: return "<40"
    if age_f < 50: return "40-49"
    if age_f < 60: return "50-59"
    if age_f < 70: return "60-69"
    return "70+"


def compute_subgroup_ece(
    predictions: pd.DataFrame,
    subgroup_lookup: pd.DataFrame,
    *,
    model_key: str,
    probability_col: str | None = None,
    label_col: str = "actual_label",
    grouping_columns: tuple[str, ...] = ("molecular_subtype", "stage"),
    include_age_bucket: bool = True,
    disparity_alert_threshold: float = 0.05,
) -> dict[str, Any]:
    """See module docstring."""

    probability_col = probability_col or f"{model_key}_probability"
    if probability_col not in predictions.columns:
        return _not_computed(f"predictions CSV missing '{probability_col}'")
    if label_col not in predictions.columns:
        return _not_computed(f"predictions CSV missing '{label_col}'")
    if "patient_id" not in predictions.columns or "patient_id" not in subgroup_lookup.columns:
        return _not_computed("patient_id is required in both predictions and subgroup_lookup")

    # Reduce the lookup table to one row per patient (the subgroup attributes
    # are patient-level, not cycle-level), preserving the FIRST observation.
    lookup_cols = [c for c in grouping_columns if c in subgroup_lookup.columns]
    if include_age_bucket and "age" in subgroup_lookup.columns:
        lookup_cols.append("age")
    if not lookup_cols:
        return _not_computed("no grouping columns present in subgroup_lookup")

    lookup = (
        subgroup_lookup[["patient_id", *lookup_cols]]
        .drop_duplicates(subset=["patient_id"], keep="first")
        .copy()
    )
    if include_age_bucket and "age" in lookup.columns:
        lookup["age_bucket"] = lookup["age"].map(_bucket_age)

    joined = predictions.merge(lookup, on="patient_id", how="left")
    probs_all = joined[probability_col].astype(float).to_numpy()
    labels_all = joined[label_col].astype(float).to_numpy()
    global_ece = expected_calibration_error(probs_all, labels_all)

    effective_groups = list(grouping_columns)
    if include_age_bucket and "age_bucket" in joined.columns:
        effective_groups.append("age_bucket")
    effective_groups = [g for g in effective_groups if g in joined.columns]

    subgroups_report: dict[str, dict[str, Any]] = {}
    max_disparity_by_column: dict[str, float | None] = {}
    performance_disparity_by_column: dict[str, dict[str, float | None]] = {}

    for column in effective_groups:
        per_group: dict[str, dict[str, Any]] = {}
        values: list[float] = []
        metric_values: dict[str, list[float]] = {
            "roc_auc": [],
            "sensitivity": [],
            "specificity": [],
            "brier_score": [],
            "precision": [],
        }
        for level, group_df in joined.groupby(column, dropna=True):
            n = int(len(group_df))
            if n < MIN_BUCKET_SIZE:
                per_group[str(level)] = {
                    "n": n,
                    "ece": None,
                    "roc_auc": None,
                    "sensitivity": None,
                    "specificity": None,
                    "brier_score": None,
                    "precision": None,
                    "positive_rate": None,
                    "status": "too_few_samples",
                }
                continue
            probs = group_df[probability_col].astype(float).to_numpy()
            labels = group_df[label_col].astype(float).astype(int).to_numpy()
            predicted = (probs >= 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
            ece = expected_calibration_error(probs, labels)
            group_metrics = {
                "n": n,
                "ece": ece,
                "roc_auc": round(float(roc_auc_score(labels, probs)), 4) if len(set(labels.tolist())) > 1 else None,
                "sensitivity": round(float(tp / max(tp + fn, 1)), 4),
                "specificity": round(float(tn / max(tn + fp, 1)), 4),
                "brier_score": round(float(brier_score_loss(labels, probs)), 4),
                "precision": round(float(precision_score(labels, predicted, zero_division=0)), 4),
                "positive_rate": round(float(labels.mean()), 4),
                "status": "passed" if ece is not None else "not_computed",
            }
            per_group[str(level)] = group_metrics
            if ece is not None:
                values.append(ece)
            for metric_name, metric_value in group_metrics.items():
                if metric_name in metric_values and metric_value is not None:
                    metric_values[metric_name].append(float(metric_value))
        subgroups_report[column] = per_group
        max_disparity_by_column[column] = (
            round(max(values) - min(values), 4) if len(values) >= 2 else None
        )
        performance_disparity_by_column[column] = {
            metric_name: round(max(metric_list) - min(metric_list), 4) if len(metric_list) >= 2 else None
            for metric_name, metric_list in metric_values.items()
        }

    valid_disparities = [v for v in max_disparity_by_column.values() if v is not None]
    max_overall = max(valid_disparities) if valid_disparities else None

    if max_overall is None:
        status = "passed_with_notes"
    elif max_overall > disparity_alert_threshold:
        status = "needs_attention"
    else:
        status = "passed"

    return {
        "status": status,
        "model": model_key,
        "n_bins": 10,
        "min_bucket_size": MIN_BUCKET_SIZE,
        "global_ece": global_ece,
        "subgroups": subgroups_report,
        "max_disparity_by_column": max_disparity_by_column,
        "performance_disparity_by_column": performance_disparity_by_column,
        "max_overall_disparity": max_overall,
        "disparity_alert_threshold": disparity_alert_threshold,
        "claim_boundary": (
            "Subgroup ECE is computed on synthetic-data buckets. The subgroup "
            "distribution reflects simulator choices, not real-world demographics. "
            "Disparity numbers here are an engineering signal, not a fairness claim."
        ),
    }


def _not_computed(reason: str) -> dict[str, Any]:
    return {
        "status": "not_computed",
        "reason": reason,
        "subgroups": {},
        "max_disparity_by_column": {},
        "performance_disparity_by_column": {},
        "max_overall_disparity": None,
    }
