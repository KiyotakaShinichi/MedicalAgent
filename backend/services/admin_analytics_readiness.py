"""Analytics inputs and the MLE readiness panel.

Every artifact the dashboard reads enters through here, so there is one place
that answers "where does this number come from" and one place that decides what
happens when the artifact is absent. Missing inputs degrade to `None` and the
panels render "unavailable" rather than raising - an analytics dashboard that
500s because an optional evaluation artifact has not been generated is worse
than one that says so.

Readiness itself is a precomputed artifact. When it is missing the panel
reports `status: unavailable` and `clinical_validation: False`, which is the
honest answer and not a claim of readiness.
"""

import json
from pathlib import Path

import pandas as pd


def _prefer_existing(candidate: str, fallback: str) -> str:
    return candidate if Path(candidate).exists() else fallback


DEFAULT_SYNTHETIC_METRICS_PATH = _prefer_existing(
    "Data/complete_synthetic_training_realism_v2/complete_synthetic_model_metrics.json",
    "Data/complete_synthetic_training/complete_synthetic_model_metrics.json",
)


DEFAULT_SYNTHETIC_PREDICTIONS_PATH = _prefer_existing(
    "Data/complete_synthetic_training_realism_v2/complete_synthetic_model_predictions.csv",
    "Data/complete_synthetic_training/complete_synthetic_model_predictions.csv",
)


DEFAULT_SYNTHETIC_TRAINING_CSV = _prefer_existing(
    "Data/complete_synthetic_breast_journeys_realism_v2/temporal_ml_rows.csv",
    "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv",
)


DEFAULT_SYNTHETIC_MRI_REPORTS_CSV = _prefer_existing(
    "Data/complete_synthetic_breast_journeys_realism_v2/mri_reports.csv",
    "Data/complete_synthetic_breast_journeys/mri_reports.csv",
)


DEFAULT_BREASTDCEDL_METRICS_PATH = "Data/breastdcedl_spy1_baseline_metrics.json"


DEFAULT_MLE_READINESS_PATH = "Data/mle_monitoring/latest_mle_readiness.json"


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
