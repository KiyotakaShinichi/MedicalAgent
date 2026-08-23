"""Fixed inputs, feature sets, seeds, and the claim boundary.

Separated so the scenario runner and the evaluation that composes it can both
read them without importing each other.

`SEED` and `REPEATED_SPLIT_SEEDS` are what make this evaluation reproducible.
Changing either changes every number in the report while leaving every
threshold untouched, so they are treated as part of the contract rather than as
tunable parameters.
"""

from __future__ import annotations

from pathlib import Path

from backend.services.synthetic_feature_policy import (
    CATEGORICAL_FEATURES as _POLICY_CATEGORICAL_FEATURES,
    CANONICAL_PROMOTION_NUMERIC_FEATURES,
    LEGACY_NUMERIC_FEATURES,
)


DEFAULT_SOURCE_PATH = Path(
    "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv"
)
REALISM_V2_PATH = Path(
    "Data/complete_synthetic_breast_journeys_realism_v2/temporal_ml_rows.csv"
)
DEFAULT_OUTPUT_PATH = Path(
    "Data/evals/models/latest_synthetic_model_perturbation_retrain_eval.json"
)
DIRECT_RESPONSE_PROXY = "mri_percent_change_from_baseline"
NUMERIC_FEATURES = list(LEGACY_NUMERIC_FEATURES)
GUARDED_NUMERIC_FEATURES = list(CANONICAL_PROMOTION_NUMERIC_FEATURES)
CATEGORICAL_FEATURES = list(_POLICY_CATEGORICAL_FEATURES)
SEED = 42
REPEATED_SPLIT_SEEDS = (11, 23, 42, 73, 101)
CLAIM_BOUNDARY = (
    "All rows and labels in this evaluation are simulator-built. Perturbation "
    "robustness measures engineering sensitivity to synthetic assumptions; it "
    "does not establish clinical realism, external validity, treatment utility, "
    "patient benefit, or production healthcare readiness."
)
