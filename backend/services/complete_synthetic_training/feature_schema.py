"""Feature, target, and path constants for complete-synthetic training.

A leaf module by design. Twenty-eight modules import ``NUMERIC_FEATURES`` and
``CATEGORICAL_FEATURES``; keeping them here rather than in the package
``__init__`` lets the training modules import them without importing the
facade, which would be circular.

Values and ordering are unchanged from the pre-decomposition module. Feature
*order* is contractual: it fixes column order into every fitted preprocessor
and therefore the meaning of stored model coefficients.
"""

from backend.services.synthetic_feature_policy import (
    CATEGORICAL_FEATURES as POLICY_CATEGORICAL_FEATURES,
    CANONICAL_PROMOTION_NUMERIC_FEATURES,
    LEGACY_NUMERIC_FEATURES,
)

DEFAULT_ML_CSV_PATH = "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv"
DEFAULT_OUTPUT_DIR = "Data/complete_synthetic_training"
RESPONSE_REGRESSION_TARGET = "response_score_percent"

# Backward-compatible aliases for serialized legacy artifacts. New promotion
# experiments must use ``PROMOTION_NUMERIC_FEATURES`` and declare
# ``PROMOTION_FEATURE_POLICY_ID``.
NUMERIC_FEATURES = list(LEGACY_NUMERIC_FEATURES)
PROMOTION_NUMERIC_FEATURES = list(CANONICAL_PROMOTION_NUMERIC_FEATURES)
CATEGORICAL_FEATURES = list(POLICY_CATEGORICAL_FEATURES)
ROW_LEVEL_TARGETS = {
    "toxicity_risk_binary",
    "support_intervention_needed",
    "urgent_intervention_needed",
}
EXCLUDED_COLUMNS = {
    "patient_id",
    "treatment_date",
    "latent_response_strength",
    "response_score_percent",
    "final_response_category",
    "final_cancer_status",
    "final_response_multiclass",
    "treatment_success_binary",
    "maintenance_needed",
    "toxicity_risk_binary",
    "support_intervention_needed",
    "urgent_intervention_needed",
    "cycle_response_trend_class",
}
