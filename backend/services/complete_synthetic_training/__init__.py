"""Complete-synthetic model training.

This package replaces a single 1037-line module. That module was split by
responsibility — schema, splitting, metrics, calibration, prediction
aggregation, three trainer families, and orchestration — so each is
independently readable and testable. No ML semantics changed: seeds, split
membership, hyperparameters, metric formulas and ordering, calibration, and
artifact layout are all byte-for-byte what they were.

This ``__init__`` is a **compatibility facade**, not a convenience. Thirty-four
modules import from ``backend.services.complete_synthetic_training``, and
several import names that begin with an underscore — ``_patient_split`` has
eight importers, ``_preprocessor`` five. Those names are private by convention
only; in practice they are repository-wide API, so they are re-exported here
unchanged rather than renamed.

Where the implementation now lives::

    feature_schema     feature/target/path constants (leaf; no package deps)
    data_preparation   validation, patient-level split, shared preprocessor
    metrics            classification/regression metrics, calibration diagnostics
    calibration        isotonic champion calibration and persistence
    predictions        row-level output aggregated to patient records
    classical_models   scikit-learn classifiers
    regression_models  response-score regressors and champion selection
    sequence_models    PyTorch temporal architectures and training loop
    pipeline           orchestration; the one genuinely public entrypoint

Import the focused module when you need one responsibility; import this
package when you want the entrypoint or the shared constants.
"""

from backend.services.complete_synthetic_training.calibration import (
    _attach_calibrated_champion,
)
from backend.services.complete_synthetic_training.classical_models import (
    _train_classical_models,
)
from backend.services.complete_synthetic_training.data_preparation import (
    _ensure_response_regression_columns,
    _patient_split,
    _preprocessor,
    _validate_training_frame,
)
from backend.services.complete_synthetic_training.feature_schema import (
    CATEGORICAL_FEATURES,
    DEFAULT_ML_CSV_PATH,
    DEFAULT_OUTPUT_DIR,
    EXCLUDED_COLUMNS,
    NUMERIC_FEATURES,
    PROMOTION_NUMERIC_FEATURES,
    RESPONSE_REGRESSION_TARGET,
    ROW_LEVEL_TARGETS,
)
from backend.services.complete_synthetic_training.metrics import (
    _binary_metrics,
    _confusion_counts,
    _expected_calibration_error,
    _probability_calibration_diagnostics,
    _regression_metrics,
    _regression_selection_score,
)
from backend.services.complete_synthetic_training.pipeline import (
    train_complete_synthetic_models,
)
from backend.services.complete_synthetic_training.predictions import (
    _aggregate_patient_predictions,
    _aggregate_patient_regression_predictions,
    _base_patient_prediction_rows,
    _base_patient_regression_rows,
)
from backend.services.complete_synthetic_training.regression_models import (
    _train_response_regression,
)
from backend.services.complete_synthetic_training.sequence_models import (
    BaselineTemporalCnn,
    TemporalCnn,
    TemporalGru,
    _dl_experiment_report,
    _false_negative_examples,
    _positive_class_weight,
    _predict_cnn,
    _sequence_tensor,
    _temporal_saliency_examples,
    _train_sequence_cnn,
    _train_sequence_cnn_baseline,
    _train_sequence_gru,
    _train_sequence_torch_model,
)

__all__ = [
    # Public API.
    "train_complete_synthetic_models",
    "BaselineTemporalCnn",
    "TemporalCnn",
    "TemporalGru",
    "CATEGORICAL_FEATURES",
    "DEFAULT_ML_CSV_PATH",
    "DEFAULT_OUTPUT_DIR",
    "EXCLUDED_COLUMNS",
    "NUMERIC_FEATURES",
    "PROMOTION_NUMERIC_FEATURES",
    "RESPONSE_REGRESSION_TARGET",
    "ROW_LEVEL_TARGETS",
    # Underscore-prefixed but imported across the repository; kept for
    # compatibility. Prefer importing these from their owning module.
    "_aggregate_patient_predictions",
    "_aggregate_patient_regression_predictions",
    "_attach_calibrated_champion",
    "_base_patient_prediction_rows",
    "_base_patient_regression_rows",
    "_binary_metrics",
    "_confusion_counts",
    "_dl_experiment_report",
    "_ensure_response_regression_columns",
    "_expected_calibration_error",
    "_false_negative_examples",
    "_patient_split",
    "_positive_class_weight",
    "_predict_cnn",
    "_preprocessor",
    "_probability_calibration_diagnostics",
    "_regression_metrics",
    "_regression_selection_score",
    "_sequence_tensor",
    "_temporal_saliency_examples",
    "_train_classical_models",
    "_train_response_regression",
    "_train_sequence_cnn",
    "_train_sequence_cnn_baseline",
    "_train_sequence_gru",
    "_train_sequence_torch_model",
    "_validate_training_frame",
]
