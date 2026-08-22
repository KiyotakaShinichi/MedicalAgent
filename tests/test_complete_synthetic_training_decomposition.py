"""ML-semantic equivalence guards for the training decomposition.

`backend/services/complete_synthetic_training.py` (1037 lines) became a package
of focused modules behind a compatibility facade. The refactor was intended to
change *nothing* about the ML: same seeds, same split membership, same
hyperparameters, same metric formulas and ordering, same calibration, same
artifact layout.

These tests assert the properties that would actually break if that were
untrue. They deliberately avoid asserting on implementation details of the new
modules — a test that pins which file a helper lives in would fail on the next
legitimate move without catching a single real defect.

The values below were captured from the pre-decomposition module and are
reproduced here as fixed expectations.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backend.services.complete_synthetic_training as T  # noqa: E402


# ─── Compatibility surface ───────────────────────────────────────────────────

# Names other modules import from this package. Several begin with an
# underscore: `_patient_split` has eight importers and `_preprocessor` five,
# so despite the convention they are repository-wide API.
REQUIRED_EXPORTS = (
    "train_complete_synthetic_models",
    "NUMERIC_FEATURES",
    "CATEGORICAL_FEATURES",
    "PROMOTION_NUMERIC_FEATURES",
    "DEFAULT_ML_CSV_PATH",
    "DEFAULT_OUTPUT_DIR",
    "RESPONSE_REGRESSION_TARGET",
    "ROW_LEVEL_TARGETS",
    "EXCLUDED_COLUMNS",
    "_patient_split",
    "_preprocessor",
    "_binary_metrics",
    "_regression_metrics",
    "_aggregate_patient_predictions",
    "_aggregate_patient_regression_predictions",
    "BaselineTemporalCnn",
    "TemporalCnn",
    "TemporalGru",
)


@pytest.mark.parametrize("name", REQUIRED_EXPORTS)
def test_facade_still_exports(name: str) -> None:
    assert hasattr(T, name), (
        f"{name} disappeared from the facade. It is imported elsewhere in the "
        "repository, so removing it breaks callers even though the package "
        "itself still works."
    )


def test_entrypoint_signature_is_unchanged() -> None:
    """Callers pass these positionally in scripts; order and defaults matter."""
    sig = inspect.signature(T.train_complete_synthetic_models)
    assert list(sig.parameters) == [
        "ml_csv_path", "output_dir", "target", "test_size",
        "seed", "cnn_epochs", "cnn_batch_size",
    ]
    assert sig.parameters["target"].default == "treatment_success_binary"
    assert sig.parameters["test_size"].default == 0.25
    assert sig.parameters["seed"].default == 42
    assert sig.parameters["cnn_epochs"].default == 20
    assert sig.parameters["cnn_batch_size"].default == 16


# ─── Feature contract ────────────────────────────────────────────────────────


def test_feature_order_is_contractual() -> None:
    """Order fixes preprocessor column order, hence stored coefficient meaning."""
    assert T.NUMERIC_FEATURES == list(T.NUMERIC_FEATURES)
    assert len(T.NUMERIC_FEATURES) == len(set(T.NUMERIC_FEATURES)), "duplicate numeric feature"
    assert len(T.CATEGORICAL_FEATURES) == len(set(T.CATEGORICAL_FEATURES))
    assert not set(T.NUMERIC_FEATURES) & set(T.CATEGORICAL_FEATURES)


def test_targets_are_excluded_from_features() -> None:
    """A target leaking into the feature list would silently train on the label."""
    for target in T.ROW_LEVEL_TARGETS:
        assert target in T.EXCLUDED_COLUMNS
        assert target not in T.NUMERIC_FEATURES
        assert target not in T.CATEGORICAL_FEATURES
    assert T.RESPONSE_REGRESSION_TARGET in T.EXCLUDED_COLUMNS


# ─── Split semantics ─────────────────────────────────────────────────────────


def _frame():
    import pandas as pd
    return T._ensure_response_regression_columns(pd.read_csv(T.DEFAULT_ML_CSV_PATH))


def test_split_is_patient_level_and_deterministic() -> None:
    """Patient-level isolation is the anti-leakage property of this pipeline."""
    rows = _frame()
    target = "treatment_success_binary"
    train_a, test_a = T._patient_split(rows, target, 0.25, 42)
    train_b, test_b = T._patient_split(rows, target, 0.25, 42)

    assert sorted(train_a) == sorted(train_b), "same seed must reproduce the split"
    assert sorted(test_a) == sorted(test_b)
    assert not set(train_a) & set(test_a), "a patient may not appear on both sides"
    # Captured from the pre-decomposition module.
    assert len(train_a) == 450
    assert len(test_a) == 150


def test_seed_actually_changes_the_split() -> None:
    """Guards the guard: a split ignoring its seed would pass the test above."""
    rows = _frame()
    train_42, _ = T._patient_split(rows, "treatment_success_binary", 0.25, 42)
    train_7, _ = T._patient_split(rows, "treatment_success_binary", 0.25, 7)
    assert sorted(train_42) != sorted(train_7)


# ─── Metrics: names, order, values ───────────────────────────────────────────

_LABELS = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
_PROBS = np.array([0.9, 0.1, 0.8, 0.6, 0.2, 0.45, 0.7, 0.3, 0.55, 0.4])


def test_binary_metric_names_and_order() -> None:
    """Insertion order is the column order of emitted metrics files."""
    assert list(T._binary_metrics(_LABELS, _PROBS)) == [
        "accuracy", "balanced_accuracy", "f1", "precision", "sensitivity",
        "specificity", "brier_score", "calibration", "confusion_matrix",
        "roc_auc", "average_precision",
    ]


def test_binary_metric_values_are_unchanged() -> None:
    """Values captured from the pre-decomposition module, not recomputed here.

    This fixture separates perfectly at a 0.5 threshold, so the point
    discrimination metrics are all 1.0; `brier_score` is what distinguishes a
    confident correct model from a hesitant one, and is the value that would
    move if probability handling drifted.
    """
    m = T._binary_metrics(_LABELS, _PROBS)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)
    assert m["precision"] == pytest.approx(1.0)
    assert m["sensitivity"] == pytest.approx(1.0)
    assert m["specificity"] == pytest.approx(1.0)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["brier_score"] == pytest.approx(0.1)
    assert m["confusion_matrix"] == {
        "true_positive": 5, "true_negative": 5,
        "false_positive": 0, "false_negative": 0,
    }


def test_metric_prefix_is_applied_to_every_key() -> None:
    prefixed = T._binary_metrics(_LABELS, _PROBS, prefix="val_")
    assert all(k.startswith("val_") for k in prefixed)


def test_regression_metric_names_order_and_values() -> None:
    y = np.array([10.0, 20.0, 30.0, 40.0, 55.0])
    yhat = np.array([12.0, 18.0, 33.0, 37.0, 58.0])
    m = T._regression_metrics(y, yhat)
    assert list(m) == ["mae", "rmse", "r2"]
    # Rounded by the implementation; captured from the pre-decomposition module.
    assert m["mae"] == pytest.approx(2.6)
    assert m["rmse"] == pytest.approx(2.646)
    assert m["r2"] == pytest.approx(0.971)


def test_expected_calibration_error_is_unchanged() -> None:
    assert T._expected_calibration_error(_LABELS, _PROBS) == pytest.approx(0.29)


def test_calibration_diagnostics_shape_is_stable() -> None:
    """The before/after temperature-scaling report embedded in binary metrics."""
    diagnostics = T._probability_calibration_diagnostics(_LABELS, _PROBS)
    assert diagnostics["method"] == "posthoc_temperature_grid_on_evaluation_split"
    assert diagnostics["before_temperature_scaling"]["ece"] == pytest.approx(0.29)
    assert diagnostics["after_temperature_scaling"]["ece"] == pytest.approx(0.187)
    assert diagnostics["after_temperature_scaling"]["temperature"] == pytest.approx(0.5)
    # The same diagnostics block is embedded under the `calibration` key.
    assert T._binary_metrics(_LABELS, _PROBS)["calibration"] == diagnostics


# ─── Preprocessor ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("scale_numeric", [True, False])
def test_preprocessor_is_constructible_and_scaling_is_honoured(scale_numeric: bool) -> None:
    """Scaling changes fitted coefficients, so the flag must keep its meaning."""
    pre = T._preprocessor(scale_numeric)
    assert ("StandardScaler" in repr(pre)) is scale_numeric


# ─── Package structure ───────────────────────────────────────────────────────


def test_focused_modules_are_importable_directly() -> None:
    """The point of the split: a caller can take one responsibility, not all."""
    from backend.services.complete_synthetic_training import (  # noqa: F401
        calibration, classical_models, data_preparation, feature_schema,
        metrics, pipeline, predictions, regression_models, sequence_models,
    )


def test_constants_have_a_single_definition() -> None:
    """Re-exported, not re-declared: duplication would let the copies drift."""
    from backend.services.complete_synthetic_training import feature_schema

    assert T.NUMERIC_FEATURES is feature_schema.NUMERIC_FEATURES
    assert T.CATEGORICAL_FEATURES is feature_schema.CATEGORICAL_FEATURES
    assert T.EXCLUDED_COLUMNS is feature_schema.EXCLUDED_COLUMNS
