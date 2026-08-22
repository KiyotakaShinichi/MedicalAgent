"""Response-score regression training and champion selection.

Trains the regressor family for ``response_score_percent`` and picks a champion
via ``_regression_selection_score``. That ordering is part of the published
contract: it decides whose artifacts are written as the promoted model.
"""

import json

import joblib
import numpy as np
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

from backend.services.complete_synthetic_training.data_preparation import _preprocessor
from backend.services.complete_synthetic_training.feature_schema import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RESPONSE_REGRESSION_TARGET,
)
from backend.services.complete_synthetic_training.metrics import (
    _regression_metrics,
    _regression_selection_score,
)
from backend.services.complete_synthetic_training.predictions import (
    _aggregate_patient_regression_predictions,
    _base_patient_regression_rows,
)


def _train_response_regression(train_rows, test_rows, output_path, seed):
    target = RESPONSE_REGRESSION_TARGET
    if target not in train_rows.columns or target not in test_rows.columns:
        return {
            "metrics": {"status": "unavailable", "reason": f"{target} column is missing"},
            "best_model": None,
            "artifacts": {},
            "predictions_csv": None,
        }

    train = train_rows.dropna(subset=[target]).copy()
    test = test_rows.dropna(subset=[target]).copy()
    if train.empty or test.empty:
        return {
            "metrics": {"status": "unavailable", "reason": "No non-null response regression labels"},
            "best_model": None,
            "artifacts": {},
            "predictions_csv": None,
        }

    X_train = train[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train[target].astype(float)
    X_test = test[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test[target].astype(float)
    models = {
        "ridge_regression": Pipeline([
            ("preprocess", _preprocessor(scale_numeric=True)),
            ("regressor", Ridge(alpha=1.0)),
        ]),
        "random_forest_regressor": Pipeline([
            ("preprocess", _preprocessor(scale_numeric=False)),
            ("regressor", RandomForestRegressor(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=3,
                random_state=seed,
            )),
        ]),
        "extra_trees_regressor": Pipeline([
            ("preprocess", _preprocessor(scale_numeric=False)),
            ("regressor", ExtraTreesRegressor(
                n_estimators=350,
                max_depth=8,
                min_samples_leaf=3,
                random_state=seed,
            )),
        ]),
        "gradient_boosting_regressor": Pipeline([
            ("preprocess", _preprocessor(scale_numeric=False)),
            ("regressor", GradientBoostingRegressor(random_state=seed)),
        ]),
        "gradient_boosting_huber_regressor": Pipeline([
            ("preprocess", _preprocessor(scale_numeric=False)),
            ("regressor", GradientBoostingRegressor(loss="huber", alpha=0.90, random_state=seed)),
        ]),
        "huber_regressor": Pipeline([
            ("preprocess", _preprocessor(scale_numeric=True)),
            ("regressor", HuberRegressor(epsilon=1.35, alpha=0.0005, max_iter=1000)),
        ]),
        "svr_rbf_regressor": Pipeline([
            ("preprocess", _preprocessor(scale_numeric=True)),
            ("regressor", SVR(C=2.0, kernel="rbf")),
        ]),
    }

    model_metrics = {}
    artifacts = {}
    predictions = _base_patient_regression_rows(test, target)
    fitted_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model
        row_predictions = model.predict(X_test)
        patient_predictions = _aggregate_patient_regression_predictions(test, target, row_predictions, name)
        model_metrics[name] = {
            **_regression_metrics(y_test, row_predictions),
            **_regression_metrics(
                patient_predictions["actual_response_score_percent"],
                patient_predictions[f"{name}_response_score_percent"],
                prefix="patient_level_",
            ),
            "model_type": "cycle_tabular_regressor",
            "target": target,
            "interpretation": "Positive values estimate tumor-size reduction percent; negative values indicate growth/progression signal in the synthetic simulator.",
        }
        artifact_path = output_path / f"{name}_{target}.joblib"
        joblib.dump(model, artifact_path)
        artifacts[f"{name}_response_regression"] = str(artifact_path)
        predictions = predictions.merge(patient_predictions, on=["patient_id", "actual_response_score_percent"], how="left")

    robust_members = [
        name for name in [
            "random_forest_regressor",
            "extra_trees_regressor",
            "gradient_boosting_regressor",
            "gradient_boosting_huber_regressor",
            "huber_regressor",
        ]
        if name in fitted_models
    ]
    if len(robust_members) >= 3:
        row_matrix = np.column_stack([fitted_models[name].predict(X_test) for name in robust_members])
        row_predictions = np.median(row_matrix, axis=1)
        patient_predictions = _aggregate_patient_regression_predictions(
            test,
            target,
            row_predictions,
            "robust_response_ensemble",
        )
        model_metrics["robust_response_ensemble"] = {
            **_regression_metrics(y_test, row_predictions),
            **_regression_metrics(
                patient_predictions["actual_response_score_percent"],
                patient_predictions["robust_response_ensemble_response_score_percent"],
                prefix="patient_level_",
            ),
            "model_type": "cycle_tabular_regressor_ensemble",
            "target": target,
            "members": robust_members,
            "selection_note": (
                "Median ensemble over tree and robust linear regressors. Selected with an outlier-aware "
                "score that combines MAE and RMSE."
            ),
            "interpretation": "Positive values estimate tumor-size reduction percent; negative values indicate growth/progression signal in the synthetic simulator.",
        }
        ensemble_path = output_path / f"robust_response_ensemble_{target}.json"
        ensemble_path.write_text(
            json.dumps(
                {
                    "model_type": "median_prediction_ensemble",
                    "target": target,
                    "members": robust_members,
                    "member_artifacts": {
                        member: artifacts.get(f"{member}_response_regression")
                        for member in robust_members
                    },
                    "warning": "Synthetic-data ensemble metadata. Recreate member models before inference.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        artifacts["robust_response_ensemble_response_regression"] = str(ensemble_path)
        predictions = predictions.merge(patient_predictions, on=["patient_id", "actual_response_score_percent"], how="left")

    best_model = min(
        model_metrics,
        key=lambda name: _regression_selection_score(model_metrics[name]),
    )
    predictions_csv = output_path / "complete_synthetic_response_regression_predictions.csv"
    predictions.to_csv(predictions_csv, index=False)
    return {
        "metrics": {
            "status": "trained",
            "task": "response_score_regression",
            "target": target,
            "models": model_metrics,
            "best_model_by_patient_level_mae": best_model,
            "best_model_selection_score": _regression_selection_score(model_metrics[best_model]),
            "selection_policy": "minimize patient_level_mae + 0.15 * patient_level_rmse to reduce large response outliers",
            "target_definition": "Continuous MRI response signal: baseline-to-current tumor-size reduction percent; higher is stronger shrinkage, negative means growth.",
        },
        "best_model": best_model,
        "artifacts": artifacts,
        "predictions_csv": str(predictions_csv),
    }
