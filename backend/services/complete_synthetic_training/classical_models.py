"""Classical (scikit-learn) classifier training for the binary targets.

Every estimator receives the caller's ``seed`` as ``random_state``; none is
constructed with a literal seed. Hyperparameters are reproduced exactly from
the pre-decomposition module.
"""

import joblib
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from backend.services.complete_synthetic_training.data_preparation import _preprocessor
from backend.services.complete_synthetic_training.feature_schema import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)
from backend.services.complete_synthetic_training.metrics import _binary_metrics
from backend.services.complete_synthetic_training.predictions import (
    _aggregate_patient_predictions,
    _base_patient_prediction_rows,
)


def _train_classical_models(train_rows, test_rows, target, output_path, seed):
    X_train = train_rows[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_rows[target].astype(int)
    X_test = test_rows[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test_rows[target].astype(int)

    models = {
        "logistic_regression": Pipeline([
            ("preprocess", _preprocessor(scale_numeric=True)),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=2000)),
        ]),
        "random_forest": Pipeline([
            ("preprocess", _preprocessor(scale_numeric=False)),
            ("classifier", RandomForestClassifier(
                n_estimators=350,
                max_depth=7,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=seed,
            )),
        ]),
        "extra_trees": Pipeline([
            ("preprocess", _preprocessor(scale_numeric=False)),
            ("classifier", ExtraTreesClassifier(
                n_estimators=400,
                max_depth=8,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=seed,
            )),
        ]),
        "gradient_boosting": Pipeline([
            ("preprocess", _preprocessor(scale_numeric=False)),
            ("classifier", GradientBoostingClassifier(random_state=seed)),
        ]),
        "svm_rbf": Pipeline([
            ("preprocess", _preprocessor(scale_numeric=True)),
            ("classifier", SVC(C=1.5, kernel="rbf", probability=True, class_weight="balanced", random_state=seed)),
        ]),
        "mlp": Pipeline([
            ("preprocess", _preprocessor(scale_numeric=True)),
            ("classifier", MLPClassifier(
                hidden_layer_sizes=(48, 24),
                alpha=1e-3,
                learning_rate_init=1e-3,
                max_iter=600,
                random_state=seed,
            )),
        ]),
    }

    model_metrics = {}
    artifacts = {}
    prediction_rows = _base_patient_prediction_rows(test_rows, target)
    for name, model in models.items():
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]
        row_metrics = _binary_metrics(y_test, probabilities)
        patient_predictions = _aggregate_patient_predictions(test_rows, target, probabilities, name)
        patient_metrics = _binary_metrics(
            patient_predictions["actual_label"].astype(int),
            patient_predictions[f"{name}_probability"],
            prefix="patient_level_",
        )
        model_metrics[name] = {**row_metrics, **patient_metrics, "model_type": "cycle_tabular_classifier"}
        artifact_path = output_path / f"{name}_{target}.joblib"
        joblib.dump(model, artifact_path)
        artifacts[name] = str(artifact_path)
        prediction_rows = prediction_rows.merge(patient_predictions, on=["patient_id", "actual_label"], how="left")

    return {
        "models": model_metrics,
        "artifacts": artifacts,
        "predictions": prediction_rows,
    }
