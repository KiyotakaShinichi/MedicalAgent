import numpy as np
import pandas as pd

from backend.services.synthetic_model_perturbation_retrain_eval import (
    _expected_calibration_error,
    _patient_split,
    perturb_features,
    perturb_training_labels,
)


def _frame():
    rows = []
    for patient in range(20):
        for cycle in range(2):
            rows.append(
                {
                    "patient_id": f"p{patient}",
                    "cycle": cycle + 1,
                    "pre_wbc": 5.0 + patient,
                    "mri_tumor_size_cm": 2.0,
                    "max_symptom_severity": 3.0,
                    "symptom_count": 1,
                    "treatment_success_binary": patient % 2,
                    "response_score_percent": float(patient),
                }
            )
    return pd.DataFrame(rows)


def test_patient_split_has_no_overlap():
    train, test = _patient_split(_frame(), seed=42)
    assert set(train.patient_id).isdisjoint(set(test.patient_id))


def test_feature_noise_does_not_change_targets():
    frame = _frame()
    perturbed = perturb_features(frame, scenario="combined_noise", seed=42)
    assert np.array_equal(
        frame["treatment_success_binary"], perturbed["treatment_success_binary"]
    )
    assert np.array_equal(
        frame["response_score_percent"], perturbed["response_score_percent"]
    )
    assert perturbed.isna().sum().sum() > frame.isna().sum().sum()


def test_label_noise_is_patient_consistent():
    frame = _frame()
    noisy = perturb_training_labels(frame, seed=42, fraction=0.1)
    changed = noisy[
        noisy["treatment_success_binary"] != frame["treatment_success_binary"]
    ]
    assert changed["patient_id"].nunique() == 2
    assert changed.groupby("patient_id").size().eq(2).all()


def test_severe_modality_dropout_is_stronger_and_preserves_targets():
    frame = _frame()
    moderate = perturb_features(frame, scenario="modality_dropout", seed=42)
    severe = perturb_features(frame, scenario="severe_modality_dropout", seed=42)
    assert severe.isna().sum().sum() > moderate.isna().sum().sum()
    assert np.array_equal(
        frame["treatment_success_binary"],
        severe["treatment_success_binary"],
    )
    assert np.array_equal(
        frame["response_score_percent"],
        severe["response_score_percent"],
    )


def test_mnar_dropout_depends_on_observed_severity_and_preserves_targets():
    frame = pd.concat([_frame()] * 4, ignore_index=True)
    frame["patient_id"] = [f"mnar-{idx}" for idx in range(len(frame))]
    frame["max_symptom_severity"] = np.tile(np.arange(8), len(frame) // 8)
    perturbed = perturb_features(
        frame,
        scenario="mnar_severity_dependent_dropout",
        seed=17,
    )
    high = frame["max_symptom_severity"] >= frame["max_symptom_severity"].median()
    missing = perturbed["mri_tumor_size_cm"].isna()
    assert missing[high].mean() >= missing[~high].mean()
    assert np.array_equal(
        frame["treatment_success_binary"],
        perturbed["treatment_success_binary"],
    )
    assert np.array_equal(
        frame["response_score_percent"],
        perturbed["response_score_percent"],
    )


def test_expected_calibration_error_is_zero_for_perfect_probabilities():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.0, 0.0, 1.0, 1.0])
    assert _expected_calibration_error(labels, probabilities) == 0.0
