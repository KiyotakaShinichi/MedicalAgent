"""Scenario execution and metric aggregation work as separate modules.

`synthetic_model_perturbation_retrain_eval` was a single 916-line module that
loaded data, perturbed it, split it, fitted models, computed metrics, applied
thresholds, and decided promotion. It is now a facade over three modules —
constants, runner, metrics — with the thresholds and the promotion decision
staying in the facade, because those are the judgement the evaluation makes.

Two properties get most of the attention here, because both are silent when
broken:

* **patient-grouped splitting** — if one patient's rows land on both sides of
  the split, the model is scored on patients it memorised and every metric
  improves for the wrong reason;
* **fixed seeds** — `SEED` and `REPEATED_SPLIT_SEEDS` are what make this
  evaluation reproducible. Changing either moves every number in the report
  without touching a single threshold.

The full-payload equivalence was verified against a pre-split run; this file
keeps the structural guarantees that would let it drift again.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services import synthetic_model_perturbation_retrain_eval as facade  # noqa: E402
from backend.services import (  # noqa: E402
    synthetic_model_perturbation_constants as constants,
    synthetic_model_perturbation_metrics as metrics,
    synthetic_model_perturbation_runner as runner,
)

ORIGINAL_PATH = "backend/services/synthetic_model_perturbation_retrain_eval.py"

RESPONSIBILITY_MODULES = (
    "backend.services.synthetic_model_perturbation_constants",
    "backend.services.synthetic_model_perturbation_metrics",
    "backend.services.synthetic_model_perturbation_runner",
)


def _split_frame() -> pd.DataFrame:
    """A frame with the patient id and label column the split stratifies on."""
    return pd.DataFrame(
        {
            "patient_id": [f"P{i % 12}" for i in range(120)],
            "treatment_success_binary": [(i % 12) % 2 for i in range(120)],
            "value": np.arange(120, dtype=float),
        }
    )


def _perturbation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": [f"P{i // 2}" for i in range(20)],
            "treatment_success_binary": [i % 2 for i in range(20)],
            **{
                name: np.linspace(0.1, 2.0, 20)
                for name in constants.GUARDED_NUMERIC_FEATURES
            },
        }
    )


# ─── the facade contract ─────────────────────────────────────────────────────


def test_public_exports_are_unchanged() -> None:
    assert facade.__all__ == [
        "build_synthetic_model_perturbation_retrain_eval",
        "perturb_features",
        "perturb_training_labels",
        "write_synthetic_model_perturbation_retrain_eval",
    ]
    for name in facade.__all__:
        assert callable(getattr(facade, name))


def test_facade_exports_every_pre_split_symbol(tmp_path) -> None:
    """Compare real module attributes, not just AST definitions.

    An earlier version of this test parsed the original source and checked only
    the names it *defined*. That missed everything the module imported at module
    scope — and the breast-monitoring suite monkeypatches
    `admin_analytics._calibration_metrics`, which was one of them. The split
    passed this test and broke that suite.
    """
    import importlib
    import importlib.util
    import subprocess
    import sys as _sys

    original_source = subprocess.run(
        ["git", "show", f"HEAD:{ORIGINAL_PATH}"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    original_file = tmp_path / "original_module.py"
    original_file.write_text(original_source, encoding="utf-8")

    spec = importlib.util.spec_from_file_location("_pre_split_original", original_file)
    original = importlib.util.module_from_spec(spec)
    _sys.modules["_pre_split_original"] = original
    spec.loader.exec_module(original)

    before = {name for name in dir(original) if not name.startswith("__")}
    after = {name for name in dir(facade) if not name.startswith("__")}

    missing = sorted(before - after)
    assert not missing, (
        f"the split dropped module attributes callers could bind to: {missing}"
    )


@pytest.mark.parametrize("module", RESPONSIBILITY_MODULES)
def test_each_module_imports_standalone(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, f"{module} failed to import alone:\n{result.stderr}"


@pytest.mark.parametrize(
    "module",
    RESPONSIBILITY_MODULES + ("backend.services.synthetic_model_perturbation_retrain_eval",),
)
def test_each_module_is_within_the_service_limit(module: str) -> None:
    path = ROOT / (module.replace(".", "/") + ".py")
    loc = len(path.read_bytes().decode("utf-8").splitlines())
    assert loc <= 500, f"{path.name} is {loc} LOC"


def test_runner_and_metrics_do_not_import_the_facade() -> None:
    """A back-import would make the evaluation and its parts mutually dependent."""
    for module in RESPONSIBILITY_MODULES:
        source = (ROOT / (module.replace(".", "/") + ".py")).read_text(encoding="utf-8")
        assert "synthetic_model_perturbation_retrain_eval import" not in source


def test_metrics_does_not_import_the_runner() -> None:
    """Metrics score predictions; they must not reach back into training."""
    source = (ROOT / "backend/services/synthetic_model_perturbation_metrics.py").read_text(
        encoding="utf-8"
    )
    assert "synthetic_model_perturbation_runner" not in source


# ─── seeds and reproducibility ───────────────────────────────────────────────


def test_seeds_are_unchanged() -> None:
    """These are contract, not tuning: they fix every number in the report."""
    assert constants.SEED == 42
    assert constants.REPEATED_SPLIT_SEEDS == (11, 23, 42, 73, 101)
    assert facade.SEED is constants.SEED
    assert facade.REPEATED_SPLIT_SEEDS is constants.REPEATED_SPLIT_SEEDS


def test_perturbation_is_deterministic_for_a_given_seed() -> None:
    """The same seed must produce the same perturbed frame, twice."""
    frame = _perturbation_frame()
    first = facade.perturb_features(frame.copy(), scenario="measurement_noise", seed=constants.SEED)
    second = facade.perturb_features(frame.copy(), scenario="measurement_noise", seed=constants.SEED)
    pd.testing.assert_frame_equal(first, second)


def test_different_seeds_produce_different_perturbations() -> None:
    """Otherwise the seed is decorative and the stability check is vacuous."""
    frame = _perturbation_frame()
    a = facade.perturb_features(frame.copy(), scenario="measurement_noise", seed=11)
    b = facade.perturb_features(frame.copy(), scenario="measurement_noise", seed=101)
    assert not a.equals(b)


def test_label_perturbation_is_deterministic() -> None:
    frame = _split_frame()
    first = facade.perturb_training_labels(frame.copy(), seed=constants.SEED, fraction=0.1)
    second = facade.perturb_training_labels(frame.copy(), seed=constants.SEED, fraction=0.1)
    pd.testing.assert_frame_equal(first, second)


# ─── patient-grouped splitting ───────────────────────────────────────────────


def test_split_never_puts_one_patient_on_both_sides() -> None:
    """Leakage here inflates every downstream metric silently."""
    train, test = runner._patient_split(_split_frame(), seed=constants.SEED)
    overlap = set(train["patient_id"]) & set(test["patient_id"])
    assert not overlap, f"patients appear in both splits: {sorted(overlap)}"


def test_split_is_reproducible_for_a_seed() -> None:
    frame = _split_frame()
    first_train, first_test = runner._patient_split(frame, seed=constants.SEED)
    second_train, second_test = runner._patient_split(frame, seed=constants.SEED)
    assert list(first_train["patient_id"]) == list(second_train["patient_id"])
    assert list(first_test["patient_id"]) == list(second_test["patient_id"])


def test_split_uses_every_row() -> None:
    frame = _split_frame()
    train, test = runner._patient_split(frame, seed=constants.SEED)
    assert len(train) + len(test) == len(frame)


# ─── metrics ─────────────────────────────────────────────────────────────────


def test_calibration_error_is_zero_for_perfect_probabilities() -> None:
    labels = np.array([0, 0, 1, 1])
    perfect = np.array([0.0, 0.0, 1.0, 1.0])
    assert metrics._expected_calibration_error(labels, perfect) == pytest.approx(0.0, abs=1e-9)


def test_calibration_error_is_positive_for_confidently_wrong_probabilities() -> None:
    labels = np.array([0, 0, 1, 1])
    inverted = np.array([1.0, 1.0, 0.0, 0.0])
    assert metrics._expected_calibration_error(labels, inverted) > 0.5


def test_percentile_interval_is_ordered() -> None:
    low, high = metrics._percentile_interval(list(np.linspace(0.0, 1.0, 101)))
    assert low <= high


def test_constant_baseline_is_a_reference_not_a_trained_model() -> None:
    """It exists so a trained model has an honest floor to beat."""
    source = (ROOT / "backend/services/synthetic_model_perturbation_metrics.py").read_text(
        encoding="utf-8"
    )
    assert "_train_only_constant_baseline" in source
    assert ".fit(" not in source.split("_train_only_constant_baseline")[1][:4000]


_CLEAN_GENERATOR = {
    "default_to_realism_delta_vs_default_internal": {},
    "realism_to_default_delta_vs_realism_internal": {},
}


def test_stress_failures_reports_nothing_when_nothing_breached() -> None:
    """A miscount here changes the promotion decision."""
    assert metrics._stress_failures([], _CLEAN_GENERATOR) == []


def test_stress_failures_flags_a_breached_threshold() -> None:
    """The thresholds are the point; a check that cannot fire protects nothing."""
    scenarios = [
        {
            "scenario": "measurement_noise",
            "retrained_delta_vs_guarded_clean": {"classification_auroc": -0.2},
        }
    ]
    failures = metrics._stress_failures(scenarios, _CLEAN_GENERATOR)
    assert [f["scenario"] for f in failures] == ["measurement_noise"]


def test_stress_failure_thresholds_are_unchanged() -> None:
    """These decide promotion, so they are pinned rather than trusted."""
    source = (ROOT / "backend/services/synthetic_model_perturbation_metrics.py").read_text(
        encoding="utf-8"
    )
    assert 'classification_auroc") or 0) < -0.05' in source
    assert 'classification_brier") or 0) > 0.03' in source
    assert 'regression_mae") or 0) > 5.0' in source


def test_generator_sensitivity_is_included_in_stress_candidates() -> None:
    """Cross-generator degradation must be able to fail the gate too."""
    generator = {
        "default_to_realism_delta_vs_default_internal": {"regression_mae": 9.0},
        "realism_to_default_delta_vs_realism_internal": {},
    }
    failures = metrics._stress_failures([], generator)
    assert [f["scenario"] for f in failures] == ["train_default_test_realism_v2"]


# ─── thresholds stay with the evaluation ─────────────────────────────────────


def test_claim_boundary_still_refuses_clinical_meaning() -> None:
    boundary = constants.CLAIM_BOUNDARY.lower()
    assert "simulator-built" in boundary
    assert "does not establish clinical realism" in boundary
    assert facade.CLAIM_BOUNDARY is constants.CLAIM_BOUNDARY


def test_artifact_path_is_unchanged() -> None:
    assert facade.DEFAULT_OUTPUT_PATH == Path(
        "Data/evals/models/latest_synthetic_model_perturbation_retrain_eval.json"
    )
