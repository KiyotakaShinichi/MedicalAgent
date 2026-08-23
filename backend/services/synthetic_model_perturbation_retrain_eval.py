"""Patient-grouped retraining stress test across perturbations and generators.

The evaluation itself: it composes the scenarios, applies the thresholds, and
decides the promotion status. The pieces it composes live alongside it:

* ``synthetic_model_perturbation_constants`` - fixed inputs, feature sets,
  seeds, and the claim boundary;
* ``synthetic_model_perturbation_runner`` - scenario execution: load, perturb,
  split by patient, fit, score, repeat across seeds;
* ``synthetic_model_perturbation_metrics`` - metric computation and
  aggregation, including the constant-predictor reference baseline.

Thresholds and the promotion decision stay here on purpose. They are the
judgement this evaluation makes; the runner and the metrics only produce the
numbers it judges.

This module remains the public import surface - the release decision surface,
the evidence maturity matrix, the focused release summary, and the ship steps
all import from it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


from backend.services.synthetic_feature_policy import POLICY_ID
from backend.services.synthetic_model_perturbation_constants import (
    # `x as x` marks a deliberate re-export: these were module attributes of
    # this module before the split, so the facade keeps them even though the
    # evaluation body does not read them itself.
    CATEGORICAL_FEATURES as CATEGORICAL_FEATURES,
    CLAIM_BOUNDARY,
    DIRECT_RESPONSE_PROXY as DIRECT_RESPONSE_PROXY,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_SOURCE_PATH,
    GUARDED_NUMERIC_FEATURES,
    NUMERIC_FEATURES,
    REALISM_V2_PATH,
    REPEATED_SPLIT_SEEDS,
    SEED,
)
from backend.services.synthetic_model_perturbation_metrics import (
    _bootstrap_metric_intervals as _bootstrap_metric_intervals,
    _classification_abstention_curve as _classification_abstention_curve,
    _expected_calibration_error as _expected_calibration_error,
    _metric_deltas,
    _metric_distribution as _metric_distribution,
    _percentile_interval as _percentile_interval,
    _stress_failures,
    _train_only_constant_baseline,
)
from backend.services.synthetic_model_perturbation_runner import (
    _fit_and_score,
    _load_frame,
    _patient_split,
    _preprocessor as _preprocessor,
    _repeated_patient_split_stability,
    perturb_features,
    perturb_training_labels,
)


def build_synthetic_model_perturbation_retrain_eval(
    source_path: str | Path = DEFAULT_SOURCE_PATH,
    realism_v2_path: str | Path = REALISM_V2_PATH,
    *,
    seed: int = SEED,
) -> dict[str, Any]:
    source = _load_frame(source_path)
    realism = _load_frame(realism_v2_path)
    train, test = _patient_split(source, seed=seed)
    realism_train, realism_test = _patient_split(realism, seed=seed)

    full_clean = _fit_and_score(
        train, test, numeric_features=NUMERIC_FEATURES, seed=seed
    )
    guarded_clean = _fit_and_score(
        train, test, numeric_features=GUARDED_NUMERIC_FEATURES, seed=seed
    )
    linear_clean = _fit_and_score(
        train,
        test,
        numeric_features=GUARDED_NUMERIC_FEATURES,
        seed=seed,
        model_family="linear",
    )
    train_only_constant = _train_only_constant_baseline(train, test, seed=seed)
    complex_vs_linear = _metric_deltas(linear_clean, guarded_clean)
    complex_model_lift = bool(
        (complex_vs_linear.get("classification_auroc") or 0.0) >= 0.02
        and (complex_vs_linear.get("classification_brier") or 0.0) <= 0.0
        and (complex_vs_linear.get("regression_mae") or 0.0) <= -1.0
    )
    scenarios = []
    for name in (
        "measurement_noise",
        "modality_dropout",
        "severe_modality_dropout",
        "mnar_severity_dependent_dropout",
        "combined_noise",
    ):
        perturbed_train = perturb_features(train, scenario=name, seed=seed)
        perturbed_test = perturb_features(test, scenario=name, seed=seed + 1)
        retrained = _fit_and_score(
            perturbed_train,
            perturbed_test,
            numeric_features=GUARDED_NUMERIC_FEATURES,
            seed=seed,
        )
        clean_model_on_perturbed = _fit_and_score(
            train,
            perturbed_test,
            numeric_features=GUARDED_NUMERIC_FEATURES,
            seed=seed,
        )
        scenarios.append(
            {
                "scenario": name,
                "retrained_on_perturbation": retrained,
                "clean_model_on_perturbed_test": clean_model_on_perturbed,
                "retrained_delta_vs_guarded_clean": _metric_deltas(
                    guarded_clean, retrained
                ),
            }
        )

    for fraction, label in (
        (0.05, "five_percent_training_label_noise"),
        (0.10, "ten_percent_training_label_noise"),
        (0.20, "twenty_percent_training_label_noise"),
    ):
        label_noisy_train = perturb_training_labels(
            train,
            seed=seed,
            fraction=fraction,
        )
        label_noise = _fit_and_score(
            label_noisy_train,
            test,
            numeric_features=GUARDED_NUMERIC_FEATURES,
            seed=seed,
        )
        scenarios.append(
            {
                "scenario": label,
                "retrained_on_perturbation": label_noise,
                "clean_model_on_perturbed_test": None,
                "retrained_delta_vs_guarded_clean": _metric_deltas(
                    guarded_clean, label_noise
                ),
            }
        )

    default_to_realism = _fit_and_score(
        train,
        realism_test,
        numeric_features=GUARDED_NUMERIC_FEATURES,
        seed=seed,
    )
    realism_to_default = _fit_and_score(
        realism_train,
        test,
        numeric_features=GUARDED_NUMERIC_FEATURES,
        seed=seed,
    )
    realism_internal = _fit_and_score(
        realism_train,
        realism_test,
        numeric_features=GUARDED_NUMERIC_FEATURES,
        seed=seed,
    )
    generator_sensitivity = {
        "default_generator_internal": guarded_clean,
        "realism_v2_generator_internal": realism_internal,
        "train_default_test_realism_v2": default_to_realism,
        "train_realism_v2_test_default": realism_to_default,
        "default_to_realism_delta_vs_default_internal": _metric_deltas(
            guarded_clean, default_to_realism
        ),
        "realism_to_default_delta_vs_realism_internal": _metric_deltas(
            realism_internal, realism_to_default
        ),
    }
    repeated_split_stability = _repeated_patient_split_stability(
        source,
        realism,
        seeds=REPEATED_SPLIT_SEEDS,
    )

    stress_failures = _stress_failures(scenarios, generator_sensitivity)
    return {
        "schema_version": "synthetic_model_perturbation_retrain_eval_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "needs_attention" if stress_failures else "acceptable_synthetic_only",
        "clinical_validation": False,
        "production_ready": False,
        "source": {
            "default_generator_path": str(source_path).replace("\\", "/"),
            "realism_v2_path": str(realism_v2_path).replace("\\", "/"),
            "default_rows": len(source),
            "default_patients": int(source["patient_id"].nunique()),
            "realism_v2_rows": len(realism),
            "realism_v2_patients": int(realism["patient_id"].nunique()),
            "patient_overlap_between_train_and_test": 0,
        },
        "feature_policies": {
            "canonical_promotion_policy_id": POLICY_ID,
            "existing_full_feature_policy": {
                "numeric_features": NUMERIC_FEATURES,
                "metrics": full_clean,
                "direct_response_proxy_present": True,
                "risk": (
                    "mri_percent_change_from_baseline is definitionally close to "
                    "response_score_percent and can inflate regression evidence."
                ),
            },
            "guarded_primary_policy": {
                "numeric_features": GUARDED_NUMERIC_FEATURES,
                "metrics": guarded_clean,
                "direct_response_proxy_present": False,
                "canonical_for_promotion_evaluation": True,
                "delta_vs_full": _metric_deltas(full_clean, guarded_clean),
            },
        },
        "proxy_removed_simple_baselines": {
            "train_only_constant": train_only_constant,
            "logistic_ridge": linear_clean,
            "gradient_boosting_huber": guarded_clean,
            "gradient_boosting_delta_vs_logistic_ridge": complex_vs_linear,
            "complex_model_lift_predeclared_threshold_met": complex_model_lift,
            "complexity_decision": (
                "retain_complex_model_for_synthetic_comparison_only"
                if complex_model_lift
                else "prefer_simple_baseline_for_parsimony"
            ),
            "decision_rule": (
                "Complex model requires AUROC delta >=0.02, no Brier regression, "
                "and regression MAE improvement >=1.0 on the same patient split."
            ),
        },
        "perturbation_scenarios": scenarios,
        "generator_version_sensitivity": generator_sensitivity,
        "repeated_patient_split_stability": repeated_split_stability,
        "stress_failures": stress_failures,
        "promotion_decision": "HOLD_SYNTHETIC_ONLY",
        "model_use_boundary": "monitor_only_engineering_signal",
        "limitations": [
            "Both generator versions share project assumptions and target semantics.",
            "Noise distributions are engineering stressors, not estimates of clinical measurement error.",
            "Cross-generator transfer is not external validation.",
            "Gradient boosting is a controlled benchmark, not a promoted clinical model.",
            "Abstention curves rank internal synthetic rows by model confidence and do not establish safe clinical abstention.",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }


def write_synthetic_model_perturbation_retrain_eval(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    **kwargs: Any,
) -> dict[str, Any]:
    payload = build_synthetic_model_perturbation_retrain_eval(**kwargs)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload

# ── compatibility re-exports ────────────────────────────────────────────────
import numpy as np  # noqa: E402,F401
import pandas as pd  # noqa: E402,F401
# Module attributes of this module before the split, for the same reason: the
# original imported them at module scope, so a caller or a test could bind to
# any of them. Keeping them makes the facade a strict superset of the pre-split
# surface instead of only the names it still defines.
from sklearn.compose import ColumnTransformer as ColumnTransformer  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    GradientBoostingClassifier as GradientBoostingClassifier,
    GradientBoostingRegressor as GradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer as SimpleImputer  # noqa: E402
from sklearn.linear_model import (  # noqa: E402
    LogisticRegression as LogisticRegression,
    Ridge as Ridge,
)
from sklearn.metrics import (  # noqa: E402
    accuracy_score as accuracy_score,
    balanced_accuracy_score as balanced_accuracy_score,
    brier_score_loss as brier_score_loss,
    mean_absolute_error as mean_absolute_error,
    roc_auc_score as roc_auc_score,
)
from sklearn.pipeline import Pipeline as Pipeline  # noqa: E402
from sklearn.preprocessing import (  # noqa: E402
    OneHotEncoder as OneHotEncoder,
    StandardScaler as StandardScaler,
)

from backend.services.synthetic_feature_policy import (  # noqa: E402
    CANONICAL_PROMOTION_NUMERIC_FEATURES as CANONICAL_PROMOTION_NUMERIC_FEATURES,
    LEGACY_NUMERIC_FEATURES as LEGACY_NUMERIC_FEATURES,
)


__all__ = [
    "build_synthetic_model_perturbation_retrain_eval",
    "perturb_features",
    "perturb_training_labels",
    "write_synthetic_model_perturbation_retrain_eval",
]
