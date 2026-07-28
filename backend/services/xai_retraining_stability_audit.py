"""Retraining-based stability audit for synthetic model explanations.

Unlike the patient bootstrap audit, this module refits the logistic baseline
across independent patient-level splits and measures global and local
explanation stability. It remains synthetic-only and does not test causal or
clinical validity.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from backend.services.complete_synthetic_training import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    _patient_split,
    _preprocessor,
)
from backend.services.complete_synthetic_xai import _clean_feature_name
from backend.services.xai_rank_stability_audit import (
    _group_for_display,
    _jaccard,
    _percentile,
    _rank,
    _spearman_on_baseline,
)


DEFAULT_INPUT_PATH = Path("Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv")
DEFAULT_OUTPUT_PATH = Path("Data/evals/models/latest_xai_retraining_stability.json")
DEFAULT_SEEDS = (11, 23, 37, 42, 59, 71, 83, 97, 109, 127, 149, 173)


def build_xai_retraining_stability_audit(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    top_k: int = 8,
    local_patient_limit: int = 80,
) -> dict[str, Any]:
    if len(seeds) < 8:
        raise ValueError("At least eight independent seeds are required")
    rows = pd.read_csv(input_path)
    required = {
        "patient_id",
        "treatment_success_binary",
        *NUMERIC_FEATURES,
        *CATEGORICAL_FEATURES,
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    seed_surfaces = [
        _fit_seed_surface(rows, seed=seed, top_k=top_k)
        for seed in seeds
    ]
    baseline = seed_surfaces[0]
    global_jaccards: list[float] = []
    global_correlations: list[float] = []
    probability_stds: list[float] = []
    patient_ids = sorted(rows["patient_id"].astype(str).unique())[:local_patient_limit]
    local_jaccards: list[float] = []

    for candidate in seed_surfaces[1:]:
        global_jaccards.append(
            _jaccard(baseline["global_ranking"][:top_k], candidate["global_ranking"][:top_k])
        )
        global_correlations.append(
            _spearman_on_baseline(
                baseline["global_ranking"][:top_k],
                candidate["global_ranking"],
            )
        )
        for patient_id in patient_ids:
            left = baseline["patient_rankings"].get(patient_id, [])[:top_k]
            right = candidate["patient_rankings"].get(patient_id, [])[:top_k]
            if left and right:
                local_jaccards.append(_jaccard(left, right))

    for patient_id in patient_ids:
        values = [
            surface["patient_probabilities"].get(patient_id)
            for surface in seed_surfaces
        ]
        finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
        if len(finite) == len(seed_surfaces):
            probability_stds.append(float(np.std(finite, ddof=1)))

    metrics = {
        "global_top_k_jaccard_p05": _percentile(global_jaccards, 0.05),
        "global_top_k_jaccard_median": _percentile(global_jaccards, 0.5),
        "global_rank_correlation_p05": _percentile(global_correlations, 0.05),
        "global_rank_correlation_median": _percentile(global_correlations, 0.5),
        "local_patient_top_k_jaccard_p05": _percentile(local_jaccards, 0.05),
        "local_patient_top_k_jaccard_median": _percentile(local_jaccards, 0.5),
        "patient_probability_std_p50": _percentile(probability_stds, 0.5),
        "patient_probability_std_p95": _percentile(probability_stds, 0.95),
    }
    status = (
        "acceptable"
        if metrics["global_top_k_jaccard_p05"] >= 0.6
        and metrics["global_rank_correlation_p05"] >= 0.5
        and metrics["local_patient_top_k_jaccard_median"] >= 0.5
        else "needs_attention"
    )
    payload = {
        "schema_version": "xai_retraining_stability_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "method": "independent_patient_split_logistic_retraining",
        "seed_count": len(seeds),
        "seeds": list(seeds),
        "patient_n": int(rows["patient_id"].nunique()),
        "temporal_row_n": int(len(rows)),
        "local_patient_n": len(patient_ids),
        "top_k": top_k,
        "baseline_seed": seeds[0],
        "baseline_grouped_ranking": baseline["global_ranking"][:top_k],
        "metrics": metrics,
        "model_retraining_stability_evaluated": True,
        "local_patient_explanation_stability_evaluated": True,
        "grouped_patient_display_features": True,
        "near_outcome_proxies_excluded": True,
        "human_participant_study_completed": False,
        "causal_interpretation_allowed": False,
        "synthetic_only": True,
        "clinical_validation": False,
        "known_limitations": [
            "Every retraining run uses the same synthetic generator output.",
            "Stable explanations can reproduce simulator shortcuts.",
            "Only the logistic engineering baseline is evaluated.",
            "No human comprehension or clinical decision-impact study was performed.",
        ],
        "claim_boundary": (
            "This audit measures explanation sensitivity to retraining and patient-level "
            "split variation on synthetic data. It is not causal explanation, clinical "
            "validation, human-factors evidence, or proof of transfer to real patients."
        ),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _fit_seed_surface(rows: pd.DataFrame, *, seed: int, top_k: int) -> dict[str, Any]:
    train_patients, _ = _patient_split(
        rows,
        "treatment_success_binary",
        0.25,
        seed,
    )
    train = rows[rows["patient_id"].isin(train_patients)]
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    model = Pipeline(
        [
            ("preprocess", _preprocessor(scale_numeric=True)),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=seed,
                ),
            ),
        ]
    )
    model.fit(train[features], train["treatment_success_binary"].astype(int))
    preprocessor = model.named_steps["preprocess"]
    classifier = model.named_steps["classifier"]
    transformed = preprocessor.transform(rows[features])
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed, dtype=float)
    names = [_clean_feature_name(name) for name in preprocessor.get_feature_names_out()]
    contributions = transformed * np.asarray(classifier.coef_[0], dtype=float)
    probabilities = model.predict_proba(rows[features])[:, 1]

    patient_rankings: dict[str, list[str]] = {}
    patient_probabilities: dict[str, float] = {}
    grouped_global: dict[str, float] = {}
    indexed = rows.reset_index(drop=True)
    for patient_id, group in indexed.groupby("patient_id"):
        indices = group.index.to_numpy()
        values = {
            names[index]: float(np.mean(np.abs(contributions[indices, index])))
            for index in range(len(names))
        }
        grouped = _group_for_display(values, exclude_proxies=True)
        patient_rankings[str(patient_id)] = _rank(grouped)
        patient_probabilities[str(patient_id)] = float(np.mean(probabilities[indices]))
        for feature, value in grouped.items():
            grouped_global[feature] = grouped_global.get(feature, 0.0) + value
    patient_count = max(1, len(patient_rankings))
    grouped_global = {
        feature: value / patient_count
        for feature, value in grouped_global.items()
    }
    return {
        "seed": seed,
        "global_ranking": _rank(grouped_global),
        "patient_rankings": patient_rankings,
        "patient_probabilities": patient_probabilities,
        "top_k": top_k,
    }


__all__ = ["build_xai_retraining_stability_audit"]
