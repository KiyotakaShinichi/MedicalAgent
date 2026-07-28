"""Bootstrap stability audit for synthetic SHAP feature rankings.

This audit measures whether the global ordering of exported SHAP
contributions is sensitive to resampling the synthetic patient explanations.
It does not retrain the model, prove causal explanations, or test human
comprehension.
"""

from __future__ import annotations

import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_INPUT_PATH = Path("Data/complete_synthetic_training/synthetic_xai_explanations.json")
DEFAULT_OUTPUT_PATH = Path("Data/evals/models/latest_xai_rank_stability.json")
DEFAULT_BOOTSTRAP_N = 300
DEFAULT_TOP_K = 8
NEAR_OUTCOME_PROXIES = {
    "mri_percent_change_from_baseline",
    "response_score_percent",
    "latent_response_strength",
}
GROUP_PREFIXES = {
    "stage_": "stage",
    "molecular_subtype_": "molecular_subtype",
    "treatment_type_": "treatment_type",
    "regimen_": "regimen",
}

CLAIM_BOUNDARY = (
    "Bootstrap ranking stability over synthetic SHAP exports is an engineering "
    "sensitivity check only. It is not causal explanation, human comprehension, "
    "clinical validation, or evidence that the synthetic feature relationships "
    "transfer to real patients."
)


def build_xai_rank_stability_audit(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    bootstrap_n: int = DEFAULT_BOOTSTRAP_N,
    top_k: int = DEFAULT_TOP_K,
    seed: int = 20260728,
) -> dict[str, Any]:
    if bootstrap_n < 30:
        raise ValueError("At least 30 bootstrap resamples are required")
    if top_k < 3:
        raise ValueError("top_k must be at least 3")

    source = json.loads(Path(input_path).read_text(encoding="utf-8"))
    patients = source.get("patients") or {}
    rows = list(patients.values()) if isinstance(patients, dict) else list(patients)
    matrices = [_patient_contributions(row) for row in rows]
    matrices = [row for row in matrices if row]
    if len(matrices) < 10:
        raise ValueError("At least 10 patient explanations are required")

    raw = _bootstrap_surface(matrices, bootstrap_n=bootstrap_n, top_k=top_k, seed=seed)
    grouped_rows = [_group_for_display(row, exclude_proxies=True) for row in matrices]
    grouped = _bootstrap_surface(
        grouped_rows,
        bootstrap_n=bootstrap_n,
        top_k=top_k,
        seed=seed + 1,
    )
    status = (
        "acceptable"
        if grouped["top_k_jaccard_p05"] >= 0.6
        and grouped["top_k_jaccard_median"] >= 0.8
        and grouped["rank_correlation_p05"] >= 0.5
        else "needs_attention"
    )
    payload = {
        "schema_version": "xai_rank_stability_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "method": "patient_bootstrap_mean_absolute_shap_ranking",
        "patient_explanation_n": len(matrices),
        "bootstrap_n": bootstrap_n,
        "top_k": top_k,
        "seed": seed,
        "raw_feature_ranking": raw,
        "patient_display_grouped_ranking": grouped,
        "display_transform": {
            "grouped_prefixes": GROUP_PREFIXES,
            "near_outcome_proxies_excluded": sorted(NEAR_OUTCOME_PROXIES),
            "aggregation": "sum absolute contributions within each display group",
        },
        "stability_scope": "global feature ranking under patient resampling",
        "model_retraining_stability_evaluated": False,
        "local_patient_explanation_stability_evaluated": False,
        "human_participant_study_completed": False,
        "causal_interpretation_allowed": False,
        "synthetic_only": True,
        "clinical_validation": False,
        "known_limitations": [
            "The bootstrap resamples exported synthetic patients; it does not retrain the model.",
            "Stable rankings can still reflect simulator shortcuts or near-label proxies.",
            "The display grouping is a presentation contract, not a clinical feature taxonomy.",
            "No human comprehension or decision-impact study has been completed.",
        ],
        "next_actions": [
            "Repeat the audit across independently trained seeds before promoting explanation stability.",
            "Keep near-outcome proxies separated from ordinary patient-facing context.",
            "Use grouped factors in the UI and retain raw contributions only in engineering traces.",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _patient_contributions(row: dict[str, Any]) -> dict[str, float]:
    contributions = list(row.get("all_contributions") or [])
    if not contributions:
        contributions = list(row.get("positive_contributions") or []) + list(
            row.get("negative_contributions") or []
        )
    values: dict[str, float] = {}
    for item in contributions:
        feature = str(item.get("feature") or "").strip()
        try:
            value = abs(float(item.get("contribution")))
        except (TypeError, ValueError):
            continue
        if feature and math.isfinite(value):
            values[feature] = values.get(feature, 0.0) + value
    return values


def _group_for_display(
    contributions: dict[str, float],
    *,
    exclude_proxies: bool,
) -> dict[str, float]:
    grouped: dict[str, float] = {}
    for feature, value in contributions.items():
        if exclude_proxies and feature in NEAR_OUTCOME_PROXIES:
            continue
        label = feature
        for prefix, group in GROUP_PREFIXES.items():
            if feature.startswith(prefix):
                label = group
                break
        grouped[label] = grouped.get(label, 0.0) + abs(value)
    return grouped


def _bootstrap_surface(
    rows: list[dict[str, float]],
    *,
    bootstrap_n: int,
    top_k: int,
    seed: int,
) -> dict[str, Any]:
    baseline_scores = _mean_scores(rows)
    baseline_ranking = _rank(baseline_scores)
    effective_k = min(top_k, len(baseline_ranking))
    baseline_top = baseline_ranking[:effective_k]
    rng = random.Random(seed)
    jaccards: list[float] = []
    correlations: list[float] = []
    top_feature_counts = {feature: 0 for feature in baseline_top}
    for _ in range(bootstrap_n):
        sample = [rows[rng.randrange(len(rows))] for _ in range(len(rows))]
        ranking = _rank(_mean_scores(sample))
        candidate_top = ranking[:effective_k]
        jaccards.append(_jaccard(baseline_top, candidate_top))
        correlations.append(_spearman_on_baseline(baseline_top, ranking))
        candidate_set = set(candidate_top)
        for feature in baseline_top:
            top_feature_counts[feature] += int(feature in candidate_set)
    return {
        "baseline_top_features": baseline_top,
        "feature_count": len(baseline_ranking),
        "top_k_jaccard_p05": _percentile(jaccards, 0.05),
        "top_k_jaccard_median": _percentile(jaccards, 0.5),
        "top_k_jaccard_p95": _percentile(jaccards, 0.95),
        "rank_correlation_p05": _percentile(correlations, 0.05),
        "rank_correlation_median": _percentile(correlations, 0.5),
        "rank_correlation_p95": _percentile(correlations, 0.95),
        "baseline_top_feature_inclusion_rate": {
            feature: round(count / bootstrap_n, 6)
            for feature, count in top_feature_counts.items()
        },
    }


def _mean_scores(rows: list[dict[str, float]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for row in rows:
        for feature, value in row.items():
            totals[feature] = totals.get(feature, 0.0) + abs(value)
    return {feature: value / len(rows) for feature, value in totals.items()}


def _rank(scores: dict[str, float]) -> list[str]:
    return [
        feature
        for feature, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ]


def _jaccard(left: list[str], right: list[str]) -> float:
    a, b = set(left), set(right)
    return len(a & b) / len(a | b) if a or b else 1.0


def _spearman_on_baseline(baseline: list[str], candidate: list[str]) -> float:
    if len(baseline) < 2:
        return 1.0
    fallback = len(candidate) + len(baseline)
    candidate_rank = {feature: index for index, feature in enumerate(candidate)}
    deltas = [
        (index - candidate_rank.get(feature, fallback)) ** 2
        for index, feature in enumerate(baseline)
    ]
    n = len(baseline)
    value = 1.0 - (6.0 * sum(deltas)) / (n * (n * n - 1))
    return max(-1.0, min(1.0, value))


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return round(ordered[lower], 6)
    fraction = position - lower
    return round(
        ordered[lower] + (ordered[upper] - ordered[lower]) * fraction,
        6,
    )


__all__ = [
    "build_xai_rank_stability_audit",
    "_group_for_display",
    "_patient_contributions",
]
