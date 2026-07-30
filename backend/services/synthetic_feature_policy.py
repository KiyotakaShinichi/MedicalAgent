"""Canonical feature policy for synthetic-only model promotion experiments.

The legacy training surface is preserved for artifact compatibility. New
promotion, comparison, and retraining evidence must use the proxy-removed
policy so a near-definition of the regression target cannot masquerade as
generalizable model signal.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_PATH = Path(
    "Data/evals/models/latest_synthetic_feature_policy.json"
)
POLICY_ID = "synthetic_proxy_removed_promotion_v1"

LEGACY_NUMERIC_FEATURES: tuple[str, ...] = (
    "cycle",
    "age",
    "pre_wbc",
    "pre_anc",
    "pre_hemoglobin",
    "pre_platelets",
    "nadir_wbc",
    "nadir_anc",
    "nadir_hemoglobin",
    "nadir_platelets",
    "recovery_wbc",
    "recovery_hemoglobin",
    "recovery_platelets",
    "mri_tumor_size_cm",
    "mri_percent_change_from_baseline",
    "max_symptom_severity",
    "symptom_count",
    "intervention_count",
    "dose_delayed",
    "dose_reduced",
)
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "stage",
    "molecular_subtype",
    "regimen",
)
DIRECT_OR_NEAR_DIRECT_TARGET_PROXIES: dict[str, tuple[str, ...]] = {
    "response_score_percent": ("mri_percent_change_from_baseline",),
}
CANONICAL_PROMOTION_NUMERIC_FEATURES: tuple[str, ...] = tuple(
    feature
    for feature in LEGACY_NUMERIC_FEATURES
    if feature
    not in {
        proxy
        for proxies in DIRECT_OR_NEAR_DIRECT_TARGET_PROXIES.values()
        for proxy in proxies
    }
)

CLAIM_BOUNDARY = (
    "This policy governs simulator-built engineering experiments only. Removing "
    "a direct target proxy improves evidence hygiene but does not establish "
    "clinical validity, real-patient generalization, treatment utility, patient "
    "benefit, or production healthcare readiness."
)


def build_synthetic_feature_policy() -> dict[str, Any]:
    removed = sorted(
        set(LEGACY_NUMERIC_FEATURES)
        - set(CANONICAL_PROMOTION_NUMERIC_FEATURES)
    )
    return {
        "schema_version": "synthetic_feature_policy_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "canonical_for_synthetic_promotion_evaluation",
        "policy_id": POLICY_ID,
        "clinical_validation": False,
        "production_ready": False,
        "canonical_scope": [
            "synthetic model promotion comparisons",
            "synthetic perturbation and retraining evaluations",
            "new synthetic champion-candidate training",
        ],
        "legacy_scope": {
            "feature_policy": list(LEGACY_NUMERIC_FEATURES),
            "allowed_use": "backward-compatible monitor-only artifact replay",
            "promotion_eligible": False,
        },
        "canonical_promotion_policy": {
            "numeric_features": list(CANONICAL_PROMOTION_NUMERIC_FEATURES),
            "categorical_features": list(CATEGORICAL_FEATURES),
            "removed_features": removed,
            "direct_or_near_direct_target_proxies": {
                target: list(features)
                for target, features in DIRECT_OR_NEAR_DIRECT_TARGET_PROXIES.items()
            },
            "promotion_eligible": True,
            "promotion_boundary": "synthetic engineering promotion only",
        },
        "migration_rule": (
            "Existing serialized models remain readable but cannot be promoted. "
            "A newly trained candidate must declare this policy_id and pass the "
            "leakage, shortcut, calibration, subgroup, perturbation, and "
            "cross-generator gates before synthetic-only promotion."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def write_synthetic_feature_policy(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    payload = build_synthetic_feature_policy()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


__all__ = [
    "POLICY_ID",
    "LEGACY_NUMERIC_FEATURES",
    "CATEGORICAL_FEATURES",
    "DIRECT_OR_NEAR_DIRECT_TARGET_PROXIES",
    "CANONICAL_PROMOTION_NUMERIC_FEATURES",
    "build_synthetic_feature_policy",
    "write_synthetic_feature_policy",
]
