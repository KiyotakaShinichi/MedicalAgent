"""Typed, patient-readable explanation contract for synthetic model signals."""

from __future__ import annotations

from typing import Any

from backend.services.xai_rank_stability_audit import (
    GROUP_PREFIXES,
    NEAR_OUTCOME_PROXIES,
)


XAI_ENVELOPE_VERSION = "patient_xai_envelope_v1_2026_07"


def build_patient_xai_envelope(
    *,
    prediction: dict[str, Any] | None,
    explanation: dict[str, Any] | None,
    hybrid_prediction: dict[str, Any] | None,
    data_availability: dict[str, Any] | None,
    reliability_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if reliability_policy is None:
        from backend.services.xai_reliability_gate import (
            load_xai_reliability_policy,
        )

        reliability_policy = load_xai_reliability_policy()
    classification = (hybrid_prediction or {}).get("classification") or {}
    evidence = classification.get("evidence") or {}
    hybrid_signal = (prediction or {}).get("hybrid_mle_signal") or {}
    abstained = bool(evidence.get("abstain"))
    has_output = bool(prediction or classification)

    status = "unavailable"
    if abstained:
        status = "abstained"
    elif has_output:
        status = "available_synthetic_signal"

    modalities_present = list(evidence.get("modalities_present") or [])
    modalities_missing = list(evidence.get("modalities_missing") or [])
    if not modalities_present and data_availability:
        modalities_present = [
            str(item.get("name"))
            for item in (data_availability.get("items") or [])
            if item.get("status") == "available"
        ]
        modalities_missing = [
            str(item.get("name"))
            for item in (data_availability.get("items") or [])
            if item.get("status") in {"missing", "insufficient_data", "model_unavailable"}
        ]

    return {
        "schema_version": XAI_ENVELOPE_VERSION,
        "status": status,
        "output": {
            "label": "Synthetic monitoring pattern",
            "hybrid_score": hybrid_signal.get("hybrid_score"),
            "classification_probability": hybrid_signal.get("classification_probability") or classification.get("probability"),
            "decision": classification.get("decision"),
            "meaning": (
                "This groups the available record against simulator-built examples. "
                "It is not the patient's chance of response or a health score."
            ),
            "calculation": (
                "When both synthetic heads are available, the displayed hybrid score combines "
                "65% calibrated classification probability and 35% normalized regression output. "
                "Unavailable heads are omitted rather than invented."
            ),
        },
        "evidence": {
            "inputs_used": modalities_present,
            "inputs_missing": modalities_missing,
            "sufficiency": evidence.get("sufficiency"),
            "abstained": abstained,
            "abstain_reason": evidence.get("reason"),
        },
        "uncertainty": {
            "confidence": classification.get("confidence"),
            "confidence_modifier": evidence.get("confidence_modifier"),
            "uncertainty_is_clinical_probability": False,
            "explanation": (
                "Confidence describes this synthetic model and the available input coverage; "
                "it does not quantify a real clinical outcome."
            ),
        },
        "top_model_factors": _top_factors(
            explanation,
            reliability_policy=reliability_policy,
        ),
        "explanation_reliability": {
            "display_mode": reliability_policy.get("mode"),
            "ranked_feature_order_allowed": bool(
                reliability_policy.get("ranked_feature_order_allowed")
            ),
            "numeric_shap_values_visible": bool(
                reliability_policy.get("show_numeric_shap_values")
            ),
            "warning": reliability_policy.get("warning"),
        },
        "provenance": {
            "synthetic_only": True,
            "model_version": classification.get("model_version"),
            "explanation_method": (explanation or {}).get("method"),
            "causal_interpretation_allowed": False,
        },
        "safe_next_steps": [
            "Review any flagged or unclear record with the care team.",
            "Add a missing record only if it is already available to you; do not infer or estimate it.",
        ],
        "clinical_validation": False,
        "claim_boundary": (
            "Synthetic engineering explanation only. It does not diagnose, predict prognosis, "
            "recommend treatment, establish causality, or replace clinician interpretation."
        ),
    }


def _top_factors(
    explanation: dict[str, Any] | None,
    *,
    reliability_policy: dict[str, Any],
) -> list[dict[str, Any]]:
    if not reliability_policy.get("show_grouped_factors"):
        return []
    limit = max(
        0,
        min(int(reliability_policy.get("maximum_factor_count") or 0), 6),
    )
    if limit <= 0:
        return []
    allowed = {
        str(value)
        for value in reliability_policy.get("stable_factor_groups") or []
        if str(value).strip()
    }
    if not allowed or reliability_policy.get("unlisted_factor_groups_allowed"):
        return []
    grouped: dict[str, dict[str, Any]] = {}
    for direction, key in (
        ("toward_synthetic_positive_class", "positive_contributions"),
        ("away_from_synthetic_positive_class", "negative_contributions"),
    ):
        for item in ((explanation or {}).get(key) or []):
            if not isinstance(item, dict):
                continue
            feature = str(item.get("feature") or "").strip()
            group = _display_group(feature)
            if not group or group not in allowed:
                continue
            try:
                raw_value = float(
                    item.get("shap_value", item.get("contribution")) or 0.0
                )
            except (TypeError, ValueError):
                raw_value = 0.0
            bucket = grouped.setdefault(
                group,
                {
                    "signed_contribution": 0.0,
                    "meaning": item.get("meaning"),
                },
            )
            bucket["signed_contribution"] += raw_value

    ordered = list(grouped.items())
    if reliability_policy.get("ranked_feature_order_allowed"):
        ordered.sort(
            key=lambda pair: abs(pair[1]["signed_contribution"]),
            reverse=True,
        )
    else:
        ordered.sort(key=lambda pair: pair[0])
    factors: list[dict[str, Any]] = []
    for group, bucket in ordered[:limit]:
        signed = float(bucket["signed_contribution"])
        factors.append(
            {
                "feature": group,
                "relative_contribution": (
                    signed
                    if reliability_policy.get("show_numeric_shap_values")
                    else None
                ),
                "direction": (
                    "toward_synthetic_positive_class"
                    if signed >= 0
                    else "away_from_synthetic_positive_class"
                ),
                "meaning": bucket.get("meaning"),
                "stability_tier": "stable_core",
                "display_order_basis": reliability_policy.get(
                    "display_order_basis"
                ),
                "clinical_causality": False,
                "rank_interpretation_allowed": bool(
                    reliability_policy.get("ranked_feature_order_allowed")
                ),
            }
        )
    return factors


def _display_group(feature: str) -> str | None:
    if not feature or feature in NEAR_OUTCOME_PROXIES:
        return None
    for prefix, group in GROUP_PREFIXES.items():
        if feature.startswith(prefix):
            return group
    return feature


__all__ = ["XAI_ENVELOPE_VERSION", "build_patient_xai_envelope"]
