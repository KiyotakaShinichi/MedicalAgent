"""Fail-closed display policy derived from synthetic XAI evidence."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.oncology_canonical_schema import ROOT_DIR


FIDELITY_PATH = Path("Data/evals/models/latest_xai_fidelity_audit.json")
RANK_PATH = Path("Data/evals/models/latest_xai_rank_stability.json")
RETRAINING_PATH = Path("Data/evals/models/latest_xai_retraining_stability.json")
DEFAULT_OUTPUT_PATH = Path("Data/evals/models/latest_xai_reliability_gate.json")
DEFAULT_DOC_PATH = Path("docs/xai_reliability_gate.md")

CLAIM_BOUNDARY = (
    "This gate controls how synthetic model explanations are displayed. It "
    "does not establish causality, clinical explainability, clinical validity, "
    "or reliable explanation behavior on real patients."
)


def build_xai_reliability_gate(
    *,
    fidelity_path: str | Path = FIDELITY_PATH,
    rank_path: str | Path = RANK_PATH,
    retraining_path: str | Path = RETRAINING_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    doc_path: str | Path = DEFAULT_DOC_PATH,
) -> dict[str, Any]:
    fidelity = _read(fidelity_path)
    rank = _read(rank_path)
    retraining = _read(retraining_path)
    retraining_metrics = (
        retraining.get("metrics")
        if isinstance(retraining.get("metrics"), dict)
        else retraining
    )
    additivity = _float(fidelity.get("additivity_pass_rate"))
    finite = _float(fidelity.get("finite_contribution_rate"))
    set_overlap_p05 = _float(retraining_metrics.get("global_top_k_jaccard_p05"))
    rank_median = _float(
        retraining_metrics.get("global_rank_correlation_median")
    )
    rank_p05 = _float(retraining_metrics.get("global_rank_correlation_p05"))
    consensus = (
        retraining.get("consensus_feature_tiers")
        if isinstance(retraining.get("consensus_feature_tiers"), dict)
        else {}
    )
    stable_groups = _consensus_group_names(
        consensus.get("stable_core_alphabetical")
    )
    suppressed_groups = _consensus_group_names(
        consensus.get("suppressed_low_consensus_alphabetical")
    )
    retraining_policy = (
        retraining.get("presentation_policy")
        if isinstance(retraining.get("presentation_policy"), dict)
        else {}
    )
    mechanical_fidelity = (
        additivity is not None
        and additivity >= 0.99
        and finite is not None
        and finite >= 0.99
    )
    grouped_presence_allowed = bool(
        mechanical_fidelity
        and set_overlap_p05 is not None
        and set_overlap_p05 >= 0.5
    )
    ranked_order_allowed = bool(
        grouped_presence_allowed
        and rank_median is not None
        and rank_median >= 0.5
        and rank_p05 is not None
        and rank_p05 >= 0.0
    )
    bounded_grouped_display = bool(
        grouped_presence_allowed
        and stable_groups
        and retraining_policy.get("enforced") is True
        and retraining_policy.get("exact_rank_display_allowed") is False
    )
    display_mode = (
        "ranked_factors_with_noncausal_boundary"
        if ranked_order_allowed
        else "grouped_factors_without_rank_claim"
        if grouped_presence_allowed
        else "suppress_feature_factors"
    )
    payload = {
        "schema_version": "xai_reliability_gate_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "acceptable"
            if ranked_order_allowed or bounded_grouped_display
            else "needs_attention"
        ),
        "status_basis": (
            "ranked_order_stable_internal_only"
            if ranked_order_allowed
            else "bounded_consensus_groups_exact_order_suppressed"
            if bounded_grouped_display
            else "xai_display_evidence_insufficient"
        ),
        "clinical_validation": False,
        "causal_interpretation_allowed": False,
        "evidence": {
            "mechanical_fidelity_passed": mechanical_fidelity,
            "additivity_pass_rate": additivity,
            "finite_contribution_rate": finite,
            "global_top_k_jaccard_p05": set_overlap_p05,
            "global_rank_correlation_median": rank_median,
            "global_rank_correlation_p05": rank_p05,
            "rank_stability_internal_only": True,
            "human_comprehension_study_completed": bool(
                rank.get("human_participant_study_completed")
            ),
            "stable_consensus_group_count": len(stable_groups),
            "suppressed_low_consensus_group_count": len(suppressed_groups),
            "bounded_display_control_passed": bounded_grouped_display,
        },
        "patient_display_policy": {
            "mode": display_mode,
            "show_grouped_factors": grouped_presence_allowed,
            "ranked_feature_order_allowed": ranked_order_allowed,
            "show_numeric_shap_values": False,
            "maximum_factor_count": 3,
            "stable_factor_groups": stable_groups,
            "suppressed_factor_groups": suppressed_groups,
            "near_outcome_proxies_allowed": False,
            "unlisted_factor_groups_allowed": False,
            "display_order_basis": (
                "internal_rank_with_noncausal_boundary"
                if ranked_order_allowed
                else "alphabetical_not_importance"
            ),
            "instability_warning_required": not ranked_order_allowed,
            "warning": (
                "These are broad synthetic model factors. Their exact order "
                "was not stable across retraining and must not be interpreted "
                "as causation or medical importance."
                if not ranked_order_allowed
                else "Factor order is an internal synthetic stability result, not clinical importance."
            ),
        },
        "promotion_blockers": [
            item
            for item, blocked in (
                ("mechanical_fidelity_below_threshold", not mechanical_fidelity),
                ("top_factor_set_unstable", not grouped_presence_allowed),
                ("exact_factor_order_unstable", not ranked_order_allowed),
                (
                    "human_comprehension_not_completed",
                    not bool(rank.get("human_participant_study_completed")),
                ),
            )
            if blocked
        ],
        "remaining_evidence_gaps": [
            item
            for item, open_gap in (
                ("exact_factor_order_unstable", not ranked_order_allowed),
                (
                    "human_comprehension_not_completed",
                    not bool(rank.get("human_participant_study_completed")),
                ),
                ("real_patient_explanation_transfer_unmeasured", True),
            )
            if open_gap
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _write(output_path, payload)
    _write_doc(doc_path, payload)
    return payload


def load_xai_reliability_policy(
    path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    payload = _read(path)
    policy = payload.get("patient_display_policy")
    if isinstance(policy, dict):
        return policy
    return {
        "mode": "suppress_feature_factors",
        "show_grouped_factors": False,
        "ranked_feature_order_allowed": False,
        "show_numeric_shap_values": False,
        "maximum_factor_count": 0,
        "stable_factor_groups": [],
        "suppressed_factor_groups": [],
        "near_outcome_proxies_allowed": False,
        "unlisted_factor_groups_allowed": False,
        "display_order_basis": "suppressed",
        "instability_warning_required": True,
        "warning": "Explanation reliability evidence is unavailable, so feature factors are hidden.",
    }


def _consensus_group_names(rows: Any) -> list[str]:
    if not isinstance(rows, list):
        return []
    return sorted(
        {
            str(row.get("feature_group") or "").strip()
            for row in rows
            if isinstance(row, dict)
            and str(row.get("feature_group") or "").strip()
        }
    )


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read(path: str | Path) -> dict[str, Any]:
    full = _full(path)
    if not full.exists():
        return {}
    try:
        payload = json.loads(full.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write(path: str | Path, payload: dict[str, Any]) -> None:
    full = _full(path)
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_doc(path: str | Path, payload: dict[str, Any]) -> None:
    full = _full(path)
    full.parent.mkdir(parents=True, exist_ok=True)
    policy = payload["patient_display_policy"]
    full.write_text(
        "\n".join(
            [
                "# XAI Reliability Gate",
                "",
                f"- Status: `{payload['status']}`",
                f"- Patient display mode: `{policy['mode']}`",
                f"- Ranked order allowed: `{policy['ranked_feature_order_allowed']}`",
                f"- Grouped factors allowed: `{policy['show_grouped_factors']}`",
                "",
                policy["warning"],
                "",
                "## Boundary",
                "",
                payload["claim_boundary"],
                "",
            ]
        ),
        encoding="utf-8",
    )


def _full(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT_DIR / candidate


__all__ = ["build_xai_reliability_gate", "load_xai_reliability_policy"]
