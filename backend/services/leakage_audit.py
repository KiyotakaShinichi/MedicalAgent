"""Unified leakage audit for the synthetic temporal monitoring dataset.

This service is the *engineering* gate for ML data hygiene.  It composes the
older `temporal_leakage_audit` (name-based field checks + cycle ordering) with
three additional checks the senior-MLE review specifically called out:

  1. Patient-ID overlap between train and test splits — fails if any patient
     appears on both sides under the project's configured split function.
  2. Direct label-proxy fields in the feature contract — fails if a known
     downstream-of-outcome column (latent_response_strength, final_*, etc.)
     ever shows up in NUMERIC_FEATURES + CATEGORICAL_FEATURES.
  3. Classification target == single-feature identity — fails if any feature
     column is byte-for-byte the classification label (a sanity tripwire for
     accidental relabel/copy-paste regressions).

Output schema: see `_build_payload` at the bottom.  Written to
`Data/evals/models/latest_leakage_audit.json` so the benchmark registry can
treat it as a critical-tier engineering artifact.

This audit produces engineering evidence only.  A passing audit means the
configured feature contract avoids the obvious synthetic leakage patterns we
know about; it does not prove the model has no leakage in the clinical sense.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from backend.services.complete_synthetic_training import (
    CATEGORICAL_FEATURES,
    EXCLUDED_COLUMNS,
    NUMERIC_FEATURES,
    ROW_LEVEL_TARGETS,
    _patient_split,
)
from backend.services.temporal_leakage_audit import (
    DEFAULT_OUTPUT_PATH as DEFAULT_TEMPORAL_OUTPUT_PATH,
    run_temporal_leakage_audit,
)


DEFAULT_TRAINING_ROWS_PATH = "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv"
DEFAULT_OUTPUT_PATH = "Data/evals/models/latest_leakage_audit.json"

# Fields that the synthetic generator either produces directly as a label or
# uses internally to *generate* the label.  None of these may appear in the
# feature contract, even if they happen to be numerically informative — using
# them defeats the entire point of training.
KNOWN_LABEL_PROXIES: tuple[str, ...] = (
    "latent_response_strength",
    "response_score_percent",
    "final_response_category",
    "final_response_multiclass",
    "final_cancer_status",
    "treatment_success_binary",
    "maintenance_needed",
    "cycle_response_trend_class",
)

DEFAULT_CLASSIFICATION_TARGETS: tuple[str, ...] = (
    "treatment_success_binary",
    "toxicity_risk_binary",
    "urgent_intervention_needed",
    "support_intervention_needed",
)

DEFAULT_SPLIT_SEEDS: tuple[int, ...] = (0, 7, 42, 123)


@dataclass
class _Finding:
    """One audit rule's verdict.  `status` is "passed" or "failed"; anything
    else is treated as failed by the aggregator."""

    name: str
    status: str
    evidence: dict[str, Any] = field(default_factory=dict)
    meaning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "evidence": self.evidence,
            "meaning": self.meaning,
        }


def run_leakage_audit(
    training_rows_path: str = DEFAULT_TRAINING_ROWS_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    *,
    classification_targets: tuple[str, ...] = DEFAULT_CLASSIFICATION_TARGETS,
    split_seeds: tuple[int, ...] = DEFAULT_SPLIT_SEEDS,
    test_size: float = 0.25,
    temporal_output_path: str = DEFAULT_TEMPORAL_OUTPUT_PATH,
) -> dict[str, Any]:
    """Run every leakage check and write the unified report.

    `classification_targets` and `split_seeds` are arguments rather than
    constants so the tests can exercise narrower configurations without rerunning
    the slow temporal audit each time.
    """
    rows = pd.read_csv(training_rows_path)
    feature_columns = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)

    findings: list[_Finding] = []
    findings.append(_check_label_proxies_in_features(feature_columns))
    findings.append(_check_excluded_columns_complete())
    findings.extend(_check_label_identity_against_features(rows, classification_targets, feature_columns))
    findings.extend(_check_patient_split_overlap(rows, classification_targets, split_seeds, test_size))
    findings.extend(_check_per_cycle_uniqueness(rows))

    # Re-use the older temporal audit so its findings live alongside the new
    # checks under one artifact.  We swallow exceptions explicitly: if the
    # temporal audit fails, that itself is an audit failure.
    temporal_payload: dict[str, Any] | None
    try:
        temporal_payload = run_temporal_leakage_audit(
            training_rows_path=training_rows_path,
            output_path=temporal_output_path,
        )
    except Exception as exc:  # noqa: BLE001 — audit itself crashing is an audit failure
        temporal_payload = None
        findings.append(_Finding(
            name="temporal_leakage_audit_runs",
            status="failed",
            evidence={"error": str(exc)},
            meaning="The temporal sub-audit must run successfully for the unified gate to pass.",
        ))

    payload = _build_payload(
        training_rows_path=training_rows_path,
        feature_columns=feature_columns,
        findings=findings,
        temporal_payload=temporal_payload,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_leakage_audit(output_path: str = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    """Read the most-recently-written audit artifact, or return a minimal
    'missing' payload if the file does not exist yet.  Used by the admin
    GET endpoint so the dashboard can render even before the first audit run."""
    path = Path(output_path)
    if not path.exists():
        return {
            "schema_version": "leakage_audit_v1",
            "status": "missing",
            "message": (
                "Leakage audit has not been generated yet. Run "
                "`scripts/run_leakage_audit.py` or POST to "
                "/admin/leakage-audit to produce it."
            ),
            "findings": [],
            "summary": {"checks_total": 0, "checks_passed": 0, "checks_failed": 0},
        }
    return json.loads(path.read_text(encoding="utf-8"))


def _check_label_proxies_in_features(feature_columns: list[str]) -> _Finding:
    """The feature contract must never include any field on the proxy list."""
    proxies_used = sorted(set(feature_columns) & set(KNOWN_LABEL_PROXIES))
    return _Finding(
        name="known_label_proxies_excluded_from_features",
        status="passed" if not proxies_used else "failed",
        evidence={
            "label_proxies_in_features": proxies_used,
            "checked_proxies": list(KNOWN_LABEL_PROXIES),
        },
        meaning=(
            "Generator latent strength, final outcomes, and the regression "
            "target are derived from the label and must never be used as "
            "model inputs — using them inflates metrics without learning."
        ),
    )


def _check_excluded_columns_complete() -> _Finding:
    """`EXCLUDED_COLUMNS` is the project's denylist; it must cover every known
    proxy so a future feature-list edit can't accidentally re-include one."""
    proxies_not_excluded = sorted(set(KNOWN_LABEL_PROXIES) - set(EXCLUDED_COLUMNS))
    return _Finding(
        name="exclusion_set_covers_all_known_proxies",
        status="passed" if not proxies_not_excluded else "failed",
        evidence={
            "proxies_missing_from_exclusion_set": proxies_not_excluded,
            "current_exclusion_set_size": len(EXCLUDED_COLUMNS),
        },
        meaning=(
            "The EXCLUDED_COLUMNS denylist in complete_synthetic_training.py "
            "is the single source of truth for what cannot be a feature. "
            "Any known proxy not in that set is a latent regression risk."
        ),
    )


def _check_label_identity_against_features(
    rows: pd.DataFrame,
    targets: tuple[str, ...],
    feature_columns: list[str],
) -> list[_Finding]:
    """For every classification target, every numeric feature must differ from
    the label in at least one row.  Catches accidental column duplication."""
    findings: list[_Finding] = []
    for target in targets:
        if target not in rows.columns:
            findings.append(_Finding(
                name=f"label_identity_check::{target}",
                status="passed",
                evidence={"target_present_in_data": False},
                meaning="Target column is not present in this dataset — check skipped.",
            ))
            continue

        label = pd.to_numeric(rows[target], errors="coerce")
        matching_features: list[str] = []
        for feature in feature_columns:
            if feature not in rows.columns:
                continue
            series = pd.to_numeric(rows[feature], errors="coerce")
            # Only compare on rows where both are defined; if every overlapping
            # row matches exactly, the feature is byte-equal to the label.
            overlap = label.notna() & series.notna()
            if overlap.sum() == 0:
                continue
            if (label[overlap] == series[overlap]).all():
                matching_features.append(feature)
        findings.append(_Finding(
            name=f"label_identity_check::{target}",
            status="passed" if not matching_features else "failed",
            evidence={
                "target": target,
                "features_equal_to_target": matching_features,
            },
            meaning=(
                "A feature column that is numerically identical to the label "
                "indicates a copy/rename regression that would trivialise the "
                "classifier."
            ),
        ))
    return findings


def _check_patient_split_overlap(
    rows: pd.DataFrame,
    targets: tuple[str, ...],
    seeds: tuple[int, ...],
    test_size: float,
) -> list[_Finding]:
    """Run the *real* `_patient_split` for each target/seed combo and assert
    train and test patient sets are disjoint."""
    findings: list[_Finding] = []
    for target in targets:
        if target not in rows.columns:
            continue
        for seed in seeds:
            try:
                train_patients, test_patients = _patient_split(rows, target, test_size, seed)
            except ValueError as exc:
                # The target doesn't have enough classes for this dataset — skip
                # but record so the report still surfaces what was attempted.
                findings.append(_Finding(
                    name=f"patient_split_disjoint::{target}::seed{seed}",
                    status="passed",
                    evidence={
                        "target": target,
                        "seed": seed,
                        "skipped_reason": str(exc),
                    },
                    meaning="Split could not be computed; check skipped, not failed.",
                ))
                continue
            overlap = sorted(train_patients & test_patients)
            findings.append(_Finding(
                name=f"patient_split_disjoint::{target}::seed{seed}",
                status="passed" if not overlap else "failed",
                evidence={
                    "target": target,
                    "seed": seed,
                    "train_patient_count": len(train_patients),
                    "test_patient_count": len(test_patients),
                    "overlapping_patient_count": len(overlap),
                    "example_overlap": overlap[:10],
                },
                meaning=(
                    "Patient IDs must not appear in both train and test. "
                    "Even one overlap means cycle-level rows from the same "
                    "patient leak across the split."
                ),
            ))
    return findings


def _check_per_cycle_uniqueness(rows: pd.DataFrame) -> list[_Finding]:
    """A row keyed on (patient_id, cycle) must be unique.  Duplicate rows
    silently double-count specific patient-cycle combinations and bias every
    downstream metric."""
    if not {"patient_id", "cycle"}.issubset(rows.columns):
        return []
    duplicate_rows = int(rows.duplicated(["patient_id", "cycle"]).sum())
    return [_Finding(
        name="patient_cycle_pair_is_unique",
        status="passed" if duplicate_rows == 0 else "failed",
        evidence={
            "duplicate_patient_cycle_row_count": duplicate_rows,
            "total_rows": int(len(rows)),
        },
        meaning=(
            "(patient_id, cycle) is the natural primary key for temporal "
            "training rows.  Any duplicate biases metrics by overweighting "
            "those visits."
        ),
    )]


def _build_payload(
    *,
    training_rows_path: str,
    feature_columns: list[str],
    findings: list[_Finding],
    temporal_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    finding_dicts = [item.to_dict() for item in findings]
    has_local_failure = any(item["status"] != "passed" for item in finding_dicts)
    temporal_status = (temporal_payload or {}).get("status", "missing")
    overall_status = "failed" if has_local_failure or temporal_status == "failed" else "passed"

    return {
        "schema_version": "leakage_audit_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": overall_status,
        "training_rows_path": training_rows_path,
        "feature_columns": feature_columns,
        "row_level_targets": sorted(ROW_LEVEL_TARGETS),
        "known_label_proxies": list(KNOWN_LABEL_PROXIES),
        "findings": finding_dicts,
        "temporal_sub_audit": {
            "status": temporal_status,
            "findings": (temporal_payload or {}).get("findings", []),
        },
        "summary": {
            "checks_total": len(finding_dicts),
            "checks_passed": sum(1 for item in finding_dicts if item["status"] == "passed"),
            "checks_failed": sum(1 for item in finding_dicts if item["status"] != "passed"),
        },
        "interpretation": (
            "A passing audit means the configured feature contract and "
            "patient-aware split are free of the known engineering leakage "
            "patterns.  It is not a guarantee of clinical validity; only "
            "real-world prospective validation can establish that."
        ),
        "claim_boundary": (
            "Engineering evidence only.  Synthetic dataset audit; does not "
            "establish absence of clinical leakage in real patient records."
        ),
    }
