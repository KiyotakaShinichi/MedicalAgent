"""Noisier synthetic-data v2 readiness scaffold.

Status is `scaffold_only`: this records what a noisier v2 dataset would need,
not a dataset that exists. No model is retrained and no clinical behaviour
changes - the artifact exists so the gap is visible instead of implied.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


NOISIER_V2_PATH = Path("Data/evals/models/latest_noisier_synthetic_v2_readiness.json")


ALLOWED_NOISIER_V2_STATUS: frozenset[str] = frozenset({
    "scaffold_only",
    "planned_not_trained",
})


def build_noisier_synthetic_v2_readiness() -> dict[str, Any]:
    noise_types = [
        {
            "name": "missingness_noise",
            "rationale": "Real cohorts have missing labs/imaging; current synthetic data has none.",
            "planned_distribution": "Bernoulli(p=0.1-0.3) per modality per cycle, with patient-block structure.",
        },
        {
            "name": "label_noise",
            "rationale": "Real outcome labels disagree across reviewers; current synthetic labels are deterministic.",
            "planned_distribution": "Symmetric noise rate eta in {0.05, 0.10, 0.15} for binary outcomes.",
        },
        {
            "name": "measurement_noise",
            "rationale": "Lab values have analytical variance; current synthetic values are exact.",
            "planned_distribution": "Multiplicative log-normal noise calibrated to assay CV bands.",
        },
        {
            "name": "date_jitter",
            "rationale": "Real records have +/- a few days of date drift around treatment events.",
            "planned_distribution": "Uniform jitter +/- 3 days per event, preserving ordering.",
        },
        {
            "name": "symptom_reporting_noise",
            "rationale": "Patient-reported severity is bursty and inconsistent.",
            "planned_distribution": "Per-patient over/under-reporting bias drawn once per patient.",
        },
        {
            "name": "imaging_report_ambiguity",
            "rationale": "Imaging reports have hedged language; current synthetic reports are crisp.",
            "planned_distribution": "Hedge-word injection rate in {0.1, 0.2}; impression vs body separation preserved.",
        },
        {
            "name": "treatment_delay_randomness",
            "rationale": "Real chemotherapy cycles slip due to non-clinical reasons.",
            "planned_distribution": "Per-cycle delay ~ Geometric(p) with p tuned to median 0 delay, p95 ~7 days.",
        },
        {
            "name": "subgroup_distribution_shift",
            "rationale": "Synthetic cohort is balanced by construction; real subgroups are not.",
            "planned_distribution": "Reweight subgroup priors per release using documented prior shifts.",
        },
    ]

    blocked_claims = [
        "this synthetic v2 represents real patients",
        "this synthetic v2 establishes clinical performance",
        "this synthetic v2 is FDA / IRB ready",
        "this synthetic v2 is sufficient for deployment",
        "this synthetic v2 replaces real-data validation",
    ]

    expected_evals_before_promotion = [
        "leakage audit re-run with patient-level temporal CV under noise",
        "subgroup metrics re-run under each noise type independently",
        "calibration + conformal coverage under noise",
        "shortcut audit re-run; toxicity AUC must drop below saturation",
        "synthetic data quality proxy with v2-specific disclaimer text",
        "release gate must continue to PASS with v2 artifacts at status: informational",
        "no metric promoted from monitor-only to treatment-influence",
    ]

    return {
        "schema_version": "noisier_synthetic_v2_readiness_v1",
        "status": "informational",
        "scaffold_status": "scaffold_only",
        "label": "noisier_synthetic_v2_readiness",
        "clinical_validation": False,
        "claim_boundary": (
            "Readiness scaffold only.  No noisier synthetic v2 data has been "
            "generated, no model has been retrained, and no live-agent behaviour "
            "has been changed by this artifact.  This is engineering planning "
            "infrastructure; it is not clinical validation, real-world readiness, "
            "or any kind of model promotion."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "why_current_synthetic_data_is_too_clean": (
            "Current temporal_ml_rows.csv has deterministic labels, no missingness, "
            "no measurement noise, no date jitter, no reporting bias, and a "
            "balanced subgroup distribution.  This saturates every metric in the "
            "MLE stack (toxicity AUC ~1.0, patient-temporal CV AUC ~0.9996) and "
            "prevents the ML and statistical-rigor dimensions of the "
            "10/10-under-constraints roadmap from moving."
        ),
        "planned_noise_types": noise_types,
        "blocked_clinical_claims": blocked_claims,
        "expected_evals_before_promotion": expected_evals_before_promotion,
        "why_this_remains_synthetic_only": (
            "Noisier synthetic v2 still has no real patient data, no clinician-"
            "reviewed labels, and no IRB.  It improves the *measurement surface* "
            "by removing saturation; it does NOT close the gap to real data."
        ),
    }


def write_noisier_synthetic_v2_readiness(path: Path = NOISIER_V2_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_noisier_synthetic_v2_readiness(), indent=2), encoding="utf-8")
    return path


__all__ = [
    "ALLOWED_NOISIER_V2_STATUS",
    "NOISIER_V2_PATH",
    "build_noisier_synthetic_v2_readiness",
    "write_noisier_synthetic_v2_readiness",
]
