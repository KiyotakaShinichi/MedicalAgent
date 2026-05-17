"""Synthetic data generator card.

What this is
------------
A structured, dashboard-readable card that documents what the synthetic
breast-cancer monitoring dataset actually IS and — more importantly —
what it ISN'T.  Reviewers will ask:

  - "Which generator produced the rows you trained on?"
  - "What seed?  Can I reproduce it?"
  - "What causal assumptions does the generator bake in?"
  - "Where could the model be exploiting shortcuts in the generator?"
  - "What clinical phenomena does this synthetic dataset NOT cover?"

This service answers all of those by combining the runtime metadata that
`complete_synthetic_dataset` writes to `summary.json` with a hand-curated
"known limitations" block reviewed by the engineering team.

It is engineering provenance, not clinical validation.  A passing
generator card is a precondition for trusting downstream metrics, not a
substitute for them.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_DATASET_DIR = "Data/complete_synthetic_breast_journeys"
DEFAULT_OUTPUT_PATH = "Data/evals/models/latest_synthetic_generator_card.json"


# Curated narrative blocks.  These describe what the *generator itself*
# does in terms a reviewer can audit — they are not derived from the data,
# they describe the code in `backend/services/complete_synthetic_dataset.py`
# and its peers.  Any time the generator's behavior changes meaningfully,
# this block must be updated in lockstep — that's why a test asserts the
# `generator_card_version` matches the dataset's `schema_version`.

GENERATOR_CARD_VERSION = "v2_2026_05"

CAUSAL_ASSUMPTIONS: tuple[str, ...] = (
    "Each patient has a latent 'response strength' drawn at birth that "
    "drives outcome category, MRI percent-change trajectory, and final "
    "label.  All cycles for the same patient share that latent.",
    "Toxicity events (low nadir CBC, dose delays/reductions) are sampled "
    "per cycle but conditioned on the same response-strength latent — "
    "non-responders trend toward higher toxicity counts on average.",
    "Imaging response (mri_percent_change_from_baseline) is monotonic in "
    "absolute terms across cycles for a given patient — synthetic patients "
    "do not stop responding mid-treatment in this generator.",
    "Symptom severity, symptom counts, and intervention counts are noisy "
    "but correlated with the same latent — no independent symptom-only "
    "subgroup is modelled.",
    "Subtype (HR+/HER2+/TNBC) influences regimen selection and baseline "
    "lab profiles, but the latent response strength is sampled independently "
    "of subtype in the balanced-subgroups configuration.",
)

KNOWN_SHORTCUTS: tuple[str, ...] = (
    "mri_percent_change_from_baseline is computed from the same latent "
    "that produces response_score_percent — leakage_audit covers this by "
    "excluding the regression target from features.",
    "Patients with very low nadir CBC + multiple dose modifications are "
    "almost always non-responders in the synthetic distribution; real-world "
    "data is much noisier.",
    "Cycle 1 features alone can already separate responders from non-"
    "responders better than would be realistic, because the latent shows "
    "up in cycle-1 imaging response.",
    "Patient IDs follow a deterministic prefix+index pattern; never use "
    "the ID string as a feature.",
)

UNSUPPORTED_CLAIMS: tuple[str, ...] = (
    "Real clinical performance — every reported metric is synthetic.",
    "Calibration of the model on real patient populations.",
    "Subgroup fairness on real demographic strata.  The generator's "
    "subgroup balance is deliberately uniform.",
    "Behavior under genuine out-of-distribution patients (rare subtypes, "
    "very old/young patients, metastatic-only journeys).",
    "Response to treatment regimens not in the generator's regimen menu.",
)

REALISM_CHECKS = (
    "synthetic_realism_candidate alignment score (separate artifact)",
    "noise_eval AUROC drop under feature corruption (separate artifact)",
    "temporal_eval generalization gap (separate artifact)",
    "drift_report subgroup performance (separate artifact)",
)


# ─── Public API ──────────────────────────────────────────────────────────────


def build_synthetic_generator_card(
    dataset_dir: str = DEFAULT_DATASET_DIR,
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    """Read the dataset's summary.json + temporal_ml_rows.csv, fuse it with
    the curated narrative blocks, write the card, and return the payload."""
    summary = _load_summary(dataset_dir)
    options = summary.get("generation_options") or {}
    table_counts = summary.get("table_counts") or {}

    distribution = _compute_distribution_summary(dataset_dir)
    rows_fingerprint = _hash_rows_csv(dataset_dir)

    schema_version = options.get("schema_version", "unknown")
    consistent = schema_version.endswith("v2") or schema_version == "unknown"

    payload: dict[str, Any] = {
        "schema_version": "synthetic_generator_card_v1",
        "generator_card_version": GENERATOR_CARD_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if consistent else "needs_attention",
        "dataset_dir": dataset_dir,
        "dataset_schema_version": schema_version,
        "card_version_matches_dataset": consistent,
        "generation_options": options,
        "cohort": {
            "patients_created": summary.get("patients_created"),
            "cycles_per_patient": summary.get("cycles_per_patient"),
            "table_counts": table_counts,
            "rows_fingerprint": rows_fingerprint,
        },
        "supported_labels": [
            "treatment_success_binary",
            "toxicity_risk_binary",
            "urgent_intervention_needed",
            "support_intervention_needed",
            "response_score_percent (regression)",
            "final_response_category (multiclass)",
        ],
        "feature_distribution_summary": distribution,
        "causal_assumptions": list(CAUSAL_ASSUMPTIONS),
        "known_shortcuts": list(KNOWN_SHORTCUTS),
        "unsupported_claims": list(UNSUPPORTED_CLAIMS),
        "realism_checks_referenced": list(REALISM_CHECKS),
        "claim_boundary": (
            "Fully synthetic data for engineering and ML practice only. "
            "Not clinical evidence.  Real patient data may violate every "
            "causal assumption listed above."
        ),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_synthetic_generator_card(
    path: str = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {
            "schema_version": "synthetic_generator_card_v1",
            "status": "missing",
            "message": (
                "Generator card has not been built yet.  Run "
                "`scripts/run_synthetic_generator_card.py` or POST to "
                "/admin/synthetic-generator-card."
            ),
            "cohort": {},
            "generation_options": {},
            "causal_assumptions": [],
            "known_shortcuts": [],
            "unsupported_claims": [],
        }
    return json.loads(file_path.read_text(encoding="utf-8"))


# ─── Loading helpers ─────────────────────────────────────────────────────────


def _load_summary(dataset_dir: str) -> dict[str, Any]:
    summary_path = Path(dataset_dir) / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _compute_distribution_summary(dataset_dir: str) -> dict[str, Any]:
    """Lightweight per-column summary used by the dashboard to spot drift
    between the documented assumptions and the actual rows."""
    csv = Path(dataset_dir) / "temporal_ml_rows.csv"
    if not csv.exists():
        return {"status": "missing", "message": "temporal_ml_rows.csv not found"}
    frame = pd.read_csv(csv)

    numeric_summary: dict[str, dict[str, float | None]] = {}
    for column in ("age", "pre_wbc", "pre_anc", "nadir_wbc", "mri_percent_change_from_baseline",
                   "max_symptom_severity", "response_score_percent"):
        if column not in frame.columns:
            continue
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if series.empty:
            numeric_summary[column] = {"count": 0, "mean": None, "std": None, "min": None, "max": None}
            continue
        numeric_summary[column] = {
            "count": int(len(series)),
            "mean": round(float(series.mean()), 4),
            "std":  round(float(series.std()), 4),
            "min":  round(float(series.min()), 4),
            "max":  round(float(series.max()), 4),
        }

    categorical_summary: dict[str, dict[str, int]] = {}
    for column in ("stage", "molecular_subtype", "regimen",
                   "final_response_category", "final_response_multiclass"):
        if column not in frame.columns:
            continue
        counts = frame[column].astype(str).value_counts().to_dict()
        categorical_summary[column] = {k: int(v) for k, v in counts.items()}

    return {
        "row_count": int(len(frame)),
        "numeric": numeric_summary,
        "categorical": categorical_summary,
        "positive_label_rate": (
            round(float(frame.get("treatment_success_binary", pd.Series([])).mean()), 4)
            if "treatment_success_binary" in frame.columns else None
        ),
    }


def _hash_rows_csv(dataset_dir: str) -> str | None:
    """Stable fingerprint of the temporal_ml_rows.csv content — lets a
    reviewer detect when the dataset was regenerated even if filenames
    didn't change."""
    csv = Path(dataset_dir) / "temporal_ml_rows.csv"
    if not csv.exists():
        return None
    hasher = hashlib.sha256()
    with csv.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


__all__ = [
    "GENERATOR_CARD_VERSION",
    "CAUSAL_ASSUMPTIONS",
    "KNOWN_SHORTCUTS",
    "UNSUPPORTED_CLAIMS",
    "build_synthetic_generator_card",
    "load_synthetic_generator_card",
]
