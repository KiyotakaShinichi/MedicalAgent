"""Evidence-aware prediction wrapper around the trained synthetic classifier.

Loads the project's champion treatment-response model + isotonic calibrator,
runs inference, and returns a structured response object that pairs every
probability with:

  - the evidence-sufficiency decision from `evidence_sufficiency`
  - an explicit ``decision`` field that can be ``"favorable_pattern"``,
    ``"concerning_pattern"``, ``"uncertain"`` or ``"insufficient_evidence"``
  - the list of modalities that were actually present on the input row
  - a non-diagnostic claim boundary

The wrapper is the single point of contact between the trained model and
anything else in the system — patient dashboard summary, clinician review
queue, batch evaluation.  Callers should never reach into the raw .joblib
files directly; using this service guarantees the abstention contract.

Importantly: when ``decision == "insufficient_evidence"``, the response
contains ``probability == None``.  Downstream code must treat that case as
"no answer", not "probability is zero".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd

from backend.services.complete_synthetic_training import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)
from backend.services.evidence_sufficiency import (
    EvidenceAssessment,
    assess_evidence,
)


# These paths mirror the artifact layout written by `complete_synthetic_training`.
DEFAULT_MODEL_PATH = "Data/complete_synthetic_training/gradient_boosting_treatment_success_binary.joblib"
DEFAULT_CALIBRATOR_PATH = "Data/complete_synthetic_training/gradient_boosting_isotonic_calibrator_treatment_success_binary.joblib"

# Decision thresholds on the *calibrated* probability.  Anything in the
# middle band is labelled "uncertain" rather than forcing a 0/1 classification;
# the band is intentionally wide because synthetic AUROC tends to be optimistic.
LOWER_DECISION_THRESHOLD = 0.40
UPPER_DECISION_THRESHOLD = 0.60


@dataclass
class EvidenceAwarePrediction:
    """The full response envelope every caller should consume."""

    decision: str  # "favorable_pattern" | "concerning_pattern" | "uncertain" | "insufficient_evidence"
    probability: float | None
    raw_probability: float | None
    calibrated: bool
    confidence: str  # "low" | "moderate" | "high"
    evidence: EvidenceAssessment
    model_version: str
    question: str
    claim_boundary: str = field(default=(
        "Synthetic monitoring-only signal.  Not a clinical diagnosis or "
        "treatment recommendation.  Routed to clinician review when partial "
        "or insufficient evidence is detected."
    ))

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "probability": self.probability,
            "raw_probability": self.raw_probability,
            "calibrated": self.calibrated,
            "confidence": self.confidence,
            "evidence": self.evidence.to_dict(),
            "model_version": self.model_version,
            "question": self.question,
            "claim_boundary": self.claim_boundary,
        }


# ─── Model artifact loading ──────────────────────────────────────────────────


@lru_cache(maxsize=4)
def _load_model(model_path: str, calibrator_path: str | None) -> tuple[Any, Any | None]:
    """Load and cache the (model, calibrator) pair.  Missing calibrator is
    fine — we still return raw probabilities, just with `calibrated=False`."""
    model = joblib.load(model_path)
    calibrator = None
    if calibrator_path and Path(calibrator_path).exists():
        try:
            calibrator = joblib.load(calibrator_path)
        except Exception:  # noqa: BLE001 — calibrator load failure is non-fatal
            calibrator = None
    return model, calibrator


def _model_version_tag(model_path: str) -> str:
    """A stable, human-readable model identifier derived from the artifact
    filename — used by traceability so downstream consumers know exactly
    which model produced a given response."""
    return Path(model_path).stem


# ─── Probability transforms ──────────────────────────────────────────────────


def _apply_confidence_modifier(probability: float, modifier: float) -> float:
    """Move a probability toward the 0.5 prior in proportion to (1 - modifier).

    A modifier of 1.0 keeps the value unchanged (full confidence).  A modifier
    of 0.0 collapses every prediction to 0.5 (no confidence whatsoever).  In
    between, partial evidence pulls the value back toward the prior — the
    classifier's signal still influences the *direction*, just less strongly.
    """
    if probability is None or not math.isfinite(probability):
        return probability
    modifier = max(0.0, min(1.0, modifier))
    return 0.5 + (probability - 0.5) * modifier


def _classify(probability: float) -> str:
    if probability >= UPPER_DECISION_THRESHOLD:
        return "favorable_pattern"
    if probability <= LOWER_DECISION_THRESHOLD:
        return "concerning_pattern"
    return "uncertain"


def _confidence_bucket(probability: float, sufficiency: str) -> str:
    """Confidence is a combination of how decisive the probability is AND
    how sufficient the evidence was.  Partial evidence caps confidence at
    'moderate' even when the probability is extreme."""
    decisive = abs(probability - 0.5) > 0.30  # |p - 0.5| > 0.30 → very strong
    moderate = abs(probability - 0.5) > 0.15
    if sufficiency == "partial":
        return "moderate" if decisive else "low"
    if decisive:
        return "high"
    if moderate:
        return "moderate"
    return "low"


# ─── Public API ──────────────────────────────────────────────────────────────


def predict_with_abstention(
    row: Mapping[str, Any],
    *,
    question: str = "response_classification",
    model_path: str = DEFAULT_MODEL_PATH,
    calibrator_path: str | None = DEFAULT_CALIBRATOR_PATH,
) -> EvidenceAwarePrediction:
    """Single-row inference.  Always emits an `EvidenceAwarePrediction` —
    if abstention kicks in, `probability` is ``None`` and `decision` is
    ``"insufficient_evidence"``."""
    evidence = assess_evidence(row, question=question)
    model_version = _model_version_tag(model_path)

    if evidence.abstain:
        return EvidenceAwarePrediction(
            decision="insufficient_evidence",
            probability=None,
            raw_probability=None,
            calibrated=False,
            confidence="low",
            evidence=evidence,
            model_version=model_version,
            question=question,
        )

    # Build the single-row feature frame in the same column order the model
    # was trained on.  Missing categorical fields are filled with empty string
    # so the model's ColumnTransformer can route them through the OHE without
    # raising; missing numerics stay NaN and the imputer in the pipeline will
    # handle them.
    feature_frame = _build_feature_frame(row)
    model, calibrator = _load_model(model_path, calibrator_path)
    raw_prob = float(model.predict_proba(feature_frame)[0, 1])
    calibrated = False
    prob = raw_prob
    if calibrator is not None:
        try:
            prob = float(calibrator.predict([raw_prob])[0])
            calibrated = True
        except Exception:  # noqa: BLE001 — fall back to raw on calibrator error
            prob = raw_prob

    # Apply the evidence-confidence modifier — partial evidence shrinks the
    # spread of the prediction toward the prior so the system never quotes
    # extreme confidence on weak inputs.
    prob = _apply_confidence_modifier(prob, evidence.confidence_modifier)

    return EvidenceAwarePrediction(
        decision=_classify(prob),
        probability=prob,
        raw_probability=raw_prob,
        calibrated=calibrated,
        confidence=_confidence_bucket(prob, evidence.sufficiency),
        evidence=evidence,
        model_version=model_version,
        question=question,
    )


def predict_batch_with_abstention(
    frame: pd.DataFrame,
    *,
    question: str = "response_classification",
    model_path: str = DEFAULT_MODEL_PATH,
    calibrator_path: str | None = DEFAULT_CALIBRATOR_PATH,
) -> list[EvidenceAwarePrediction]:
    """Vectorised wrapper.  Used by the evaluation pipeline to score the
    entire test split in one pass."""
    return [
        predict_with_abstention(
            row.to_dict(),
            question=question,
            model_path=model_path,
            calibrator_path=calibrator_path,
        )
        for _, row in frame.iterrows()
    ]


def _build_feature_frame(row: Mapping[str, Any]) -> pd.DataFrame:
    """Coerce a single row dict to a one-row DataFrame in the model's expected
    column order.  Missing fields become NaN (numeric) or "" (categorical)."""
    values: dict[str, Any] = {}
    for column in NUMERIC_FEATURES:
        raw = row.get(column)
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            values[column] = float("nan")
        else:
            try:
                values[column] = float(raw)
            except (TypeError, ValueError):
                values[column] = float("nan")
    for column in CATEGORICAL_FEATURES:
        raw = row.get(column)
        values[column] = "" if raw is None else str(raw)
    return pd.DataFrame([values], columns=NUMERIC_FEATURES + CATEGORICAL_FEATURES)


__all__ = [
    "EvidenceAwarePrediction",
    "predict_with_abstention",
    "predict_batch_with_abstention",
]
