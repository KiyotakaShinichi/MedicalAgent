"""Hybrid evidence-aware prediction: classification + regression + toxicity.

The classification head answers "is the response pattern favorable?".  The
regression head answers "how strong is the response signal, 0–1?".  The
toxicity head answers "is this cycle on the high-toxicity risk path?".  All
three go through the same evidence-sufficiency layer, so the patient view
never sees a forced number when the evidence doesn't support it.

The hybrid envelope is what makes the system honest about its limits:
  - response classification can be confident while the regression score
    abstains, if the evidence is partial in a way that supports one
    question but not the other,
  - toxicity can abstain independently — it has its own sufficiency rules
    that don't require imaging,
  - every signal carries its own ``model_version`` + ``confidence_modifier``
    so the prediction trace records exactly which model produced which
    decision under which evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import joblib

from backend.services.evidence_sufficiency import (
    EvidenceAssessment,
    assess_evidence,
)
from backend.services.predict_with_abstention import (
    DEFAULT_CALIBRATOR_PATH,
    DEFAULT_MODEL_PATH,
    EvidenceAwarePrediction,
    LOWER_DECISION_THRESHOLD,
    UPPER_DECISION_THRESHOLD,
    _apply_confidence_modifier,
    _build_feature_frame,
    _confidence_bucket,
    _model_version_tag,
    predict_with_abstention,
)
from backend.services.response_conformal_calibration import conformal_adjustment


DEFAULT_REGRESSION_MODEL_PATH = (
    "Data/complete_synthetic_training/random_forest_regressor_response_score_percent.joblib"
)
DEFAULT_TOXICITY_MODEL_PATH = (
    "Data/complete_synthetic_training/gradient_boosting_toxicity_risk_binary.joblib"
)

# Quantile-regression artifacts trained by ``quantile_regression_training``.
# When all three are present, the regression head uses them for a genuine
# 80% prediction interval drawn from the model's own residual distribution
# instead of the heuristic band.  If any are missing, we silently fall back
# to the heuristic so deployments without the extra training step still
# return a sensible response score.
DEFAULT_QUANTILE_P10_PATH = (
    "Data/complete_synthetic_training/quantile_gbm_p10_response_score_percent.joblib"
)
DEFAULT_QUANTILE_P50_PATH = (
    "Data/complete_synthetic_training/quantile_gbm_p50_response_score_percent.joblib"
)
DEFAULT_QUANTILE_P90_PATH = (
    "Data/complete_synthetic_training/quantile_gbm_p90_response_score_percent.joblib"
)
DEFAULT_ROBUST_QUANTILE_P10_PATH = (
    "Data/complete_synthetic_training/modality_dropout_quantile_gbm_p10_response_score_percent.joblib"
)
DEFAULT_ROBUST_QUANTILE_P50_PATH = (
    "Data/complete_synthetic_training/modality_dropout_quantile_gbm_p50_response_score_percent.joblib"
)
DEFAULT_ROBUST_QUANTILE_P90_PATH = (
    "Data/complete_synthetic_training/modality_dropout_quantile_gbm_p90_response_score_percent.joblib"
)

# Response-score thresholds on the normalised [0,1] scale.  Anything in the
# middle band is labelled "moderate_response_signal" rather than committing
# to a strong/weak direction.
STRONG_RESPONSE_THRESHOLD = 0.66
WEAK_RESPONSE_THRESHOLD = 0.34

# Toxicity thresholds on the calibrated probability.  Slightly tighter
# bands than the classification head because toxicity signal swings faster.
HIGH_TOXICITY_THRESHOLD = 0.66
LOW_TOXICITY_THRESHOLD = 0.40


# ─── Regression envelope ─────────────────────────────────────────────────────


@dataclass
class EvidenceAwareRegression:
    """The regression head's equivalent of `EvidenceAwarePrediction`.  When
    abstained, `response_score` is ``None`` (not zero) so downstream code
    cannot interpret an empty answer as a weak response."""

    decision: str  # "strong_response_signal" | "moderate_response_signal" | "weak_response_signal" | "insufficient_evidence"
    response_score: float | None  # 0-1, normalised from synthetic 0-100 scale
    raw_response_score: float | None  # 0-1 before confidence shrinkage
    uncertainty_band: tuple[float, float] | None  # rough ± band scaled to confidence_modifier
    confidence: str  # "low" | "moderate" | "high"
    evidence: EvidenceAssessment
    model_version: str
    question: str
    uncertainty_method: str = "unspecified"
    claim_boundary: str = field(default=(
        "Synthetic monitoring-only response-strength estimate. Not a "
        "clinical prediction of treatment effect."
    ))

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "response_score": self.response_score,
            "raw_response_score": self.raw_response_score,
            "uncertainty_band": (
                list(self.uncertainty_band) if self.uncertainty_band is not None else None
            ),
            "confidence": self.confidence,
            "evidence": self.evidence.to_dict(),
            "model_version": self.model_version,
            "question": self.question,
            "uncertainty_method": self.uncertainty_method,
            "claim_boundary": self.claim_boundary,
        }


# ─── Hybrid bundle ───────────────────────────────────────────────────────────


@dataclass
class HybridPrediction:
    """All three heads together, ready to embed in the patient report."""

    classification: EvidenceAwarePrediction
    response_score: EvidenceAwareRegression
    toxicity: EvidenceAwarePrediction
    claim_boundary: str = field(default=(
        "Hybrid synthetic monitoring signals. None of these outputs are a "
        "clinical diagnosis, prognosis, or treatment recommendation."
    ))

    def to_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification.to_dict(),
            "response_score": self.response_score.to_dict(),
            "toxicity": self.toxicity.to_dict(),
            "claim_boundary": self.claim_boundary,
        }


# ─── Model loading (cached) ──────────────────────────────────────────────────


@lru_cache(maxsize=4)
def _load_pipeline(model_path: str) -> Any | None:
    """Return the model pipeline, or ``None`` if the artifact is missing.
    Callers must handle ``None`` by abstaining rather than crashing."""
    if not Path(model_path).exists():
        return None
    return joblib.load(model_path)


def _quantile_candidate_paths(
    override_p10: str | None,
    override_p50: str | None,
    override_p90: str | None,
) -> list[tuple[str, str, str, str]]:
    """Return quantile model trios in preference order.

    Explicit overrides are used exactly as provided for tests/backward
    compatibility. In normal runtime, modality-dropout quantile heads are
    preferred because they were trained to handle missing-modality patterns;
    plain quantile heads remain the fallback.
    """
    if override_p10 or override_p50 or override_p90:
        if override_p10 and override_p50 and override_p90:
            return [(
                override_p10,
                override_p50,
                override_p90,
                "custom_quantile_gbm_p10_p50_p90_response_score_percent",
            )]
        return []
    return [
        (
            DEFAULT_ROBUST_QUANTILE_P10_PATH,
            DEFAULT_ROBUST_QUANTILE_P50_PATH,
            DEFAULT_ROBUST_QUANTILE_P90_PATH,
            "modality_dropout_quantile_gbm_p10_p50_p90_response_score_percent",
        ),
        (
            DEFAULT_QUANTILE_P10_PATH,
            DEFAULT_QUANTILE_P50_PATH,
            DEFAULT_QUANTILE_P90_PATH,
            "quantile_gbm_p10_p50_p90_response_score_percent",
        ),
    ]


# ─── Per-head predict functions ──────────────────────────────────────────────


def predict_response_score_with_abstention(
    row: Mapping[str, Any],
    *,
    model_path: str = DEFAULT_REGRESSION_MODEL_PATH,
    quantile_p10_path: str | None = None,
    quantile_p50_path: str | None = None,
    quantile_p90_path: str | None = None,
) -> EvidenceAwareRegression:
    """Regression head.  Uses the same sufficiency rules as the classification
    head — both require imaging OR longitudinal CBC — but emits a continuous
    score plus a prediction interval.

    Inference strategy
    ~~~~~~~~~~~~~~~~~~
    If the quantile-regression artifacts (p10/p50/p90) are all present on
    disk, the p50 head produces the central estimate and (p10, p90) the
    band — drawn from the model's own residual distribution.  Independent
    quantile heads can cross, so the trio is **sorted per row** at read
    time to enforce p10 ≤ p50 ≤ p90 regardless of training-time crossing.

    When the quantile artifacts are missing, we fall back to the legacy
    point-estimate regressor + heuristic band (the previous behavior),
    so deployments without the extra training step still get a sensible
    response score.
    """
    evidence = assess_evidence(row, question="response_regression")

    if evidence.abstain:
        return EvidenceAwareRegression(
            decision="insufficient_evidence",
            response_score=None,
            raw_response_score=None,
            uncertainty_band=None,
            confidence="low",
            evidence=evidence,
            model_version=_model_version_tag(model_path),
            question="response_score_regression",
            uncertainty_method="abstained_insufficient_evidence",
        )

    feature_frame = _build_feature_frame(row)
    modifier = evidence.confidence_modifier

    # Preferred path: three quantile heads, sorted per row.
    for quantile_p10_path, quantile_p50_path, quantile_p90_path, version_tag in _quantile_candidate_paths(
        quantile_p10_path,
        quantile_p50_path,
        quantile_p90_path,
    ):
        p10_model = _load_pipeline(quantile_p10_path)
        p50_model = _load_pipeline(quantile_p50_path)
        p90_model = _load_pipeline(quantile_p90_path)
        if p10_model is not None and p50_model is not None and p90_model is not None:
            raw_percentages = sorted([
                float(p10_model.predict(feature_frame)[0]),
                float(p50_model.predict(feature_frame)[0]),
                float(p90_model.predict(feature_frame)[0]),
            ])
            # Normalise + clip the sorted trio to 0-1.
            raw_lo, raw_mid, raw_hi = (
                max(0.0, min(1.0, v / 100.0)) for v in raw_percentages
            )
            # Shrink all three toward the 0.5 prior in proportion to (1 - modifier).
            shrunk_lo = 0.5 + (raw_lo - 0.5) * modifier
            shrunk_mid = 0.5 + (raw_mid - 0.5) * modifier
            shrunk_hi = 0.5 + (raw_hi - 0.5) * modifier
            band = _evidence_adjusted_interval(
                center=shrunk_mid,
                lower=shrunk_lo,
                upper=shrunk_hi,
                evidence=evidence,
            )
            band = _apply_conformal_adjustment(band, version_tag)
            return EvidenceAwareRegression(
                decision=_response_decision(shrunk_mid),
                response_score=shrunk_mid,
                raw_response_score=raw_mid,
                uncertainty_band=band,
                confidence=_response_confidence_bucket(shrunk_mid, evidence.sufficiency, band),
                evidence=evidence,
                model_version=version_tag,
                question="response_score_regression",
                uncertainty_method=(
                    "quantile_gbm_p10_p90_evidence_adjusted"
                    + ("_conformal" if conformal_adjustment() > 0 else "")
                ),
            )

    # Fallback: legacy point regressor + heuristic band.
    model = _load_pipeline(model_path)
    if model is None:
        return EvidenceAwareRegression(
            decision="insufficient_evidence",
            response_score=None,
            raw_response_score=None,
            uncertainty_band=None,
            confidence="low",
            evidence=evidence,
            model_version=_model_version_tag(model_path),
            question="response_score_regression",
            uncertainty_method="missing_regression_artifact",
        )

    raw_percent = float(model.predict(feature_frame)[0])
    raw_score = max(0.0, min(1.0, raw_percent / 100.0))
    shrunk = 0.5 + (raw_score - 0.5) * modifier
    half_width = max(0.05, 0.20 * (1.0 - modifier) + 0.05)
    band = _evidence_adjusted_interval(
        center=shrunk,
        lower=max(0.0, shrunk - half_width),
        upper=min(1.0, shrunk + half_width),
        evidence=evidence,
    )

    return EvidenceAwareRegression(
        decision=_response_decision(shrunk),
        response_score=shrunk,
        raw_response_score=raw_score,
        uncertainty_band=band,
        confidence=_response_confidence_bucket(shrunk, evidence.sufficiency, band),
        evidence=evidence,
        model_version=_model_version_tag(model_path),
        question="response_score_regression",
        uncertainty_method="heuristic_point_regressor_evidence_adjusted",
    )


def predict_toxicity_with_abstention(
    row: Mapping[str, Any],
    *,
    model_path: str = DEFAULT_TOXICITY_MODEL_PATH,
) -> EvidenceAwarePrediction:
    """Toxicity head.  Uses the toxicity-specific sufficiency rules — needs
    CBC of any flavour OR symptoms; doesn't require imaging."""
    evidence = assess_evidence(row, question="toxicity_classification")
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
            question="toxicity_classification",
        )

    model = _load_pipeline(model_path)
    if model is None:
        return EvidenceAwarePrediction(
            decision="insufficient_evidence",
            probability=None,
            raw_probability=None,
            calibrated=False,
            confidence="low",
            evidence=evidence,
            model_version=model_version,
            question="toxicity_classification",
        )

    feature_frame = _build_feature_frame(row)
    raw_prob = float(model.predict_proba(feature_frame)[0, 1])
    prob = _apply_confidence_modifier(raw_prob, evidence.confidence_modifier)
    return EvidenceAwarePrediction(
        decision=_toxicity_decision(prob),
        probability=prob,
        raw_probability=raw_prob,
        calibrated=False,  # toxicity head not calibrated yet
        confidence=_confidence_bucket(prob, evidence.sufficiency),
        evidence=evidence,
        model_version=model_version,
        question="toxicity_classification",
    )


# ─── Hybrid bundle ───────────────────────────────────────────────────────────


def predict_hybrid(
    row: Mapping[str, Any],
    *,
    classification_model_path: str = DEFAULT_MODEL_PATH,
    calibrator_path: str | None = DEFAULT_CALIBRATOR_PATH,
    regression_model_path: str = DEFAULT_REGRESSION_MODEL_PATH,
    toxicity_model_path: str = DEFAULT_TOXICITY_MODEL_PATH,
) -> HybridPrediction:
    """One call → all three signals, each with its own evidence envelope."""
    classification = predict_with_abstention(
        row,
        question="response_classification",
        model_path=classification_model_path,
        calibrator_path=calibrator_path,
    )
    response_score = predict_response_score_with_abstention(
        row,
        model_path=regression_model_path,
    )
    toxicity = predict_toxicity_with_abstention(
        row,
        model_path=toxicity_model_path,
    )
    return HybridPrediction(
        classification=classification,
        response_score=response_score,
        toxicity=toxicity,
    )


# ─── Decision rules ──────────────────────────────────────────────────────────


def _response_decision(score: float) -> str:
    if score >= STRONG_RESPONSE_THRESHOLD:
        return "strong_response_signal"
    if score <= WEAK_RESPONSE_THRESHOLD:
        return "weak_response_signal"
    return "moderate_response_signal"


def _toxicity_decision(prob: float) -> str:
    if prob >= HIGH_TOXICITY_THRESHOLD:
        return "high_toxicity_signal"
    if prob <= LOW_TOXICITY_THRESHOLD:
        return "low_toxicity_signal"
    return "moderate_toxicity_signal"


def _evidence_adjusted_interval(
    *,
    center: float,
    lower: float,
    upper: float,
    evidence: EvidenceAssessment,
) -> tuple[float, float]:
    """Keep the model direction but widen intervals when evidence is weaker.

    The raw quantile heads can be sharp on full synthetic rows. If evidence is
    partial, shrinking all quantiles toward the 0.5 prior can collapse the
    interval and imply false precision. This helper enforces a minimum
    half-width based on missing modalities and the confidence modifier.
    """
    lo = min(lower, center, upper)
    hi = max(lower, center, upper)
    observed_half_width = max(center - lo, hi - center, 0.0)
    total_modalities = max(1, len(evidence.modalities_present) + len(evidence.modalities_missing))
    missing_fraction = len(evidence.modalities_missing) / total_modalities
    evidence_penalty = 1.0 - evidence.confidence_modifier
    minimum_half_width = 0.05 + 0.18 * evidence_penalty + 0.10 * missing_fraction
    if evidence.sufficiency == "partial":
        minimum_half_width = max(minimum_half_width, 0.16)
    half_width = max(observed_half_width, minimum_half_width)
    return (
        max(0.0, center - half_width),
        min(1.0, center + half_width),
    )


def _apply_conformal_adjustment(
    band: tuple[float, float],
    model_version: str,
) -> tuple[float, float]:
    """Apply split-conformal widening to robust quantile intervals only."""
    if not model_version.startswith("modality_dropout_quantile"):
        return band
    qhat = conformal_adjustment()
    if qhat <= 0:
        return band
    return (
        max(0.0, band[0] - qhat),
        min(1.0, band[1] + qhat),
    )


def _response_confidence_bucket(
    score: float,
    sufficiency: str,
    uncertainty_band: tuple[float, float] | None = None,
) -> str:
    """Same shape as `_confidence_bucket` but for the regression score —
    confidence is decisive when the score is far from 0.5 AND evidence is
    sufficient."""
    band_width = (
        uncertainty_band[1] - uncertainty_band[0]
        if uncertainty_band is not None
        else 0.0
    )
    if band_width >= 0.45:
        return "low"
    if band_width >= 0.28 and sufficiency == "partial":
        return "low"
    decisive = abs(score - 0.5) > 0.30
    moderate = abs(score - 0.5) > 0.15
    if sufficiency == "partial":
        return "moderate" if decisive else "low"
    if band_width >= 0.28:
        return "moderate" if decisive else "low"
    if decisive:
        return "high"
    if moderate:
        return "moderate"
    return "low"


# Mirror the classification thresholds for callers that need them at
# read time (e.g. for the trace's threshold_config_json).
DEFAULT_HYBRID_THRESHOLD_CONFIG: dict[str, float] = {
    "classification_lower": LOWER_DECISION_THRESHOLD,
    "classification_upper": UPPER_DECISION_THRESHOLD,
    "response_strong":      STRONG_RESPONSE_THRESHOLD,
    "response_weak":        WEAK_RESPONSE_THRESHOLD,
    "toxicity_high":        HIGH_TOXICITY_THRESHOLD,
    "toxicity_low":         LOW_TOXICITY_THRESHOLD,
}


__all__ = [
    "EvidenceAwareRegression",
    "HybridPrediction",
    "DEFAULT_HYBRID_THRESHOLD_CONFIG",
    "DEFAULT_REGRESSION_MODEL_PATH",
    "DEFAULT_TOXICITY_MODEL_PATH",
    "predict_hybrid",
    "predict_response_score_with_abstention",
    "predict_toxicity_with_abstention",
]
