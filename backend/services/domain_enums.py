from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ValidationError


class RagMode(StrEnum):
    education = "education"
    urgent_safety = "urgent_safety"
    record_explanation = "record_explanation"
    clinician_context = "clinician_context"
    portal_help = "portal_help"


class SourceTier(StrEnum):
    T1 = "T1"
    T2 = "T2"
    T3 = "T3"
    T4 = "T4"
    T5 = "T5"


class AllowedUse(StrEnum):
    education = "education"
    patient_safety = "patient_safety"
    monitoring_context = "monitoring_context"
    portal_help = "portal_help"
    clinician_only = "clinician_only"


class ModelHead(StrEnum):
    response_classification = "response_classification"
    response_regression = "response_regression"
    toxicity_review = "toxicity_review"
    abstention = "abstention"


class EvidenceSufficiency(StrEnum):
    sufficient = "sufficient"
    partial = "partial"
    insufficient = "insufficient"


class PromotionDecision(StrEnum):
    monitor_only = "monitor_only"
    candidate_only = "candidate_only"
    keep_current_default = "keep_current_default"
    blocked = "blocked"


class BoundaryEvent(BaseModel):
    rag_mode: RagMode | None = None
    source_tier: SourceTier | None = None
    allowed_use: AllowedUse | None = None
    model_head: ModelHead | None = None
    evidence_sufficiency: EvidenceSufficiency | None = None
    promotion_decision: PromotionDecision | None = None


def validate_boundary_values(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        parsed = BoundaryEvent(**payload)
        return {"valid": True, "normalized": parsed.model_dump(mode="json"), "errors": []}
    except ValidationError as exc:
        return {"valid": False, "normalized": None, "errors": exc.errors()}


def domain_enum_manifest() -> dict[str, Any]:
    return {
        "schema_version": "domain_enums_v1",
        "enums": {
            "rag_mode": [item.value for item in RagMode],
            "source_tier": [item.value for item in SourceTier],
            "allowed_use": [item.value for item in AllowedUse],
            "model_head": [item.value for item in ModelHead],
            "evidence_sufficiency": [item.value for item in EvidenceSufficiency],
            "promotion_decision": [item.value for item in PromotionDecision],
        },
        "claim_boundary": "Domain enums enforce wire/schema discipline only; they do not validate medical correctness.",
    }
