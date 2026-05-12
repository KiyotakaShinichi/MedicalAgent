"""
Pydantic response models for key API endpoints.

Using explicit response models:
- Documents the API contract in OpenAPI/Swagger
- Ensures PHI-like fields are not accidentally included
- Makes type errors visible at serialisation time rather than runtime
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


# ─── Auth ─────────────────────────────────────────────────────────────────────

class LoginResponse(BaseModel):
    role: str = Field(..., examples=["patient", "clinician", "admin"])
    access_token: str
    patient_id: str | None = None


class DemoPatientItem(BaseModel):
    id: str
    label: str
    hint: str


class DemoPatientsResponse(BaseModel):
    patients: list[DemoPatientItem]


class WhoAmIResponse(BaseModel):
    role: str
    patient_id: str | None = None


# ─── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    database: str


# ─── Chat ─────────────────────────────────────────────────────────────────────

class SavedAction(BaseModel):
    type: str
    data: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    reply: str
    saved_actions: list[SavedAction] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    intent: str | None = None
    safety_level: str | None = None
    cache_status: str | None = None
    latency_ms: float | None = None


# ─── Review queue ─────────────────────────────────────────────────────────────

class ReviewQueueItem(BaseModel):
    patient_id: str
    patient_name: str
    overall_status: str
    priority_score: float
    urgent_flags: list[str] = Field(default_factory=list)
    latest_decision: str | None = None


class ReviewQueueResponse(BaseModel):
    queue: list[ReviewQueueItem]


# ─── Summary review ───────────────────────────────────────────────────────────

class SummaryReviewRecord(BaseModel):
    id: int
    patient_id: str
    decision: str
    clinician_notes: str
    edited_patient_summary: str | None = None
    explanation_quality_score: float | None = None
    model_usefulness_score: float | None = None
    created_at: str


class SummaryReviewResponse(BaseModel):
    message: str
    review: SummaryReviewRecord


# ─── Admin analytics ─────────────────────────────────────────────────────────

class RagEvaluationSummary(BaseModel):
    evaluations: int | None = None
    grounding_score: float | None = None
    hallucination_score: float | None = None
    cache_hit_rate: float | None = None
    precision_at_3: float | None = None
    estimated_cost_usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    p95_latency_ms: float | None = None


class GuardrailSummary(BaseModel):
    input_blocks: int = 0
    output_blocks: int = 0
    attack_block_rate: float | None = None
    pass_rate: float | None = None


class MleReadinessSummary(BaseModel):
    status: str
    release_recommendation: str
    hard_gate_status: str
    hard_gate_failures: int
    poc_demo_readiness: str
    poc_demo_recommendation: str | None = None
    category_statuses: dict[str, str] = Field(default_factory=dict)


class AgentFeedbackSummary(BaseModel):
    count: int = 0
    average_rating: float | None = None
    thumbs_up_rate: float | None = None


class AdminAnalyticsResponse(BaseModel):
    rag_evaluation: RagEvaluationSummary
    guardrails: GuardrailSummary
    mle_readiness: MleReadinessSummary
    agent_feedback: AgentFeedbackSummary


# ─── Agent regression ─────────────────────────────────────────────────────────

class AgentRegressionResult(BaseModel):
    case_count: int
    status: str
    pass_rate: float
    attack_block_rate: float
    expected_source_hit_rate: float


class AgentRegressionResponse(BaseModel):
    message: str
    result: AgentRegressionResult


# ─── Structured error ─────────────────────────────────────────────────────────

class ErrorDetail(BaseModel):
    error: str
    code: str | None = None


class ValidationErrorResponse(BaseModel):
    error: str
    field_errors: list[str] = Field(default_factory=list)
