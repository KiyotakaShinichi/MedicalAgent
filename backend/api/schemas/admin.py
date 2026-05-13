"""
Pydantic response schemas for Admin and MLE endpoints.

All metrics described here are from synthetic data pipelines unless
explicitly labelled 'locked_holdout' or 'external_validation'.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


# ─── MLE readiness ───────────────────────────────────────────────────────────

class MleReadinessCheck(BaseModel):
    name: str
    category: str
    status: str
    value: Any = None
    threshold: Any = None
    hard_gate: bool = False
    meaning: str = ""
    remediation: str = ""


class MleReadinessSummaryResponse(BaseModel):
    status: str
    release_recommendation: str
    hard_gate_status: str
    hard_gate_failures: int | list[str]
    poc_demo_readiness: str
    poc_demo_recommendation: str | None = None
    category_statuses: dict[str, str] = Field(default_factory=dict)
    checks: list[MleReadinessCheck] = Field(default_factory=list)
    advisory_gaps: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    claim_boundary: str = (
        "All MLE readiness metrics are computed on synthetic data unless "
        "explicitly labelled 'locked_holdout' or 'external_validation'."
    )


# ─── Safety / agent regression ───────────────────────────────────────────────

class RegressionCaseCheck(BaseModel):
    name: str
    passed: bool
    expected: Any = None
    observed: Any = None


class RegressionCase(BaseModel):
    id: str
    category: str
    query: str
    status: str
    checks: list[RegressionCaseCheck] = Field(default_factory=list)


class AgentRegressionSummary(BaseModel):
    case_count: int
    pass_rate: float
    attack_block_rate: float
    expected_source_hit_rate: float
    status: str


class AgentRegressionResponse(BaseModel):
    message: str
    result: AgentRegressionSummary


# ─── RAG evaluation summary ──────────────────────────────────────────────────

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


# ─── Guardrails summary ───────────────────────────────────────────────────────

class GuardrailSummary(BaseModel):
    input_blocks: int = 0
    output_blocks: int = 0
    attack_block_rate: float | None = None
    pass_rate: float | None = None


# ─── Model quality report ────────────────────────────────────────────────────

class CalibrationBin(BaseModel):
    range: str
    count: int
    mean_predicted: float | None = None
    observed_rate: float | None = None
    gap: float | None = None


class CalibrationReport(BaseModel):
    ece: float | None = None
    brier_score: float | None = None
    bins: list[CalibrationBin] = Field(default_factory=list)
    claim_boundary: str = "Calibration computed on synthetic holdout data only."


class FeatureDrift(BaseModel):
    feature: str
    std_mean_shift: float
    status: str


class DriftMonitoringSummary(BaseModel):
    status: str
    evaluated_features: int
    drifted_features: int
    drift_entries: list[FeatureDrift] = Field(default_factory=list)
    claim_boundary: str = "Drift computed on synthetic data. Not production deployment drift."


class SubgroupMetrics(BaseModel):
    subgroup: str
    n: int
    auroc: float | None = None
    brier: float | None = None
    sensitivity: float | None = None


class SubgroupPerformanceSummary(BaseModel):
    status: str
    subgroups: list[SubgroupMetrics] = Field(default_factory=list)
    disparity_warning: bool = False
    claim_boundary: str = "Subgroup metrics are from synthetic data. Subgroup distributions reflect simulator design."


class ModelQualityReport(BaseModel):
    auroc: float | None = None
    brier_score: float | None = None
    ece: float | None = None
    sensitivity: float | None = None
    specificity: float | None = None
    fnr: float | None = None
    mae: float | None = None
    rmse: float | None = None
    threshold: float | None = None
    best_classifier: str | None = None
    best_regressor: str | None = None
    test_patients: int | None = None
    calibration: CalibrationReport | None = None
    subgroups: SubgroupPerformanceSummary | None = None
    claim_boundary: str = "All metrics are on synthetic data and do not represent clinical performance."


# ─── Dataset lineage ─────────────────────────────────────────────────────────

class DatasetLineageEntry(BaseModel):
    dataset_name: str
    row_count: int | None = None
    schema_signature: str | None = None
    content_hash: str | None = None
    created_at: str | None = None
    purpose: str = ""


class DatasetLineageSummary(BaseModel):
    datasets: list[DatasetLineageEntry] = Field(default_factory=list)
    claim_boundary: str = "Dataset lineage covers synthetic data pipelines only."


# ─── Error taxonomy ──────────────────────────────────────────────────────────

class ErrorTaxonomyEntry(BaseModel):
    error_type: str
    count: int
    examples: list[str] = Field(default_factory=list)
    remediation_hint: str = ""


class ErrorTaxonomySummary(BaseModel):
    false_negatives: list[ErrorTaxonomyEntry] = Field(default_factory=list)
    false_positives: list[ErrorTaxonomyEntry] = Field(default_factory=list)
    total_fn: int = 0
    total_fp: int = 0
    fnr: float | None = None
    fpr: float | None = None
    claim_boundary: str = "Error taxonomy from synthetic holdout data only."


# ─── Training pipeline status ─────────────────────────────────────────────────

class TrainingPipelineStatus(BaseModel):
    status: str
    run_id: str | None = None
    experiment_name: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    tags: dict[str, Any] = Field(default_factory=dict)
    warning: str = "Synthetic training pipeline. Not clinical evidence."


# ─── Admin analytics ─────────────────────────────────────────────────────────

class AgentFeedbackSummary(BaseModel):
    count: int = 0
    average_rating: float | None = None
    thumbs_up_rate: float | None = None


class AdminAnalyticsResponse(BaseModel):
    rag_evaluation: RagEvaluationSummary
    guardrails: GuardrailSummary
    mle_readiness: MleReadinessSummaryResponse
    agent_feedback: AgentFeedbackSummary
