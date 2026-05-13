"""
Pydantic response schemas for model lifecycle, registry, and pipeline endpoints.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


# ─── Model registry ──────────────────────────────────────────────────────────

class ModelRegistryEntry(BaseModel):
    model_name: str
    version: str
    promotion_status: str
    promoted_at: str | None = None
    promotion_reason: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    training_data_path: str | None = None
    registered_at: str | None = None
    warning: str = "Synthetic model registry. Not for clinical deployment."


class ModelListResponse(BaseModel):
    models: list[ModelRegistryEntry] = Field(default_factory=list)


class ModelActionResponse(BaseModel):
    message: str
    model: ModelRegistryEntry
    safety_note: str = (
        "Model lifecycle actions change registry status only. "
        "They do not imply clinical validation."
    )


# ─── Training run ─────────────────────────────────────────────────────────────

class TrainingRunResponse(BaseModel):
    message: str
    mlops_run_id: str | None = None
    result: dict[str, Any] = Field(default_factory=dict)
    warning: str = "Synthetic training only. Not clinical evidence."


# ─── Prediction audit ────────────────────────────────────────────────────────

class PredictionAuditEntry(BaseModel):
    id: int
    patient_id: str
    model_name: str
    model_version: str
    predicted_probability: float | None = None
    predicted_class: int | None = None
    actual_label: int | None = None
    threshold: float | None = None
    created_at: str


class PredictionAuditResponse(BaseModel):
    prediction_audits: list[PredictionAuditEntry] = Field(default_factory=list)


# ─── MLOps runs ──────────────────────────────────────────────────────────────

class MlopsRunEntry(BaseModel):
    run_id: str
    experiment_name: str
    run_name: str | None = None
    status: str
    started_at: str | None = None
    finished_at: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    tags: dict[str, Any] = Field(default_factory=dict)


class MlopsRunsResponse(BaseModel):
    runs: list[MlopsRunEntry] = Field(default_factory=list)
    purpose: str = "Local experiment tracking for params, metrics, artifacts, hashes, and run status."


# ─── Inference service ───────────────────────────────────────────────────────

class InferenceServiceDescription(BaseModel):
    status: str
    available_models: list[str] = Field(default_factory=list)
    serving_mode: str = ""
    note: str = ""


# ─── XAI ─────────────────────────────────────────────────────────────────────

class FeatureContribution(BaseModel):
    feature: str
    contribution: float
    shap_value: float


class PatientXaiExplanation(BaseModel):
    patient_id: str
    positive_contributions: list[FeatureContribution] = Field(default_factory=list)
    negative_contributions: list[FeatureContribution] = Field(default_factory=list)
    claim_boundary: str = "SHAP values explain synthetic model behavior, not clinical causality."
