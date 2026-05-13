"""Shared Pydantic models used across multiple routers."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class MessageResponse(BaseModel):
    message: str


class ErrorDetail(BaseModel):
    error: str
    code: str | None = None
    field_errors: list[str] = Field(default_factory=list)


class PaginatedMeta(BaseModel):
    total: int | None = None
    limit: int
    offset: int = 0


class SavedAction(BaseModel):
    type: str
    data: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    reply: str
    saved_actions: list[SavedAction] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    intent: str | None = None
    safety_level: str | None = None
    cache_status: str | None = None
    latency_ms: float | None = None


class ValidationWarning(BaseModel):
    field: str
    message: str


class AuditNote(BaseModel):
    safety_note: str
