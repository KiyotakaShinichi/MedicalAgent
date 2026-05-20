"""FHIR-aligned internal clinical objects.

These dataclasses borrow shape ideas from FHIR resources but are not certified
FHIR resources and do not imply hospital interoperability.  They give the app
a stable internal vocabulary for future mapping, validation, and reviewer
readiness while allowing unknown/unmapped codes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Coding:
    system: str | None = None
    code: str | None = None
    display: str | None = None


@dataclass
class ReferenceRange:
    low: float | None = None
    high: float | None = None
    unit: str | None = None
    text: str | None = None


@dataclass
class CanonicalObservation:
    resource_type: str = "ObservationLike"
    id: str | None = None
    status: str = "available"
    category: str = "laboratory"
    coding: Coding = field(default_factory=Coding)
    value: float | str | None = None
    unit: str | None = None
    reference_range: ReferenceRange | None = None
    effective_datetime: str | None = None
    source: str | None = None
    unmapped_fields: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CanonicalMedicationStatement:
    resource_type: str = "MedicationStatementLike"
    id: str | None = None
    status: str = "recorded"
    medication_text: str | None = None
    coding: Coding = field(default_factory=Coding)
    dose_text: str | None = None
    effective_datetime: str | None = None
    source: str | None = None
    unmapped_fields: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CanonicalDiagnosticReport:
    resource_type: str = "DiagnosticReportLike"
    id: str | None = None
    status: str = "available"
    report_type: str = "imaging"
    modality: str | None = None
    coding: Coding = field(default_factory=Coding)
    effective_datetime: str | None = None
    findings_text: str | None = None
    impression_text: str | None = None
    source: str | None = None
    unmapped_fields: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CanonicalFamilyMemberHistory:
    resource_type: str = "FamilyMemberHistoryLike"
    id: str | None = None
    status: str = "recorded"
    relationship: str | None = None
    condition_text: str | None = None
    age_at_diagnosis: int | None = None
    side: str | None = None
    coding: Coding = field(default_factory=Coding)
    source: str | None = None
    unmapped_fields: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CanonicalCondition:
    resource_type: str = "ConditionLike"
    id: str | None = None
    clinical_status: str = "recorded"
    condition_text: str | None = None
    coding: Coding = field(default_factory=Coding)
    recorded_datetime: str | None = None
    source: str | None = None
    unmapped_fields: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "CanonicalCondition",
    "CanonicalDiagnosticReport",
    "CanonicalFamilyMemberHistory",
    "CanonicalMedicationStatement",
    "CanonicalObservation",
    "Coding",
    "ReferenceRange",
]
