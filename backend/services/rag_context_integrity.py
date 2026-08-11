"""Prompt/data separation checks for retrieved RAG chunks.

Retrieved documents are untrusted data even when their source is otherwise
allowed.  This module removes chunks that attempt to direct the assistant,
alter policy, invoke tools, or conceal provenance.  It deliberately does not
judge the medical truth of ordinary research prose.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


_INSTRUCTION_TARGET = re.compile(
    r"\b(assistant|model|system|agent|chatbot|you)\b.{0,45}\b"
    r"(must|should|will|need\s+to|are\s+required\s+to)\b|"
    r"\b(system\s+prompt|developer\s+message|hidden\s+instruction|instruction\s+hierarchy)\b",
    re.IGNORECASE | re.DOTALL,
)
_CONTROL_OVERRIDE = re.compile(
    r"\b(ignore|disregard|override|bypass|disable|forget|supersede|set\s+aside|"
    r"skip|evade)\b.{0,60}\b(rule|policy|instruction|prompt|validator|guardrail|"
    r"safety|refusal|restriction)\b",
    re.IGNORECASE | re.DOTALL,
)
_TOOL_DIRECTIVE = re.compile(
    r"\b(call|invoke|execute|run|trigger|write|delete|export)\b.{0,45}\b"
    r"(tool|function|database|record|webhook|api|command)\b",
    re.IGNORECASE | re.DOTALL,
)
_PROVENANCE_EVASION = re.compile(
    r"\b(do\s+not|don't|never)\b.{0,35}\b(cite|attribute|mention|show)\b.{0,35}\b"
    r"(source|origin|document|citation|provenance)\b",
    re.IGNORECASE | re.DOTALL,
)
_CONTROL_TITLE = re.compile(
    r"\b(system|developer|assistant|model)\s+(instruction|message|prompt|directive)\b",
    re.IGNORECASE,
)
_PRIVATE_DATA_DIRECTIVE = re.compile(
    r"\b(reveal|expose|export|share|send|copy|retrieve|open)\b.{0,55}\b"
    r"(another|other|someone\s+else(?:'s)?)\b.{0,35}\b"
    r"(patient|record|chart|identifier|account|lab|result)\b",
    re.IGNORECASE | re.DOTALL,
)
_DATA_QUOTATION = re.compile(
    r"\b(study|paper|article|authors?|participants?|patients?|results?|methods?|"
    r"guideline|review|analysis|trial)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ContextIntegrityDecision:
    chunk_id: str
    safe_for_generation: bool
    reason: str
    matched_rules: tuple[str, ...] = ()


@dataclass
class ContextIntegrityResult:
    kept_chunks: list[dict[str, Any]] = field(default_factory=list)
    dropped_chunks: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[ContextIntegrityDecision] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kept_count": len(self.kept_chunks),
            "dropped_count": len(self.dropped_chunks),
            "kept_chunk_ids": [row.get("id") or row.get("chunk_id") for row in self.kept_chunks],
            "dropped_chunk_ids": [row.get("id") or row.get("chunk_id") for row in self.dropped_chunks],
            "decisions": [
                {
                    "chunk_id": item.chunk_id,
                    "safe_for_generation": item.safe_for_generation,
                    "reason": item.reason,
                    "matched_rules": list(item.matched_rules),
                }
                for item in self.decisions
            ],
        }


def inspect_retrieved_chunk(chunk: Mapping[str, Any]) -> ContextIntegrityDecision:
    chunk_id = str(chunk.get("id") or chunk.get("chunk_id") or "unknown")
    title = str(chunk.get("title") or "")
    section = str(chunk.get("section") or "")
    text = str(chunk.get("text") or "")
    payload = f"{title}\n{section}\n{text}"
    matches: list[str] = []
    provenance = str(
        chunk.get("provenance_integrity")
        or chunk.get("provenance_status")
        or ""
    ).strip().lower()
    if provenance in {"failed", "invalid", "spoofed", "tampered", "quarantined"}:
        matches.append("provenance_integrity_failure")
    if bool(chunk.get("retracted")):
        matches.append("retracted_source")
    if _CONTROL_OVERRIDE.search(payload):
        matches.append("control_override")
    if _CONTROL_TITLE.search(title):
        matches.append("control_directive_in_metadata")
    if _TOOL_DIRECTIVE.search(payload):
        matches.append("tool_directive")
    if _PRIVATE_DATA_DIRECTIVE.search(payload):
        matches.append("private_data_directive")
    if _PROVENANCE_EVASION.search(payload):
        matches.append("provenance_evasion")
    if _INSTRUCTION_TARGET.search(payload) and not _DATA_QUOTATION.search(payload):
        matches.append("assistant_directive")
    if matches:
        return ContextIntegrityDecision(
            chunk_id,
            False,
            "Retrieved content contains executable-looking instructions and is treated as untrusted data.",
            tuple(sorted(set(matches))),
        )
    return ContextIntegrityDecision(
        chunk_id,
        True,
        "No prompt/data boundary violation detected.",
        (),
    )


def sanitize_retrieved_context(chunks: Iterable[Mapping[str, Any]]) -> ContextIntegrityResult:
    result = ContextIntegrityResult()
    for chunk in chunks:
        row = dict(chunk)
        decision = inspect_retrieved_chunk(row)
        result.decisions.append(decision)
        if decision.safe_for_generation:
            row["context_integrity"] = "passed"
            result.kept_chunks.append(row)
        else:
            row["context_integrity"] = "blocked"
            row["context_integrity_reasons"] = list(decision.matched_rules)
            result.dropped_chunks.append(row)
    return result


__all__ = [
    "ContextIntegrityDecision",
    "ContextIntegrityResult",
    "inspect_retrieved_chunk",
    "sanitize_retrieved_context",
]
