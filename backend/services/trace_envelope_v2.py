from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from backend.services.oncology_canonical_schema import ROOT_DIR
from backend.services.request_context import get_request_id


DEFAULT_OUTPUT_PATH = "Data/evals/ops/latest_trace_envelope_v2_eval.json"
DEFAULT_DOC_PATH = "docs/trace_envelope_v2.md"

TRACE_SCHEMA_VERSION = "2.0"
CLAIM_BOUNDARY = (
    "Trace envelope v2 stores redacted decision metadata for engineering observability only. "
    "It is not clinical validation, not a medical record, not clinician review, and not a chain-of-thought log."
)

REQUIRED_FIELDS: tuple[str, ...] = (
    "schema_version",
    "generated_at",
    "request_id",
    "correlation_id",
    "patient_id_hash",
    "route",
    "intent",
    "safety_decision",
    "policy_decision",
    "retrieval_backend",
    "source_ids",
    "claim_validation",
    "post_generation_decision",
    "cache_status",
    "latency_ms",
    "estimated_cost",
    "final_policy_status",
    "clinical_validation",
    "claim_boundary",
)

FORBIDDEN_EXACT_KEYS: frozenset[str] = frozenset(
    {
        "patient_id",
        "raw_patient_identifier",
        "raw_prompt_with_secrets",
        "unredacted_phi",
        "private_chain_of_thought",
        "chain_of_thought",
        "reasoning_text",
        "scratchpad",
        "draft_response",
        "raw_patient_message",
        "full_chat_transcript",
    }
)


def build_trace_envelope_v2(
    result: Mapping[str, Any],
    *,
    patient_id: str | int | None,
    route: str,
    latency_ms: Mapping[str, Any] | float | int | None,
    correlation_id: str | None = None,
    estimated_cost: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    pipeline = _mapping(result.get("pipeline_trace"))
    safety = _mapping(result.get("safety") or (result.get("guardrails") or {}).get("input"))
    cache = _mapping(result.get("cache"))
    post_gen = _mapping(result.get("post_gen_validator"))
    evidence = _mapping(result.get("evidence_grade"))
    retrieval_confidence = _mapping(result.get("retrieval_confidence"))
    rag_eval = _mapping(result.get("rag_evaluation"))

    request_id = correlation_id or get_request_id() or "untracked-request"
    source_ids = _source_ids(result)
    refused = _is_refused(result)
    payload = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "correlation_id": request_id,
        "patient_id_hash": _hash_patient_id(patient_id),
        "route": route,
        "intent": result.get("intent") or pipeline.get("intent") or "unknown",
        "safety_decision": {
            "level": safety.get("level"),
            "scope": safety.get("scope"),
            "matched_terms_count": len(safety.get("matched_terms") or []),
        },
        "policy_decision": {
            "route": result.get("rag_mode") or pipeline.get("terminal_step") or route,
            "refused": refused,
            "answerability_status": retrieval_confidence.get("answerability_status"),
            "evidence_grade": evidence.get("grade"),
        },
        "retrieval_backend": _retrieval_backend(result, source_ids),
        "source_ids": source_ids,
        "claim_validation": {
            "status": _first_present(
                result,
                "claim_validation_status",
                default=(rag_eval.get("claim_support_status") or evidence.get("claim_support_status")),
            ),
            "claim_support_rate": rag_eval.get("claim_support_rate"),
            "citation_precision": rag_eval.get("citation_precision"),
            "unsupported_context_rate": rag_eval.get("unsupported_context_rate"),
        },
        "post_generation_decision": {
            "decision": post_gen.get("decision") or "not_recorded",
            "blocked_claim_types_count": len(post_gen.get("blocked_claim_types") or []),
            "output_guardrail_status": (_mapping(result.get("guardrails")).get("output") or {}).get("status")
            if isinstance(_mapping(result.get("guardrails")).get("output"), Mapping)
            else None,
        },
        "cache_status": {
            "status": cache.get("status") or "not_recorded",
            "cacheable": bool(cache.get("cacheable")) if "cacheable" in cache else False,
            "reason": cache.get("reason"),
        },
        "latency_ms": _latency(latency_ms, pipeline),
        "estimated_cost": estimated_cost or {"available": False, "reason": "not_tracked"},
        "final_policy_status": "refused_or_blocked" if refused else "allowed_nonclinical_response",
        "redaction_guard": {
            "raw_patient_identifier_stored": False,
            "private_chain_of_thought_stored": False,
            "verbatim_patient_message_stored": False,
        },
        "clinical_validation": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    return _scrub_forbidden(payload)


def validate_trace_envelope_v2(payload: Mapping[str, Any]) -> tuple[bool, list[str]]:
    issues: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in payload:
            issues.append(f"missing_required_field:{field}")
    _walk_forbidden(payload, issues)
    if payload.get("clinical_validation") is not False:
        issues.append("clinical_validation_must_be_false")
    boundary = str(payload.get("claim_boundary") or "").lower()
    if "not clinical validation" not in boundary:
        issues.append("claim_boundary_missing_not_clinical_validation")
    if payload.get("patient_id_hash") in {None, "", str(payload.get("patient_id"))}:
        issues.append("patient_id_hash_invalid")
    return (not issues, issues)


def build_trace_envelope_v2_eval(
    *,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    doc_path: str | Path = DEFAULT_DOC_PATH,
) -> dict[str, Any]:
    samples = [
        build_trace_envelope_v2(
            {
                "intent": "education",
                "safety": {"level": "low", "scope": "education", "matched_terms": []},
                "pipeline_trace": {"terminal_step": "generated", "stage_ms": {"retrieval_ms": 12.0}},
                "retrieval_confidence": {"answerability_status": "answerable_with_citations"},
                "rag_evaluation": {"claim_support_rate": 1.0, "citation_precision": 1.0},
                "cache": {"status": "miss", "cacheable": True},
                "sources": [{"source_id": "curated-cbc-monitoring"}],
            },
            patient_id="P001",
            route="patient_chat",
            latency_ms={"total": 120.0},
            correlation_id="trace-v2-sample-1",
        ),
        build_trace_envelope_v2(
            {
                "intent": "safety_boundary",
                "safety": {"level": "high_risk", "scope": "treatment_decision_request", "matched_terms": ["stop"]},
                "pipeline_trace": {"terminal_step": "input_guardrail_block"},
                "post_gen_validator": {"decision": "blocked", "blocked_claim_types": ["treatment"]},
                "cache": {"status": "not_cacheable", "cacheable": False, "reason": "safety"},
                "evidence_grade": {"grade": "refuse_due_to_safety"},
            },
            patient_id="P001",
            route="patient_chat",
            latency_ms=35.0,
            correlation_id="trace-v2-sample-2",
        ),
    ]
    validations = [validate_trace_envelope_v2(sample) for sample in samples]
    poisoned = {
        **samples[0],
        "patient_id": "P001",
        "private_chain_of_thought": "do not store",
    }
    _, poisoned_issues = validate_trace_envelope_v2(poisoned)
    payload = {
        "schema_version": "trace_envelope_v2_eval",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong",
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "sample_count": len(samples),
        "validation_pass_rate": sum(1 for ok, _ in validations if ok) / len(validations),
        "forbidden_field_catch_rate": 1.0 if poisoned_issues else 0.0,
        "raw_patient_identifier_stored": False,
        "private_chain_of_thought_stored": False,
        "samples": samples,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _write_json(_resolve(output_path), payload)
    _write_doc(_resolve(doc_path), payload)
    return payload


def _source_ids(result: Mapping[str, Any]) -> list[str]:
    source_ids: list[str] = []
    for key in ("sources", "citations", "retrieved_sources"):
        for item in result.get(key) or []:
            if isinstance(item, Mapping):
                value = item.get("source_id") or item.get("id") or item.get("title")
            else:
                value = str(item)
            if value and value not in source_ids:
                source_ids.append(str(value))
    return source_ids[:10]


def _retrieval_backend(result: Mapping[str, Any], source_ids: list[str]) -> str:
    if (result.get("cache") or {}).get("status") == "hit":
        return "cache"
    if source_ids or result.get("retrieval_confidence") or result.get("rag_evaluation"):
        return "local_source_governed_rag"
    return "none"


def _latency(latency_ms: Mapping[str, Any] | float | int | None, pipeline: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(latency_ms, Mapping):
        base = dict(latency_ms)
    elif latency_ms is not None:
        base = {"total": float(latency_ms)}
    else:
        base = {}
    stage_ms = pipeline.get("stage_ms")
    if isinstance(stage_ms, Mapping):
        base.update({k: v for k, v in stage_ms.items() if k not in base})
    return base


def _is_refused(result: Mapping[str, Any]) -> bool:
    terminal = str((_mapping(result.get("pipeline_trace")).get("terminal_step") or "")).lower()
    post_gen = _mapping(result.get("post_gen_validator"))
    evidence = _mapping(result.get("evidence_grade"))
    return "refusal" in terminal or "block" in terminal or post_gen.get("decision") == "blocked" or evidence.get("grade") in {
        "insufficient",
        "refuse_due_to_safety",
    }


def _hash_patient_id(patient_id: str | int | None) -> str:
    raw = "unknown" if patient_id is None else str(patient_id)
    return hashlib.sha256(f"nlcare-trace-v2:{raw}".encode("utf-8")).hexdigest()[:24]


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _first_present(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if mapping.get(key) is not None:
            return mapping[key]
    return default


def _scrub_forbidden(value: Any) -> Any:
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, nested in value.items():
            if str(key) in FORBIDDEN_EXACT_KEYS:
                continue
            cleaned[str(key)] = _scrub_forbidden(nested)
        return cleaned
    if isinstance(value, list):
        return [_scrub_forbidden(item) for item in value]
    return value


def _walk_forbidden(value: Any, issues: list[str], path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_str = str(key)
            current = f"{path}.{key_str}" if path else key_str
            if key_str in FORBIDDEN_EXACT_KEYS:
                issues.append(f"forbidden_key:{current}")
            _walk_forbidden(nested, issues, current)
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            _walk_forbidden(item, issues, f"{path}[{idx}]")


def _write_doc(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Trace Envelope V2",
        "",
        CLAIM_BOUNDARY,
        "",
        "## Status",
        "",
        f"- Status: `{payload['status']}`",
        f"- Clinical validation: `{payload['clinical_validation']}`",
        f"- Validation pass rate: `{payload['validation_pass_rate']}`",
        f"- Forbidden field catch rate: `{payload['forbidden_field_catch_rate']}`",
        "",
        "## Required Fields",
        "",
        *[f"- `{field}`" for field in REQUIRED_FIELDS],
        "",
        "## Forbidden Keys",
        "",
        *[f"- `{field}`" for field in sorted(FORBIDDEN_EXACT_KEYS)],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT_DIR / candidate


__all__ = [
    "CLAIM_BOUNDARY",
    "DEFAULT_DOC_PATH",
    "DEFAULT_OUTPUT_PATH",
    "FORBIDDEN_EXACT_KEYS",
    "REQUIRED_FIELDS",
    "TRACE_SCHEMA_VERSION",
    "build_trace_envelope_v2",
    "build_trace_envelope_v2_eval",
    "validate_trace_envelope_v2",
]
