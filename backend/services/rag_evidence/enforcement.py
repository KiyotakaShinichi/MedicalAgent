"""Fail-closed enforcement at generation and transport boundaries."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, MutableMapping, Sequence
from uuid import uuid4

from backend.services.rag_evidence.assembly import build_evidence_envelope
from backend.services.rag_evidence.authorization import (
    authorize_evidence_release,
    parse_evidence_envelope,
)
from backend.services.rag_evidence.metrics import increment
from backend.services.rag_evidence.responses import build_safe_abstention
from backend.services.rag_evidence.telemetry import record_event, record_simple_event
from backend.services.rag_evidence.types import (
    EVIDENCE_ENVELOPE_VERSION,
    EVIDENCE_POLICY_VERSION,
    SAFETY_POLICY_VERSION,
    VALIDATOR_POLICY_VERSION,
    AuthorizationDecision,
    EvidenceDisposition,
    EvidenceEnvelope,
)
from backend.services.rag_evidence.utilities import current_request_id, response_digest


def enforce_evidence_release(
    result: MutableMapping[str, Any],
    *,
    query: str = "",
    retrieved: Sequence[Mapping[str, Any]] | None = None,
    input_guardrails: Mapping[str, Any] | None = None,
    request_id: str | None = None,
    validation_errors: Sequence[str] | None = None,
    validation_warnings: Sequence[str] | None = None,
    evidence_required: bool | None = None,
) -> MutableMapping[str, Any]:
    """Create, authorize, and enforce the final evidence envelope in place."""

    boundary_request_id = str(request_id or current_request_id() or f"local-{uuid4().hex}")
    try:
        record_simple_event(
            result,
            "evidence_validation_started",
            request_id=boundary_request_id,
            evidence_required=evidence_required,
        )
        envelope = build_evidence_envelope(
            result,
            query=query,
            retrieved=retrieved,
            input_guardrails=input_guardrails,
            request_id=boundary_request_id,
            validation_errors=validation_errors,
            validation_warnings=validation_warnings,
            evidence_required=evidence_required,
        )
        decision = authorize_evidence_release(envelope)
        record_simple_event(
            result,
            "evidence_validation_completed",
            request_id=boundary_request_id,
            evidence_required=envelope.evidence_required,
            reason=decision.reason,
        )
        if envelope.validation_errors:
            record_simple_event(
                result,
                "evidence_validator_failure",
                request_id=boundary_request_id,
                evidence_required=envelope.evidence_required,
                reason="validation_component_failure",
            )
    except Exception as exc:  # noqa: BLE001 - final boundary must deny on all failures
        return build_fail_closed_error_result(
            query=query,
            request_id=boundary_request_id,
            error_code=f"evidence_boundary_exception:{type(exc).__name__}",
            result=result,
        )

    if decision.disposition is not EvidenceDisposition.ALLOW:
        result["reply"] = build_safe_abstention(
            decision.disposition,
            query=query,
            input_guardrails=input_guardrails,
            existing_reply=("" if envelope.evidence_required else str(result.get("reply") or "")),
        )
        result["citations"] = []
    envelope_dict = envelope.to_dict()
    envelope_dict["final_disposition"] = decision.disposition.value
    envelope_dict["abstention_reason"] = (
        None if decision.disposition is EvidenceDisposition.ALLOW else decision.reason
    )
    envelope_dict["response_digest"] = response_digest(result.get("reply"))
    result["evidence_envelope"] = envelope_dict
    result["release_authorization"] = decision.to_dict()
    record_event(result, "evidence_envelope_created", envelope_dict, decision)
    record_event(
        result,
        "rag_release_allowed" if decision.disposition is EvidenceDisposition.ALLOW else "rag_release_denied",
        envelope_dict,
        decision,
    )
    if decision.disposition is not EvidenceDisposition.ALLOW:
        record_simple_event(
            result,
            "rag_abstention_reason",
            request_id=boundary_request_id,
            evidence_required=envelope.evidence_required,
            reason=decision.reason,
        )
    increment(
        "rag_release_allowed_total"
        if decision.disposition is EvidenceDisposition.ALLOW
        else "rag_release_denied_total"
    )
    if decision.disposition is not EvidenceDisposition.ALLOW:
        increment("rag_abstention_total")
    if decision.disposition is EvidenceDisposition.ABSTAIN_VALIDATION_FAILURE:
        increment("rag_validation_failure_total")
    if decision.disposition is EvidenceDisposition.ABSTAIN_UNSUPPORTED_CLAIMS:
        increment("rag_unsupported_claim_total")
    return result


def enforce_transport_release(
    result: MutableMapping[str, Any],
    *,
    query: str,
    actionability_classifier: Callable[[str], Any],
) -> MutableMapping[str, Any]:
    """Recheck a completed JSON/SSE payload immediately before transport."""

    container: MutableMapping[str, Any] = result
    nested = result.get("agent_pipeline")
    if isinstance(nested, MutableMapping):
        container = nested
    envelope_raw = container.get("evidence_envelope") or result.get("evidence_envelope")
    envelope, error = parse_evidence_envelope(envelope_raw)
    reply = result.get("reply") if "reply" in result else container.get("reply")
    try:
        output_guard = actionability_classifier(reply if isinstance(reply, str) else "")
    except Exception as exc:
        from backend.services.dep001d_output_actionability import OutputActionabilityDecision

        output_guard = OutputActionabilityDecision(
            decision="blocked",
            actionable_probability=1.0,
            uncertainty=1.0,
            threshold=0.0,
            uncertainty_threshold=0.0,
            model_version="unavailable",
            reason="output_actionability_validation_unavailable",
            failure_reason=f"transport_validator_exception:{type(exc).__name__}",
        )
    if output_guard.blocked:
        reason = f"semantic_output_guard:{output_guard.reason}"
        increment("rag_release_denied_total")
        increment("rag_validation_failure_total")
        increment("rag_abstention_total")
        failed = build_fail_closed_error_result(
            query=query,
            request_id=(envelope.request_id if envelope else None),
            error_code=reason,
            result=container,
        )
        trace = failed["evidence_envelope"].setdefault("trace_metadata", {})
        trace["semantic_output_actionability"] = output_guard.to_dict()
        safe_reply = failed["reply"]
        result["reply"] = safe_reply
        result["citations"] = []
        result["evidence_envelope"] = failed["evidence_envelope"]
        result["release_authorization"] = failed["release_authorization"]
        if container is not result:
            container["reply"] = safe_reply
            container["citations"] = []
            container["evidence_envelope"] = failed["evidence_envelope"]
            container["release_authorization"] = failed["release_authorization"]
        return result
    if envelope is None or response_digest(reply) != envelope.response_digest:
        reason = error or "response_changed_after_authorization"
        increment("rag_release_denied_total")
        increment("rag_validation_failure_total")
        increment("rag_abstention_total")
        failed = build_fail_closed_error_result(
            query=query,
            request_id=(envelope.request_id if envelope else None),
            error_code=reason,
            result=container,
        )
        safe_reply = failed["reply"]
        result["reply"] = safe_reply
        result["citations"] = []
        result["evidence_envelope"] = failed["evidence_envelope"]
        result["release_authorization"] = failed["release_authorization"]
        if container is not result:
            container["reply"] = safe_reply
            container["citations"] = []
            container["evidence_envelope"] = failed["evidence_envelope"]
            container["release_authorization"] = failed["release_authorization"]
        return result

    decision = authorize_evidence_release(envelope)
    if decision.disposition is EvidenceDisposition.ALLOW:
        return result
    transported_citations = result.get("citations") or container.get("citations") or []
    if transported_citations:
        failed = build_fail_closed_error_result(
            query=query,
            request_id=envelope.request_id,
            error_code="non_allow_payload_contains_citations",
            result=container,
        )
        result["reply"] = failed["reply"]
        result["citations"] = []
        result["evidence_envelope"] = failed["evidence_envelope"]
        result["release_authorization"] = failed["release_authorization"]
        if container is not result:
            container.update({
                "reply": failed["reply"],
                "citations": [],
                "evidence_envelope": failed["evidence_envelope"],
                "release_authorization": failed["release_authorization"],
            })
    return result


def build_fail_closed_error_result(
    *,
    query: str = "",
    request_id: str | None = None,
    error_code: str = "internal_pipeline_failure",
    result: MutableMapping[str, Any] | None = None,
) -> MutableMapping[str, Any]:
    """Build a self-contained safe result when the final boundary fails."""

    target = result if isinstance(result, MutableMapping) else {}
    safe_reply = build_safe_abstention(EvidenceDisposition.INTERNAL_ERROR, query=query)
    rid = str(request_id or current_request_id() or f"local-{uuid4().hex}")
    now = datetime.now(timezone.utc).isoformat()
    digest = response_digest(safe_reply)
    envelope = EvidenceEnvelope(
        request_id=rid,
        version=EVIDENCE_ENVELOPE_VERSION,
        policy_version=EVIDENCE_POLICY_VERSION,
        safety_policy_version=SAFETY_POLICY_VERSION,
        validator_version=VALIDATOR_POLICY_VERSION,
        evidence_required=True,
        response_kind="safe_abstention",
        retrieval_status="failed",
        answerability_status="validation_failure",
        citation_validation_status="failed",
        claim_support_status="failed",
        evidence_coverage_status="failed",
        conflict_status="unknown",
        safety_validation_status="failed",
        validation_errors=(str(error_code)[:160],),
        abstention_reason=str(error_code)[:160],
        final_disposition=EvidenceDisposition.INTERNAL_ERROR,
        candidate_response_digest=response_digest(target.get("reply")),
        response_digest=digest,
        created_at=now,
        trace_metadata={
            "query_hash": hashlib.sha256(str(query or "").lower().encode("utf-8")).hexdigest(),
        },
    )
    target.update({
        "reply": safe_reply,
        "citations": [],
        "intent": target.get("intent") or "internal_error",
        "safety": target.get("safety") or {
            "level": "unknown",
            "scope": "internal_failure",
            "cache_allowed": False,
        },
        "cache": {
            "status": "not_stored_validation_failure",
            "cacheable": False,
            "reason": str(error_code)[:160],
        },
        "evidence_envelope": envelope.to_dict(),
        "release_authorization": AuthorizationDecision(
            disposition=EvidenceDisposition.INTERNAL_ERROR,
            release_evidence_answer=False,
            release_safe_response=True,
            reason=str(error_code)[:160],
        ).to_dict(),
    })
    decision = AuthorizationDecision(
        disposition=EvidenceDisposition.INTERNAL_ERROR,
        release_evidence_answer=False,
        release_safe_response=True,
        reason=str(error_code)[:160],
    )
    try:
        record_event(target, "rag_release_denied", envelope.to_dict(), decision)
    except (TypeError, AttributeError):
        target["evidence_envelope_events"] = []
        record_event(target, "rag_release_denied", envelope.to_dict(), decision)
    increment("rag_release_denied_total")
    increment("rag_validation_failure_total")
    increment("rag_abstention_total")
    return target
