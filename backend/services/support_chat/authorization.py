from backend.services.agent_input_gate import input_guardrail_check
from backend.services.agent_output_gate import output_guardrail_check
from backend.services.agent_post_gen import apply_post_gen_validator
from backend.services.rag_evidence_envelope import (
    build_fail_closed_error_result,
    enforce_evidence_release,
    enforce_transport_release,
    parse_evidence_envelope,
)


def authorize_final_support_response(
    agent_result,
    *,
    query,
    routing_safety,
    deterministic_tool_confirmation=False,
):
    """Authorize the exact reply that the support API persists and sends.

    Evidence-dependent answers cannot be re-authorized after the outer support
    layer mutates their text because claim/citation validation applied to the
    original candidate. Deterministic support replies are rechecked from
    scratch because they do not depend on retrieved medical evidence.
    """

    if not isinstance(agent_result, dict):
        return build_fail_closed_error_result(
            query=query,
            error_code="support_result_malformed",
        )
    existing_envelope, _ = parse_evidence_envelope(agent_result.get("evidence_envelope"))
    if existing_envelope is not None and existing_envelope.evidence_required:
        return enforce_transport_release(agent_result, query=query)

    try:
        input_guardrails = input_guardrail_check(query, routing_safety or {})
        validation = agent_result.get("validation")
        if not isinstance(validation, dict) or validation.get("status") != "passed":
            agent_result["validation"] = {
                "status": "passed",
                "issues": [],
                "citation_count": 0,
                "validation_scope": "deterministic_non_evidence_support",
            }
        output_candidate = agent_result
        if deterministic_tool_confirmation:
            output_candidate = dict(agent_result)
            output_candidate["safety"] = {
                **(agent_result.get("safety") or {}),
                "level": "deterministic_tool_confirmation",
            }
        output_guardrails = output_guardrail_check(output_candidate)
        output_guardrails, _ = apply_post_gen_validator(agent_result, output_guardrails)
        agent_result["guardrails"] = {
            "input": input_guardrails,
            "output": output_guardrails,
        }
        errors = []
        if (output_guardrails or {}).get("status") != "passed":
            errors.append("support_output_guardrail_failed")
        enforce_evidence_release(
            agent_result,
            query=query,
            input_guardrails=input_guardrails,
            validation_errors=errors,
            evidence_required=False,
        )
        return enforce_transport_release(agent_result, query=query)
    except Exception as exc:  # noqa: BLE001 - alternate entry point must deny
        return build_fail_closed_error_result(
            query=query,
            error_code=f"support_final_authorization_exception:{type(exc).__name__}",
            result=agent_result,
        )
