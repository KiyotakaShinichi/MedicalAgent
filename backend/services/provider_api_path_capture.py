"""Controlled provider-usage capture through the normal synthetic chat API."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse


DEFAULT_OUTPUT_PATH = Path("Data/evals/ops/latest_provider_api_path_capture.json")
DEFAULT_BASE_URL = "http://127.0.0.1:8017"
MIN_REQUESTS = 30
MIN_PROVIDER_COVERAGE = 0.8

CLAIM_BOUNDARY = (
    "This is a controlled synthetic, non-patient operational probe. Provider-reported "
    "usage remains unaudited telemetry, not billing truth. The probe is not clinical "
    "validation, a production SLO, or production healthcare evidence."
)

# Fixed low-risk education and portal prompts only. Contents are sent to the local API but
# never written to the artifact; only stable IDs and SHA-256 hashes are persisted.
PROMPT_BANK: tuple[tuple[str, str, str], ...] = (
    ("portal_01", "portal_help", "How do I open the lab trends view in this demo portal?"),
    ("portal_02", "portal_help", "Where can I review synthetic CBC entries in the portal?"),
    ("portal_03", "portal_help", "How do I open the support chat tools menu?"),
    ("portal_04", "portal_help", "Where are imaging report records shown in this demo?"),
    ("portal_05", "portal_help", "How can I view the synthetic treatment timeline?"),
    ("portal_06", "portal_help", "Where can I find family and genetics records?"),
    ("portal_07", "portal_help", "How do I return from support chat to the overview?"),
    ("portal_08", "portal_help", "Where does this portal show missing data?"),
    ("portal_09", "portal_help", "How can I view citations attached to an educational answer?"),
    ("portal_10", "portal_help", "Where can an admin inspect an agent trace in this demo?"),
    ("education_01", "general_education", "What does CBC stand for in general oncology education?"),
    ("education_02", "general_education", "What is the general difference between MRI and CT imaging?"),
    ("education_03", "general_education", "What does HER2 mean as a general breast cancer term?"),
    ("education_04", "general_education", "What is a variant of uncertain significance in general terms?"),
    ("education_05", "general_education", "Why should a VUS not be treated as a positive result?"),
    ("education_06", "general_education", "What is a tumor marker in general educational terms?"),
    ("education_07", "general_education", "Why are tumor markers not diagnostic by themselves?"),
    ("education_08", "general_education", "What is source-grounded health education?"),
    ("education_09", "general_education", "What does evidence insufficiency mean in this prototype?"),
    ("education_10", "general_education", "Why can this prototype abstain from answering?"),
    ("governance_01", "system_boundary", "Is this demo clinically validated?"),
    ("governance_02", "system_boundary", "Can this prototype diagnose breast cancer?"),
    ("governance_03", "system_boundary", "Can this system choose a treatment?"),
    ("governance_04", "system_boundary", "What does synthetic-only mean for the model signals?"),
    ("governance_05", "system_boundary", "What is the purpose of the source-tier policy?"),
    ("governance_06", "system_boundary", "Why are clinician-only sources filtered from patient answers?"),
    ("governance_07", "system_boundary", "What does claim-level citation checking do?"),
    ("governance_08", "system_boundary", "What does the post-generation validator check?"),
    ("governance_09", "system_boundary", "What is the difference between an engineering gate and clinical validation?"),
    ("governance_10", "system_boundary", "Why is missing provider usage not treated as zero cost?"),
)

RequestFn = Callable[[str, str, dict[str, str], dict[str, Any]], tuple[int, dict[str, Any]]]


def build_provider_api_path_capture(
    *,
    execute: bool = False,
    base_url: str = DEFAULT_BASE_URL,
    request_count: int = MIN_REQUESTS,
    env: dict[str, str] | None = None,
    request_fn: RequestFn | None = None,
) -> dict[str, Any]:
    """Run an explicitly approved probe or emit an honest blocked/readiness artifact."""
    values = dict(os.environ if env is None else env)
    count = max(MIN_REQUESTS, min(int(request_count), len(PROMPT_BANK)))
    configured = _provider_configured(values)
    paid_probe_allowed = _truthy(values.get("NLCARE_ALLOW_PAID_PROVIDER_PROBE"))
    target_allowed = _target_allowed(base_url, values)
    execution_allowed = execute and configured and paid_probe_allowed and target_allowed

    blockers: list[str] = []
    if not configured:
        blockers.append("provider_credential_missing")
    if not paid_probe_allowed:
        blockers.append("paid_provider_probe_not_approved")
    if not target_allowed:
        blockers.append("non_loopback_target_not_approved")
    if not execute:
        blockers.append("explicit_execute_flag_missing")

    rows: list[dict[str, Any]] = []
    login_status: int | None = None
    execution_error: str | None = None
    if execution_allowed:
        requester = request_fn or _request_json
        try:
            login_status, login = requester(
                "POST",
                f"{base_url.rstrip('/')}/auth/demo-credential-login",
                {"Content-Type": "application/json", "X-NLCare-Data-Class": "synthetic"},
                {"username": "P001", "password": "patient-demo"},
            )
            token = str(login.get("access_token") or "")
            if login_status != 200 or not token:
                raise RuntimeError(f"demo_login_failed_http_{login_status}")
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "X-NLCare-Data-Class": "synthetic",
            }
            for prompt_id, category, prompt in PROMPT_BANK[:count]:
                started = time.perf_counter()
                status_code, response = requester(
                    "POST",
                    f"{base_url.rstrip('/')}/me/chat",
                    headers,
                    {"message": prompt},
                )
                elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
                rows.append(
                    _observation(
                        prompt_id=prompt_id,
                        category=category,
                        prompt=prompt,
                        status_code=status_code,
                        response=response,
                        elapsed_ms=elapsed_ms,
                    )
                )
        except Exception as exc:  # noqa: BLE001 - artifact must retain failure, not content
            execution_error = f"{type(exc).__name__}:{str(exc)[:120]}"

    successful = [row for row in rows if row["http_status"] == 200]
    provider_rows = [row for row in rows if row["provider_reported_total_tokens"] is not None]
    coverage = round(len(provider_rows) / len(rows), 4) if rows else 0.0
    completed = (
        len(rows) >= MIN_REQUESTS
        and len(successful) == len(rows)
        and coverage >= MIN_PROVIDER_COVERAGE
        and execution_error is None
    )
    if completed:
        status = "completed_controlled_probe"
        reason = "Normal synthetic API path met the request-count and provider-usage coverage contract."
    elif rows:
        status = "insufficient_provider_capture"
        reason = "The controlled probe ran, but success or provider-usage coverage stayed below contract."
    elif execute and execution_error:
        status = "execution_failed"
        reason = "The explicitly requested controlled probe failed closed."
    elif configured:
        status = "ready_for_explicit_execution"
        reason = "Provider configuration exists, but all explicit execution gates have not been approved."
    else:
        status = "blocked_configuration"
        reason = "No provider credential is available; no paid request was attempted."

    return {
        "schema_version": "provider_api_path_capture_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "completed": completed,
        "reason": reason,
        "normal_api_path": True,
        "base_url_origin": _origin(base_url),
        "provider_configured": configured,
        "explicit_execute_requested": bool(execute),
        "paid_provider_probe_approved": paid_probe_allowed,
        "target_allowed": target_allowed,
        "execution_blockers": blockers,
        "execution_error": execution_error,
        "login_http_status": login_status,
        "request_count": len(rows),
        "successful_request_count": len(successful),
        "provider_usage_observed_request_count": len(provider_rows),
        "provider_usage_coverage_rate": coverage,
        "minimum_request_count": MIN_REQUESTS,
        "minimum_provider_usage_coverage_rate": MIN_PROVIDER_COVERAGE,
        "provider_reported_total_tokens": sum(
            int(row["provider_reported_total_tokens"] or 0) for row in provider_rows
        ),
        "estimated_total_tokens": sum(int(row["estimated_total_tokens"] or 0) for row in rows),
        "estimated_cost_usd": (
            round(sum(float(row["estimated_cost_usd"] or 0.0) for row in rows), 8)
            if rows
            else None
        ),
        "requests": rows,
        "content_retained": False,
        "patient_data_processed": False,
        "synthetic_only": True,
        "automatic_paid_provider_probe": False,
        "audited_billing": False,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def write_provider_api_path_capture(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    **kwargs: Any,
) -> dict[str, Any]:
    payload = build_provider_api_path_capture(**kwargs)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _observation(
    *,
    prompt_id: str,
    category: str,
    prompt: str,
    status_code: int,
    response: dict[str, Any],
    elapsed_ms: float,
) -> dict[str, Any]:
    telemetry = response.get("llm_telemetry") if isinstance(response, dict) else {}
    telemetry = telemetry if isinstance(telemetry, dict) else {}
    provider_calls = int(telemetry.get("provider_reported_call_count") or 0)
    total_tokens = _positive_int(telemetry.get("total_tokens"))
    actual = total_tokens if provider_calls > 0 else None
    estimated = total_tokens if total_tokens is not None else None
    return {
        "request_id": f"provider-probe-{prompt_id}",
        "prompt_id": prompt_id,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "category": category,
        "route": "/me/chat",
        "http_status": int(status_code),
        "latency_ms": elapsed_ms,
        "provider_reported_total_tokens": actual,
        "estimated_total_tokens": estimated,
        "provider_reported_call_count": provider_calls,
        "estimated_cost_usd": _number(telemetry.get("estimated_cost_usd")),
        "claim_validation_passed": _claim_validation_passed(response),
        "source_tier_correct": _source_tier_correct(response),
        "unsafe_leakage": _unsafe_leakage(response),
        "content_retained": False,
    }


def _claim_validation_passed(response: dict[str, Any]) -> bool | None:
    rag = response.get("rag_evaluation") if isinstance(response, dict) else {}
    rag = rag if isinstance(rag, dict) else {}
    for key in ("claim_validation_passed", "claim_support_passed"):
        if isinstance(rag.get(key), bool):
            return rag[key]
    status = str(rag.get("status") or "").lower()
    return True if status in {"acceptable", "strong", "passed"} else None


def _source_tier_correct(response: dict[str, Any]) -> bool | None:
    rag = response.get("rag_evaluation") if isinstance(response, dict) else {}
    rag = rag if isinstance(rag, dict) else {}
    value = rag.get("source_tier_correct")
    return value if isinstance(value, bool) else None


def _unsafe_leakage(response: dict[str, Any]) -> bool | None:
    guardrails = response.get("guardrails") if isinstance(response, dict) else {}
    guardrails = guardrails if isinstance(guardrails, dict) else {}
    output = guardrails.get("output") if isinstance(guardrails.get("output"), dict) else {}
    for key in ("unsafe_leakage", "unsafe_answer"):
        if isinstance(output.get(key), bool):
            return output[key]
    return None


def _request_json(
    method: str,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 - guarded target
            data = json.loads(response.read().decode("utf-8"))
            return int(response.status), data if isinstance(data, dict) else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {}
        return int(exc.code), parsed if isinstance(parsed, dict) else {}


def _provider_configured(env: dict[str, str]) -> bool:
    for name in ("GROQ_API_KEY", "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY"):
        value = str(env.get(name) or "").strip()
        if value and "replace" not in value.lower() and "placeholder" not in value.lower():
            return True
    return False


def _target_allowed(base_url: str, env: dict[str, str]) -> bool:
    hostname = (urlparse(base_url).hostname or "").lower()
    if hostname in {"127.0.0.1", "localhost", "::1"}:
        return True
    return _truthy(env.get("NLCARE_ALLOW_NON_LOOPBACK_PROVIDER_PROBE"))


def _origin(base_url: str) -> str:
    parsed = urlparse(base_url)
    return f"{parsed.scheme}://{parsed.netloc}"


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "PROMPT_BANK",
    "build_provider_api_path_capture",
    "write_provider_api_path_capture",
]
