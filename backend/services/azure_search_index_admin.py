"""Azure AI Search index readiness and opt-in provisioning.

The default path is a local schema validation only. Network execution requires
both an explicit ``apply`` argument and the managed-vector network gates.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import quote

from backend.services.managed_vector_store import (
    ManagedVectorConfig,
    VectorStoreError,
    load_managed_vector_config,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA_PATH = Path("config/vector_indexes/azure_ai_search_nlcare_kb.json")
DEFAULT_OUTPUT_PATH = Path("Data/evals/rag/latest_azure_search_index_readiness.json")
REQUIRED_FILTER_FIELDS = {
    "allowed_use",
    "clinical_validation",
    "data_scope",
    "patient_facing",
    "source_tier",
    "staleness_status",
}
Transport = Callable[
    [str, str, Mapping[str, str], Mapping[str, Any], float],
    Mapping[str, Any],
]


def build_azure_search_index_readiness(
    *,
    root_dir: str | Path = ROOT_DIR,
    schema_path: str | Path = DEFAULT_SCHEMA_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    environment: Mapping[str, str] | None = None,
    apply: bool = False,
    transport: Transport | None = None,
) -> dict[str, Any]:
    root = Path(root_dir)
    schema_file = _resolve(root, schema_path)
    schema = json.loads(schema_file.read_text(encoding="utf-8"))
    validation = validate_azure_search_schema(schema)
    config = load_managed_vector_config(environment)
    configured = config.provider == "azure_ai_search" and config.configured
    network_allowed = bool(configured and config.allow_network)
    applied = False
    apply_result: dict[str, Any] | None = None

    if apply:
        if not network_allowed:
            raise VectorStoreError(
                "Azure index apply requires the azure_ai_search backend, explicit shadow enablement, "
                "credentials, and NLCARE_MANAGED_VECTOR_ALLOW_NETWORK=true."
            )
        payload = dict(schema)
        payload["name"] = config.index_name
        request_transport = transport or _default_transport
        url = (
            f"{config.endpoint}/indexes/{quote(config.index_name)}"
            f"?api-version={quote(config.api_version)}"
        )
        apply_result = dict(
            request_transport(
                "PUT",
                url,
                _headers(config),
                payload,
                30.0,
            )
        )
        applied = True

    status = (
        "applied_shadow_index"
        if applied
        else "ready_for_opt_in_provisioning"
        if validation["valid"]
        else "needs_attention"
    )
    payload = {
        "schema_version": "nlcare_azure_search_index_readiness_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "patient_data_allowed": False,
        "data_scope": "curated_non_patient_kb",
        "schema_path": Path(schema_path).as_posix(),
        "schema_sha256": hashlib.sha256(schema_file.read_bytes()).hexdigest(),
        "index_schema_name": schema.get("name"),
        "embedding_dimension": validation["embedding_dimension"],
        "validation": validation,
        "configured": configured,
        "network_allowed": network_allowed,
        "network_request_performed": applied,
        "index_apply_completed": applied,
        "idempotent_method": "PUT create-or-update",
        "apply_result_summary": _redacted_apply_summary(apply_result),
        "promotion_boundary": {
            "live_patient_route_changed": False,
            "shadow_only": True,
            "retrieval_improvement_proven": False,
            "requires_frozen_comparison": True,
        },
        "claim_boundary": (
            "This validates or provisions an engineering-only Azure AI Search shadow index for curated "
            "non-patient knowledge. It does not prove retrieval improvement, clinical validity, patient "
            "benefit, security certification, or production healthcare readiness."
        ),
    }
    destination = _resolve(root, output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def validate_azure_search_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        str(field.get("name")): dict(field)
        for field in schema.get("fields", [])
        if isinstance(field, Mapping) and field.get("name")
    }
    issues: list[str] = []
    if fields.get("id", {}).get("key") is not True:
        issues.append("id_must_be_key")
    vector_field = fields.get("content_vector") or {}
    dimension = int(vector_field.get("dimensions") or 0)
    if vector_field.get("type") != "Collection(Edm.Single)":
        issues.append("content_vector_type_invalid")
    if dimension <= 0:
        issues.append("content_vector_dimension_missing")
    if not vector_field.get("vectorSearchProfile"):
        issues.append("content_vector_profile_missing")

    missing_filter_fields = sorted(REQUIRED_FILTER_FIELDS - set(fields))
    if missing_filter_fields:
        issues.append("missing_governance_fields")
    non_filterable = sorted(
        name
        for name in REQUIRED_FILTER_FIELDS
        if name in fields and fields[name].get("filterable") is not True
    )
    if non_filterable:
        issues.append("governance_fields_not_filterable")
    profiles = {
        str(row.get("name"))
        for row in (schema.get("vectorSearch", {}).get("profiles") or [])
        if isinstance(row, Mapping)
    }
    if vector_field.get("vectorSearchProfile") not in profiles:
        issues.append("vector_profile_not_declared")
    return {
        "valid": not issues,
        "issues": issues,
        "field_count": len(fields),
        "embedding_dimension": dimension,
        "required_filter_fields_present": not missing_filter_fields,
        "required_filter_fields_filterable": not non_filterable,
        "vector_profile_declared": vector_field.get("vectorSearchProfile") in profiles,
    }


def _headers(config: ManagedVectorConfig) -> dict[str, str]:
    auth = (
        {"Authorization": f"Bearer {config.credential}"}
        if config.credential.count(".") == 2
        else {"api-key": config.credential}
    )
    return {"Content-Type": "application/json", **auth}


def _default_transport(
    method: str,
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout: float,
) -> Mapping[str, Any]:
    from backend.services.managed_vector_store import _json_transport

    return _json_transport(method, url, headers, payload, timeout)


def _redacted_apply_summary(payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    return {
        "name": payload.get("name"),
        "field_count": len(payload.get("fields") or []),
        "etag_present": bool(payload.get("@odata.etag")),
    }


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


__all__ = [
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_SCHEMA_PATH",
    "REQUIRED_FILTER_FIELDS",
    "ROOT_DIR",
    "build_azure_search_index_readiness",
    "validate_azure_search_schema",
]
