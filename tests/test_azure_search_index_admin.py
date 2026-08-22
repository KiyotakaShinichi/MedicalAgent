from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest

from backend.services.azure_search_index_admin import (
    build_azure_search_index_readiness,
    validate_azure_search_schema,
)
from backend.services.managed_vector_store import VectorStoreError


ROOT_DIR = Path(__file__).resolve().parents[1]


def _configured_environment() -> dict[str, str]:
    return {
        "NLCARE_VECTOR_BACKEND": "azure_ai_search",
        "NLCARE_MANAGED_VECTOR_SHADOW_ENABLED": "true",
        "NLCARE_MANAGED_VECTOR_ALLOW_NETWORK": "true",
        "AZURE_SEARCH_ENDPOINT": "https://fixture.search.windows.net",
        "AZURE_SEARCH_INDEX_NAME": "fixture-index",
        "AZURE_SEARCH_API_KEY": "fixture-key",
        "NLCARE_VECTOR_DIMENSION": "384",
    }


def test_schema_has_filterable_governance_fields_and_vector_profile(tmp_path: Path):
    # `output_path` defaults to the committed
    # Data/evals/rag/latest_azure_search_index_readiness.json, so leaving it
    # unset rewrote tracked evidence on every run. The assertions read the
    # returned report, never the file.
    report = build_azure_search_index_readiness(
        root_dir=ROOT_DIR, output_path=tmp_path / "readiness.json"
    )
    assert report["status"] == "ready_for_opt_in_provisioning"
    assert report["validation"]["valid"] is True
    assert report["validation"]["required_filter_fields_filterable"] is True
    assert report["embedding_dimension"] == 384
    assert report["network_request_performed"] is False
    assert report["clinical_validation"] is False


def test_apply_is_blocked_without_explicit_network_configuration():
    with pytest.raises(VectorStoreError, match="explicit shadow enablement"):
        build_azure_search_index_readiness(root_dir=ROOT_DIR, apply=True, environment={})


def test_apply_uses_idempotent_put_and_redacts_response(tmp_path: Path):
    calls: list[dict[str, Any]] = []

    def transport(
        method: str,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout: float,
    ) -> Mapping[str, Any]:
        calls.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers),
                "payload": dict(payload),
                "timeout": timeout,
            }
        )
        return {"name": payload["name"], "fields": payload["fields"], "@odata.etag": "fixture"}

    report = build_azure_search_index_readiness(
        root_dir=ROOT_DIR,
        output_path=tmp_path / "readiness.json",
        apply=True,
        environment=_configured_environment(),
        transport=transport,
    )
    assert report["status"] == "applied_shadow_index"
    assert report["network_request_performed"] is True
    assert calls[0]["method"] == "PUT"
    assert calls[0]["payload"]["name"] == "fixture-index"
    assert "fixture-key" not in str(report)


def test_schema_validator_rejects_missing_governance_fields():
    report = validate_azure_search_schema(
        {
            "fields": [
                {"name": "id", "type": "Edm.String", "key": True},
                {
                    "name": "content_vector",
                    "type": "Collection(Edm.Single)",
                    "dimensions": 4,
                    "vectorSearchProfile": "profile",
                },
            ],
            "vectorSearch": {"profiles": [{"name": "profile", "algorithm": "hnsw"}]},
        }
    )
    assert report["valid"] is False
    assert "missing_governance_fields" in report["issues"]
