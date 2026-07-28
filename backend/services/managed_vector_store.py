"""Managed vector-store boundary for shadow retrieval experiments.

The live patient route still uses ``rag_vector_index.py``.  This module keeps
Azure AI Search and Pinecone behind one small contract so their payloads,
metadata filters, and failure behavior can be tested without credentials or
network access.
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Protocol
from urllib.parse import quote


ALLOWED_PROVIDERS = {"local_faiss", "azure_ai_search", "pinecone"}
REMOTE_SAFE_NAMESPACES = {
    "nlcare_kb_demo_t1_t3",
    "nlcare_eval_synthetic",
    "nlcare_portal_help",
}
BANNED_METADATA_KEYS = {
    "patient_id",
    "patient_name",
    "email",
    "phone",
    "address",
    "raw_chat",
    "raw_prompt",
    "raw_response",
    "medical_record_number",
    "mrn",
}
CANONICAL_METADATA_FIELDS = {
    "source_id",
    "chunk_id",
    "parent_id",
    "source_tier",
    "allowed_use",
    "patient_facing",
    "staleness_status",
    "kb_fingerprint",
    "doc_type",
    "data_scope",
    "clinical_validation",
    "title",
    "source_name",
    "source_url",
    "topic",
    "section",
    "tags",
}


class VectorStoreError(RuntimeError):
    """Raised when a managed-vector request violates the local contract."""


@dataclass(frozen=True)
class VectorRecord:
    record_id: str
    vector: tuple[float, ...]
    text: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class VectorSearchRequest:
    query_vector: tuple[float, ...]
    text_query: str
    top_k: int = 10
    allowed_tiers: tuple[str, ...] = ("T1", "T2", "T3")
    allowed_uses: tuple[str, ...] = ("education", "patient_safety", "monitoring_context")


@dataclass(frozen=True)
class VectorSearchResult:
    record_id: str
    score: float
    text: str
    metadata: Mapping[str, Any]
    provider: str


@dataclass(frozen=True)
class ManagedVectorConfig:
    provider: str
    enabled: bool
    shadow_only: bool
    allow_network: bool
    namespace: str
    endpoint: str
    index_name: str
    api_version: str
    credential: str
    embedding_dimension: int

    @property
    def configured(self) -> bool:
        if self.provider == "local_faiss":
            return True
        return bool(self.enabled and self.endpoint and self.index_name and self.credential)


class VectorStoreAdapter(Protocol):
    provider: str

    def upsert(self, records: Iterable[VectorRecord]) -> dict[str, Any]:
        ...

    def search(self, request: VectorSearchRequest) -> list[VectorSearchResult]:
        ...


Transport = Callable[[str, str, Mapping[str, str], Mapping[str, Any], float], Mapping[str, Any]]


def load_managed_vector_config(environment: Mapping[str, str] | None = None) -> ManagedVectorConfig:
    env = dict(os.environ if environment is None else environment)
    provider = (env.get("NLCARE_VECTOR_BACKEND") or "local_faiss").strip().lower()
    if provider not in ALLOWED_PROVIDERS:
        raise VectorStoreError(f"Unsupported vector provider: {provider}")

    enabled = _truthy(env.get("NLCARE_MANAGED_VECTOR_SHADOW_ENABLED"))
    allow_network = _truthy(env.get("NLCARE_MANAGED_VECTOR_ALLOW_NETWORK"))
    dimension = _positive_int(env.get("NLCARE_VECTOR_DIMENSION"), default=384)

    if provider == "azure_ai_search":
        return ManagedVectorConfig(
            provider=provider,
            enabled=enabled,
            shadow_only=True,
            allow_network=allow_network,
            namespace=(env.get("AZURE_SEARCH_NAMESPACE") or "nlcare_kb_demo_t1_t3").strip(),
            endpoint=(env.get("AZURE_SEARCH_ENDPOINT") or "").rstrip("/"),
            index_name=(env.get("AZURE_SEARCH_INDEX_NAME") or "").strip(),
            api_version=(env.get("AZURE_SEARCH_API_VERSION") or "2026-04-01").strip(),
            credential=(
                env.get("AZURE_SEARCH_BEARER_TOKEN")
                or env.get("AZURE_SEARCH_API_KEY")
                or ""
            ).strip(),
            embedding_dimension=dimension,
        )
    if provider == "pinecone":
        return ManagedVectorConfig(
            provider=provider,
            enabled=enabled,
            shadow_only=True,
            allow_network=allow_network,
            namespace=(env.get("PINECONE_NAMESPACE_KB") or "nlcare_kb_demo_t1_t3").strip(),
            endpoint=(env.get("PINECONE_INDEX_HOST") or "").rstrip("/"),
            index_name=(env.get("PINECONE_INDEX_NAME") or "nlcare-kb-shadow").strip(),
            api_version=(env.get("PINECONE_API_VERSION") or "2025-10").strip(),
            credential=(env.get("PINECONE_API_KEY") or "").strip(),
            embedding_dimension=dimension,
        )
    return ManagedVectorConfig(
        provider="local_faiss",
        enabled=True,
        shadow_only=False,
        allow_network=False,
        namespace="local",
        endpoint="",
        index_name="local_hybrid_rag_index",
        api_version="local",
        credential="",
        embedding_dimension=dimension,
    )


def build_managed_adapter(
    config: ManagedVectorConfig,
    *,
    transport: Transport | None = None,
) -> VectorStoreAdapter:
    if config.provider == "azure_ai_search":
        return AzureAISearchAdapter(config, transport=transport)
    if config.provider == "pinecone":
        return PineconeAdapter(config, transport=transport)
    return InMemoryVectorStore(dimension=config.embedding_dimension)


def validate_remote_record(record: VectorRecord, *, namespace: str) -> dict[str, Any]:
    if namespace not in REMOTE_SAFE_NAMESPACES:
        raise VectorStoreError(f"Remote namespace is not approved for shadow use: {namespace}")
    if not record.record_id or not record.text.strip():
        raise VectorStoreError("Vector records require a stable ID and non-empty text.")
    if not record.vector or any(not math.isfinite(value) for value in record.vector):
        raise VectorStoreError("Vector records require finite embedding values.")

    metadata = flatten_metadata(record.metadata)
    forbidden = sorted(BANNED_METADATA_KEYS & set(metadata))
    if forbidden:
        raise VectorStoreError(f"Forbidden metadata keys: {', '.join(forbidden)}")
    unknown = sorted(set(metadata) - CANONICAL_METADATA_FIELDS)
    if unknown:
        raise VectorStoreError(f"Unknown remote metadata keys: {', '.join(unknown)}")
    if metadata.get("data_scope") != "curated_non_patient_kb":
        raise VectorStoreError("Remote shadow records must declare curated_non_patient_kb scope.")
    if metadata.get("clinical_validation") is not False:
        raise VectorStoreError("Remote shadow records must declare clinical_validation=false.")
    if metadata.get("patient_facing") is not True:
        raise VectorStoreError("Remote shadow records must be explicitly patient-facing.")
    if metadata.get("source_tier") not in {"T1", "T2", "T3", "T4"}:
        raise VectorStoreError("Remote shadow records require a governed source tier.")
    if not metadata.get("kb_fingerprint"):
        raise VectorStoreError("Remote shadow records require a KB fingerprint.")
    return metadata


def flatten_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in metadata.items():
        normalized_key = str(key).strip()
        if normalized_key not in CANONICAL_METADATA_FIELDS and normalized_key in BANNED_METADATA_KEYS:
            output[normalized_key] = value
            continue
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            output[normalized_key] = value
        elif isinstance(value, (list, tuple, set)):
            output[normalized_key] = [str(item) for item in value if item is not None]
        else:
            output[normalized_key] = json.dumps(value, sort_keys=True, default=str)
    return output


class InMemoryVectorStore:
    """Deterministic contract adapter used by tests and offline parity probes."""

    provider = "in_memory_contract"

    def __init__(self, *, dimension: int) -> None:
        self.dimension = dimension
        self._records: dict[str, VectorRecord] = {}

    def upsert(self, records: Iterable[VectorRecord]) -> dict[str, Any]:
        count = 0
        for record in records:
            self._validate_dimension(record.vector)
            self._records[record.record_id] = record
            count += 1
        return {"provider": self.provider, "upserted_count": count}

    def search(self, request: VectorSearchRequest) -> list[VectorSearchResult]:
        self._validate_dimension(request.query_vector)
        rows: list[VectorSearchResult] = []
        for record in self._records.values():
            metadata = flatten_metadata(record.metadata)
            if metadata.get("clinical_validation") is not False:
                continue
            if metadata.get("patient_facing") is not True:
                continue
            if metadata.get("data_scope") != "curated_non_patient_kb":
                continue
            if metadata.get("source_tier") not in request.allowed_tiers:
                continue
            uses = set(metadata.get("allowed_use") or [])
            if uses and not uses.intersection(request.allowed_uses):
                continue
            rows.append(
                VectorSearchResult(
                    record_id=record.record_id,
                    score=_cosine(request.query_vector, record.vector),
                    text=record.text,
                    metadata=metadata,
                    provider=self.provider,
                )
            )
        rows.sort(key=lambda row: (-row.score, row.record_id))
        return rows[: request.top_k]

    def _validate_dimension(self, vector: tuple[float, ...]) -> None:
        if len(vector) != self.dimension:
            raise VectorStoreError(
                f"Embedding dimension mismatch: expected {self.dimension}, got {len(vector)}"
            )


class AzureAISearchAdapter:
    provider = "azure_ai_search"

    def __init__(self, config: ManagedVectorConfig, *, transport: Transport | None = None) -> None:
        self.config = config
        self.transport = transport or _json_transport

    def upsert(self, records: Iterable[VectorRecord]) -> dict[str, Any]:
        self._require_network()
        documents = []
        for record in records:
            metadata = validate_remote_record(record, namespace=self.config.namespace)
            self._validate_dimension(record.vector)
            documents.append(
                {
                    "@search.action": "mergeOrUpload",
                    "id": record.record_id,
                    "content": record.text,
                    "content_vector": list(record.vector),
                    **metadata,
                }
            )
        url = (
            f"{self.config.endpoint}/indexes/{quote(self.config.index_name)}/docs/index"
            f"?api-version={quote(self.config.api_version)}"
        )
        return dict(self.transport("POST", url, self._headers(), {"value": documents}, 20.0))

    def search(self, request: VectorSearchRequest) -> list[VectorSearchResult]:
        self._require_network()
        self._validate_dimension(request.query_vector)
        body = self.build_search_payload(request)
        url = (
            f"{self.config.endpoint}/indexes/{quote(self.config.index_name)}/docs/search"
            f"?api-version={quote(self.config.api_version)}"
        )
        payload = self.transport("POST", url, self._headers(), body, 10.0)
        rows = payload.get("value") or []
        return [
            VectorSearchResult(
                record_id=str(row.get("id") or ""),
                score=float(row.get("@search.score") or 0.0),
                text=str(row.get("content") or ""),
                metadata={key: value for key, value in row.items() if key not in {"id", "content", "@search.score"}},
                provider=self.provider,
            )
            for row in rows
        ]

    def build_search_payload(self, request: VectorSearchRequest) -> dict[str, Any]:
        tier_filter = " or ".join(f"source_tier eq '{tier}'" for tier in request.allowed_tiers)
        use_values = ",".join(_escape_filter_literal(value) for value in request.allowed_uses)
        use_filter = f"allowed_use/any(u: search.in(u, '{use_values}', ','))"
        return {
            "search": request.text_query or "*",
            "top": request.top_k,
            "select": "id,content,source_id,chunk_id,parent_id,source_tier,allowed_use,"
            "patient_facing,staleness_status,kb_fingerprint,doc_type,data_scope,clinical_validation,"
            "title,source_name,source_url,topic,section,tags",
            "filter": (
                "clinical_validation eq false and patient_facing eq true and "
                "data_scope eq 'curated_non_patient_kb' and "
                f"({tier_filter}) and {use_filter}"
            ),
            "vectorFilterMode": "preFilter",
            "vectorQueries": [
                {
                    "kind": "vector",
                    "vector": list(request.query_vector),
                    "fields": "content_vector",
                    "k": max(request.top_k, 10),
                }
            ],
        }

    def _headers(self) -> dict[str, str]:
        auth_header = (
            {"Authorization": f"Bearer {self.config.credential}"}
            if self.config.credential.count(".") == 2
            else {"api-key": self.config.credential}
        )
        return {"Content-Type": "application/json", **auth_header}

    def _require_network(self) -> None:
        if not self.config.configured or not self.config.allow_network:
            raise VectorStoreError("Azure AI Search is shadow-only and network execution is disabled.")

    def _validate_dimension(self, vector: tuple[float, ...]) -> None:
        if len(vector) != self.config.embedding_dimension:
            raise VectorStoreError("Azure AI Search embedding dimension does not match the configured index.")


class PineconeAdapter:
    provider = "pinecone"

    def __init__(self, config: ManagedVectorConfig, *, transport: Transport | None = None) -> None:
        self.config = config
        self.transport = transport or _json_transport

    def upsert(self, records: Iterable[VectorRecord]) -> dict[str, Any]:
        self._require_network()
        vectors = []
        for record in records:
            metadata = validate_remote_record(record, namespace=self.config.namespace)
            self._validate_dimension(record.vector)
            vectors.append(
                {
                    "id": record.record_id,
                    "values": list(record.vector),
                    "metadata": {"content": record.text, **metadata},
                }
            )
        return dict(
            self.transport(
                "POST",
                f"{self.config.endpoint}/vectors/upsert",
                self._headers(),
                {"namespace": self.config.namespace, "vectors": vectors},
                20.0,
            )
        )

    def search(self, request: VectorSearchRequest) -> list[VectorSearchResult]:
        self._require_network()
        self._validate_dimension(request.query_vector)
        payload = self.transport(
            "POST",
            f"{self.config.endpoint}/query",
            self._headers(),
            self.build_search_payload(request),
            10.0,
        )
        return [
            VectorSearchResult(
                record_id=str(row.get("id") or ""),
                score=float(row.get("score") or 0.0),
                text=str((row.get("metadata") or {}).get("content") or ""),
                metadata=row.get("metadata") or {},
                provider=self.provider,
            )
            for row in (payload.get("matches") or [])
        ]

    def build_search_payload(self, request: VectorSearchRequest) -> dict[str, Any]:
        return {
            "namespace": self.config.namespace,
            "vector": list(request.query_vector),
            "topK": request.top_k,
            "includeMetadata": True,
            "includeValues": False,
            "filter": {
                "$and": [
                    {"clinical_validation": {"$eq": False}},
                    {"patient_facing": {"$eq": True}},
                    {"data_scope": {"$eq": "curated_non_patient_kb"}},
                    {"source_tier": {"$in": list(request.allowed_tiers)}},
                    {"allowed_use": {"$in": list(request.allowed_uses)}},
                ]
            },
        }

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Api-Key": self.config.credential,
            "X-Pinecone-Api-Version": self.config.api_version,
        }

    def _require_network(self) -> None:
        if not self.config.configured or not self.config.allow_network:
            raise VectorStoreError("Pinecone is shadow-only and network execution is disabled.")

    def _validate_dimension(self, vector: tuple[float, ...]) -> None:
        if len(vector) != self.config.embedding_dimension:
            raise VectorStoreError("Pinecone embedding dimension does not match the configured index.")


def _json_transport(
    method: str,
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout: float,
) -> Mapping[str, Any]:
    request = urllib.request.Request(
        url,
        method=method,
        headers=dict(headers),
        data=json.dumps(payload).encode("utf-8"),
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise VectorStoreError(f"Managed vector request failed ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise VectorStoreError(f"Managed vector request failed: {exc.reason}") from exc
    return json.loads(body) if body else {}


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _escape_filter_literal(value: str) -> str:
    return str(value).replace("'", "''").replace(",", "")


def _positive_int(value: str | None, *, default: int) -> int:
    try:
        parsed = int(str(value or default))
    except ValueError as exc:
        raise VectorStoreError("Vector dimension must be an integer.") from exc
    if parsed <= 0:
        raise VectorStoreError("Vector dimension must be positive.")
    return parsed


__all__ = [
    "ALLOWED_PROVIDERS",
    "AzureAISearchAdapter",
    "BANNED_METADATA_KEYS",
    "CANONICAL_METADATA_FIELDS",
    "InMemoryVectorStore",
    "ManagedVectorConfig",
    "PineconeAdapter",
    "REMOTE_SAFE_NAMESPACES",
    "VectorRecord",
    "VectorSearchRequest",
    "VectorSearchResult",
    "VectorStoreAdapter",
    "VectorStoreError",
    "build_managed_adapter",
    "flatten_metadata",
    "load_managed_vector_config",
    "validate_remote_record",
]
