"""Machine-readable inventory of mutating HTTP validation boundaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


MUTATING_METHODS = ("post", "put", "patch", "delete")
RAW_BODY_JUSTIFICATIONS = {
    ("/admin/automation/delivery-receipts", "post"): (
        "The HMAC signature covers the exact request bytes before the validated "
        "receipt object is constructed. Parsing first would invalidate the security contract."
    ),
}


@dataclass(frozen=True)
class BoundaryEntry:
    path: str
    method: str
    classification: str
    request_schema: str | None
    justification: str | None = None


def build_mutating_boundary_inventory(
    openapi: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Classify every mutating operation from its published request contract."""
    entries: list[dict[str, Any]] = []
    for path, path_item in sorted(openapi.get("paths", {}).items()):
        if not isinstance(path_item, Mapping):
            continue
        for method in MUTATING_METHODS:
            operation = path_item.get(method)
            if not isinstance(operation, Mapping):
                continue
            override = RAW_BODY_JUSTIFICATIONS.get((path, method))
            request_body = operation.get("requestBody")
            if override:
                classification = "explicit_raw_body_exception"
                schema = None
                justification = override
            elif not isinstance(request_body, Mapping):
                classification = "query_or_path_only"
                schema = None
                justification = None
            else:
                content = request_body.get("content", {})
                if "multipart/form-data" in content:
                    classification = "multipart_or_file_upload"
                    media = content["multipart/form-data"]
                else:
                    classification = "typed_request_body"
                    media = content.get("application/json", {})
                schema_payload = media.get("schema", {}) if isinstance(media, Mapping) else {}
                schema = _schema_name(schema_payload)
                justification = None
            entries.append(
                asdict(
                    BoundaryEntry(
                        path=path,
                        method=method.upper(),
                        classification=classification,
                        request_schema=schema,
                        justification=justification,
                    )
                )
            )
    return entries


def _schema_name(schema: Any) -> str | None:
    if not isinstance(schema, Mapping):
        return None
    reference = schema.get("$ref")
    if isinstance(reference, str):
        return reference.rsplit("/", 1)[-1]
    if schema:
        return str(schema.get("type") or "inline_schema")
    return None


__all__ = [
    "BoundaryEntry",
    "MUTATING_METHODS",
    "RAW_BODY_JUSTIFICATIONS",
    "build_mutating_boundary_inventory",
]
