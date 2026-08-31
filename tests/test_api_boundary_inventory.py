"""Every mutating route has an explicit, auditable validation boundary."""

from __future__ import annotations

from backend.api.main import app
from backend.services.api_boundary_inventory import build_mutating_boundary_inventory


def test_mutating_boundary_inventory_is_complete_and_machine_readable() -> None:
    inventory = build_mutating_boundary_inventory(app.openapi())
    openapi = app.openapi()
    expected = sum(
        method in path_item
        for path_item in openapi["paths"].values()
        for method in ("post", "put", "patch", "delete")
    )

    assert len(inventory) == expected
    assert inventory == sorted(inventory, key=lambda row: (row["path"], row["method"]))
    assert all(
        row["classification"]
        in {
            "typed_request_body",
            "query_or_path_only",
            "multipart_or_file_upload",
            "explicit_raw_body_exception",
        }
        for row in inventory
    )


def test_raw_body_exception_is_unique_and_justified() -> None:
    inventory = build_mutating_boundary_inventory(app.openapi())
    raw = [row for row in inventory if row["classification"] == "explicit_raw_body_exception"]

    assert raw == [
        {
            "path": "/admin/automation/delivery-receipts",
            "method": "POST",
            "classification": "explicit_raw_body_exception",
            "request_schema": None,
            "justification": (
                "The HMAC signature covers the exact request bytes before the validated "
                "receipt object is constructed. Parsing first would invalidate the security contract."
            ),
        }
    ]


def test_typed_request_bodies_publish_a_schema() -> None:
    inventory = build_mutating_boundary_inventory(app.openapi())
    typed = [row for row in inventory if row["classification"] == "typed_request_body"]
    assert typed
    assert all(row["request_schema"] for row in typed)
