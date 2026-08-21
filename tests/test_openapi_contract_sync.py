"""Contract-synchronisation regression tests for the OpenAPI surface.

Background
----------
`Data/openapi.json` and `frontend-react/src/types/generated-openapi.d.ts` are
generated artifacts derived from the FastAPI app. They drifted 26 endpoints
behind the implementation because nothing failed fast when they went stale —
the only detection lived in `quality-gates` and Ship Gate, which run late.

These tests move that detection into the offline suite. They *strengthen* the
existing gates rather than replacing them: the CI drift checks are unchanged
and still authoritative for the generated TypeScript.

They also pin the one genuine implementation defect found during that
remediation: two child routers repeated their parent's tag, so 26 operations
were emitted with a duplicated tag such as `["patient", "patient"]`.
"""

from __future__ import annotations

import json
from pathlib import Path

from backend.api.main import app

ROOT = Path(__file__).resolve().parents[1]
COMMITTED_SCHEMA = ROOT / "Data" / "openapi.json"

HTTP_METHODS = {"get", "post", "put", "patch", "delete", "options", "head", "trace"}


def _operations(spec: dict) -> list[tuple[str, str, dict]]:
    return [
        (method.upper(), route, operation)
        for route, item in spec["paths"].items()
        for method, operation in item.items()
        if method in HTTP_METHODS
    ]


def test_no_operation_declares_duplicate_tags() -> None:
    """A child router must not repeat a tag its parent already declares.

    FastAPI concatenates parent and child tags, so declaring the same tag in
    both places emits it twice. Harmless at runtime, but it corrupts the
    published contract and the generated client grouping.
    """
    offenders = [
        f"{method} {route} -> {operation.get('tags')}"
        for method, route, operation in _operations(app.openapi())
        if len(operation.get("tags") or []) != len(set(operation.get("tags") or []))
    ]
    assert not offenders, "operations with duplicated tags:\n  " + "\n  ".join(offenders)


def test_openapi_generation_is_deterministic() -> None:
    """Two generations of the same app must be byte-identical.

    A non-deterministic generator would make the committed artifact
    unmaintainable: every regeneration would produce spurious drift.
    """
    first = json.dumps(app.openapi(), indent=2, sort_keys=True)
    app.openapi_schema = None  # force a full rebuild rather than the cached copy
    second = json.dumps(app.openapi(), indent=2, sort_keys=True)
    app.openapi_schema = None
    assert first == second


def test_committed_schema_matches_the_implementation() -> None:
    """`Data/openapi.json` must be regenerated whenever the API changes.

    This is the fast-failing counterpart to the `quality-gates` and Ship Gate
    drift checks. If it fails, regenerate rather than editing the artifact:

        python scripts/export_openapi_schema.py
        cd frontend-react && npm run typegen:file
    """
    assert COMMITTED_SCHEMA.is_file(), "Data/openapi.json is missing"
    committed = json.loads(COMMITTED_SCHEMA.read_text(encoding="utf-8"))
    live = app.openapi()

    live_ops = {(m, r) for m, r, _ in _operations(live)}
    committed_ops = {(m, r) for m, r, _ in _operations(committed)}

    missing = sorted(live_ops - committed_ops)
    extra = sorted(committed_ops - live_ops)
    assert not missing, f"endpoints implemented but absent from Data/openapi.json: {missing}"
    assert not extra, f"endpoints in Data/openapi.json but no longer implemented: {extra}"

    # Full structural equality, not just the route surface — a changed request
    # or response schema is drift too.
    assert json.dumps(live, sort_keys=True) == json.dumps(committed, sort_keys=True), (
        "Data/openapi.json is stale; regenerate with scripts/export_openapi_schema.py"
    )


# Liveness/readiness probes sit outside the domain tag groups by design: they
# are infrastructure endpoints, not part of the patient/clinician/admin API
# surface, and they carried no tag before the duplicate-tag remediation too.
UNTAGGED_INFRA_ROUTES = {"/health", "/ready"}


def test_every_domain_operation_carries_at_least_one_tag() -> None:
    """Untagged operations fall out of the generated client's grouping.

    Guards the fix for the duplicate-tag defect: the redundant tag was removed
    from two child routers, which is only correct because the parent router
    still supplies one. If a parent ever loses its tag, these operations would
    silently become untagged and this test fails.
    """
    untagged = [
        f"{method} {route}"
        for method, route, operation in _operations(app.openapi())
        if not operation.get("tags") and route not in UNTAGGED_INFRA_ROUTES
    ]
    assert not untagged, "domain operations with no tag:\n  " + "\n  ".join(untagged)
