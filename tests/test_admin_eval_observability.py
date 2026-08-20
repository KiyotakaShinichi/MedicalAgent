from __future__ import annotations

from backend.api.main import app
from backend.api.routers.admin_eval_observability import (
    _loads,
    build_admin_observability_router,
)


def _dependency_stub():
    yield None


def test_observability_router_owns_expected_contracts() -> None:
    router = build_admin_observability_router(_dependency_stub, _dependency_stub)
    contracts = {(route.path, method) for route in router.routes for method in route.methods}
    assert contracts == {
        ("/admin/agent-trace-logs", "GET"),
        ("/admin/rag-trace-replay", "GET"),
        ("/admin/system-health", "GET"),
        ("/admin/system-health", "POST"),
    }


def test_main_app_registers_each_observability_contract_once() -> None:
    contracts = [
        (route.path, method)
        for route in app.routes
        for method in getattr(route, "methods", set())
        if route.path in {
            "/admin/agent-trace-logs",
            "/admin/rag-trace-replay",
            "/admin/system-health",
        }
    ]
    assert len(contracts) == len(set(contracts)) == 4


def test_trace_json_loader_preserves_contract_defaults() -> None:
    assert _loads('["source-a"]', default=[]) == ["source-a"]
    assert _loads(None, default=[]) == []
    assert _loads("malformed", default=None) is None
