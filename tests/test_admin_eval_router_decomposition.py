from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from backend.api.routers.admin_eval import build_admin_eval_router
from backend.api.routers.admin_eval_core import build_admin_eval_core_router
from backend.api.routers.admin_eval_data_imaging import build_admin_eval_data_imaging_router
from backend.api.routers.admin_eval_lifecycle import build_admin_eval_lifecycle_router
from backend.api.routers.admin_eval_medical_data import build_admin_eval_medical_data_router
from backend.api.routers.admin_eval_ml import build_admin_eval_ml_router
from backend.api.routers.admin_eval_observability import build_admin_observability_router
from backend.api.routers.admin_eval_rag import build_admin_eval_rag_router
from backend.api.routers.admin_eval_reporting import build_admin_eval_reporting_router


EXPECTED_CONTRACT_SHA256 = "a250f1de8ea6c05321b1505bfe69f12a4d38f16b1dcc9e6511e6e50cfe3ac590"


def admin_dep() -> dict[str, str]:
    return {"role": "admin"}


def db_dep():
    yield object()


def _build_app(admin_dependency: Callable = admin_dep) -> FastAPI:
    app = FastAPI()
    app.include_router(build_admin_eval_router(admin_dependency, db_dep))
    return app


def _api_routes(app: FastAPI) -> list[APIRoute]:
    return [
        route
        for route in app.routes
        if isinstance(route, APIRoute) and route.path.startswith("/admin/")
    ]


def _normalized_contract(app: FastAPI) -> list[dict[str, object]]:
    schema = app.openapi()
    inventory: list[dict[str, object]] = []
    for route in _api_routes(app):
        for method in sorted(value.lower() for value in route.methods if value not in {"HEAD", "OPTIONS"}):
            operation = schema["paths"][route.path][method]
            inventory.append(
                {
                    "method": method.upper(),
                    "path": route.path,
                    "name": route.name,
                    "explicit_operation_id": route.operation_id,
                    "openapi_operation_id": operation.get("operationId"),
                    "tags": route.tags,
                    "status_code": route.status_code,
                    "response_model": repr(route.response_model),
                    "dependencies": [
                        getattr(dependency.call, "__name__", repr(dependency.call))
                        for dependency in route.dependant.dependencies
                    ],
                    "request_body": operation.get("requestBody"),
                    "responses": operation.get("responses"),
                    "parameters": operation.get("parameters", []),
                }
            )
    return inventory


def _endpoint(router, path: str, method: str) -> Callable:
    return next(
        route.endpoint
        for route in router.routes
        if isinstance(route, APIRoute) and route.path == path and method in route.methods
    )


def test_facade_preserves_complete_pre_refactor_openapi_contract() -> None:
    inventory = _normalized_contract(_build_app())
    payload = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()

    assert len(inventory) == 103
    assert len({(row["method"], row["path"]) for row in inventory}) == 103
    assert hashlib.sha256(payload).hexdigest() == EXPECTED_CONTRACT_SHA256


def test_focused_routers_own_each_original_route_exactly_once() -> None:
    router_counts = [
        (build_admin_observability_router(admin_dep, db_dep), 4),
        (build_admin_eval_core_router(admin_dep, db_dep), 18),
        (build_admin_eval_data_imaging_router(admin_dep), 14),
        (build_admin_eval_reporting_router(admin_dep, db_dep), 15),
        (build_admin_eval_ml_router(admin_dep, db_dep), 12),
        (build_admin_eval_rag_router(admin_dep), 11),
        (build_admin_eval_lifecycle_router(admin_dep), 18),
        (build_admin_eval_medical_data_router(admin_dep), 11),
    ]

    contracts: list[tuple[str, str]] = []
    for router, expected_count in router_counts:
        owned = [
            (route.path, method)
            for route in router.routes
            for method in route.methods
            if method not in {"HEAD", "OPTIONS"}
        ]
        assert len(owned) == expected_count
        contracts.extend(owned)

    assert len(contracts) == len(set(contracts)) == 103


def test_every_admin_eval_route_keeps_the_injected_admin_dependency() -> None:
    app = _build_app()

    for route in _api_routes(app):
        dependency_calls = {dependency.call for dependency in route.dependant.dependencies}
        assert admin_dep in dependency_calls, route.path


def test_every_admin_eval_route_fails_before_handler_for_non_admin_access() -> None:
    def deny_admin_access() -> None:
        raise HTTPException(status_code=403, detail="admin role required")

    app = _build_app(deny_admin_access)
    client = TestClient(app)

    for route in _api_routes(app):
        path = re.sub(r"\{[^}]+\}", "sample-artifact", route.path)
        for method in route.methods - {"HEAD", "OPTIONS"}:
            response = client.request(method, path)
            assert response.status_code == 403, (method, path, response.text)


@pytest.mark.parametrize(
    ("service_module", "service_name", "path", "message"),
    [
        (
            "backend.services.agent_regression_eval",
            "run_agent_regression_suite",
            "/admin/agent-regression",
            "Agent regression suite completed.",
        ),
        (
            "backend.services.public_data_manifest",
            "build_public_data_manifest",
            "/admin/public-data-manifest",
            "Public data manifest rebuilt.",
        ),
        (
            "backend.services.evaluation_narrative_report",
            "build_ai_ml_narrative_report",
            "/admin/ai-ml-narrative-report",
            "AI/ML narrative report generated.",
        ),
        (
            "backend.services.biomarker_feature_benchmark",
            "run_biomarker_feature_benchmark",
            "/admin/biomarker-feature-benchmark",
            "Biomarker/tumor-marker feature benchmark completed.",
        ),
        (
            "backend.services.kb_source_governance",
            "build_kb_source_governance",
            "/admin/kb-source-governance",
            "KB source governance rebuilt.",
        ),
        (
            "backend.services.synthetic_generator_card",
            "build_synthetic_generator_card",
            "/admin/synthetic-generator-card",
            "Generator card refreshed.",
        ),
        (
            "backend.services.medical_safety_contract",
            "build_medical_safety_contract",
            "/admin/medical-safety-contract",
            "Medical safety contract generated.",
        ),
    ],
)
def test_representative_endpoint_payloads_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
    service_module: str,
    service_name: str,
    path: str,
    message: str,
) -> None:
    module = __import__(service_module, fromlist=[service_name])
    sentinel = {"artifact": path}
    monkeypatch.setattr(module, service_name, lambda *args, **kwargs: sentinel)
    endpoint = _endpoint(build_admin_eval_router(admin_dep, db_dep), path, "POST")

    assert endpoint(context=admin_dep()) == {"message": message, "result": sentinel}


def test_endpoint_service_failures_still_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.services import agent_regression_eval

    def fail() -> None:
        raise RuntimeError("evaluation failed")

    monkeypatch.setattr(agent_regression_eval, "run_agent_regression_suite", fail)
    endpoint = _endpoint(
        build_admin_eval_router(admin_dep, db_dep),
        "/admin/agent-regression",
        "POST",
    )

    with pytest.raises(RuntimeError, match="evaluation failed"):
        endpoint(context=admin_dep())
