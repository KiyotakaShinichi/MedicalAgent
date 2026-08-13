"""Authorization lock-ins for expensive and filesystem-capable model routes."""

from fastapi.routing import APIRoute
import pytest
from fastapi import HTTPException

from backend.api.main import app
from backend.api.routers.model import CSVImportRequest, import_csv_payload


ADMIN_ONLY_MODEL_ROUTES = {
    ("GET", "/import-schema"),
    ("POST", "/import-qin-breast-02"),
    ("POST", "/generate-qin-synthetic-cbc"),
    ("POST", "/generate-synthetic-breast-journeys"),
    ("POST", "/generate-temporal-synthetic-breast-journeys"),
    ("POST", "/generate-complete-synthetic-breast-dataset"),
    ("POST", "/train-complete-synthetic-models"),
    ("POST", "/generate-complete-synthetic-xai"),
    ("POST", "/index-qin-mri"),
    ("POST", "/build-qin-mri-manifest"),
    ("POST", "/preprocess-qin-mri-previews"),
    ("POST", "/inspect-breastdcedl"),
    ("POST", "/build-breastdcedl-manifest"),
    ("POST", "/run-breastdcedl-baseline"),
    ("POST", "/generate-breastdcedl-previews"),
    ("POST", "/import-breastdcedl-patients"),
    ("POST", "/run-breastdcedl-cnn"),
    ("POST", "/generate-breastdcedl-xai"),
    ("POST", "/models/breastdcedl/train-final"),
}


def _route_map() -> dict[tuple[str, str], APIRoute]:
    routes: dict[tuple[str, str], APIRoute] = {}
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        for method in route.methods or set():
            routes[(method, route.path)] = route
    return routes


def test_expensive_model_routes_require_admin_context() -> None:
    routes = _route_map()
    assert ADMIN_ONLY_MODEL_ROUTES <= routes.keys()

    for key in ADMIN_ONLY_MODEL_ROUTES:
        dependency_names = {
            dependency.call.__name__
            for dependency in routes[key].dependant.dependencies
            if dependency.call is not None
        }
        assert "get_admin_access_context" in dependency_names, key


def test_clinician_model_reads_and_predictions_keep_scoped_access() -> None:
    routes = _route_map()
    for key in {
        ("GET", "/models"),
        ("POST", "/models/breastdcedl/predict/{patient_id}"),
        ("GET", "/prediction-audits"),
    }:
        dependency_names = {
            dependency.call.__name__
            for dependency in routes[key].dependant.dependencies
            if dependency.call is not None
        }
        assert "get_clinician_or_admin_context" in dependency_names, key


def test_csv_import_api_rejects_server_file_paths_before_io() -> None:
    payload = CSVImportRequest(import_type="labs", file_path="../../secrets.txt")

    with pytest.raises(HTTPException) as exc_info:
        import_csv_payload(payload, context=object(), db=None)

    assert exc_info.value.status_code == 400
    assert "file_path imports are disabled" in str(exc_info.value.detail)
