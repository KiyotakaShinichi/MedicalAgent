"""Compose admin evaluation API routers without changing their public contracts."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter

from backend.api.routers.admin_eval_core import build_admin_eval_core_router
from backend.api.routers.admin_eval_data_imaging import build_admin_eval_data_imaging_router
from backend.api.routers.admin_eval_lifecycle import build_admin_eval_lifecycle_router
from backend.api.routers.admin_eval_medical_data import build_admin_eval_medical_data_router
from backend.api.routers.admin_eval_ml import build_admin_eval_ml_router
from backend.api.routers.admin_eval_observability import build_admin_observability_router
from backend.api.routers.admin_eval_rag import build_admin_eval_rag_router
from backend.api.routers.admin_eval_reporting import build_admin_eval_reporting_router


def build_admin_eval_router(get_admin_access_context: Callable, get_db: Callable) -> APIRouter:
    """Build the compatibility facade for all admin evaluation routes."""
    router = APIRouter(tags=["admin-evaluation"])
    router.include_router(build_admin_observability_router(get_admin_access_context, get_db))
    router.include_router(build_admin_eval_core_router(get_admin_access_context, get_db))
    router.include_router(build_admin_eval_data_imaging_router(get_admin_access_context))
    router.include_router(build_admin_eval_reporting_router(get_admin_access_context, get_db))
    router.include_router(build_admin_eval_ml_router(get_admin_access_context, get_db))
    router.include_router(build_admin_eval_rag_router(get_admin_access_context))
    router.include_router(build_admin_eval_lifecycle_router(get_admin_access_context))
    router.include_router(build_admin_eval_medical_data_router(get_admin_access_context))
    return router
