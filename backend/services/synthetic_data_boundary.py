"""Runtime lock preventing strict prototype profiles from accepting unlabeled data."""

from __future__ import annotations

import os
import re

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


SYNTHETIC_PATIENT_ID = re.compile(
    r"^(?:P\d{3,}|TEST[-_].+|SYN[-_].+|DEMO[-_].+)$",
    re.IGNORECASE,
)
MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
EXEMPT_PATH_PREFIXES = ("/auth/", "/health", "/ready", "/docs", "/openapi.json")


def synthetic_only_enabled() -> bool:
    return _bool(os.environ.get("NLCARE_SYNTHETIC_ONLY"))


def assert_synthetic_patient_id(patient_id: str | None) -> None:
    if synthetic_only_enabled() and not SYNTHETIC_PATIENT_ID.fullmatch(str(patient_id or "")):
        raise PermissionError("Synthetic-only runtime rejected a non-synthetic patient namespace")


class SyntheticDataBoundaryMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # noqa: ANN001
        if not synthetic_only_enabled():
            return await call_next(request)
        if request.method not in MUTATING_METHODS or request.url.path.startswith(EXEMPT_PATH_PREFIXES):
            return await call_next(request)
        classification = request.headers.get("x-nlcare-data-class", "").strip().lower()
        if classification != "synthetic":
            return JSONResponse(
                status_code=422,
                content={
                    "detail": "This deployment accepts explicitly labeled synthetic data only.",
                    "required_data_class": "synthetic",
                    "clinical_validation": False,
                    "healthcare_production_ready": False,
                },
            )
        response = await call_next(request)
        response.headers.setdefault("X-NLCare-Data-Boundary", "synthetic_only")
        return response


def _bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


__all__ = [
    "SyntheticDataBoundaryMiddleware",
    "assert_synthetic_patient_id",
    "synthetic_only_enabled",
]
