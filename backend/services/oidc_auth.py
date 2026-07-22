"""Feature-flagged OIDC bearer validation for non-demo deployments.

Only signed access tokens from an explicitly configured issuer, audience, and
JWKS endpoint are accepted. This adapter does not implement browser login,
consent, identity proofing, or healthcare compliance.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping
from urllib.parse import urlparse


VALID_ROLES = {"patient", "clinician", "admin"}
ALLOWED_ALGORITHMS = {"RS256"}


class OIDCAuthError(PermissionError):
    """Fail-closed authentication error safe to translate to HTTP 401."""


@dataclass(frozen=True)
class OIDCConfig:
    enabled: bool
    issuer: str
    audience: str
    jwks_url: str
    algorithms: tuple[str, ...]
    role_claim: str
    patient_id_claim: str
    subject_claim: str
    role_map: dict[str, str]
    leeway_seconds: int


@dataclass(frozen=True)
class OIDCIdentity:
    role: str
    patient_id: str | None
    subject: str
    issuer: str


def _bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def load_oidc_config(environment: Mapping[str, str] | None = None) -> OIDCConfig:
    env = os.environ if environment is None else environment
    algorithms = tuple(
        item.strip().upper()
        for item in str(env.get("NLCARE_OIDC_ALGORITHMS") or "RS256").split(",")
        if item.strip()
    )
    raw_map = str(env.get("NLCARE_OIDC_ROLE_MAP_JSON") or "{}").strip()
    try:
        parsed_map = json.loads(raw_map)
    except json.JSONDecodeError as exc:
        raise OIDCAuthError("OIDC role map is not valid JSON") from exc
    if not isinstance(parsed_map, dict):
        raise OIDCAuthError("OIDC role map must be a JSON object")
    role_map = {str(key): str(value).lower() for key, value in parsed_map.items()}
    try:
        leeway = int(str(env.get("NLCARE_OIDC_LEEWAY_SECONDS") or "30"))
    except ValueError as exc:
        raise OIDCAuthError("OIDC leeway must be an integer") from exc
    return OIDCConfig(
        enabled=_bool(env.get("NLCARE_OIDC_ENABLED")),
        issuer=str(env.get("NLCARE_OIDC_ISSUER") or "").rstrip("/"),
        audience=str(env.get("NLCARE_OIDC_AUDIENCE") or "").strip(),
        jwks_url=str(env.get("NLCARE_OIDC_JWKS_URL") or "").strip(),
        algorithms=algorithms,
        role_claim=str(env.get("NLCARE_OIDC_ROLE_CLAIM") or "roles").strip(),
        patient_id_claim=str(env.get("NLCARE_OIDC_PATIENT_ID_CLAIM") or "patient_id").strip(),
        subject_claim=str(env.get("NLCARE_OIDC_SUBJECT_CLAIM") or "sub").strip(),
        role_map=role_map,
        leeway_seconds=max(0, min(leeway, 300)),
    )


def validate_oidc_config(config: OIDCConfig, *, strict: bool) -> list[str]:
    if not config.enabled:
        return ["OIDC is disabled"] if strict else []
    issues: list[str] = []
    if not config.issuer:
        issues.append("issuer is required")
    if not config.audience:
        issues.append("audience is required")
    if not config.jwks_url:
        issues.append("JWKS URL is required")
    if not config.role_claim:
        issues.append("role claim is required")
    if not config.subject_claim:
        issues.append("subject claim is required")
    if not config.algorithms or any(item not in ALLOWED_ALGORITHMS for item in config.algorithms):
        issues.append("only RS256 is permitted")
    if any(value not in VALID_ROLES for value in config.role_map.values()):
        issues.append("role map values must be patient, clinician, or admin")
    if strict:
        if urlparse(config.issuer).scheme != "https":
            issues.append("strict-profile issuer must use HTTPS")
        if urlparse(config.jwks_url).scheme != "https":
            issues.append("strict-profile JWKS URL must use HTTPS")
    return issues


def _claim_value(claims: Mapping[str, Any], path: str) -> Any:
    value: Any = claims
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _resolve_role(raw: Any, role_map: Mapping[str, str]) -> str:
    values = raw if isinstance(raw, list) else [raw]
    roles = {
        role_map.get(str(value), str(value).lower())
        for value in values
        if value is not None and str(value).strip()
    }
    roles &= VALID_ROLES
    if len(roles) != 1:
        raise OIDCAuthError("OIDC token must resolve to exactly one allowed application role")
    return next(iter(roles))


def authenticate_oidc_token(
    token: str,
    *,
    environment: Mapping[str, str] | None = None,
    jwks_client_factory: Callable[[str], Any] | None = None,
    decode_function: Callable[..., dict[str, Any]] | None = None,
) -> OIDCIdentity:
    config = load_oidc_config(environment)
    if not config.enabled:
        raise OIDCAuthError("OIDC authentication is disabled")
    profile = str((environment or os.environ).get("ENVIRONMENT") or (environment or os.environ).get("APP_ENV") or "development").lower()
    issues = validate_oidc_config(config, strict=profile in {"staging", "production", "prod"})
    if issues:
        raise OIDCAuthError("OIDC configuration is incomplete or unsafe")
    try:
        import jwt
        from jwt import PyJWKClient

        factory = jwks_client_factory or (lambda url: PyJWKClient(url, cache_jwk_set=True, lifespan=300))
        decoder = decode_function or jwt.decode
        signing_key = factory(config.jwks_url).get_signing_key_from_jwt(token).key
        claims = decoder(
            token,
            signing_key,
            algorithms=list(config.algorithms),
            audience=config.audience,
            issuer=config.issuer,
            leeway=config.leeway_seconds,
            options={"require": ["exp", "iat", config.subject_claim]},
        )
    except OIDCAuthError:
        raise
    except Exception as exc:  # PyJWT errors intentionally collapse to one boundary
        raise OIDCAuthError("Invalid OIDC bearer token") from exc

    role = _resolve_role(_claim_value(claims, config.role_claim), config.role_map)
    subject = str(_claim_value(claims, config.subject_claim) or "").strip()
    if not subject:
        raise OIDCAuthError("OIDC subject claim is missing")
    patient_id: str | None = None
    if role == "patient":
        patient_id = str(_claim_value(claims, config.patient_id_claim) or "").strip()
        if not patient_id:
            raise OIDCAuthError("Patient role requires a patient identifier claim")
    return OIDCIdentity(role=role, patient_id=patient_id, subject=subject, issuer=config.issuer)


__all__ = [
    "OIDCAuthError", "OIDCConfig", "OIDCIdentity", "authenticate_oidc_token",
    "load_oidc_config", "validate_oidc_config",
]
