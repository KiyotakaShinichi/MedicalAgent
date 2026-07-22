from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from backend.services.oidc_auth import (
    OIDCAuthError,
    authenticate_oidc_token,
    load_oidc_config,
    validate_oidc_config,
)


BASE_ENV = {
    "ENVIRONMENT": "production",
    "NLCARE_OIDC_ENABLED": "true",
    "NLCARE_OIDC_ISSUER": "https://identity.example",
    "NLCARE_OIDC_AUDIENCE": "nlcare-api",
    "NLCARE_OIDC_JWKS_URL": "https://identity.example/.well-known/jwks.json",
    "NLCARE_OIDC_ALGORITHMS": "RS256",
    "NLCARE_OIDC_ROLE_CLAIM": "roles",
    "NLCARE_OIDC_PATIENT_ID_CLAIM": "patient_id",
}


class _Key:
    def __init__(self, key):
        self.key = key


class _Client:
    def __init__(self, key):
        self.key = key

    def get_signing_key_from_jwt(self, _token):
        return _Key(self.key)


def _token(*, audience: str = "nlcare-api", roles=None, patient_id: str | None = "P001"):
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = datetime.now(timezone.utc)
    claims = {
        "iss": "https://identity.example",
        "aud": audience,
        "sub": "external-user-123",
        "iat": now,
        "exp": now + timedelta(minutes=5),
        "roles": roles or ["patient"],
    }
    if patient_id is not None:
        claims["patient_id"] = patient_id
    encoded = jwt.encode(claims, private, algorithm="RS256", headers={"kid": "test-key"})
    return encoded, private.public_key()


def test_strict_oidc_config_requires_https_and_rs256():
    env = {**BASE_ENV, "NLCARE_OIDC_JWKS_URL": "http://identity.example/keys", "NLCARE_OIDC_ALGORITHMS": "HS256"}
    issues = validate_oidc_config(load_oidc_config(env), strict=True)
    assert "only RS256 is permitted" in issues
    assert "strict-profile JWKS URL must use HTTPS" in issues


def test_valid_signed_patient_token_maps_role_and_patient_scope():
    token, public = _token()
    identity = authenticate_oidc_token(
        token,
        environment=BASE_ENV,
        jwks_client_factory=lambda _url: _Client(public),
    )
    assert identity.role == "patient"
    assert identity.patient_id == "P001"
    assert identity.subject == "external-user-123"


def test_wrong_audience_is_rejected_fail_closed():
    token, public = _token(audience="another-api")
    with pytest.raises(OIDCAuthError, match="Invalid OIDC bearer token"):
        authenticate_oidc_token(
            token,
            environment=BASE_ENV,
            jwks_client_factory=lambda _url: _Client(public),
        )


def test_patient_role_requires_explicit_patient_mapping_claim():
    token, public = _token(patient_id=None)
    with pytest.raises(OIDCAuthError, match="patient identifier"):
        authenticate_oidc_token(
            token,
            environment=BASE_ENV,
            jwks_client_factory=lambda _url: _Client(public),
        )


def test_ambiguous_application_roles_are_rejected():
    token, public = _token(roles=["patient", "admin"])
    with pytest.raises(OIDCAuthError, match="exactly one"):
        authenticate_oidc_token(
            token,
            environment=BASE_ENV,
            jwks_client_factory=lambda _url: _Client(public),
        )
