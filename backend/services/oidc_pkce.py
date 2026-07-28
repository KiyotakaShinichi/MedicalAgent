"""Browser OIDC Authorization Code + PKCE primitives and readiness checks.

This module prepares a standards-aligned flow but deliberately does not claim
that an identity provider, consent screen, callback exchange, or logout path
has been demonstrated in this repository.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping
from urllib.parse import urlencode, urlparse


OUTPUT_PATH = Path("Data/evals/ops/latest_oidc_browser_pkce_readiness.json")


@dataclass(frozen=True)
class OIDCBrowserConfig:
    authorization_endpoint: str
    token_endpoint: str
    client_id: str
    redirect_uri: str
    scopes: tuple[str, ...]


@dataclass(frozen=True)
class PKCETransaction:
    state: str
    nonce: str
    code_verifier: str
    code_challenge: str
    authorization_url: str
    created_at: str


class OIDCPKCEError(ValueError):
    pass


def _urlsafe(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def load_browser_oidc_config(environment: Mapping[str, str] | None = None) -> OIDCBrowserConfig:
    env = os.environ if environment is None else environment
    scopes = tuple(item for item in str(env.get("NLCARE_OIDC_SCOPES") or "openid profile").split() if item)
    return OIDCBrowserConfig(
        authorization_endpoint=str(env.get("NLCARE_OIDC_AUTHORIZATION_ENDPOINT") or "").strip(),
        token_endpoint=str(env.get("NLCARE_OIDC_TOKEN_ENDPOINT") or "").strip(),
        client_id=str(env.get("NLCARE_OIDC_CLIENT_ID") or "").strip(),
        redirect_uri=str(env.get("NLCARE_OIDC_REDIRECT_URI") or "").strip(),
        scopes=scopes,
    )


def validate_browser_oidc_config(config: OIDCBrowserConfig, *, strict: bool = True) -> list[str]:
    issues: list[str] = []
    for label, value in (
        ("authorization endpoint", config.authorization_endpoint),
        ("token endpoint", config.token_endpoint),
        ("client ID", config.client_id),
        ("redirect URI", config.redirect_uri),
    ):
        if not value:
            issues.append(f"{label} is required")
    if "openid" not in config.scopes:
        issues.append("openid scope is required")
    if strict:
        for label, value in (
            ("authorization endpoint", config.authorization_endpoint),
            ("token endpoint", config.token_endpoint),
            ("redirect URI", config.redirect_uri),
        ):
            if value and urlparse(value).scheme != "https":
                issues.append(f"{label} must use HTTPS")
    return issues


def create_pkce_transaction(config: OIDCBrowserConfig) -> PKCETransaction:
    issues = validate_browser_oidc_config(config, strict=True)
    if issues:
        raise OIDCPKCEError("Unsafe or incomplete OIDC browser configuration")
    verifier = _urlsafe(secrets.token_bytes(48))
    challenge = _urlsafe(hashlib.sha256(verifier.encode("ascii")).digest())
    state = _urlsafe(secrets.token_bytes(32))
    nonce = _urlsafe(secrets.token_bytes(32))
    parameters = {
        "response_type": "code",
        "client_id": config.client_id,
        "redirect_uri": config.redirect_uri,
        "scope": " ".join(config.scopes),
        "state": state,
        "nonce": nonce,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    return PKCETransaction(
        state=state,
        nonce=nonce,
        code_verifier=verifier,
        code_challenge=challenge,
        authorization_url=f"{config.authorization_endpoint}?{urlencode(parameters)}",
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def validate_callback(*, expected_state: str, received_state: str | None, code: str | None, error: str | None = None) -> str:
    if error:
        raise OIDCPKCEError("Identity provider returned an authorization error")
    if not received_state or not hmac.compare_digest(expected_state, received_state):
        raise OIDCPKCEError("OIDC callback state mismatch")
    if not code or not code.strip():
        raise OIDCPKCEError("OIDC callback authorization code is missing")
    return code.strip()


def build_oidc_browser_pkce_readiness(
    output_path: Path = OUTPUT_PATH,
    *,
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    config = load_browser_oidc_config(environment)
    issues = validate_browser_oidc_config(config, strict=True)
    configured = not issues
    report: dict[str, object] = {
        "schema_version": "oidc_browser_pkce_readiness_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "prepared_not_integrated" if configured else "blocked_configuration",
        "clinical_validation": False,
        "configuration": {**asdict(config), "client_id": "configured" if config.client_id else "missing"},
        "configuration_issues": issues,
        "implemented_primitives": [
            "authorization code request",
            "S256 PKCE verifier and challenge",
            "cryptographic state and nonce",
            "constant-time callback state validation",
            "HTTPS strict-profile validation",
        ],
        "not_demonstrated": [
            "live identity-provider login",
            "server-side transaction persistence",
            "authorization-code token exchange",
            "refresh-token rotation",
            "provider logout and session revocation",
            "consent or identity proofing",
        ],
        "browser_login_completed": False,
        "production_auth_ready": False,
        "claim_boundary": (
            "PKCE primitives and readiness checks are engineering preparation only. No live provider flow, "
            "healthcare compliance, clinical validation, or production authentication readiness is claimed."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


__all__ = [
    "OIDCBrowserConfig",
    "OIDCPKCEError",
    "PKCETransaction",
    "build_oidc_browser_pkce_readiness",
    "create_pkce_transaction",
    "load_browser_oidc_config",
    "validate_browser_oidc_config",
    "validate_callback",
]
