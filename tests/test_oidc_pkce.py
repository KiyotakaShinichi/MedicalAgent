from urllib.parse import parse_qs, urlparse

import pytest

from backend.services.oidc_pkce import OIDCBrowserConfig, OIDCPKCEError, create_pkce_transaction, validate_callback


def _config() -> OIDCBrowserConfig:
    return OIDCBrowserConfig(
        authorization_endpoint="https://identity.example/authorize",
        token_endpoint="https://identity.example/token",
        client_id="nlcare-browser",
        redirect_uri="https://nlcare.example/auth/callback",
        scopes=("openid", "profile"),
    )


def test_pkce_transaction_uses_s256_state_and_nonce():
    transaction = create_pkce_transaction(_config())
    params = parse_qs(urlparse(transaction.authorization_url).query)
    assert params["code_challenge_method"] == ["S256"]
    assert params["state"] == [transaction.state]
    assert params["nonce"] == [transaction.nonce]
    assert len(transaction.code_verifier) >= 43
    assert transaction.code_verifier != transaction.code_challenge


def test_callback_fails_closed_on_state_mismatch():
    with pytest.raises(OIDCPKCEError, match="state mismatch"):
        validate_callback(expected_state="expected", received_state="wrong", code="abc")


def test_callback_accepts_matching_state_and_code():
    assert validate_callback(expected_state="same", received_state="same", code=" abc ") == "abc"


def test_http_endpoints_are_rejected():
    bad = OIDCBrowserConfig("http://id/authorize", "https://id/token", "client", "https://app/cb", ("openid",))
    with pytest.raises(OIDCPKCEError):
        create_pkce_transaction(bad)
