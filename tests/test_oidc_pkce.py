from urllib.parse import parse_qs, urlparse

import pytest

from backend.services.oidc_pkce import OIDCBrowserConfig, OIDCPKCEError, create_pkce_transaction, validate_callback
from backend.services.oidc_pkce import (
    InMemoryPKCETransactionStore,
    begin_pkce_login,
    consume_pkce_callback,
)


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


def test_pkce_store_hides_verifier_and_consumes_callback_once():
    store = InMemoryPKCETransactionStore()
    pending = begin_pkce_login(_config(), store)
    params = parse_qs(urlparse(pending.authorization_url).query)

    assert "code_verifier" not in pending.authorization_url
    code, verifier, nonce = consume_pkce_callback(
        store,
        transaction_id=pending.transaction_id,
        received_state=params["state"][0],
        code="authorization-code",
    )
    assert code == "authorization-code"
    assert len(verifier) >= 43
    assert nonce

    with pytest.raises(OIDCPKCEError, match="already used"):
        consume_pkce_callback(
            store,
            transaction_id=pending.transaction_id,
            received_state=params["state"][0],
            code="replay",
        )


def test_pkce_store_rejects_expired_transaction():
    clock = [1_000.0]
    store = InMemoryPKCETransactionStore(ttl_seconds=30, clock=lambda: clock[0])
    pending = begin_pkce_login(_config(), store)
    params = parse_qs(urlparse(pending.authorization_url).query)
    clock[0] += 31

    with pytest.raises(OIDCPKCEError, match="expired"):
        consume_pkce_callback(
            store,
            transaction_id=pending.transaction_id,
            received_state=params["state"][0],
            code="late",
        )


def test_state_mismatch_consumes_transaction_fail_closed():
    store = InMemoryPKCETransactionStore()
    pending = begin_pkce_login(_config(), store)

    with pytest.raises(OIDCPKCEError, match="state mismatch"):
        consume_pkce_callback(
            store,
            transaction_id=pending.transaction_id,
            received_state="wrong",
            code="code",
        )
    with pytest.raises(OIDCPKCEError, match="already used"):
        consume_pkce_callback(
            store,
            transaction_id=pending.transaction_id,
            received_state="wrong",
            code="code",
        )
