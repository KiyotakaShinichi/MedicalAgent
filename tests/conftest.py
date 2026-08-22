"""Pytest bootstrap and the default test-environment contract.

Two responsibilities:

1. Put the repository root on ``sys.path``. Some Windows environments invoke
   ``pytest`` through a console script that does not, and the release gate
   documents ``pytest tests/test_breast_monitoring.py -q`` as the smoke
   command, so that exact entrypoint has to stay reliable.

2. Enforce that the default suite is **hermetic**: no outbound network, no
   third-party credentials. A fresh clone with no `.env`, no API keys, and no
   running Ollama/n8n/MLflow must produce the same results as a developer
   machine that happens to have all of them.

Why the egress guard exists
---------------------------
The backend contains real adapters for Groq, Ollama, Pinecone, Azure Search,
n8n, cBioPortal, and MLflow. Each is *supposed* to be gated behind
configuration, but "supposed to be" is not a contract — a single call site
that forgets its guard turns the suite into something that passes only where
those services answer, and fails or hangs everywhere else. Blocking egress
converts that from a silent environmental dependency into an immediate,
attributable test failure.

This blocks **transport**, never behaviour. No safety policy, DEP-001
classifier, evaluator, or medical logic is stubbed here: those are exercised
for real. A test that genuinely needs the network marks itself `network` and
is deselected from the default run by `pytest.ini`.
"""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Credentials that must never be required by the default suite. They are
# cleared rather than merely ignored so that a stray live code path fails on a
# missing key instead of quietly authenticating against a real account with a
# developer's credentials.
_THIRD_PARTY_CREDENTIAL_VARS = (
    "GROQ_API_KEY",
    "PINECONE_API_KEY",
    "AZURE_SEARCH_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "N8N_WEBHOOK_URL",
    "N8N_API_KEY",
    "MLFLOW_TRACKING_URI",
    "OLLAMA_BASE_URL",
)

# Offline flags the safety runtimes rely on. Set here as well as in CI so a
# local run without the CI environment behaves identically.
_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}


def _is_loopback(host: object) -> bool:
    """Loopback stays reachable: SQLite, local fixtures, and in-process servers."""
    if not isinstance(host, str):
        return False
    return host in {"localhost", "127.0.0.1", "::1", "0.0.0.0", ""}


class NetworkEgressBlocked(RuntimeError):
    """Raised when the default suite attempts a real outbound connection."""


@pytest.fixture(scope="session", autouse=True)
def _hermetic_test_environment() -> object:
    """Clear third-party credentials and block non-loopback egress."""
    saved = {name: os.environ.pop(name, None) for name in _THIRD_PARTY_CREDENTIAL_VARS}
    saved_offline = {k: os.environ.get(k) for k in _OFFLINE_ENV}
    os.environ.update(_OFFLINE_ENV)

    # Opt-out for the explicitly-marked network suite and for debugging.
    allow = os.environ.get("NLCARE_ALLOW_TEST_NETWORK", "").strip().lower() in {
        "1", "true", "yes", "on",
    }

    real_connect = socket.socket.connect
    real_create_connection = socket.create_connection

    def guarded_connect(self, address, *args, **kwargs):  # type: ignore[no-untyped-def]
        host = address[0] if isinstance(address, tuple) else address
        if not _is_loopback(host):
            raise NetworkEgressBlocked(
                f"Blocked outbound connection to {host!r}. The default test suite is "
                "hermetic: it must pass on a fresh clone with no credentials and no "
                "reachable third-party services. Either stub this call, or mark the "
                "test `@pytest.mark.network` (it will then be excluded from the "
                "default run)."
            )
        return real_connect(self, address, *args, **kwargs)

    def guarded_create_connection(address, *args, **kwargs):  # type: ignore[no-untyped-def]
        host = address[0] if isinstance(address, tuple) else address
        if not _is_loopback(host):
            raise NetworkEgressBlocked(f"Blocked outbound connection to {host!r}.")
        return real_create_connection(address, *args, **kwargs)

    if not allow:
        socket.socket.connect = guarded_connect  # type: ignore[method-assign]
        socket.create_connection = guarded_create_connection  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket.connect = real_connect  # type: ignore[method-assign]
        socket.create_connection = real_create_connection  # type: ignore[assignment]
        for name, value in saved.items():
            if value is not None:
                os.environ[name] = value
        for name, value in saved_offline.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
