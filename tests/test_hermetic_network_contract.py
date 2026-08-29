"""Executable contract for the default offline test transport boundary."""

from __future__ import annotations

import os
import socket

import pytest

from conftest import NetworkEgressBlocked, network_block_required


def test_canonical_offline_mode_and_provider_credentials() -> None:
    assert os.environ["NLCARE_TEST_OFFLINE"] == "true"
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    for name in ("GROQ_API_KEY", "OPENAI_API_KEY", "PINECONE_API_KEY", "N8N_API_KEY"):
        assert name not in os.environ


def test_external_socket_attempt_fails_with_target() -> None:
    with pytest.raises(NetworkEgressBlocked, match="example.com"):
        socket.create_connection(("example.com", 443), timeout=0.01)


def test_external_dns_attempt_fails_with_target() -> None:
    with pytest.raises(NetworkEgressBlocked, match="example.com"):
        socket.getaddrinfo("example.com", 443)


def test_loopback_dns_and_socket_remain_available() -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    try:
        client = socket.create_connection(listener.getsockname(), timeout=1)
        server, _ = listener.accept()
        client.close()
        server.close()
    finally:
        listener.close()


def test_requires_network_marker_explicitly_opts_out() -> None:
    class Node:
        def get_closest_marker(self, name: str):
            return object() if name == "requires_network" else None

    assert network_block_required(Node()) is False  # type: ignore[arg-type]
