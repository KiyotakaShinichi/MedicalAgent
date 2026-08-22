"""Suite-local deterministic environment, applied before production imports."""

import os


def configure_breast_monitoring_test_environment() -> None:
    os.environ.setdefault("RAG_FORCE_SPARSE", "1")
    os.environ.setdefault("ONCOTRACK_FAST_MODE", "1")


configure_breast_monitoring_test_environment()
