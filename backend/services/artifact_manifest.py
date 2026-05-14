"""Reproducibility manifest helpers for evaluation artifacts.

Every artifact written by an eval or monitoring runner should carry a
``reproducibility`` block that captures the exact state of the world at
generation time. Without this, comparing two artifacts is guesswork: you
cannot tell whether a metric shift came from code, data, KB, or model changes.

The block is best-effort and intentionally account-free. It captures git state,
Python/runtime details, dataset fingerprints, KB/model fingerprints, seeds, and
freshness TTLs. Missing files are represented as ``None`` rather than failing
the evaluation runner.
"""

from __future__ import annotations

import hashlib
import platform as platform_module
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping


DEFAULT_FRESHNESS_TTL_SECONDS = 24 * 60 * 60

DEFAULT_DATASET_PATHS: dict[str, str] = {
    "synthetic_training_rows": "Data/complete_synthetic_breast_journeys/temporal_ml_rows.csv",
    "synthetic_model_metrics": "Data/complete_synthetic_training/complete_synthetic_model_metrics.json",
}


def build_artifact_manifest(
    *,
    seed: int | None = None,
    dataset_paths: Mapping[str, str | Path] | None = None,
    ttl_seconds: int = DEFAULT_FRESHNESS_TTL_SECONDS,
) -> dict:
    """Return a reproducibility + freshness manifest block."""

    now = datetime.now(timezone.utc)
    merged_paths: dict[str, str | Path] = dict(DEFAULT_DATASET_PATHS)
    merged_paths.update(dataset_paths or {})
    fingerprints: dict[str, str | None] = {
        label: _file_fingerprint(path)
        for label, path in merged_paths.items()
    }

    return {
        "reproducibility": {
            "git_commit": _git_commit(),
            "git_dirty": _git_dirty(),
            "python_version": _python_version(),
            "platform": _platform_descriptor(),
            "dataset_fingerprints": fingerprints,
            "knowledge_base_fingerprint": _knowledge_base_fingerprint(),
            "model_registry_fingerprint": _model_registry_fingerprint(),
            "seed": seed,
            "generated_at": now.isoformat(),
        },
        "artifact_freshness": {
            "generated_at": now.isoformat(),
            "ttl_seconds": ttl_seconds,
            "expires_at": (now + timedelta(seconds=ttl_seconds)).isoformat(),
            "status": "fresh",
        },
    }


def freshness_status(
    generated_at: str | None,
    ttl_seconds: int = DEFAULT_FRESHNESS_TTL_SECONDS,
) -> str:
    """Classify an artifact age as ``fresh``, ``stale``, or ``unknown``."""

    if not generated_at:
        return "unknown"
    try:
        ts = datetime.fromisoformat(generated_at)
    except (TypeError, ValueError):
        return "unknown"
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - ts).total_seconds()
    if age < 0:
        return "fresh"
    return "fresh" if age <= ttl_seconds else "stale"


def _file_fingerprint(path: str | Path) -> str | None:
    """Return ``size:mtime:sha256_prefix`` for a file, or ``None`` if missing."""

    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return None
    stat = file_path.stat()
    try:
        digest = hashlib.sha256()
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
        sha256_prefix = digest.hexdigest()[:12]
    except OSError:
        sha256_prefix = "unreadable"
    return f"{stat.st_size}:{int(stat.st_mtime)}:{sha256_prefix}"


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


def _python_version() -> str:
    info = sys.version_info
    return f"{info.major}.{info.minor}.{info.micro}"


def _platform_descriptor() -> str:
    return f"{platform_module.system()}-{platform_module.machine()}"


def _knowledge_base_fingerprint() -> str | None:
    candidates = [
        Path("Data/rag_index/local_hybrid_rag_index.joblib"),
        Path("KnowledgeBase/rag_chunks.json"),
        Path("KnowledgeBase/processed/rag_chunks.json"),
        Path("Data/rag_chunks.json"),
    ]
    for candidate in candidates:
        fingerprint = _file_fingerprint(candidate)
        if fingerprint:
            return f"{candidate.name}:{fingerprint}"
    return None


def _model_registry_fingerprint() -> str | None:
    candidates = [
        Path("Data/model_registry/registry.json"),
        Path("Data/complete_synthetic_training/complete_synthetic_model_metrics.json"),
    ]
    for candidate in candidates:
        fingerprint = _file_fingerprint(candidate)
        if fingerprint:
            return f"{candidate.name}:{fingerprint}"
    return None
