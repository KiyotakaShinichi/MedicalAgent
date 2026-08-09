"""Fail-closed upload inspection for the synthetic engineering prototype."""

from __future__ import annotations

import base64
import binascii
import hashlib
import os
import shlex
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping


ALLOWED_CONTENT_TYPES = {
    "application/pdf": {".pdf"},
    "image/jpeg": {".jpg", ".jpeg"},
    "image/png": {".png"},
    "text/plain": {".txt", ".md", ".csv", ".tsv"},
    "application/json": {".json"},
}


@dataclass(frozen=True)
class UploadSecurityPolicy:
    enabled: bool
    strict_profile: bool
    scanner_mode: str
    scanner_command: str
    scanner_timeout_seconds: int


@dataclass(frozen=True)
class UploadInspection:
    sha256: str
    size_bytes: int
    detected_content_type: str
    scanner_mode: str
    scanner_status: str
    synthetic_only: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_upload_security_policy(
    environment: Mapping[str, str] | None = None,
) -> UploadSecurityPolicy:
    env = os.environ if environment is None else environment
    profile = str(env.get("ENVIRONMENT") or env.get("APP_ENV") or "development").lower()
    strict = profile in {"staging", "production", "prod"}
    enabled_value = env.get("NLCARE_UPLOADS_ENABLED")
    enabled = (not strict) if enabled_value is None else _bool(enabled_value)
    mode = str(env.get("NLCARE_UPLOAD_SCANNER_MODE") or ("builtin" if not strict else "disabled")).lower()
    command = str(env.get("NLCARE_UPLOAD_SCANNER_COMMAND") or "").strip()
    try:
        timeout = int(str(env.get("NLCARE_UPLOAD_SCANNER_TIMEOUT_SECONDS") or "20"))
    except ValueError as exc:
        raise ValueError("Upload scanner timeout must be an integer") from exc
    policy = UploadSecurityPolicy(enabled, strict, mode, command, max(1, min(timeout, 120)))
    validate_upload_security_policy(policy)
    return policy


def validate_upload_security_policy(policy: UploadSecurityPolicy) -> None:
    if not policy.enabled:
        return
    if policy.scanner_mode not in {"builtin", "external"}:
        raise ValueError("Uploads require a configured scanner mode")
    if policy.strict_profile and policy.scanner_mode != "external":
        raise ValueError("Staging/production uploads require an external scanner")
    if policy.scanner_mode == "external" and not policy.scanner_command:
        raise ValueError("External upload scanner command is required")


def decode_upload_payload(payload: str) -> bytes:
    if not isinstance(payload, str) or not payload.strip():
        raise ValueError("Upload payload is empty")
    encoded = payload.split(",", 1)[1] if "," in payload else payload
    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Upload payload is not valid strict base64") from exc


def inspect_quarantined_upload(
    path: Path,
    *,
    file_name: str,
    declared_content_type: str | None,
    policy: UploadSecurityPolicy,
) -> UploadInspection:
    if not policy.enabled:
        raise ValueError("Uploads are disabled for this deployment profile")
    data = path.read_bytes()
    detected = _detect_content_type(data)
    suffix = Path(file_name).suffix.lower()
    if detected not in ALLOWED_CONTENT_TYPES:
        raise ValueError("Upload type is not allowed")
    if suffix not in ALLOWED_CONTENT_TYPES[detected]:
        raise ValueError("Upload filename extension does not match file content")
    declared = str(declared_content_type or "").split(";", 1)[0].strip().lower()
    if declared and declared != detected:
        raise ValueError("Declared content type does not match file content")
    _builtin_malicious_content_checks(data, detected)
    scanner_status = "builtin_passed"
    if policy.scanner_mode == "external":
        scanner_status = _run_external_scanner(path, policy)
    return UploadInspection(
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        detected_content_type=detected,
        scanner_mode=policy.scanner_mode,
        scanner_status=scanner_status,
    )


def _detect_content_type(data: bytes) -> str:
    if data.startswith(b"%PDF-"):
        return "application/pdf"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return "application/octet-stream"
    if "\x00" in text:
        return "application/octet-stream"
    stripped = text.lstrip()
    return "application/json" if stripped.startswith(("{", "[")) else "text/plain"


def _builtin_malicious_content_checks(data: bytes, content_type: str) -> None:
    lowered = data[:1_000_000].lower()
    if data.startswith((b"MZ", b"\x7fELF")) or b"<script" in lowered:
        raise ValueError("Executable or active content is not allowed")
    if content_type == "application/pdf" and any(
        marker in lowered for marker in (b"/javascript", b"/launch", b"/embeddedfile")
    ):
        raise ValueError("Active or embedded PDF content is not allowed")


def _run_external_scanner(path: Path, policy: UploadSecurityPolicy) -> str:
    command = [*shlex.split(policy.scanner_command), str(path)]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=policy.scanner_timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("Upload scanner unavailable or timed out") from exc
    if result.returncode != 0:
        raise ValueError("Upload scanner rejected the file")
    return "external_passed"


def _bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


__all__ = [
    "UploadInspection",
    "UploadSecurityPolicy",
    "decode_upload_payload",
    "inspect_quarantined_upload",
    "load_upload_security_policy",
    "validate_upload_security_policy",
]
