"""Deterministic export of tracked buyer-candidate content."""

from __future__ import annotations

import fnmatch
import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from scripts.buyer.contracts import (
    ROOT,
    git_sha,
    load_json,
    tracked_files,
)


FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
MANIFEST_NAME = "BUYER_PACKAGE_MANIFEST.json"


class PackageError(ValueError):
    """Raised when a buyer package would violate its transfer policy."""


def _matches(path: str, patterns: list[str]) -> bool:
    normalized = PurePosixPath(path).as_posix()
    return any(
        fnmatch.fnmatch(normalized, pattern)
        or fnmatch.fnmatch(PurePosixPath(normalized).name, pattern)
        for pattern in patterns
    )


def selected_files(policy: dict[str, Any] | None = None) -> list[str]:
    policy = policy or load_json("config/buyer/package_policy.json")
    excluded = policy["excluded_patterns"]
    selected = [path for path in tracked_files() if not _matches(path, excluded)]
    forbidden = [path for path in selected if _matches(path, policy["forbidden_in_archive"])]
    if forbidden:
        raise PackageError(f"Forbidden files selected: {forbidden[:10]}")
    missing = [path for path in policy["required_paths"] if path not in selected]
    if missing:
        raise PackageError(f"Required package paths are missing: {missing}")
    protected = load_json("config/buyer/protected_evidence_manifest.json")["files"]
    omitted = [entry["path"] for entry in protected if entry["path"] not in selected]
    if omitted:
        raise PackageError(f"Protected evidence omitted: {omitted[:10]}")
    return sorted(selected)


def _tracked_snapshot(paths: list[str]) -> dict[str, bytes]:
    """Read HEAD once so package bytes are canonical without per-file Git processes."""
    tree = subprocess.check_output(["git", "ls-tree", "-r", "-z", "HEAD"], cwd=ROOT)
    wanted = set(paths)
    refs: dict[str, str] = {}
    for record in tree.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        _mode, object_type, raw_oid = metadata.split()
        path = raw_path.decode("utf-8")
        if path in wanted and object_type == b"blob":
            refs[path] = raw_oid.decode("ascii")
    missing = sorted(wanted - set(refs))
    if missing:
        raise PackageError(f"Tracked snapshot omitted files: {missing[:10]}")

    object_ids = sorted(set(refs.values()))
    process = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=ROOT,
        input="".join(f"{oid}\n" for oid in object_ids).encode("ascii"),
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        raise PackageError(process.stderr.decode("utf-8", errors="replace").strip())

    stream = io.BytesIO(process.stdout)
    objects: dict[str, bytes] = {}
    for expected_oid in object_ids:
        header = stream.readline().decode("ascii").strip().split()
        if len(header) != 3 or header[0] != expected_oid or header[1] != "blob":
            raise PackageError(f"Unexpected git cat-file response for {expected_oid}")
        size = int(header[2])
        objects[expected_oid] = stream.read(size)
        if stream.read(1) != b"\n":
            raise PackageError(f"Malformed git cat-file response for {expected_oid}")
    return {path: objects[oid] for path, oid in refs.items()}


def build_manifest(paths: list[str], snapshot: dict[str, bytes]) -> dict[str, Any]:
    return {
        "schema_version": "nlcare_buyer_package_manifest_v1",
        "candidate_type": "BUYER_CANDIDATE",
        "source_sha": git_sha(),
        "data_boundary": "synthetic/research artifacts only; no real patient data",
        "file_count": len(paths) + 1,
        "files": [
            {
                "path": path,
                "sha256": hashlib.sha256(snapshot[path]).hexdigest(),
                "size_bytes": len(snapshot[path]),
            }
            for path in paths
        ],
    }


def _zip_info(path: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(path, FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def archive_bytes(paths: list[str]) -> tuple[bytes, dict[str, Any]]:
    snapshot = _tracked_snapshot(paths)
    manifest = build_manifest(paths, snapshot)
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in paths:
            archive.writestr(_zip_info(path), snapshot[path])
        archive.writestr(_zip_info(MANIFEST_NAME), manifest_bytes)
    return buffer.getvalue(), manifest


def verify_archive(payload: bytes) -> dict[str, Any]:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = sorted(archive.namelist())
        if MANIFEST_NAME not in names:
            raise PackageError("Archive has no package manifest")
        manifest = json.loads(archive.read(MANIFEST_NAME))
        expected = sorted([entry["path"] for entry in manifest["files"]] + [MANIFEST_NAME])
        if names != expected:
            raise PackageError("Archive contents do not match its manifest")
        for entry in manifest["files"]:
            digest = hashlib.sha256(archive.read(entry["path"])).hexdigest()
            if digest != entry["sha256"]:
                raise PackageError(f"Archive hash mismatch: {entry['path']}")
    return manifest


def build_archive(output: Path) -> dict[str, Any]:
    paths = selected_files()
    payload, manifest = archive_bytes(paths)
    verify_archive(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    return {
        "archive": str(output),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "file_count": manifest["file_count"],
        "source_sha": manifest["source_sha"],
    }
