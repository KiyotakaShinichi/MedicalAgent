"""Disposable, deterministic synthetic buyer-demo database lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEMO_ROOT = (ROOT / "Data" / "test_tmp" / "buyer_demo").resolve()
DEFAULT_DATABASE = DEMO_ROOT / "buyer_demo.db"
MARKER_SUFFIX = ".nlcare-buyer-demo.json"
CORE_TABLES = {
    "patients": ("id", "name", "diagnosis"),
    "breast_cancer_profiles": (
        "id",
        "patient_id",
        "cancer_stage",
        "er_status",
        "pr_status",
        "her2_status",
        "molecular_subtype",
        "treatment_intent",
        "menopausal_status",
    ),
    "lab_results": ("id", "patient_id", "date", "wbc", "hemoglobin", "platelets", "source", "source_note"),
    "symptom_reports": ("id", "patient_id", "date", "symptom", "severity", "notes"),
    "treatments": ("id", "patient_id", "date", "cycle", "drug"),
    "imaging_reports": (
        "id",
        "patient_id",
        "date",
        "modality",
        "report_type",
        "body_site",
        "findings",
        "impression",
    ),
}


class DemoSafetyError(ValueError):
    """Raised when a demo lifecycle command could touch non-demo data."""


def resolve_demo_database(value: str | Path | None = None) -> Path:
    candidate = Path(value) if value else DEFAULT_DATABASE
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    resolved = candidate.resolve()
    if resolved == DEMO_ROOT or DEMO_ROOT not in resolved.parents:
        raise DemoSafetyError(f"Buyer demo database must stay under {DEMO_ROOT}")
    if resolved.suffix.lower() not in {".db", ".sqlite", ".sqlite3"}:
        raise DemoSafetyError("Buyer demo database must use a SQLite file extension")
    return resolved


def marker_path(database: Path) -> Path:
    return database.with_name(database.name + MARKER_SUFFIX)


def database_url(database: Path) -> str:
    relative = database.relative_to(ROOT).as_posix()
    return f"sqlite:///./{relative}"


def logical_fingerprint(database: Path) -> str:
    digest = hashlib.sha256()
    with closing(sqlite3.connect(database)) as connection:
        for table, columns in CORE_TABLES.items():
            available = {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}
            if not set(columns) <= available:
                raise DemoSafetyError(f"Expected demo table is missing: {table}")
            order = ", ".join(f'"{column}"' for column in columns)
            rows = connection.execute(f'SELECT {order} FROM "{table}" ORDER BY {order}').fetchall()
            digest.update(table.encode("utf-8"))
            digest.update(json.dumps(rows, default=str, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def _write_marker(database: Path, fingerprint: str) -> None:
    marker_path(database).write_text(
        json.dumps(
            {
                "schema_version": "nlcare_buyer_demo_marker_v1",
                "data_classification": "synthetic",
                "database": database.relative_to(ROOT).as_posix(),
                "logical_fingerprint": fingerprint,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _validated_marker(database: Path) -> dict[str, Any]:
    marker = marker_path(database)
    if not marker.is_file():
        raise DemoSafetyError(f"Refusing to alter unmarked database: {database}")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "nlcare_buyer_demo_marker_v1":
        raise DemoSafetyError("Unrecognized buyer-demo marker")
    if payload.get("data_classification") != "synthetic":
        raise DemoSafetyError("Buyer-demo marker is not synthetic")
    if payload.get("database") != database.relative_to(ROOT).as_posix():
        raise DemoSafetyError("Buyer-demo marker does not match the target database")
    return payload


def reset_demo(value: str | Path | None = None) -> None:
    database = resolve_demo_database(value)
    if not database.exists() and not marker_path(database).exists():
        return
    _validated_marker(database)
    for candidate in (
        database,
        database.with_name(database.name + "-wal"),
        database.with_name(database.name + "-shm"),
        marker_path(database),
    ):
        for attempt in range(50):
            if not candidate.exists():
                break
            try:
                candidate.unlink()
                break
            except PermissionError:
                if attempt == 49:
                    raise
                time.sleep(0.1)


def seed_demo(value: str | Path | None = None) -> dict[str, Any]:
    database = resolve_demo_database(value)
    database.parent.mkdir(parents=True, exist_ok=True)
    if database.exists() or marker_path(database).exists():
        reset_demo(database)

    env = os.environ.copy()
    env.update(
        {
            "DATABASE_URL": database_url(database),
            "APP_ENV": "development",
            "ENVIRONMENT": "development",
            "ALLOW_DEMO_AUTH": "true",
            "NLCARE_SYNTHETIC_ONLY": "true",
            "NLCARE_TEST_OFFLINE": "true",
        }
    )
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=ROOT,
        env=env,
        check=True,
    )
    subprocess.run([sys.executable, "seed_db.py"], cwd=ROOT, env=env, check=True)
    fingerprint = logical_fingerprint(database)
    _write_marker(database, fingerprint)
    return {
        "database": database.relative_to(ROOT).as_posix(),
        "data_classification": "synthetic",
        "logical_fingerprint": fingerprint,
    }


def backup_demo(output: str | Path, value: str | Path | None = None) -> Path:
    database = resolve_demo_database(value)
    _validated_marker(database)
    destination = Path(output)
    if not destination.is_absolute():
        destination = ROOT / destination
    destination = resolve_demo_database(destination)
    if destination.exists():
        raise DemoSafetyError(f"Refusing to overwrite backup: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database)) as source, closing(sqlite3.connect(destination)) as target:
        source.backup(target)
    return destination


def restore_demo(backup: str | Path, value: str | Path | None = None) -> dict[str, Any]:
    source = resolve_demo_database(backup)
    if not source.is_file():
        raise DemoSafetyError(f"Backup does not exist: {source}")
    target = resolve_demo_database(value)
    if target.exists() or marker_path(target).exists():
        reset_demo(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(source)) as source_connection, closing(
        sqlite3.connect(target)
    ) as target_connection:
        if source_connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise DemoSafetyError("Backup failed SQLite integrity_check")
        source_connection.backup(target_connection)
    fingerprint = logical_fingerprint(target)
    _write_marker(target, fingerprint)
    return {
        "database": target.relative_to(ROOT).as_posix(),
        "data_classification": "synthetic",
        "logical_fingerprint": fingerprint,
    }
