"""Local-only backup and restore drill for deployment engineering evidence."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "Data" / "evals" / "ops" / "latest_deployment_recovery_drill.json"


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_hash(connection: sqlite3.Connection) -> str:
    rows = connection.execute(
        "SELECT event_id, event_type, status FROM synthetic_ops_events ORDER BY event_id"
    ).fetchall()
    body = json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _integrity(connection: sqlite3.Connection) -> str:
    return str(connection.execute("PRAGMA integrity_check").fetchone()[0])


def run_local_recovery_drill(output_path: Path = OUTPUT_PATH) -> dict[str, Any]:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="nlcare-recovery-") as tmp:
        directory = Path(tmp)
        source_path = directory / "source.db"
        backup_path = directory / "backup.db"
        restored_path = directory / "restored.db"

        with closing(sqlite3.connect(source_path)) as source:
            source.execute(
                "CREATE TABLE synthetic_ops_events "
                "(event_id TEXT PRIMARY KEY, event_type TEXT NOT NULL, status TEXT NOT NULL)"
            )
            source.executemany(
                "INSERT INTO synthetic_ops_events VALUES (?, ?, ?)",
                [
                    ("evt-001", "release_gate", "passed"),
                    ("evt-002", "automation_dispatch", "delivered"),
                    ("evt-003", "quality_sentinel", "warning"),
                ],
            )
            source.commit()
            expected_count = int(source.execute("SELECT COUNT(*) FROM synthetic_ops_events").fetchone()[0])
            expected_content_hash = _content_hash(source)
            source_integrity = _integrity(source)
            with closing(sqlite3.connect(backup_path)) as backup:
                source.backup(backup)
            source.execute("INSERT INTO synthetic_ops_events VALUES (?, ?, ?)", ("evt-004", "post_backup", "ignored"))
            source.commit()

        backup_sha256 = _file_hash(backup_path)
        restore_started = time.perf_counter()
        with closing(sqlite3.connect(backup_path)) as backup, closing(sqlite3.connect(restored_path)) as restored:
            backup.backup(restored)
            restored_count = int(restored.execute("SELECT COUNT(*) FROM synthetic_ops_events").fetchone()[0])
            restored_content_hash = _content_hash(restored)
            restored_integrity = _integrity(restored)
        restore_seconds = time.perf_counter() - restore_started

    passed = all([
        source_integrity == "ok",
        restored_integrity == "ok",
        expected_count == restored_count,
        expected_content_hash == restored_content_hash,
    ])
    payload = {
        "schema_version": "deployment_recovery_drill_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong_local_only" if passed else "failed_local_drill",
        "passed": passed,
        "scope": "temporary SQLite synthetic operational records only",
        "clinical_validation": False,
        "healthcare_production_ready": False,
        "contains_patient_data": False,
        "strict_profile_validated": False,
        "postgres_restore_tested": False,
        "multi_instance_restore_tested": False,
        "backup": {
            "sha256": backup_sha256,
            "expected_row_count": expected_count,
            "source_integrity_check": source_integrity,
        },
        "restore": {
            "restored_row_count": restored_count,
            "content_hash_match": expected_content_hash == restored_content_hash,
            "integrity_check": restored_integrity,
            "restore_seconds": round(restore_seconds, 6),
        },
        "total_drill_seconds": round(time.perf_counter() - started, 6),
        "remaining_requirements": [
            "managed PostgreSQL backup and point-in-time recovery drill",
            "encrypted object-store retention policy",
            "multi-instance restore validation",
            "documented production RPO and RTO approval",
        ],
        "claim_boundary": (
            "This local synthetic SQLite drill proves only that the repository's backup/restore procedure works "
            "in one process. It is not production disaster-recovery evidence or clinical readiness."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
