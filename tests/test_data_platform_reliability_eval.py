from __future__ import annotations

from pathlib import Path

import pytest

from backend.services.data_platform_reliability_eval import (
    apply_tombstones,
    build_backfill_batches,
    build_data_platform_reliability_eval,
    migrate_gold_record,
)


ROOT_DIR = Path(__file__).resolve().parents[1]


def test_reliability_eval_runs_all_offline_drills(tmp_path):
    report = build_data_platform_reliability_eval(
        root_dir=ROOT_DIR,
        output_path=tmp_path / "report.json",
    )
    assert report["status"] == "strong_offline_drill"
    assert report["n_cases"] >= 6
    assert report["failed"] == 0
    assert report["external_cloud_write_performed"] is False
    assert report["delete_propagation"]["managed_index_delete_completed"] is False
    assert report["recovery"]["azure_restore_drill_completed"] is False
    assert report["partition_scale_replay"]["scale_multiplier"] == 100
    assert report["partition_scale_replay"]["deterministic"] is True
    assert report["partition_scale_replay"]["managed_cloud_throughput_measured"] is False
    assert report["clinical_validation"] is False


def test_schema_migration_adds_explicit_governance_defaults():
    migrated = migrate_gold_record(
        {
            "record_id": "a",
            "metadata": {
                "source_id": "s",
                "chunk_id": "a",
                "parent_id": "s",
                "source_tier": "T2",
                "allowed_use": ["education"],
                "patient_facing": True,
                "staleness_status": "current",
                "kb_fingerprint": "fixture",
            },
        }
    )
    assert migrated["data_contract"] == "vector_record_gold_v1"
    assert migrated["metadata"]["data_scope"] == "curated_non_patient_kb"
    assert migrated["metadata"]["clinical_validation"] is False


def test_tombstones_and_backfill_are_deterministic():
    rows = [{"record_id": "b"}, {"record_id": "a"}, {"record_id": "c"}]
    active, deleted = apply_tombstones(rows, {"b"})
    assert deleted == ["b"]
    assert {row["record_id"] for row in active} == {"a", "c"}
    assert build_backfill_batches(["c", "a", "b"], batch_size=2) == [["a", "b"], ["c"]]
    assert build_backfill_batches(["b", "c", "a"], batch_size=2) == [["a", "b"], ["c"]]


def test_backfill_rejects_non_positive_batch_size():
    with pytest.raises(ValueError, match="positive"):
        build_backfill_batches(["a"], batch_size=0)
