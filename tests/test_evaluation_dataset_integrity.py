import hashlib
import json

import pytest

from backend.services.evaluation_dataset_integrity import (
    FrozenDatasetIntegrityError,
    verify_registry,
    write_integrity_report,
)


def _hash(path):
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def test_registry_verifies_frozen_content_and_marks_external_block(tmp_path):
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text('{"case_id":"one","query":"hello"}\n', encoding="utf-8")
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({
        "registry_version": "test",
        "datasets": [
            {
                "evaluation_class": "internal_holdout",
                "dataset_name": "frozen",
                "path": str(dataset),
                "frozen": True,
                "was_used_for_tuning": False,
                "normalized_content_sha256": _hash(dataset),
            },
            {
                "evaluation_class": "external_no_read_holdout",
                "dataset_name": "external",
                "path": str(tmp_path / "missing.jsonl"),
                "frozen": True,
                "external_status": "BLOCKED_EXTERNAL",
            },
        ],
    }), encoding="utf-8")
    report = verify_registry(registry)
    assert report["integrity_failure_count"] == 0
    assert report["external_review_status"] == "BLOCKED_EXTERNAL"
    assert report["clinical_validation"] is False


def test_frozen_hash_mismatch_fails_closed(tmp_path):
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text('{"case_id":"one","query":"changed"}\n', encoding="utf-8")
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({
        "datasets": [{
            "evaluation_class": "internal_holdout",
            "dataset_name": "frozen",
            "path": str(dataset),
            "frozen": True,
            "normalized_content_sha256": "0" * 64,
        }],
    }), encoding="utf-8")
    with pytest.raises(FrozenDatasetIntegrityError):
        write_integrity_report(tmp_path / "output.json", registry)
