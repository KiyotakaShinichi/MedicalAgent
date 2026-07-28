from __future__ import annotations

import json
from pathlib import Path

from backend.services.data_platform_pipeline import run_data_platform_pipeline


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_fixture(root: Path) -> None:
    source = root / "KnowledgeBase" / "raw" / "guide.md"
    source.parent.mkdir(parents=True)
    source.write_text("# Guide\n\nSynthetic education content.", encoding="utf-8")

    chunks = {
        "chunks": [
            {
                "id": "chunk-1",
                "parent_id": "source-1",
                "text": "Synthetic education content.",
                "title": "Guide",
                "source_name": "Fixture",
                "source_url": "https://example.test/guide",
                "source_path": "KnowledgeBase/raw/guide.md",
                "trust_level": "research_paper",
                "tags": ["education"],
            }
        ]
    }
    chunk_path = root / "Data" / "rag_knowledge_base_chunks.json"
    chunk_path.parent.mkdir(parents=True)
    chunk_path.write_text(json.dumps(chunks), encoding="utf-8")

    governance = {
        "sources": [
            {
                "source_id": "source-1",
                "tier": "T2",
                "allowed_use": ["education"],
                "staleness_status": "current",
            }
        ]
    }
    governance_path = root / "Data" / "evals" / "rag" / "latest_kb_source_governance.json"
    governance_path.parent.mkdir(parents=True)
    governance_path.write_text(json.dumps(governance), encoding="utf-8")

    contract_path = root / "config" / "data_contracts.json"
    contract_path.parent.mkdir(parents=True)
    contract_path.write_text(
        (REPO_ROOT / "config" / "data_contracts.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )


def test_pipeline_materializes_medallion_layers_and_lineage(tmp_path: Path):
    _write_fixture(tmp_path)
    report = run_data_platform_pipeline(root_dir=tmp_path)
    assert report["status"] == "strong"
    assert report["patient_data_processed"] is False
    assert report["layers"]["bronze"]["record_count"] == 1
    assert report["layers"]["silver"]["record_count"] == 1
    assert report["layers"]["gold"]["record_count"] == 1
    assert report["lineage"]["complete"] is True
    assert (tmp_path / report["layers"]["gold"]["path"]).exists()


def test_second_pipeline_run_is_incremental_no_change(tmp_path: Path):
    _write_fixture(tmp_path)
    run_data_platform_pipeline(root_dir=tmp_path)
    gold_path = tmp_path / "Data" / "lakehouse" / "gold" / "vector_records.jsonl"
    first_gold_mtime = gold_path.stat().st_mtime_ns
    report = run_data_platform_pipeline(root_dir=tmp_path)
    assert report["incremental_run"]["new"] == []
    assert report["incremental_run"]["changed"] == []
    assert report["incremental_run"]["source_change_detected"] is False
    assert report["incremental_run"]["rebuild_performed"] is False
    assert gold_path.stat().st_mtime_ns == first_gold_mtime


def test_changed_source_is_detected(tmp_path: Path):
    _write_fixture(tmp_path)
    run_data_platform_pipeline(root_dir=tmp_path)
    source = tmp_path / "KnowledgeBase" / "raw" / "guide.md"
    source.write_text("# Guide\n\nChanged synthetic education content.", encoding="utf-8")
    report = run_data_platform_pipeline(root_dir=tmp_path)
    assert report["incremental_run"]["changed"] == ["KnowledgeBase/raw/guide.md"]
    assert report["incremental_run"]["rebuild_performed"] is True


def test_changed_chunk_artifact_forces_upstream_rebuild(tmp_path: Path):
    _write_fixture(tmp_path)
    run_data_platform_pipeline(root_dir=tmp_path)
    chunk_path = tmp_path / "Data" / "rag_knowledge_base_chunks.json"
    chunks = json.loads(chunk_path.read_text(encoding="utf-8"))
    chunks["chunks"][0]["text"] = "Changed governed chunk materialization."
    chunk_path.write_text(json.dumps(chunks), encoding="utf-8")

    report = run_data_platform_pipeline(root_dir=tmp_path)

    assert report["incremental_run"]["source_change_detected"] is False
    assert report["incremental_run"]["upstream_change_detected"] is True
    assert report["incremental_run"]["upstream_changed"] == ["chunk_artifact"]
    assert report["incremental_run"]["rebuild_performed"] is True
    assert report["lineage"]["kb_fingerprint"]


def test_invalid_chunk_is_quarantined(tmp_path: Path):
    _write_fixture(tmp_path)
    chunk_path = tmp_path / "Data" / "rag_knowledge_base_chunks.json"
    chunk_path.write_text(
        json.dumps({"chunks": [{"id": "", "parent_id": "", "text": ""}]}),
        encoding="utf-8",
    )
    report = run_data_platform_pipeline(root_dir=tmp_path)
    assert report["status"] == "needs_attention"
    assert report["layers"]["quarantine"]["record_count"] == 1
    assert report["layers"]["silver"]["record_count"] == 0


def test_contract_refuses_patient_data_configuration(tmp_path: Path):
    _write_fixture(tmp_path)
    contract_path = tmp_path / "config" / "data_contracts.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["patient_data_allowed"] = True
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    try:
        run_data_platform_pipeline(root_dir=tmp_path)
    except ValueError as exc:
        assert "patient data" in str(exc)
    else:
        raise AssertionError("Patient-data configuration should have been rejected.")
