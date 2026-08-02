from __future__ import annotations

import json
from pathlib import Path

from backend.services.kb_ingestion import ingest_knowledge_base, load_ingested_chunks


def test_embedded_pmc_link_does_not_become_curated_source_identity(tmp_path: Path) -> None:
    source_dir = tmp_path / "kb"
    source_dir.mkdir()
    (source_dir / "curated_summary.md").write_text(
        "# Curated summary\n\nReference: https://pmc.ncbi.nlm.nih.gov/articles/PMC2793754/",
        encoding="utf-8",
    )
    output = tmp_path / "chunks.json"

    ingest_knowledge_base(source_dir, output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["chunks"][0]["pmcid"] is None


def test_pmc_filename_remains_a_source_identity(tmp_path: Path) -> None:
    source_dir = tmp_path / "research_papers"
    source_dir.mkdir()
    (source_dir / "PMC123456_example.txt").write_text(
        "Abstract\n\nA synthetic ingestion fixture for provenance testing.",
        encoding="utf-8",
    )
    output = tmp_path / "chunks.json"

    ingest_knowledge_base(source_dir, output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["chunks"][0]["pmcid"] == "PMC123456"


def test_runtime_loader_preserves_auditable_research_metadata(tmp_path: Path) -> None:
    payload = {
        "chunks": [
            {
                "id": "chunk-1",
                "parent_id": "paper-1",
                "title": "Paper title",
                "source_name": "Paper title",
                "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC123456/",
                "source_path": "KnowledgeBase/raw/research_papers/PMC123456_example.txt",
                "source_type": "txt",
                "trust_level": "research_paper",
                "topic": "test",
                "modality": ["MRI"],
                "care_stage": "monitoring",
                "section": "methods",
                "section_rank": 5,
                "chunk_index": 3,
                "confidence": "peer_reviewed_open_access",
                "pmcid": "PMC123456",
                "ingested_at": "2026-08-02T00:00:00+00:00",
                "tags": ["test"],
                "text": "Methods fixture.",
            }
        ]
    }
    output = tmp_path / "chunks.json"
    output.write_text(json.dumps(payload), encoding="utf-8")

    row = load_ingested_chunks(output)[0]

    for key in (
        "pmcid",
        "source_path",
        "source_type",
        "section_rank",
        "chunk_index",
        "ingested_at",
    ):
        assert row[key] == payload["chunks"][0][key]
