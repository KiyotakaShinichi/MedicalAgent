import json

from backend.services.kb_ingestion import (
    _chunk_text_by_section,
    ingest_knowledge_base,
    load_ingested_chunks,
)
from backend.services.kb_source_governance import build_kb_source_governance


def _write_fixture(tmp_path, *, retracted=False):
    source_dir = tmp_path / "KnowledgeBase" / "raw" / "research_papers"
    source_dir.mkdir(parents=True)
    file_name = "PMC123456_fixture-paper.txt"
    (source_dir / file_name).write_text(
        "Abstract\n\n" + "patient reported symptom monitoring evidence " * 120,
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "research_paper_manifest_v2",
        "clinical_validation": False,
        "items": [
            {
                "pmcid": "PMC123456",
                "pmid": "123456",
                "doi": "10.0000/fixture",
                "publication_date": "2026 Jan",
                "journal": "Fixture Journal",
                "license": "CC BY",
                "title": "Fixture paper",
                "landing_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC123456/",
                "file_name": file_name,
                "path": f"KnowledgeBase/raw/research_papers/{file_name}",
                "topic": "symptom_monitoring",
                "modality": ["symptoms"],
                "stage": "treatment_monitoring",
                "confidence": "peer_reviewed_open_access",
                "trust_level": "research_paper",
                "allowed_use": ["education"],
                "patient_facing_suitability": "education_with_boundary",
                "evidence_role": "workflow_design_evidence",
                "not_allowed_for": ["diagnosis", "treatment_selection_or_change"],
                "selection_rationale": "Fixture",
                "retracted": retracted,
            }
        ],
    }
    (source_dir / "research_papers_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return source_dir.parent, file_name


def test_ingestion_preserves_bibliography_and_use_boundaries(tmp_path):
    input_dir, _ = _write_fixture(tmp_path)
    output = tmp_path / "chunks.json"
    ingest_knowledge_base(input_dir=input_dir, output_path=output)
    chunks = load_ingested_chunks(output)
    paper = next(chunk for chunk in chunks if chunk["pmcid"] == "PMC123456")
    assert paper["doi"] == "10.0000/fixture"
    assert paper["pmid"] == "123456"
    assert paper["publication_date"] == "2026 Jan"
    assert paper["license"] == "CC BY"
    assert paper["allowed_use"] == ["education"]
    assert paper["patient_facing_suitability"] == "education_with_boundary"
    assert paper["not_allowed_for"] == ["diagnosis", "treatment_selection_or_change"]


def test_governance_uses_narrower_source_manifest_policy(tmp_path):
    input_dir, _ = _write_fixture(tmp_path)
    chunks_path = tmp_path / "chunks.json"
    governance_path = tmp_path / "governance.json"
    ingest_knowledge_base(input_dir=input_dir, output_path=chunks_path)
    report = build_kb_source_governance(
        kb_chunks_path=str(chunks_path),
        output_path=str(governance_path),
    )
    source = next(row for row in report["sources"] if row["pmcid"] == "PMC123456")
    assert source["tier"] == "T2"
    assert source["allowed_use"] == ["education"]
    assert source["allowed_use_source"] == "source_manifest"
    assert source["patient_facing_suitability"] == "education_with_boundary"


def test_retracted_source_is_fail_closed(tmp_path):
    input_dir, _ = _write_fixture(tmp_path, retracted=True)
    chunks_path = tmp_path / "chunks.json"
    governance_path = tmp_path / "governance.json"
    ingest_knowledge_base(input_dir=input_dir, output_path=chunks_path)
    report = build_kb_source_governance(
        kb_chunks_path=str(chunks_path),
        output_path=str(governance_path),
    )
    source = next(row for row in report["sources"] if row["pmcid"] == "PMC123456")
    assert source["allowed_use"] == []
    assert report["status"] == "needs_attention"
    assert any(issue["code"] == "retracted_source" for issue in report["governance_issues"])


def test_article_without_named_sections_is_not_discarded_as_front_matter():
    text = (
        "A paper title\n\n" + "patient reported outcome evidence " * 150
        + "\n\nReferences\n\n1. Example citation"
    )

    chunks = _chunk_text_by_section(text, chunk_chars=500, overlap_chars=50)

    assert chunks
    assert all(chunk["section"] == "body" for chunk in chunks)
    assert all("Example citation" not in chunk["text"] for chunk in chunks)


def test_guideline_highlights_and_extended_headings_are_sectioned():
    text = (
        "Article title\n\nHighlights\n\nA short evidence summary.\n\n"
        "Introduction\n\nBackground prose.\n\nRisk factors\n\nRisk-factor prose.\n\n"
        "Methodology\n\nMethod prose.\n\nRecommendations\n\nRecommendation prose.\n\n"
        "References\n\n1. Example citation"
    )

    chunks = _chunk_text_by_section(text, chunk_chars=500, overlap_chars=50)

    assert {chunk["section"] for chunk in chunks} >= {"abstract", "introduction", "body", "methods", "conclusion"}
    assert any(chunk["section_heading"] == "Highlights" for chunk in chunks)
    assert all("Example citation" not in chunk["text"] for chunk in chunks)
