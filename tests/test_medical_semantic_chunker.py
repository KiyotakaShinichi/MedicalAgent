from backend.services.medical_semantic_chunker import evaluate_chunking_quality, semantic_chunk_markdown


def test_semantic_chunker_preserves_heading_metadata():
    text = "# CBC monitoring\n\nWBC collected 2026-01-01: 5.2 10^9/L.\n\n## Imaging\n\nFindings: mass smaller.\n\nImpression: response context."
    chunks = semantic_chunk_markdown(
        text,
        source_id="sample",
        metadata={"allowed_use": ["education"], "source_tier": "T2", "staleness": "current"},
    )
    assert chunks
    assert all(chunk["source_id"] == "sample" for chunk in chunks)
    assert any(chunk["section_heading"] == "CBC monitoring" for chunk in chunks)
    assert any(chunk["parent_heading"] == "CBC monitoring" for chunk in chunks if chunk["section_heading"] == "Imaging")


def test_chunking_quality_reports_no_critical_splits_for_short_sections():
    payload = evaluate_chunking_quality([
        (
            "sample",
            "# Family history\n\nMother had breast cancer at age 52.\n\n# Medication\n\nTamoxifen 20 mg daily is listed as context only.",
            {"allowed_use": ["education"], "source_tier": "T3"},
        )
    ])
    assert payload["chunk_count"] >= 1
    assert payload["critical_context_split_rate"] == 0
    assert payload["chunk_source_traceability"] == 1.0
