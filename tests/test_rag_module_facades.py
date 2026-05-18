from backend.services import (
    citation_assembler,
    claim_level_citation_validator,
    intent_classification,
    rag_cache,
    response_finalizer,
    retrieval_pipeline,
    source_tier_filtering,
)


def test_rag_responsibility_facades_export_expected_callables():
    assert callable(intent_classification.route_intent)
    assert callable(retrieval_pipeline.hybrid_retrieval)
    assert callable(source_tier_filtering.filter_chunks_by_mode)
    assert callable(claim_level_citation_validator.validate_claims)
    assert callable(rag_cache.is_cacheable)
    assert callable(response_finalizer.generate_answer)


def test_citation_assembler_uses_stable_source_order():
    ids = citation_assembler.assemble_citation_ids([
        {"id": "chunk-1", "parent_id": "source-a"},
        {"id": "chunk-2", "parent_id": "source-a"},
        {"id": "chunk-3", "source_id": "source-b"},
    ])

    assert ids == ["source-a", "source-b"]
