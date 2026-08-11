from backend.services.agent_query_rewriting import tokenize
from backend.services.agent_retrieval import rerank_context, title_match_boost


def _paper(title, *, score=0.1, builtin=False, pmcid=None):
    return {
        "id": title.lower().replace(" ", "-")[:30],
        "parent_id": "parent",
        "title": title,
        "source_name": title,
        "source_url": "https://example.test/paper",
        "text": "Source-backed educational context.",
        "tags": ["patient education"],
        "topic": "research",
        "source_tier": "T2",
        "allowed_use": ["education"],
        "retrieval_score": score,
        "builtin": builtin,
        "pmcid": pmcid,
    }


def test_exact_title_lookup_gets_strong_case_agnostic_boost():
    query = set(tokenize("Find the paper titled Use of PRO-CTCAE in oncology clinical trials"))
    item = _paper("Use of PRO-CTCAE in oncology clinical trials")

    assert title_match_boost(query, item) == 2.0


def test_unrelated_title_gets_no_boost():
    query = set(tokenize("What does PRO-CTCAE measure?"))
    item = _paper("CT ascites report wording monitoring")

    assert title_match_boost(query, item) == 0.0


def test_source_identifier_lookup_gets_strong_boost():
    query = set(tokenize("Summarize PMC12452844 with citations"))
    item = _paper("A research paper", pmcid="PMC12452844")

    assert title_match_boost(query, item) == 2.0


def test_title_match_can_outrank_irrelevant_curated_chunk(monkeypatch):
    monkeypatch.setenv("RAG_ENABLE_CROSS_ENCODER", "false")
    query = "Find the paper titled Use of PRO-CTCAE in oncology clinical trials"
    rewritten = {"expanded_query": query}
    matching = _paper("Use of PRO-CTCAE in oncology clinical trials", score=0.1)
    irrelevant_curated = _paper("CT ascites report wording monitoring", score=0.9, builtin=True)

    rows = rerank_context(
        [irrelevant_curated, matching],
        rewritten,
        "education",
        {"level": "low_risk"},
    )

    assert rows[0]["title"] == matching["title"]
    assert rows[0]["title_match_boost"] == 2.0

