from pathlib import Path

from backend.services.research_paper_query_telemetry import (
    QUERY_CASES,
    _build_row,
    run_research_paper_query_telemetry,
)


def _fixture_result(*, intent="education", provider_reported=False):
    calls = []
    if provider_reported:
        calls.append({"usage_basis": "provider_reported", "total_tokens": 42})
    return {
        "intent": intent,
        "rag_mode": "general_education",
        "citations": [{"source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11400212/"}],
        "retrieval_context": [{"pmcid": "PMC11400212"}],
        "cache": {"status": "stored"},
        "pipeline_trace": {"terminal_step": "generated", "stage_ms": {"retrieval_ms": 3.5}},
        "rag_evaluation": {
            "cost_latency": {
                "latency_ms": 12.5,
                "estimated_input_tokens": 100,
                "estimated_output_tokens": 20,
                "estimated_total_tokens": 120,
                "estimated_llm_cost_usd": 0.0,
                "provider_token_usage": {
                    "call_count": len(calls),
                    "actual_usage_coverage_rate": 1.0 if calls else 0.0,
                    "calls": calls,
                },
            }
        },
    }


def test_fixed_query_suite_is_diverse_and_synthetic():
    assert len(QUERY_CASES) == 30
    assert len({row["id"] for row in QUERY_CASES}) == 30
    assert len({row["category"] for row in QUERY_CASES}) >= 10
    assert any(row["style"] == "taglish" for row in QUERY_CASES)
    assert all(row["allowed_intents"] for row in QUERY_CASES)


def test_row_keeps_provider_tokens_separate_from_estimates():
    case = {
        "id": "fixture",
        "category": "fixture",
        "style": "formal",
        "query": "fixture query",
        "allowed_intents": ["education"],
    }
    row = _build_row(case, _fixture_result(provider_reported=True), 14.0)
    assert row["estimated_total_tokens"] == 120
    assert row["provider_reported_total_tokens"] == 42
    assert row["token_measurement_basis"] == "provider_reported"
    assert row["cited_pmcids"] == ["PMC11400212"]
    assert row["response_content_retained"] is False
    assert "reply" not in row


def test_row_labels_chars_estimate_when_provider_usage_is_absent():
    case = {
        "id": "fixture",
        "category": "fixture",
        "style": "formal",
        "query": "fixture query",
        "allowed_intents": ["education"],
    }
    row = _build_row(case, _fixture_result(), 14.0)
    assert row["provider_reported_total_tokens"] == 0
    assert row["token_measurement_basis"] == "chars_div_4_estimate"


def test_runner_retains_no_generated_response(monkeypatch, tmp_path):
    import backend.services.agent_rag as agent_rag

    monkeypatch.setattr(agent_rag, "run_patient_agent_pipeline", lambda **_: _fixture_result())
    case = {
        "id": "fixture",
        "category": "fixture",
        "style": "formal",
        "query": "fixture query",
        "allowed_intents": ["education"],
    }
    output = tmp_path / "telemetry.json"
    failures = tmp_path / "failures.json"
    report = run_research_paper_query_telemetry(
        output_path=output,
        failures_path=failures,
        cases=[case],
    )
    assert report["query_count"] == 1
    assert report["response_content_retained"] is False
    assert report["internal_vs_external_authored"] == "internal"
    assert report["was_used_for_tuning"] is True
    assert "development evidence" in report["tuning_note"]
    assert len(report["query_suite_sha256"]) == 64
    assert report["environment"]["fast_mode"] is True
    assert report["summary"]["cold_start_excluded_from_warm_metrics"] is True
    assert report["summary"]["cold_start_query_id"] == "fixture"
    assert report["summary"]["latency_max_ms"] >= report["summary"]["latency_p95_ms"]
    assert "reply" not in report["rows"][0]
    assert output.exists()
    assert failures.exists()
    assert report["clinical_validation"] is False
    assert report["healthcare_production_ready"] is False


def test_service_source_does_not_persist_reply_text():
    source = Path("backend/services/research_paper_query_telemetry.py").read_text(encoding="utf-8")
    assert '"response_content_retained": False' in source
    assert '"reply":' not in source
