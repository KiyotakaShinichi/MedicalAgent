from __future__ import annotations

import json
from pathlib import Path

from backend.services.prototype_independent_prompt_bank_v2 import freeze_prompt_bank_v2
from backend.services.real_pipeline_scale_eval import (
    build_real_pipeline_cases,
    run_real_pipeline_scale_eval,
)


def test_real_pipeline_case_builder_has_300_mixed_calls(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    freeze_prompt_bank_v2(bank, tmp_path / "manifest.json")
    cases = build_real_pipeline_cases(bank)
    assert len(cases) == 300
    assert sum(case["source_suite"] == "frozen_prototype_independent_v2" for case in cases) == 210
    assert sum(case["source_suite"] == "research_query_telemetry" for case in cases) == 60
    assert sum(case["source_suite"] == "fixed_direct_support" for case in cases) == 20
    assert sum(case["source_suite"] == "cache_repeat" for case in cases) == 10


def test_small_fake_run_preserves_claim_boundary_and_no_response_text(tmp_path: Path) -> None:
    cases = [
        {
            "id": "fake_1",
            "category": "education",
            "style": "formal",
            "query": "What is monitoring?",
            "allowed_intents": ["education"],
            "expected_behavior": "source_grounded_or_abstained",
            "source_suite": "test",
        }
    ]

    def fake_runner(**kwargs):
        return {
            "reply": "Synthetic response text",
            "intent": "education",
            "citations": [{"source_id": "demo"}],
            "pipeline_trace": {"terminal_step": "generated", "stage_ms": {"generation_ms": 1}},
            "rag_evaluation": {
                "cost_latency": {
                    "latency_ms": 3,
                    "estimated_input_tokens": 4,
                    "estimated_output_tokens": 5,
                    "estimated_total_tokens": 9,
                    "provider_token_usage": {"calls": [], "call_count": 0},
                }
            },
            "post_gen_validator": {"decision": "allow"},
            "output_guardrail": {"status": "passed"},
        }

    output = tmp_path / "out.json"
    result = run_real_pipeline_scale_eval(
        output_path=output,
        failure_path=tmp_path / "fail.json",
        cases=cases,
        pipeline_runner=fake_runner,
        enable_prewarm=False,
    )
    serialized = output.read_text(encoding="utf-8")
    assert "Synthetic response text" not in serialized
    assert result["summary"]["pipeline_completion_rate"] == 1.0
    assert result["summary"]["provider_usage_coverage_rate"] == 0.0
    assert result["clinical_validation"] is False
    assert "not clinical validation" in result["claim_boundary"].lower()
    assert json.loads(serialized)["rows"][0]["response_character_count"] > 0
