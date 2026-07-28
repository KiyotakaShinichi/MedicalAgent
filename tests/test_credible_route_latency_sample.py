from backend.services.credible_route_latency_sample import (
    MEASURED_ROUTES,
    build_probe_queries,
    build_profile_from_probe,
)


def test_probe_builder_has_minimum_samples_per_measured_route():
    rows = build_probe_queries(samples_per_route=30)
    for route in MEASURED_ROUTES:
        assert sum(identifier.startswith(f"{route}__") for identifier, _ in rows) == 30


def test_profile_marks_measured_routes_credible_and_keeps_unmeasured_distinct(tmp_path):
    rows = []
    for route in MEASURED_ROUTES:
        for index in range(30):
            rows.append(
                {
                    "id": f"{route}__{index:03d}",
                    "query": f"{route} query {index}",
                    "terminal_step": (
                        "input_guardrail_block"
                        if route == "deterministic_safety_refusal"
                        else "cache_hit"
                        if route == "cached_educational_answer"
                        else "direct_support"
                        if route == "emotional_distress_support"
                        else "generated"
                    ),
                    "total_ms": 10 + index,
                    "stage_ms": {"generation_ms": 8 + index},
                }
            )
    payload = build_profile_from_probe(
        {"per_query": rows, "summary": {}, "warmup": {}},
        output_path=tmp_path / "profile.json",
        samples_per_route=30,
    )
    by_route = {row["route"]: row for row in payload["routes"]}
    for route in MEASURED_ROUTES:
        assert by_route[route]["sample_count"] == 30
        assert by_route[route]["percentile_credible"] is True
        assert by_route[route]["route_integrity_rate"] == 1.0
    assert by_route["rag_plus_reranker"]["measurement_status"] == "not_sampled"
    assert by_route["hybrid_prediction"]["measurement_status"] == "not_sampled"
    assert payload["production_ready"] is False
    assert payload["clinical_validation"] is False


def test_profile_rejects_false_credibility_from_small_sample(tmp_path):
    rows = [
        {
            "id": f"normal_rag__{index:03d}",
            "query": "q",
            "terminal_step": "generated",
            "total_ms": 20,
            "stage_ms": {},
        }
        for index in range(4)
    ]
    payload = build_profile_from_probe(
        {"per_query": rows, "summary": {}, "warmup": {}},
        output_path=tmp_path / "profile.json",
        samples_per_route=30,
    )
    normal = next(row for row in payload["routes"] if row["route"] == "normal_rag")
    assert normal["measurement_status"] == "insufficient_samples"
    assert normal["percentile_credible"] is False
    assert payload["status"] == "needs_attention"
