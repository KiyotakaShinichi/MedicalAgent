import json

from backend.services.retrieval_runtime_cache_eval import (
    build_retrieval_runtime_cache_eval,
)


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_reports_local_improvement_without_production_claim(tmp_path):
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    _write(
        baseline,
        {
            "retrieval_p50_ms": 1000,
            "retrieval_p95_ms": 1800,
            "total_p50_ms": 1200,
            "total_p95_ms": 2000,
            "workload": {"retrieval_mode": "forced_sparse"},
        },
    )
    _write(
        current,
        {
            "summary_by_route": {
                "normal_rag": {
                    "retrieval_ms": {"samples": 30, "median_ms": 100, "p95_ms": 200},
                    "total_ms": {"samples": 30, "median_ms": 150, "p95_ms": 250},
                }
            }
        },
    )
    result = build_retrieval_runtime_cache_eval(baseline, current)
    assert result["status"] == "improved_local_regression"
    assert result["local_regression_improvement_observed"] is True
    assert result["production_ready"] is False
    assert result["clinical_validation"] is False
    assert result["dense_unique_query_latency_measured"] is False


def test_requires_credible_sample_size(tmp_path):
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    _write(
        baseline,
        {
            "retrieval_p50_ms": 1000,
            "retrieval_p95_ms": 1800,
            "total_p50_ms": 1200,
            "total_p95_ms": 2000,
        },
    )
    _write(
        current,
        {
            "summary_by_route": {
                "normal_rag": {
                    "retrieval_ms": {"samples": 2, "median_ms": 100, "p95_ms": 200},
                    "total_ms": {"samples": 2, "median_ms": 150, "p95_ms": 250},
                }
            }
        },
    )
    result = build_retrieval_runtime_cache_eval(baseline, current)
    assert result["status"] == "needs_attention"
    assert result["sample_contract_met"] is False


def test_regression_stays_visible(tmp_path):
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    _write(
        baseline,
        {
            "retrieval_p50_ms": 100,
            "retrieval_p95_ms": 200,
            "total_p50_ms": 150,
            "total_p95_ms": 250,
        },
    )
    _write(
        current,
        {
            "summary_by_route": {
                "normal_rag": {
                    "retrieval_ms": {"samples": 30, "median_ms": 150, "p95_ms": 250},
                    "total_ms": {"samples": 30, "median_ms": 200, "p95_ms": 300},
                }
            }
        },
    )
    result = build_retrieval_runtime_cache_eval(baseline, current)
    assert result["status"] == "needs_attention"
    assert result["comparison"]["retrieval_p95_ms"]["delta_ms"] == 50
