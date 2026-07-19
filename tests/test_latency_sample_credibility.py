from scripts.run_latency_profile import (
    MIN_CREDIBLE_PERCENTILE_SAMPLES,
    ROUTE_BUDGETS,
    _latency_status,
    _measurement_status,
)


def test_tiny_route_sample_cannot_be_called_ideal():
    budget = ROUTE_BUDGETS["normal_rag"]
    assert _measurement_status(4) == "insufficient_samples"
    assert _latency_status(10.0, budget, sample_count=4) == "insufficient_samples"


def test_budget_label_only_after_minimum_sample_count():
    budget = ROUTE_BUDGETS["normal_rag"]
    assert MIN_CREDIBLE_PERCENTILE_SAMPLES == 30
    assert _latency_status(10.0, budget, sample_count=30) == "ideal"


def test_zero_samples_are_distinct_from_insufficient_samples():
    budget = ROUTE_BUDGETS["normal_rag"]
    assert _measurement_status(0) == "not_sampled"
    assert _latency_status(None, budget, sample_count=0) == "not_sampled"
