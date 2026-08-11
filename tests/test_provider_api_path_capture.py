from backend.services.provider_api_path_capture import build_provider_api_path_capture


def _env(**values):
    base = {"OPENAI_API_KEY": "test-key", "NLCARE_ALLOW_PAID_PROVIDER_PROBE": "true"}
    base.update(values)
    return base


def test_probe_never_calls_network_without_explicit_execute():
    calls = []

    def requester(*args):
        calls.append(args)
        raise AssertionError("request must not run")

    result = build_provider_api_path_capture(env=_env(), request_fn=requester)
    assert result["status"] == "ready_for_explicit_execution"
    assert result["completed"] is False
    assert calls == []


def test_probe_blocks_missing_provider_configuration():
    result = build_provider_api_path_capture(
        execute=True,
        env={"NLCARE_ALLOW_PAID_PROVIDER_PROBE": "true"},
    )
    assert result["status"] == "blocked_configuration"
    assert "provider_credential_missing" in result["execution_blockers"]
    assert result["request_count"] == 0


def test_probe_blocks_non_loopback_without_second_approval():
    result = build_provider_api_path_capture(
        execute=True,
        base_url="https://staging.example.test",
        env=_env(),
    )
    assert result["target_allowed"] is False
    assert result["request_count"] == 0


def test_completed_probe_keeps_only_hashes_and_operational_metadata():
    def requester(method, url, headers, payload):
        if url.endswith("/auth/demo-credential-login"):
            return 200, {"access_token": "token"}
        return 200, {
            "reply": "content that must not be retained",
            "llm_telemetry": {
                "provider_reported_call_count": 1,
                "total_tokens": 100,
                "estimated_cost_usd": 0.0001,
            },
            "rag_evaluation": {
                "status": "strong",
                "source_tier_correct": True,
            },
            "guardrails": {"output": {"unsafe_leakage": False}},
        }

    result = build_provider_api_path_capture(
        execute=True,
        request_count=30,
        env=_env(),
        request_fn=requester,
    )
    assert result["status"] == "completed_controlled_probe"
    assert result["provider_usage_coverage_rate"] == 1.0
    assert result["request_count"] == 30
    assert result["provider_reported_total_tokens"] == 3000
    assert result["content_retained"] is False
    assert result["patient_data_processed"] is False
    serialized = str(result)
    assert "content that must not be retained" not in serialized
    assert "How do I open" not in serialized
    assert all(len(row["prompt_sha256"]) == 64 for row in result["requests"])


def test_estimated_only_telemetry_does_not_count_as_provider_reported():
    def requester(method, url, headers, payload):
        if url.endswith("/auth/demo-credential-login"):
            return 200, {"access_token": "token"}
        return 200, {
            "llm_telemetry": {
                "provider_reported_call_count": 0,
                "total_tokens": 80,
                "estimated_cost_usd": 0.00008,
            }
        }

    result = build_provider_api_path_capture(
        execute=True,
        env=_env(),
        request_fn=requester,
    )
    assert result["status"] == "insufficient_provider_capture"
    assert result["provider_usage_coverage_rate"] == 0.0
    assert result["completed"] is False


def test_artifact_retains_nonclinical_boundaries():
    result = build_provider_api_path_capture(env={})
    assert result["estimated_cost_usd"] is None
    assert result["clinical_validation"] is False
    assert result["healthcare_production_ready"] is False
    assert result["audited_billing"] is False
    assert "not clinical validation" in result["claim_boundary"]
