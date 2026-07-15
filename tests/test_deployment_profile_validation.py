from backend.services.deployment_profile_validation import build_report


def test_development_profile_is_honest_and_non_clinical():
    report = build_report({"ENVIRONMENT": "development", "DATABASE_URL": "sqlite:///demo.db"})
    assert report["status"] == "development_profile"
    assert report["strict_profile"] is False
    assert report["clinical_validation"] is False
    assert report["healthcare_production_ready"] is False
    assert report["secrets_included_in_artifact"] is False


def test_production_profile_rejects_unsafe_configuration():
    report = build_report({
        "ENVIRONMENT": "production",
        "DATABASE_URL": "postgresql+psycopg2://medical_agent:change_me@db/medical_agent",
        "REDIS_URL": "redis://redis:6379/0",
        "ALLOW_DEMO_AUTH": "true",
        "ONCOTRACK_CORS_ORIGINS": "*",
    })
    assert report["status"] == "blocked"
    failed = {item["name"] for item in report["failed_checks"]}
    assert {"database_password_not_placeholder", "demo_auth_disabled", "cors_allowlist_explicit", "production_cors_uses_https"} <= failed


def test_valid_production_shaped_profile_passes_runtime_configuration_checks():
    report = build_report({
        "ENVIRONMENT": "production",
        "DATABASE_URL": "postgresql+psycopg2://medical_agent:a-unique-secret@db/medical_agent",
        "REDIS_URL": "rediss://redis.example:6380/0",
        "ALLOW_DEMO_AUTH": "false",
        "ONCOTRACK_CORS_ORIGINS": "https://nlcare.example",
    })
    assert report["status"] == "strong"
    assert report["n_failed"] == 0
    assert report["healthcare_production_ready"] is False
