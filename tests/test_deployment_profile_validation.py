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
    assert report["status"] == "blocked"
    failed = {item["name"] for item in report["failed_checks"]}
    assert failed == {"non_demo_identity_provider_integrated"}
    assert report["deployment_capability"] == "strict_profile_blocked"
    assert report["healthcare_production_ready"] is False


def test_production_shaped_profile_accepts_complete_oidc_configuration_only_as_engineering_readiness():
    report = build_report({
        "ENVIRONMENT": "production",
        "DATABASE_URL": "postgresql+psycopg2://medical_agent:a-unique-secret@db/medical_agent",
        "REDIS_URL": "rediss://redis.example:6380/0",
        "ALLOW_DEMO_AUTH": "false",
        "NLCARE_CORS_ORIGINS": "https://nlcare.example",
        "NLCARE_OIDC_ENABLED": "true",
        "NLCARE_OIDC_ISSUER": "https://identity.example",
        "NLCARE_OIDC_AUDIENCE": "nlcare-api",
        "NLCARE_OIDC_JWKS_URL": "https://identity.example/.well-known/jwks.json",
        "NLCARE_OIDC_ALGORITHMS": "RS256",
    })
    assert report["status"] == "strong"
    assert report["oidc_config_valid"] is True
    assert report["deployment_capability"] == "strict_profile_configured_not_healthcare_ready"
    assert report["healthcare_production_ready"] is False


def test_enabled_external_dispatch_fails_closed_on_insecure_configuration():
    report = build_report({
        "ENVIRONMENT": "development",
        "DATABASE_URL": "sqlite:///demo.db",
        "N8N_WEBHOOK_DISPATCH_ENABLED": "true",
        "N8N_WEBHOOK_BASE_URL": "http://127.0.0.1:5678/webhook/nlcare",
        "N8N_WEBHOOK_SIGNING_SECRET": "replace_with_a_long_random_secret",
        "NLCARE_ALERT_TEST_RECIPIENT_ONLY": "false",
    })
    failed = {item["name"] for item in report["failed_checks"]}
    assert {
        "external_dispatch_uses_https",
        "external_dispatch_secret_strong",
        "external_dispatch_test_recipient_only",
    } <= failed


def test_disabled_external_dispatch_does_not_require_channel_secrets():
    report = build_report({
        "ENVIRONMENT": "development",
        "DATABASE_URL": "sqlite:///demo.db",
        "N8N_WEBHOOK_DISPATCH_ENABLED": "false",
    })
    failed = {item["name"] for item in report["failed_checks"]}
    assert not any(name.startswith("external_dispatch_") for name in failed)
