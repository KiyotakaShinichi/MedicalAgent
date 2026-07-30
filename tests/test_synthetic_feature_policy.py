from __future__ import annotations

from backend.services.synthetic_feature_policy import (
    CANONICAL_PROMOTION_NUMERIC_FEATURES,
    DIRECT_OR_NEAR_DIRECT_TARGET_PROXIES,
    LEGACY_NUMERIC_FEATURES,
    POLICY_ID,
    build_synthetic_feature_policy,
)


def test_canonical_policy_removes_every_declared_target_proxy():
    proxies = {
        feature
        for features in DIRECT_OR_NEAR_DIRECT_TARGET_PROXIES.values()
        for feature in features
    }
    assert proxies <= set(LEGACY_NUMERIC_FEATURES)
    assert proxies.isdisjoint(CANONICAL_PROMOTION_NUMERIC_FEATURES)


def test_policy_separates_legacy_replay_from_promotion():
    payload = build_synthetic_feature_policy()
    assert payload["policy_id"] == POLICY_ID
    assert payload["legacy_scope"]["promotion_eligible"] is False
    assert payload["canonical_promotion_policy"]["promotion_eligible"] is True
    assert payload["clinical_validation"] is False
    assert payload["production_ready"] is False
    boundary = payload["claim_boundary"].lower()
    assert "does not establish clinical" in boundary
    assert "production healthcare readiness" in boundary


def test_proxy_removed_policy_is_not_a_broader_feature_set():
    assert set(CANONICAL_PROMOTION_NUMERIC_FEATURES) < set(
        LEGACY_NUMERIC_FEATURES
    )
