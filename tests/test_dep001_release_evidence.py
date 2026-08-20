from __future__ import annotations

from copy import deepcopy

import pytest

from backend.services.dep001_release_evidence import build_release_compatible_evidence


def _canonical() -> dict:
    return {
        "schema_version": "dep001_external_safety_evaluation_v1",
        "status": "failed",
        "dep001_complete": False,
        "dep001_decision": "BLOCKED",
        "metrics": {
            "unsafe_released_output_rate": 0.0,
            "en_taglish_absolute_gap": 0.03,
            "indirect_intent_unsafe_recall": 0.94,
            "obfuscated_unsafe_recall": 1.0,
            "multi_turn_unsafe_recall": 1.0,
            "rag_conditioned_unsafe_recall": 1.0,
        },
        "fault_injection": {"passed": True},
    }


def test_compatibility_aliases_preserve_official_failure() -> None:
    canonical = _canonical()
    derived = build_release_compatible_evidence(
        canonical,
        source_path="canonical.json",
        source_sha256="a" * 64,
    )
    assert derived["status"] == "failed"
    assert derived["dep001_complete"] is False
    assert derived["dep001_decision"] == "BLOCKED"
    assert derived["metrics"]["unsafe_pass_rate"] == 0.0
    assert derived["metrics"]["en_taglish_parity"] == pytest.approx(0.97)
    assert derived["metrics"]["adversarial_paraphrase_robustness"] == 0.94
    assert derived["metrics"]["multi_turn_safety"] == 1.0
    assert derived["metrics"]["rag_conditioned_safety"] == 1.0
    assert derived["metrics"]["failure_path_safety"] == 1.0
    assert derived["release_compatibility"]["official_decision_unchanged"] is True


def test_builder_does_not_mutate_canonical() -> None:
    canonical = _canonical()
    before = deepcopy(canonical)
    build_release_compatible_evidence(
        canonical,
        source_path="canonical.json",
        source_sha256="b" * 64,
    )
    assert canonical == before


def test_missing_canonical_metric_fails_closed() -> None:
    canonical = _canonical()
    del canonical["metrics"]["multi_turn_unsafe_recall"]
    with pytest.raises(ValueError, match="missing required metrics"):
        build_release_compatible_evidence(
            canonical,
            source_path="canonical.json",
            source_sha256="c" * 64,
        )
