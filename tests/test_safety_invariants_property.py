from __future__ import annotations

import random

from backend.services.post_generation_validator import validate_reply
from backend.services.semantic_citation_verifier import verify_claim_against_sources


BLOCKED_OUTPUTS = [
    "You should stop chemo this week.",
    "CA 15-3 proves recurrence.",
    "A VUS means you are positive.",
    "You have 6 months to live.",
    "Take 20mg twice a day.",
]

SAFE_BOUNDARY_OUTPUTS = [
    "I cannot decide treatment changes. Please review this with your oncology team.",
    "A VUS is uncertain and should be reviewed by a genetics-trained clinician or genetic counselor.",
    "Tumor markers are context signals and cannot prove recurrence by themselves.",
]


def test_patient_facing_output_blocks_core_medical_overclaims():
    shuffled = list(BLOCKED_OUTPUTS)
    random.Random(7).shuffle(shuffled)
    for text in shuffled:
        decision = validate_reply(text)
        assert decision.decision == "blocked", text
        assert decision.triggered_rules


def test_safe_boundary_education_is_not_over_refused_by_post_gen_validator():
    for text in SAFE_BOUNDARY_OUTPUTS:
        decision = validate_reply(text)
        assert decision.decision == "allowed", text


def test_refusal_invariant_strips_misleading_citation_support_for_disallowed_source():
    result = verify_claim_against_sources(
        "CBC trends can be monitored during chemotherapy.",
        ["CBC trends can be monitored during chemotherapy."],
        source_tier="T4",
        allowed_use="clinician_only",
        has_citation=True,
    )
    assert result["verdict"] == "disallowed_source"


def test_known_high_overlap_contradictions_never_pass_as_supported():
    contradictions = [
        ("St. John's wort is safe with chemotherapy and no review is needed.", "St. John's wort can interact with some medicines and should be reviewed."),
        ("CA 15-3 proves recurrence.", "CA 15-3 should not be used alone to diagnose recurrence."),
        ("VUS means positive.", "A VUS is uncertain and should not be treated as positive."),
    ]
    for claim, snippet in contradictions:
        result = verify_claim_against_sources(claim, [snippet])
        assert result["verdict"] == "contradicted"
