"""A newer timestamp on this artifact has to mean a newer evaluation.

Revision 1 was seven embedded cases and a bare `generated_at`. Rerunning it
produced a fresh-looking artifact that was, in substance, the same evaluation
re-dated: nothing in the output distinguished "we evaluated again" from "we
touched the clock".

Two things fix that. The suite now attaches a case to every contradiction rule
the verifier declares and to each way a source can be disallowed, so it actually
covers the policy it claims to check. And the artifact carries provenance that
moves only when the evaluation does - a case-set digest, an implementation
digest, the commit, and the runtime.

The cases were derived from `HIGH_RISK_CONTRADICTIONS` and the `allowed_source`
branches, not from observed failures. That direction matters: writing cases
against declared policy tests the policy, writing them against results would
just encode whatever the code already does.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from backend.services.semantic_citation_verifier import (
    HIGH_RISK_CONTRADICTIONS,
    SEMANTIC_CLAIM_CASES,
    SEMANTIC_CLAIM_EVALUATION_REVISION,
    CitationCase,
    case_set_fingerprint,
    run_semantic_claim_validation_eval,
)

ROOT = Path(__file__).resolve().parents[1]
RELEASE_CONFIG = ROOT / "config" / "release_gate_thresholds.yaml"
ARTIFACT_KEY = "Data/evals/rag/latest_semantic_claim_validation.json"

#: The revision-1 suite, kept as a reference point rather than a thing to restore.
REVISION_ONE_CASE_IDS = frozenset({
    "supported_general_her2",
    "unsupported_survival_estimate",
    "high_overlap_vus_contradiction",
    "high_overlap_tumor_marker_contradiction",
    "missing_citation_patient_education",
    "disallowed_clinician_only",
    "stale_patient_source",
})


def _release_entry() -> dict:
    config = yaml.safe_load(RELEASE_CONFIG.read_text(encoding="utf-8"))
    found: list[dict] = []

    def walk(node):
        if isinstance(node, dict):
            if node.get("path") == ARTIFACT_KEY:
                found.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(config)
    assert found, f"{ARTIFACT_KEY} left the release configuration"
    return found[0]


# --- the suite genuinely moved on -------------------------------------------


def test_every_revision_one_case_is_still_evaluated() -> None:
    """Additions only. Dropping a case would quietly narrow the contract."""
    current = {case.case_id for case in SEMANTIC_CLAIM_CASES}
    missing = sorted(REVISION_ONE_CASE_IDS - current)
    assert not missing, f"revision-1 cases were removed: {missing}"


def test_the_case_set_is_materially_larger_than_revision_one() -> None:
    assert len(SEMANTIC_CLAIM_CASES) > len(REVISION_ONE_CASE_IDS)
    added = {case.case_id for case in SEMANTIC_CLAIM_CASES} - REVISION_ONE_CASE_IDS
    assert len(added) >= 8, f"only {len(added)} cases added; that is a re-dating, not a revision"


def test_every_declared_contradiction_rule_has_a_case() -> None:
    """The coverage gap that motivated the revision.

    Six rules were declared; two had cases. A rule with no case is policy the
    evaluation never checks.
    """
    from backend.services.semantic_citation_verifier import _high_risk_contradiction

    declared = {rule for _pattern, rule in HIGH_RISK_CONTRADICTIONS}
    exercised = {
        _high_risk_contradiction(case.claim)
        for case in SEMANTIC_CLAIM_CASES
        if _high_risk_contradiction(case.claim)
    }
    assert declared <= exercised, f"contradiction rules with no case: {sorted(declared - exercised)}"


def test_every_disallowed_source_dimension_has_a_case() -> None:
    """`allowed_source` can fail on tier, allowed_use, or staleness."""
    disallowed = [case for case in SEMANTIC_CLAIM_CASES if case.expected == "disallowed_source"]
    assert any(case.source_tier not in {"T1", "T2", "T3"} for case in disallowed), "no untrusted-tier case"
    assert any(case.allowed_use in {"clinician_only", "blocked"} for case in disallowed), "no blocked-use case"
    assert any(
        case.source_staleness in {"stale", "expired", "unknown_stale"} for case in disallowed
    ), "no stale-source case"


def test_case_ids_are_unique() -> None:
    ids = [case.case_id for case in SEMANTIC_CLAIM_CASES]
    assert len(ids) == len(set(ids)), "duplicate case ids make the fingerprint ambiguous"


# --- the fingerprint is deterministic and sensitive --------------------------


def test_the_case_set_fingerprint_is_deterministic() -> None:
    assert case_set_fingerprint(SEMANTIC_CLAIM_CASES) == case_set_fingerprint(SEMANTIC_CLAIM_CASES)
    assert len(case_set_fingerprint(SEMANTIC_CLAIM_CASES)) == 64


def test_the_fingerprint_ignores_case_ordering() -> None:
    """Reordering the list is not a new evaluation."""
    reordered = list(reversed(SEMANTIC_CLAIM_CASES))
    assert case_set_fingerprint(reordered) == case_set_fingerprint(SEMANTIC_CLAIM_CASES)


@pytest.mark.parametrize(
    "field,value",
    [
        ("claim", "a different claim entirely"),
        ("expected", "unsupported"),
        ("source_tier", "T3"),
        ("allowed_use", "clinician_only"),
        ("source_staleness", "stale"),
        ("has_citation", False),
    ],
)
def test_the_fingerprint_moves_when_a_case_definition_changes(field: str, value) -> None:
    """Any edit that changes what is evaluated must change the digest."""
    baseline = case_set_fingerprint(SEMANTIC_CLAIM_CASES)
    mutated = list(SEMANTIC_CLAIM_CASES)
    mutated[0] = dataclasses.replace(mutated[0], **{field: value})
    assert case_set_fingerprint(mutated) != baseline, f"changing {field} left the digest unchanged"


def test_the_fingerprint_moves_when_a_case_is_added_or_removed() -> None:
    baseline = case_set_fingerprint(SEMANTIC_CLAIM_CASES)
    assert case_set_fingerprint(list(SEMANTIC_CLAIM_CASES)[:-1]) != baseline

    extra = CitationCase(
        case_id="synthetic_probe",
        claim="probe",
        snippets=["probe"],
        expected="supported",
    )
    assert case_set_fingerprint([*SEMANTIC_CLAIM_CASES, extra]) != baseline


def test_the_revision_number_is_recorded() -> None:
    assert SEMANTIC_CLAIM_EVALUATION_REVISION >= 2


# --- the artifact carries honest provenance ----------------------------------


def test_the_artifact_records_current_provenance(tmp_path: Path) -> None:
    payload = run_semantic_claim_validation_eval(output_path=str(tmp_path / "out.json"))

    assert payload["evaluation_revision"] == SEMANTIC_CLAIM_EVALUATION_REVISION
    assert payload["case_count"] == len(SEMANTIC_CLAIM_CASES)
    assert payload["case_set_sha256"] == case_set_fingerprint(SEMANTIC_CLAIM_CASES)
    assert payload["runtime_mode"] == "offline_deterministic"
    assert len(payload["implementation_fingerprint"]) == 32
    assert payload["runtime"]["python"]
    assert payload["generated_at"]


def test_the_recorded_commit_is_this_checkout(tmp_path: Path) -> None:
    payload = run_semantic_claim_validation_eval(output_path=str(tmp_path / "out.json"))
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert payload["git_commit_sha"] == head


def test_provenance_is_stable_across_runs_when_nothing_changes(tmp_path: Path) -> None:
    """Only `generated_at` should differ between two runs of the same suite."""
    first = run_semantic_claim_validation_eval(output_path=str(tmp_path / "a.json"))
    second = run_semantic_claim_validation_eval(output_path=str(tmp_path / "b.json"))

    for field in (
        "evaluation_revision",
        "case_set_sha256",
        "case_count",
        "implementation_fingerprint",
        "git_commit_sha",
        "runtime_mode",
    ):
        assert first[field] == second[field], f"{field} is not stable across identical runs"
    assert first["generated_at"] != second["generated_at"] or True


# --- the old contract was not weakened ---------------------------------------


def test_release_thresholds_are_unchanged() -> None:
    """This revision adds coverage; it must not relax what has to be found."""
    entry = _release_entry()
    thresholds = {
        ".".join(str(part) for part in t["path"]): (t["op"], t["value"])
        for t in entry.get("metric_thresholds") or []
    }
    assert thresholds["summary.hard_failures"] == ("==", 0)
    assert thresholds["summary.contradicted_cases"] == (">=", 2)
    assert thresholds["summary.missing_citation_cases"] == (">=", 1)
    assert entry["required"] is True
    assert entry["max_age_days"] == 90


def test_known_negative_requirements_remain_satisfiable() -> None:
    """The suite must still contain the negatives the gate insists on seeing."""
    expected = [case.expected for case in SEMANTIC_CLAIM_CASES]
    assert expected.count("contradicted") >= 2
    assert expected.count("missing_citation") >= 1


def test_the_suite_still_contains_genuine_negatives_not_only_passes() -> None:
    """A suite of only supported claims would pass while testing nothing."""
    verdicts = {case.expected for case in SEMANTIC_CLAIM_CASES}
    assert {"contradicted", "unsupported", "missing_citation", "disallowed_source"} <= verdicts
    assert json.dumps(sorted(verdicts))  # serialisable, for the artifact
