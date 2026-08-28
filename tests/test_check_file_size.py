"""The backend service size ratchet does what it claims.

`scripts/check_file_size.py` is only worth having if it actually blocks the two
things it exists to block: a new oversized service module, and an existing
oversized one getting bigger. A ratchet that silently passes is worse than no
ratchet, because it looks like coverage.

These tests drive `evaluate` and `write_baseline` directly with synthetic
sizes, so they assert the policy rather than the current contents of the
repository — which will change as the debt shrinks.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_file_size import (  # noqa: E402
    DEFAULT_BASELINE,
    DEFAULT_MAX_LOC,
    evaluate,
    is_generated,
    load_baseline,
    main,
    physical_loc,
    tracked_source_files,
    write_baseline,
)

MAX = 500


# ─── the policy ──────────────────────────────────────────────────────────────


def test_a_new_oversized_file_fails() -> None:
    """New code has no excuse: not in the baseline and over the limit."""
    problems = evaluate({"backend/services/new_service.py": 501}, {}, MAX)
    assert len(problems) == 1
    assert "new_service.py" in problems[0]
    assert "501" in problems[0]


def test_a_new_file_at_the_limit_passes() -> None:
    """The limit is inclusive; 500 is allowed, 501 is not."""
    assert evaluate({"backend/services/new_service.py": MAX}, {}, MAX) == []


def test_a_baseline_file_that_grows_fails() -> None:
    """Known debt is frozen: it may shrink, never grow."""
    problems = evaluate(
        {"backend/services/legacy.py": 901}, {"backend/services/legacy.py": 900}, MAX
    )
    assert len(problems) == 1
    assert "above its recorded baseline" in problems[0]


def test_a_baseline_file_that_shrinks_passes() -> None:
    assert (
        evaluate({"backend/services/legacy.py": 700}, {"backend/services/legacy.py": 900}, MAX)
        == []
    )


def test_a_baseline_file_that_stays_the_same_passes() -> None:
    assert (
        evaluate({"backend/services/legacy.py": 900}, {"backend/services/legacy.py": 900}, MAX)
        == []
    )


def test_multiple_violations_are_all_reported() -> None:
    """One message per file, so a contributor fixes them in one pass."""
    problems = evaluate(
        {"backend/services/a.py": 600, "backend/services/b.py": 901, "backend/services/c.py": 10},
        {"backend/services/b.py": 900},
        MAX,
    )
    assert len(problems) == 2


def test_violation_messages_say_what_to_do() -> None:
    """A failing gate that does not explain itself gets bypassed."""
    new_file = evaluate({"backend/services/new.py": 700}, {}, MAX)[0]
    grown = evaluate({"backend/services/old.py": 901}, {"backend/services/old.py": 900}, MAX)[0]
    assert "Split it by responsibility" in new_file
    assert "focused module" in grown


# ─── the baseline is shrink-only ─────────────────────────────────────────────


def test_update_lowers_a_recorded_size(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    result = write_baseline(
        baseline_path,
        {"backend/services/legacy.py": 700},
        {"backend/services/legacy.py": 900},
        MAX,
    )
    assert load_baseline(baseline_path)["backend/services/legacy.py"] == 700
    assert result["lowered"] == ["backend/services/legacy.py 900 -> 700"]


def test_update_refuses_to_raise_a_recorded_size(tmp_path: Path) -> None:
    """This is the property that makes the ratchet a ratchet.

    Without it, `--update` would be a one-command way to legalise growth, and
    the check would enforce nothing at all.
    """
    baseline_path = tmp_path / "baseline.json"
    write_baseline(
        baseline_path,
        {"backend/services/legacy.py": 950},
        {"backend/services/legacy.py": 900},
        MAX,
    )
    assert load_baseline(baseline_path)["backend/services/legacy.py"] == 900

    # And the frozen value still fails the grown file.
    assert evaluate({"backend/services/legacy.py": 950}, load_baseline(baseline_path), MAX)


def test_update_drops_a_file_that_came_under_the_limit(tmp_path: Path) -> None:
    """A remediated file leaves the debt list and cannot silently regrow."""
    baseline_path = tmp_path / "baseline.json"
    result = write_baseline(
        baseline_path,
        {"backend/services/fixed.py": 200},
        {"backend/services/fixed.py": 900},
        MAX,
    )
    assert "backend/services/fixed.py" not in load_baseline(baseline_path)
    assert "backend/services/fixed.py" in result["dropped"]

    # Regrowing it now fails as a brand-new offender.
    assert evaluate({"backend/services/fixed.py": 600}, load_baseline(baseline_path), MAX)


def test_update_drops_a_deleted_file(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    result = write_baseline(baseline_path, {}, {"backend/services/gone.py": 900}, MAX)
    assert load_baseline(baseline_path) == {}
    assert "backend/services/gone.py" in result["dropped"]


def test_baseline_file_explains_itself(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    write_baseline(baseline_path, {"backend/services/legacy.py": 900}, {}, MAX)
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert "shrink-only" in payload["_comment"]
    assert payload["max_loc"] == MAX


# ─── measurement ─────────────────────────────────────────────────────────────


def test_line_count_is_newline_style_independent(tmp_path: Path) -> None:
    """A CRLF checkout and an LF checkout must agree, or the baseline is not portable."""
    lf = tmp_path / "lf.py"
    crlf = tmp_path / "crlf.py"
    lf.write_bytes(b"a = 1\nb = 2\nc = 3\n")
    crlf.write_bytes(b"a = 1\r\nb = 2\r\nc = 3\r\n")
    assert physical_loc(lf) == physical_loc(crlf) == 3


def test_generated_trees_are_excluded_narrowly() -> None:
    assert is_generated("backend/migrations/versions/0012_x.py")
    assert is_generated("backend/services/__pycache__/x.py")
    assert not is_generated("backend/services/saas_jobs.py")
    assert not is_generated("backend/services/migrations_helper.py")
    assert is_generated("frontend-react/src/types/generated-openapi.d.ts")


def test_tracked_source_files_supports_frontend_suffixes() -> None:
    files = tracked_source_files(ROOT, ROOT / "frontend-react/src", (".ts", ".tsx", ".css"))
    relative = {path.relative_to(ROOT).as_posix() for path in files}
    assert "frontend-react/src/api/client.ts" in relative
    assert "frontend-react/src/types/generated-openapi.d.ts" not in relative


# ─── the repository's own state ──────────────────────────────────────────────


def test_the_repository_currently_passes_its_own_ratchet() -> None:
    assert (
        main(
            [
                "backend",
                "--baseline",
                "tests/contracts/backend_authored_loc_baseline.json",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "frontend-react/src",
                "--extensions",
                ".ts,.tsx,.css",
                "--baseline",
                "tests/contracts/frontend_authored_loc_baseline.json",
            ]
        )
        == 0
    )


def test_the_committed_baseline_excludes_the_remediated_modules() -> None:
    """The two modules split in this sprint must be under the limit for real."""
    baseline = load_baseline(DEFAULT_BASELINE)
    for remediated in (
        "backend/services/saas_control_plane.py",
        "backend/services/governance_credibility_artifacts.py",
    ):
        assert remediated not in baseline, f"{remediated} should be under the limit now"
        assert physical_loc(ROOT / remediated) <= DEFAULT_MAX_LOC


@pytest.mark.parametrize(
    "module",
    [
        "backend/services/saas_common.py",
        "backend/services/saas_organizations.py",
        "backend/services/saas_projects.py",
        "backend/services/saas_jobs.py",
        "backend/services/governance_artifacts/negative_results.py",
        "backend/services/governance_artifacts/portfolio_claim_safety.py",
        "backend/services/governance_artifacts/contamination_harmonization.py",
        "backend/services/governance_artifacts/noisier_v2_readiness.py",
        "backend/services/agent_safety.py",
        "backend/services/agent_safety_vocab.py",
        "backend/services/agent_safety_rules.py",
        "backend/services/unsafe_intent_semantic_classifier.py",
        "backend/services/unsafe_intent_families.py",
        "backend/services/unsafe_intent_compositional_rules.py",
        "backend/api/routers/patient_interactions.py",
        "backend/api/routers/patient_interaction_records.py",
        "backend/api/routers/patient_interaction_genetics.py",
        "backend/api/routers/patient_interaction_support.py",
    ],
)
def test_every_module_created_by_this_split_is_within_the_limit(module: str) -> None:
    loc = physical_loc(ROOT / module)
    assert loc <= DEFAULT_MAX_LOC, f"{module} is {loc} LOC"


def test_missing_target_directory_is_an_error_not_a_pass() -> None:
    """A typo'd path in CI must not look like a green check."""
    assert main(["backend/does_not_exist"]) == 2
