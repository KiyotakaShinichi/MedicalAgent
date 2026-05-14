"""Cross-language wire-contract test for review / risk / confidence constants.

The backend Python module ``backend/services/review_constants.py`` and the
frontend TypeScript module ``frontend-react/src/lib/constants.ts`` define
the same canonical strings (review decisions, targets, reason categories,
risk levels, confidence levels, artifact statuses).

This test parses the TypeScript file textually (no JS toolchain in the
Python CI step) and asserts every enum array contains the same values
as the Python tuple. A linter or hand-edit that desyncs the two sides
will fail this test, preventing a wire-incompatibility regression
between the React frontend and the FastAPI backend.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from backend.services.review_constants import (
    ARTIFACT_STATUSES,
    CONFIDENCE_LEVELS,
    EVENT_TYPES,
    REVIEW_DECISIONS,
    REVIEW_REASON_CATEGORIES,
    REVIEW_TARGETS,
    RISK_LEVELS,
    SAFETY_REFUSAL_TYPES,
)


CONSTANTS_TS = (
    Path(__file__).resolve().parents[1]
    / "frontend-react"
    / "src"
    / "lib"
    / "constants.ts"
)


def _extract_string_array(source: str, name: str) -> list[str]:
    """Parse `export const NAME = [...] as const;` and return the strings."""

    pattern = rf"export\s+const\s+{re.escape(name)}\s*=\s*\[([^\]]*)\]\s*as\s+const"
    match = re.search(pattern, source, re.MULTILINE | re.DOTALL)
    if not match:
        raise AssertionError(f"frontend constants.ts is missing array `{name}`")
    body = match.group(1)
    return [m.group(1) for m in re.finditer(r"[\"']([^\"']+)[\"']", body)]


class ConstantsSyncTests(unittest.TestCase):
    """Each enum must contain the same set of strings on both sides."""

    @classmethod
    def setUpClass(cls):
        if not CONSTANTS_TS.exists():
            raise unittest.SkipTest(
                f"frontend constants file not found at {CONSTANTS_TS}"
            )
        cls.source = CONSTANTS_TS.read_text(encoding="utf-8")

    def _assert_matches(self, name: str, python_values: tuple[str, ...]) -> None:
        ts_values = _extract_string_array(self.source, name)
        self.assertEqual(
            set(ts_values),
            set(python_values),
            f"frontend `{name}` and backend `{name}` diverged: "
            f"FE-only={sorted(set(ts_values) - set(python_values))}, "
            f"BE-only={sorted(set(python_values) - set(ts_values))}",
        )

    def test_review_decisions_match(self):
        self._assert_matches("REVIEW_DECISIONS", REVIEW_DECISIONS)

    def test_review_targets_match(self):
        self._assert_matches("REVIEW_TARGETS", REVIEW_TARGETS)

    def test_review_reason_categories_match(self):
        self._assert_matches("REVIEW_REASON_CATEGORIES", REVIEW_REASON_CATEGORIES)

    def test_risk_levels_match(self):
        self._assert_matches("RISK_LEVELS", RISK_LEVELS)

    def test_confidence_levels_match(self):
        self._assert_matches("CONFIDENCE_LEVELS", CONFIDENCE_LEVELS)

    def test_artifact_statuses_match(self):
        self._assert_matches("ARTIFACT_STATUSES", ARTIFACT_STATUSES)

    def test_event_types_are_documented(self):
        # EVENT_TYPES and SAFETY_REFUSAL_TYPES are not mirrored as TS arrays yet
        # (they're surfaced only as backend artifacts). Pin them as a contract
        # so adding them to the frontend in the future has a clear target.
        for value in EVENT_TYPES:
            self.assertIsInstance(value, str)
        for value in SAFETY_REFUSAL_TYPES:
            self.assertIsInstance(value, str)


if __name__ == "__main__":
    unittest.main()
