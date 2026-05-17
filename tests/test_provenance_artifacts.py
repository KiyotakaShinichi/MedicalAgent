"""Tests for the synthetic generator card + failure-mode registry.

These two artifacts are the project's "documented provenance" surfaces.
The tests guard the *contract*, not the curated content — a future reviewer
should be able to trust the structure even if the narrative wording changes.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from backend.services.failure_mode_registry import (
    ENGINEERING_RISKS,
    build_failure_mode_registry,
    load_failure_mode_registry,
)
from backend.services.synthetic_generator_card import (
    CAUSAL_ASSUMPTIONS,
    GENERATOR_CARD_VERSION,
    KNOWN_SHORTCUTS,
    UNSUPPORTED_CLAIMS,
    build_synthetic_generator_card,
    load_synthetic_generator_card,
)


# ─── Synthetic generator card ────────────────────────────────────────────────


class GeneratorCardContract(unittest.TestCase):
    """The card must aggregate dataset metadata into a stable, JSON-safe
    structure with every required block populated."""

    REQUIRED_TOP_LEVEL = (
        "schema_version", "generator_card_version", "status",
        "dataset_dir", "dataset_schema_version",
        "card_version_matches_dataset", "generation_options", "cohort",
        "supported_labels", "feature_distribution_summary",
        "causal_assumptions", "known_shortcuts", "unsupported_claims",
        "realism_checks_referenced", "claim_boundary",
    )

    def test_curated_narrative_blocks_are_non_empty(self) -> None:
        self.assertGreater(len(CAUSAL_ASSUMPTIONS), 0)
        self.assertGreater(len(KNOWN_SHORTCUTS), 0)
        self.assertGreater(len(UNSUPPORTED_CLAIMS), 0)
        # Each entry should be a sentence-length string, not a placeholder.
        for entry in (*CAUSAL_ASSUMPTIONS, *KNOWN_SHORTCUTS, *UNSUPPORTED_CLAIMS):
            self.assertGreater(len(entry), 20, f"narrative entry too short: {entry!r}")

    def test_card_builds_from_tiny_synthetic_dataset(self) -> None:
        with TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "ds"
            dataset_dir.mkdir()
            (dataset_dir / "summary.json").write_text(json.dumps({
                "patients_created": 5,
                "cycles_per_patient": 2,
                "generation_options": {
                    "seed": 7,
                    "schema_version": "complete_synthetic_breast_journey_v2",
                },
                "table_counts": {"patients": 5, "temporal_ml_rows": 10},
            }))
            # Minimal temporal CSV with the columns the distribution summary reads.
            pd.DataFrame([{
                "patient_id": f"P{i}",
                "cycle": (i % 2) + 1,
                "age": 50 + i,
                "stage": "II",
                "molecular_subtype": "HR_positive",
                "regimen": "AC_T",
                "pre_wbc": 6.0, "pre_anc": 3.0, "nadir_wbc": 2.0,
                "mri_percent_change_from_baseline": -10.0,
                "max_symptom_severity": 3,
                "response_score_percent": 12.0,
                "final_response_category": "partial_response",
                "final_response_multiclass": "PR",
                "treatment_success_binary": i % 2,
            } for i in range(10)]).to_csv(dataset_dir / "temporal_ml_rows.csv", index=False)

            out_path = Path(tmp) / "card.json"
            payload = build_synthetic_generator_card(
                dataset_dir=str(dataset_dir), output_path=str(out_path),
            )

            self.assertEqual(payload["status"], "passed")
            for key in self.REQUIRED_TOP_LEVEL:
                self.assertIn(key, payload, f"generator card missing {key}")
            self.assertEqual(payload["generator_card_version"], GENERATOR_CARD_VERSION)
            self.assertTrue(payload["card_version_matches_dataset"])
            self.assertEqual(payload["cohort"]["patients_created"], 5)
            self.assertIsNotNone(payload["cohort"]["rows_fingerprint"])
            self.assertGreaterEqual(payload["feature_distribution_summary"]["row_count"], 10)
            self.assertTrue(out_path.exists())

    def test_loader_returns_missing_shell_when_file_absent(self) -> None:
        with TemporaryDirectory() as tmp:
            payload = load_synthetic_generator_card(
                path=str(Path(tmp) / "does_not_exist.json"),
            )
            self.assertEqual(payload["status"], "missing")
            self.assertIn("message", payload)


# ─── Failure-mode registry ───────────────────────────────────────────────────


class FailureModeRegistryContract(unittest.TestCase):
    """The registry must always include the hand-curated engineering risks
    and gracefully aggregate optional sources without crashing when they're
    missing."""

    REQUIRED_ENTRY_FIELDS = (
        "name", "category", "example", "risk",
        "detection", "mitigation", "benchmark_coverage",
        "remaining_gap", "severity",
    )

    def test_engineering_risks_all_have_required_fields(self) -> None:
        for risk in ENGINEERING_RISKS:
            for field in self.REQUIRED_ENTRY_FIELDS:
                self.assertIn(
                    field, risk,
                    f"engineering risk {risk.get('name')} missing {field}",
                )
            self.assertIsInstance(risk["benchmark_coverage"], list)
            self.assertIn(risk["severity"], {"high", "medium", "low"})

    def test_registry_builds_with_no_optional_sources_present(self) -> None:
        with TemporaryDirectory() as tmp:
            payload = build_failure_mode_registry(
                failure_gallery_path=str(Path(tmp) / "absent.json"),
                safety_red_team_path=str(Path(tmp) / "absent.json"),
                drift_report_path=str(Path(tmp) / "absent.json"),
                output_path=str(Path(tmp) / "registry.json"),
            )
            # Only engineering risks should remain.
            self.assertEqual(payload["entry_count"], len(ENGINEERING_RISKS))
            self.assertEqual(payload["summary"]["by_severity"]["high"],
                             sum(1 for r in ENGINEERING_RISKS if r["severity"] == "high"))
            self.assertIn(payload["status"], {"strong", "acceptable", "needs_attention"})

    def test_failure_gallery_cases_are_imported(self) -> None:
        with TemporaryDirectory() as tmp:
            gallery_path = Path(tmp) / "gallery.json"
            gallery_path.write_text(json.dumps({
                "schema_version": "failure_case_gallery_v1",
                "cases": [
                    {
                        "id": "gallery_case_alpha",
                        "what_happened": "Synthetic test case description.",
                        "why_risky": "Test risk explanation.",
                        "system_response": "Detected during manual review.",
                        "mitigation": "Patched in commit X.",
                        "unresolved": "Edge cases not yet enumerated.",
                        "severity": "medium",
                    },
                ],
            }))
            payload = build_failure_mode_registry(
                failure_gallery_path=str(gallery_path),
                safety_red_team_path=str(Path(tmp) / "absent.json"),
                drift_report_path=str(Path(tmp) / "absent.json"),
                output_path=str(Path(tmp) / "registry.json"),
            )
            entry = next(
                e for e in payload["entries"] if e["name"] == "gallery_case_alpha"
            )
            self.assertEqual(entry["category"], "narrative_case")
            self.assertEqual(entry["severity"], "medium")
            self.assertIn("failure_case_gallery", entry["benchmark_coverage"])

    def test_drift_findings_only_added_when_status_is_not_stable(self) -> None:
        with TemporaryDirectory() as tmp:
            drift_path = Path(tmp) / "drift.json"
            drift_path.write_text(json.dumps({
                "lab_distribution_shift": {"status": "stable"},
                "subgroup_performance_drift": {"status": "drift_detected"},
            }))
            payload = build_failure_mode_registry(
                failure_gallery_path=str(Path(tmp) / "absent.json"),
                safety_red_team_path=str(Path(tmp) / "absent.json"),
                drift_report_path=str(drift_path),
                output_path=str(Path(tmp) / "registry.json"),
            )
            drift_entries = [e for e in payload["entries"] if e["name"].startswith("drift::")]
            self.assertEqual(len(drift_entries), 1)
            self.assertEqual(drift_entries[0]["name"], "drift::subgroup_performance_drift")

    def test_safety_red_team_failures_summarised_into_one_entry(self) -> None:
        with TemporaryDirectory() as tmp:
            sr_path = Path(tmp) / "sr.json"
            sr_path.write_text(json.dumps({
                "summary": {
                    "status": "needs_attention",
                    "failed_cases": ["case_001", "case_002", "case_003"],
                },
            }))
            payload = build_failure_mode_registry(
                failure_gallery_path=str(Path(tmp) / "absent.json"),
                safety_red_team_path=str(sr_path),
                drift_report_path=str(Path(tmp) / "absent.json"),
                output_path=str(Path(tmp) / "registry.json"),
            )
            sr_entry = next(
                e for e in payload["entries"] if e["name"] == "safety_red_team_failures"
            )
            self.assertEqual(sr_entry["severity"], "high")
            self.assertIn("case_001", sr_entry["example"])

    def test_loader_returns_missing_shell(self) -> None:
        with TemporaryDirectory() as tmp:
            payload = load_failure_mode_registry(path=str(Path(tmp) / "absent.json"))
            self.assertEqual(payload["status"], "missing")
            self.assertEqual(payload["entries"], [])


if __name__ == "__main__":
    unittest.main()
