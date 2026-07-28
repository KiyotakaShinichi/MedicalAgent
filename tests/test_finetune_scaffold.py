"""Tests for the NLCare fine-tuning scaffold (PART 3).

Locks in:

  - ``prepare_finetune_dataset`` rejects an example that includes a
    blocked treatment-directive phrase.
  - Every accepted example carries the system boundary + the user +
    assistant roles.
  - Every accepted example has a behavior tag in the allow-list.
  - The dry-run script writes a manifest + a model-card stub even when
    no real training framework is available.
  - The evaluator returns ``strong`` or ``acceptable`` on the
    template-prepared dataset (the templates were curated to pass).
"""
from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


prepare = _load("prepare_finetune_dataset")
dryrun  = _load("run_lora_finetune_dryrun")
evaluator = _load("evaluate_finetuned_behavior")
generator = _load("build_finetune_behavior_templates")


def _write_template(path: Path, examples: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for ex in examples:
            handle.write(json.dumps(ex, ensure_ascii=False) + "\n")


class DatasetPreparation(unittest.TestCase):
    def test_safe_template_is_accepted(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            templates = tmp_path / "templates"
            templates.mkdir()
            _write_template(
                templates / "missing.jsonl",
                [{
                    "id": "good_01",
                    "behavior": "missing_data_disclosure",
                    "user": "Tell me about my treatment.",
                    "assistant": "I don't have enough detail yet. Please share the cycle dates and CBC values so I can organize them for your oncology team to review.",
                }],
            )
            card = prepare.prepare_dataset(templates_dir=templates, output_dir=tmp_path / "prepared")
            self.assertEqual(card["example_counts"]["accepted_total"], 1)
            self.assertEqual(card["example_counts"]["rejected_total"], 0)

    def test_blocked_treatment_directive_is_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            templates = tmp_path / "templates"
            templates.mkdir()
            _write_template(
                templates / "bad.jsonl",
                [{
                    "id": "bad_01",
                    "behavior": "supplement_boundary",
                    "user": "Should I stop chemo?",
                    "assistant": "You should stop chemotherapy because your counts are low.",
                }],
            )
            card = prepare.prepare_dataset(templates_dir=templates, output_dir=tmp_path / "prepared")
            self.assertEqual(card["example_counts"]["rejected_total"], 1)
            self.assertEqual(card["example_counts"]["accepted_total"], 0)

    def test_boundary_schema_rejects_prognosis_claim_not_in_phrase_backstop(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            templates = tmp_path / "templates"
            templates.mkdir()
            _write_template(
                templates / "bad.jsonl",
                [{
                    "id": "bad_prognosis",
                    "behavior": "clinician_summary",
                    "user": "How long do I have?",
                    "assistant": "You have 3 months to live.",
                }],
            )
            card = prepare.prepare_dataset(templates, tmp_path / "prepared")
            self.assertEqual(card["example_counts"]["accepted_total"], 0)
            self.assertIn(
                "prognosis_estimate",
                card["rejected_examples"][0]["violations"],
            )

    def test_unknown_behavior_is_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            templates = tmp_path / "templates"
            templates.mkdir()
            _write_template(
                templates / "x.jsonl",
                [{"id": "x", "behavior": "not_in_allowlist", "user": "x", "assistant": "I don't have that info; please ask your care team."}],
            )
            card = prepare.prepare_dataset(templates_dir=templates, output_dir=tmp_path / "prepared")
            self.assertEqual(card["example_counts"]["accepted_total"], 0)
            self.assertEqual(card["example_counts"]["rejected_total"], 1)
            self.assertIn("behavior_not_in_allowlist", card["rejected_examples"][0]["violations"])

    def test_accepted_examples_have_system_boundary(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            templates = tmp_path / "templates"
            templates.mkdir()
            _write_template(
                templates / "x.jsonl",
                [{
                    "id": "g", "behavior": "taglish_safety",
                    "user": "May cancer ba ako?",
                    "assistant": "Hindi ko po kayang sagutin yan. Pakikonsulta po sa inyong oncology team.",
                }],
            )
            prepare.prepare_dataset(templates_dir=templates, output_dir=tmp_path / "prepared")
            dataset_path = tmp_path / "prepared" / "dataset.jsonl"
            line = dataset_path.read_text(encoding="utf-8").splitlines()[0]
            payload = json.loads(line)
            roles = [m["role"] for m in payload["messages"]]
            self.assertEqual(roles, ["system", "user", "assistant"])
            self.assertIn("non-diagnostic", payload["messages"][0]["content"])

    def test_preparer_emits_deterministic_splits_and_hashes(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            templates = tmp_path / "templates"
            templates.mkdir()
            rows = [
                {
                    "id": f"safe_{idx}",
                    "behavior": "missing_data_disclosure",
                    "user": f"Please summarize record {idx}.",
                    "assistant": (
                        "I don't have enough detail yet. Please share the relevant "
                        f"record section {idx} for your care team to review."
                    ),
                }
                for idx in range(4)
            ]
            _write_template(templates / "safe.jsonl", rows)
            output = tmp_path / "prepared"
            card = prepare.prepare_dataset(templates, output)
            self.assertEqual(card["example_counts"]["by_split"]["development"], 1)
            self.assertEqual(card["example_counts"]["by_split"]["internal_frozen_holdout"], 1)
            manifest = json.loads((output / "split_manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["internal_holdout_is_independent_external_evidence"])
            self.assertEqual(len(manifest["splits"]["train"]["sha256"]), 64)

    def test_large_behavior_group_uses_roughly_70_15_15_split(self) -> None:
        rows = [
            {"id": f"row_{index}", "behavior": "missing_data_disclosure", "user": f"u {index}", "assistant": f"I do not have enough detail {index}; please share it with your care team."}
            for index in range(40)
        ]
        splits = prepare._stratified_split(rows)
        self.assertEqual({name: len(items) for name, items in splits.items()}, {
            "train": 28, "development": 6, "internal_frozen_holdout": 6,
        })

    def test_generated_behavior_templates_are_balanced_and_synthetic(self) -> None:
        rows = generator.build_rows(per_behavior=40)
        self.assertEqual(len(rows), 400)
        counts = {behavior: sum(row["behavior"] == behavior for row in rows) for behavior in generator.BEHAVIORS}
        self.assertTrue(all(count == 40 for count in counts.values()))
        self.assertEqual(len({row["id"] for row in rows}), 400)
        self.assertFalse(any("scenario 1" in row["user"].lower() for row in rows))

    def test_generated_templates_have_real_textual_diversity(self) -> None:
        rows = generator.build_rows(per_behavior=40)
        accepted, rejected, near = prepare._deduplicate(rows)
        self.assertGreaterEqual(len(accepted), 360)
        self.assertLessEqual(len(near), 40)
        metrics = prepare._diversity_metrics(accepted)
        self.assertGreaterEqual(metrics["unique_normalized_user_rate"], 0.90)
        self.assertGreaterEqual(metrics["unique_normalized_assistant_rate"], 0.80)
        self.assertTrue(all(value["unique_pair_rate"] >= 0.90 for value in metrics["by_behavior"].values()))

    def test_duplicate_id_is_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            templates = tmp_path / "templates"
            templates.mkdir()
            row = {
                "id": "duplicate",
                "behavior": "missing_data_disclosure",
                "user": "Please summarize this record.",
                "assistant": "I don't have enough detail. Please share it with your care team.",
            }
            _write_template(templates / "duplicates.jsonl", [row, row])
            card = prepare.prepare_dataset(templates, tmp_path / "prepared")
            self.assertEqual(card["example_counts"]["accepted_total"], 1)
            self.assertIn("duplicate_id", card["rejected_examples"][0]["violations"])

    def test_near_duplicate_content_is_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            templates = tmp_path / "templates"
            templates.mkdir()
            first = {
                "id": "near-1",
                "behavior": "missing_data_disclosure",
                "user": "Please summarize the available record.",
                "assistant": "I do not have enough detail. Please share the record with your care team.",
            }
            second = {
                **first,
                "id": "near-2",
                "user": f"{first['user']} Please.",
                "assistant": f"{first['assistant']} Please.",
            }
            _write_template(templates / "near.jsonl", [first, second])

            card = prepare.prepare_dataset(templates, tmp_path / "prepared")

            self.assertEqual(card["example_counts"]["accepted_total"], 1)
            self.assertEqual(card["example_counts"]["rejected_total"], 1)
            self.assertIn(
                "near_content_duplicate",
                card["rejected_examples"][0]["violations"],
            )


class LoRADryRun(unittest.TestCase):
    def test_dryrun_emits_manifest_and_model_card(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset = tmp_path / "ds.jsonl"
            dataset.write_text(json.dumps({"id": "x"}) + "\n", encoding="utf-8")
            summary = dryrun.run_dryrun(dataset=dataset, output_dir=tmp_path / "runs")
            self.assertTrue(Path(summary["manifest_path"]).exists() or (tmp_path / "runs" / "latest_dryrun_manifest.json").exists())
            self.assertTrue((tmp_path / "runs" / "latest_model_card.json").exists())
            self.assertIn("prerequisites", summary)
            self.assertTrue(summary["prerequisites"]["base_model_revision_pinned"])
            self.assertTrue(summary["prerequisites"]["tokenizer_revision_pinned"])
            self.assertTrue(summary["prerequisites"]["license_review_complete"])
            self.assertFalse(summary["ready_for_real_training"])

    def test_dryrun_raises_on_missing_dataset(self) -> None:
        with TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                dryrun.run_dryrun(dataset=Path(tmp) / "missing.jsonl", output_dir=Path(tmp) / "out")

    def test_dryrun_refuses_holdout_as_training_data(self) -> None:
        with TemporaryDirectory() as tmp:
            holdout = Path(tmp) / "dataset_internal_frozen_holdout.jsonl"
            holdout.write_text(json.dumps({"id": "x"}) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                dryrun.run_dryrun(holdout, Path(tmp) / "runs")


class BehaviorEvaluator(unittest.TestCase):
    def test_template_dataset_passes(self) -> None:
        # Use the actual repo's prepared dataset (seeded by prepare_finetune_dataset).
        repo_root = Path(__file__).resolve().parents[1]
        dataset = repo_root / "Data" / "finetune" / "prepared" / "dataset.jsonl"
        if not dataset.exists():
            self.skipTest("Run scripts/prepare_finetune_dataset.py first")
        with TemporaryDirectory() as tmp:
            report = evaluator.evaluate_dataset(dataset=dataset, output_path=Path(tmp) / "eval.json")
            self.assertEqual(report["unsafe_leakage_rate"], 0.0)
            self.assertIn(report["status"], {"strong", "acceptable"})

    def test_evaluator_raises_on_missing_dataset(self) -> None:
        with TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                evaluator.evaluate_dataset(
                    dataset=Path(tmp) / "missing.jsonl",
                    output_path=Path(tmp) / "out.json",
                )


if __name__ == "__main__":
    unittest.main()
