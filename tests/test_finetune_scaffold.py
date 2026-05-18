"""Tests for the OncoTrack fine-tuning scaffold (PART 3).

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


class LoRADryRun(unittest.TestCase):
    def test_dryrun_emits_manifest_and_model_card(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset = tmp_path / "ds.jsonl"
            dataset.write_text(json.dumps({"id": "x"}) + "\n", encoding="utf-8")
            summary = dryrun.run_dryrun(dataset=dataset, output_dir=tmp_path / "runs")
            self.assertTrue(Path(summary["manifest_path"]).exists() or (tmp_path / "runs" / "latest_dryrun_manifest.json").exists())
            self.assertTrue((tmp_path / "runs" / "latest_model_card.json").exists())

    def test_dryrun_raises_on_missing_dataset(self) -> None:
        with TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                dryrun.run_dryrun(dataset=Path(tmp) / "missing.jsonl", output_dir=Path(tmp) / "out")


class BehaviorEvaluator(unittest.TestCase):
    def test_template_dataset_passes(self) -> None:
        # Use the actual repo's prepared dataset (seeded by prepare_finetune_dataset).
        repo_root = Path(__file__).resolve().parents[1]
        dataset = repo_root / "data" / "finetune" / "prepared" / "dataset.jsonl"
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
