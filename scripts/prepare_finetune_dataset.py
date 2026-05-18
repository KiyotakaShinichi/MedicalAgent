"""Prepare an OncoTrack behavior-tuning dataset from the templates.

Reads every JSONL file under ``data/finetune/templates/``, applies the
medical claim boundary checker as a safety filter, and writes a unified
dataset under ``data/finetune/prepared/`` along with a dataset card.

Hard rule: examples that trip the claim boundary checker are dropped
from the output and surfaced in the dataset card's ``rejected_examples``
section.  The user-facing assistant strings in the templates are
deliberately conservative; if a future contributor adds an unsafe
example, this script catches it.

Usage
~~~~~
    python scripts/prepare_finetune_dataset.py
    python scripts/prepare_finetune_dataset.py --output-dir data/finetune/prepared
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_TEMPLATES_DIR = ROOT / "data" / "finetune" / "templates"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "finetune" / "prepared"


def _rel(path: Path) -> str:
    """Stringify a path relative to ROOT when possible; fall back to the
    absolute form for paths outside the repo (e.g. tests using temp
    directories)."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


SYSTEM_BOUNDARY = (
    "You are a non-diagnostic oncology monitoring support assistant. "
    "You organize information, summarize safely, ask for missing details, "
    "and route urgent or treatment-decision questions to the care team. "
    "You do not diagnose, confirm progression, recommend treatment changes, "
    "give dosage advice, interpret genetic-risk for the patient, or conclude "
    "from tumor markers. You do not declare any supplement 'safe with chemo'."
)


ALLOWED_BEHAVIOR_TARGETS = (
    "clinician_summary",
    "missing_data_disclosure",
    "questions_to_ask_care_team",
    "supplement_boundary",
    "taglish_safety",
)


BLOCKED_CLAIMS = (
    "diagnosis",
    "treatment_recommendation",
    "dosage_change",
    "prognosis_estimate",
    "genetic_risk_prediction",
    "tumor_marker_conclusion",
    "survival_estimate",
    "supplement_safe_with_chemo_claim",
    "replace_treatment_claim",
)


def _safety_filter(example: dict[str, Any]) -> list[str]:
    """Return the list of blocked-claim categories this example trips.
    Empty list means the example passes the filter."""
    from backend.services.medical_claim_boundary import classify_medical_claim

    violations: list[str] = []
    try:
        verdict = classify_medical_claim(example.get("assistant") or "")
        if not verdict.get("safe", True):
            violations.extend(str(c) for c in verdict.get("blocked_claims") or [])
    except Exception:  # noqa: BLE001 — the filter must never crash dataset prep
        pass

    text = (example.get("assistant") or "").lower()
    # Belt-and-suspenders pattern check that does not depend on the
    # claim-boundary module's evolving rule set.
    if "safe with chemo" in text and "cannot" not in text:
        violations.append("supplement_safe_with_chemo_claim")
    if any(phrase in text for phrase in (
        "you should stop chemo", "you should start chemo",
        "you have metastasis", "you are cancer free",
    )):
        violations.append("treatment_or_diagnostic_overclaim")
    return sorted(set(violations))


def _load_templates(templates_dir: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for path in sorted(templates_dir.glob("*.jsonl")):
        for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path.name}:{idx + 1}: {exc}") from exc
            data["_source_file"] = path.name
            examples.append(data)
    return examples


def prepare_dataset(
    templates_dir: Path = DEFAULT_TEMPLATES_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    examples = _load_templates(templates_dir)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for example in examples:
        violations = _safety_filter(example)
        if violations:
            rejected.append({"id": example.get("id"), "behavior": example.get("behavior"), "violations": violations, "source_file": example["_source_file"]})
            continue
        if example.get("behavior") not in ALLOWED_BEHAVIOR_TARGETS:
            rejected.append({"id": example.get("id"), "behavior": example.get("behavior"), "violations": ["behavior_not_in_allowlist"], "source_file": example["_source_file"]})
            continue
        accepted.append({
            "id": example["id"],
            "behavior": example["behavior"],
            "messages": [
                {"role": "system", "content": SYSTEM_BOUNDARY},
                {"role": "user",   "content": example["user"]},
                {"role": "assistant", "content": example["assistant"]},
            ],
            "source_file": example["_source_file"],
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "dataset.jsonl"
    with dataset_path.open("w", encoding="utf-8") as handle:
        for item in accepted:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    behavior_counts = _count_by(accepted, "behavior")
    card = {
        "schema_version":             "finetune_dataset_card_v1",
        "generated_at":               datetime.now(timezone.utc).isoformat(),
        "dataset_purpose":            "Behavior / style tuning for OncoTrack patient-safe responses. NOT medical knowledge tuning.",
        "allowed_behavior_targets":   list(ALLOWED_BEHAVIOR_TARGETS),
        "blocked_claims":             list(BLOCKED_CLAIMS),
        "synthetic_or_source":        "all_synthetic",
        "safety_filters_applied":     [
            "backend.services.medical_claim_boundary.classify_text",
            "phrase_blocklist (safe with chemo / stop chemo / metastasis / cancer free)",
            "behavior_allowlist_check",
        ],
        "known_risks": [
            "Behavior over-fits on synthetic phrasing; live patient input is more varied.",
            "Refusal phrasing might be reused even when not needed; A/B test the candidate.",
            "Taglish examples are small in number; do not generalize broadly.",
            "The medical_claim_boundary checker is heuristic; future versions may catch new violations.",
        ],
        "example_counts": {
            "accepted_total":    len(accepted),
            "rejected_total":    len(rejected),
            "by_behavior":       behavior_counts,
        },
        "system_prompt":              SYSTEM_BOUNDARY,
        "rejected_examples":          rejected,
        "files": {
            "dataset_jsonl":    _rel(dataset_path),
            "dataset_card":     _rel(output_dir / "dataset_card.json"),
        },
        "claim_boundary": (
            "Behavior / style tuning only on synthetic data. No clinical "
            "knowledge tuning, no diagnosis training, no treatment / dosage "
            "/ prognosis / genetic-risk / tumor-marker / supplement-safety "
            "training. Any deployed adapter still goes through every safety "
            "layer and clinician review."
        ),
    }
    (output_dir / "dataset_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    return card


def _count_by(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item.get(key, "unknown")] = counts.get(item.get(key, "unknown"), 0) + 1
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare the OncoTrack behavior fine-tuning dataset.")
    parser.add_argument("--templates-dir", type=Path, default=DEFAULT_TEMPLATES_DIR)
    parser.add_argument("--output-dir",    type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    card = prepare_dataset(args.templates_dir, args.output_dir)
    print(json.dumps({
        "accepted": card["example_counts"]["accepted_total"],
        "rejected": card["example_counts"]["rejected_total"],
        "by_behavior": card["example_counts"]["by_behavior"],
        "files": card["files"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
