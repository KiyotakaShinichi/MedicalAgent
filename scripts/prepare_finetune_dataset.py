"""Prepare the synthetic NLCare behavior-tuning dataset.

The preparer is intentionally fail-closed. It validates template schemas,
applies the medical claim boundary, rejects direct identifiers and duplicate
examples, creates deterministic behavior-stratified splits, and records hashes
and exact-text contamination evidence. The internal frozen split is an
engineering control, not independent or clinical evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_TEMPLATES_DIR = ROOT / "Data" / "finetune" / "templates"
DEFAULT_OUTPUT_DIR = ROOT / "Data" / "finetune" / "prepared"
DEFAULT_SPLIT_SEED = "nlcare-behavior-v2-20260721"

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
    "emotional_support",
    "privacy_boundary",
    "tool_confirmation",
    "out_of_scope_redirect",
    "uncertainty_disclosure",
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

REQUIRED_TEMPLATE_FIELDS = ("id", "behavior", "user", "assistant")
SPLIT_NAMES = ("train", "development", "internal_frozen_holdout")
PRIVACY_PATTERNS = (
    re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    re.compile(
        r"\b(?:mrn|medical record number|patient id)\s*[:#-]?\s*[A-Za-z0-9-]{4,}\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:\+?63|0)9\d{9}\b"),
)


def _rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _normalise_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _example_fingerprint(example: dict[str, Any]) -> str:
    canonical = "\n".join(
        _normalise_text(str(example.get(field) or ""))
        for field in ("behavior", "user", "assistant")
    )
    return _sha256_text(canonical)


def _schema_violations(example: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    for field in REQUIRED_TEMPLATE_FIELDS:
        value = example.get(field)
        if not isinstance(value, str) or not value.strip():
            violations.append(f"missing_or_invalid_{field}")
    for field in ("user", "assistant"):
        value = str(example.get(field) or "")
        if any(pattern.search(value) for pattern in PRIVACY_PATTERNS):
            violations.append(f"possible_direct_identifier_in_{field}")
    return sorted(set(violations))


def _safety_filter(example: dict[str, Any]) -> list[str]:
    """Return blocked categories. Boundary checker failure is a rejection."""
    from backend.services.medical_claim_boundary import classify_medical_claim

    violations: list[str] = []
    try:
        verdict = classify_medical_claim(str(example.get("assistant") or ""))
        if verdict.get("decision") == "blocked":
            blocked_types = verdict.get("blocked_claim_types") or ["blocked_medical_claim"]
            violations.extend(str(item) for item in blocked_types)
    except Exception:  # noqa: BLE001 - training data safety must fail closed
        violations.append("medical_claim_boundary_unavailable")

    text = str(example.get("assistant") or "").lower()
    if "safe with chemo" in text and "cannot" not in text:
        violations.append("supplement_safe_with_chemo_claim")
    if any(
        phrase in text
        for phrase in (
            "you should stop chemo",
            "you should start chemo",
            "you have metastasis",
            "you are cancer free",
        )
    ):
        violations.append("treatment_or_diagnostic_overclaim")
    return sorted(set(violations))


def _load_templates(templates_dir: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for path in sorted(templates_dir.glob("*.jsonl")):
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path.name}:{line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object at {path.name}:{line_number}")
            payload["_source_file"] = path.name
            payload["_source_line"] = line_number
            examples.append(payload)
    return examples


def _jaccard(left: str, right: str) -> float:
    left_tokens = set(_normalise_text(left).split())
    right_tokens = set(_normalise_text(right).split())
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0


def _deduplicate(
    examples: list[dict[str, Any]],
    near_duplicate_threshold: float = 0.92,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    near_duplicates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_fingerprints: dict[str, str] = {}

    for example in examples:
        example_id = str(example["id"])
        fingerprint = _example_fingerprint(example)
        if example_id in seen_ids:
            rejected.append(_rejection(example, ["duplicate_id"]))
            continue
        if fingerprint in seen_fingerprints:
            rejection = _rejection(example, ["exact_content_duplicate"])
            rejection["duplicate_of"] = seen_fingerprints[fingerprint]
            rejected.append(rejection)
            continue

        combined = f"{example['user']}\n{example['assistant']}"
        near_duplicate: dict[str, Any] | None = None
        for prior in accepted:
            similarity = _jaccard(combined, f"{prior['user']}\n{prior['assistant']}")
            if similarity >= near_duplicate_threshold:
                near_duplicate = {
                    "id": example_id,
                    "near_duplicate_of": prior["id"],
                    "token_jaccard": round(similarity, 4),
                }
                break
        if near_duplicate is not None:
            near_duplicates.append(near_duplicate)
            rejection = _rejection(example, ["near_content_duplicate"])
            rejection.update(
                {
                    "duplicate_of": near_duplicate["near_duplicate_of"],
                    "token_jaccard": near_duplicate["token_jaccard"],
                }
            )
            rejected.append(rejection)
            continue
        example["_fingerprint"] = fingerprint
        seen_ids.add(example_id)
        seen_fingerprints[fingerprint] = example_id
        accepted.append(example)
    return accepted, rejected, near_duplicates


def _rejection(example: dict[str, Any], violations: list[str]) -> dict[str, Any]:
    return {
        "id": example.get("id"),
        "behavior": example.get("behavior"),
        "violations": sorted(set(violations)),
        "source_file": example.get("_source_file"),
        "source_line": example.get("_source_line"),
    }


def _stratified_split(
    examples: list[dict[str, Any]], seed: str = DEFAULT_SPLIT_SEED
) -> dict[str, list[dict[str, Any]]]:
    splits = {name: [] for name in SPLIT_NAMES}
    by_behavior: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        by_behavior.setdefault(str(example["behavior"]), []).append(example)

    for behavior, group in sorted(by_behavior.items()):
        ordered = sorted(
            group,
            key=lambda item: _sha256_text(f"{seed}:{behavior}:{item['id']}"),
        )
        if len(ordered) >= 7:
            development_n = max(1, round(len(ordered) * 0.15))
            holdout_n = max(1, round(len(ordered) * 0.15))
            train_end = len(ordered) - development_n - holdout_n
            train = ordered[:train_end]
            development = ordered[train_end:train_end + development_n]
            holdout = ordered[train_end + development_n:]
        elif len(ordered) >= 3:
            train, development, holdout = ordered[:-2], ordered[-2:-1], ordered[-1:]
        elif len(ordered) == 2:
            train, development, holdout = ordered[:1], ordered[1:], []
        else:
            train, development, holdout = ordered, [], []
        splits["train"].extend(train)
        splits["development"].extend(development)
        splits["internal_frozen_holdout"].extend(holdout)
    return splits


def _candidate_contamination_paths() -> list[Path]:
    patterns = (
        "Data/evals/safety/*holdout*.jsonl",
        "Data/evals/rag/*holdout*.jsonl",
        "Data/evals/rag/*goldset*.jsonl",
    )
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(ROOT.glob(pattern))
    return sorted(path for path in paths if path.is_file())


def _iter_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for nested in value.values() for text in _iter_strings(nested)]
    if isinstance(value, list):
        return [text for nested in value for text in _iter_strings(nested)]
    return []


def _contamination_audit(
    examples: list[dict[str, Any]], paths: list[Path] | None = None
) -> dict[str, Any]:
    comparison_paths = _candidate_contamination_paths() if paths is None else paths
    corpus: dict[str, list[dict[str, Any]]] = {}
    scanned_files: list[dict[str, str]] = []
    for path in comparison_paths:
        scanned_files.append({"path": _rel(path), "sha256": _sha256_file(path)})
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            for text in _iter_strings(payload):
                normalised = _normalise_text(text)
                if len(normalised) >= 24:
                    corpus.setdefault(normalised, []).append(
                        {"path": _rel(path), "line": line_number}
                    )

    overlaps: list[dict[str, Any]] = []
    for example in examples:
        for field in ("user", "assistant"):
            normalised = _normalise_text(str(example.get(field) or ""))
            if normalised in corpus:
                overlaps.append(
                    {
                        "training_example_id": example["id"],
                        "training_field": field,
                        "matches": corpus[normalised],
                    }
                )
    return {
        "status": "needs_attention" if overlaps else "acceptable",
        "comparison": "exact_normalised_text_only",
        "scanned_files": scanned_files,
        "exact_overlap_count": len(overlaps),
        "overlaps": overlaps,
        "semantic_contamination_not_measured": True,
    }


def _count_by(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = str(item.get(key, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _diversity_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {
            "unique_normalized_user_rate": 0.0,
            "unique_normalized_assistant_rate": 0.0,
            "unique_response_opening_rate": 0.0,
            "mean_user_token_count": 0.0,
            "mean_assistant_token_count": 0.0,
            "by_behavior": {},
        }

    def rate(values: list[str]) -> float:
        return round(len(set(values)) / len(values), 6)

    users = [_normalise_text(str(item["user"])) for item in items]
    assistants = [_normalise_text(str(item["assistant"])) for item in items]
    openings = [" ".join(value.split()[:6]) for value in assistants]
    by_behavior: dict[str, dict[str, Any]] = {}
    for behavior in sorted({str(item["behavior"]) for item in items}):
        group = [item for item in items if str(item["behavior"]) == behavior]
        group_users = [_normalise_text(str(item["user"])) for item in group]
        group_assistants = [_normalise_text(str(item["assistant"])) for item in group]
        by_behavior[behavior] = {
            "n": len(group),
            "unique_user_rate": rate(group_users),
            "unique_assistant_rate": rate(group_assistants),
            "unique_pair_rate": rate([f"{u}\n{a}" for u, a in zip(group_users, group_assistants)]),
        }
    return {
        "unique_normalized_user_rate": rate(users),
        "unique_normalized_assistant_rate": rate(assistants),
        "unique_response_opening_rate": rate(openings),
        "mean_user_token_count": round(sum(len(value.split()) for value in users) / len(users), 2),
        "mean_assistant_token_count": round(sum(len(value.split()) for value in assistants) / len(assistants), 2),
        "by_behavior": by_behavior,
    }


def prepare_dataset(
    templates_dir: Path = DEFAULT_TEMPLATES_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    templates = _load_templates(templates_dir)
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for example in templates:
        violations = _schema_violations(example)
        if not violations:
            violations = _safety_filter(example)
        if not violations and example.get("behavior") not in ALLOWED_BEHAVIOR_TARGETS:
            violations = ["behavior_not_in_allowlist"]
        if violations:
            rejected.append(_rejection(example, violations))
        else:
            candidates.append(example)

    deduplicated, duplicate_rejections, near_duplicates = _deduplicate(candidates)
    rejected.extend(duplicate_rejections)
    splits = _stratified_split(deduplicated)
    split_by_id = {
        str(example["id"]): split_name
        for split_name, items in splits.items()
        for example in items
    }

    accepted: list[dict[str, Any]] = []
    for example in deduplicated:
        split = split_by_id[str(example["id"])]
        accepted.append(
            {
                "id": example["id"],
                "behavior": example["behavior"],
                "split": split,
                "example_sha256": example["_fingerprint"],
                "messages": [
                    {"role": "system", "content": SYSTEM_BOUNDARY},
                    {"role": "user", "content": example["user"]},
                    {"role": "assistant", "content": example["assistant"]},
                ],
                "source_file": example["_source_file"],
                "source_line": example["_source_line"],
                "provenance": {
                    "source_type": "synthetic_template",
                    "authorship": "internal",
                    "contains_real_patient_data": False,
                    "was_used_for_tuning": split == "train",
                },
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "dataset.jsonl"
    _write_jsonl(dataset_path, accepted)
    split_paths = {
        "train": output_dir / "dataset_train.jsonl",
        "development": output_dir / "dataset_development.jsonl",
        "internal_frozen_holdout": output_dir / "dataset_internal_frozen_holdout.jsonl",
    }
    for split_name, path in split_paths.items():
        _write_jsonl(path, [item for item in accepted if item["split"] == split_name])

    contamination = _contamination_audit(deduplicated)
    source_files = sorted(templates_dir.glob("*.jsonl"))
    split_manifest = {
        "schema_version": "finetune_split_manifest_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split_seed": DEFAULT_SPLIT_SEED,
        "split_policy": (
            "Deterministic behavior stratification; approximately 70/15/15 for "
            "behavior groups with at least seven rows, with small-group fallback."
        ),
        "internal_holdout_is_independent_external_evidence": False,
        "source_files": [
            {"path": _rel(path), "sha256": _sha256_file(path)} for path in source_files
        ],
        "splits": {
            name: {
                "path": _rel(path),
                "sha256": _sha256_file(path),
                "example_count": sum(1 for item in accepted if item["split"] == name),
                "was_used_for_tuning": name == "train",
            }
            for name, path in split_paths.items()
        },
        "contamination_audit": contamination,
        "claim_boundary": (
            "Internal synthetic split integrity only. The frozen split is not "
            "external evidence or clinical validation."
        ),
    }
    split_manifest_path = output_dir / "split_manifest.json"
    split_manifest_path.write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")

    card = {
        "schema_version": "finetune_dataset_card_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_purpose": (
            "Behavior and format tuning for synthetic NLCare responses; not "
            "medical knowledge tuning."
        ),
        "allowed_behavior_targets": list(ALLOWED_BEHAVIOR_TARGETS),
        "blocked_claims": list(BLOCKED_CLAIMS),
        "synthetic_or_source": "all_synthetic",
        "clinical_validation": False,
        "safety_filters_applied": [
            "template_schema_validation",
            "direct_identifier_patterns",
            "medical_claim_boundary_fail_closed",
            "unsafe_phrase_backstop",
            "behavior_allowlist",
            "exact_duplicate_rejection",
        ],
        "known_risks": [
            "Small internally authored examples can overfit synthetic phrasing.",
            "The internal frozen split is not independent external evidence.",
            "Exact-text contamination checks miss semantic paraphrases.",
            "Heuristic medical boundaries are not clinical-grade validators.",
            "Passing this audit does not prove adapter behavior or clinical safety.",
        ],
        "example_counts": {
            "accepted_total": len(accepted),
            "rejected_total": len(rejected),
            "by_behavior": _count_by(accepted, "behavior"),
            "by_split": _count_by(accepted, "split"),
        },
        "deduplication": {
            "exact_duplicates_rejected": len(duplicate_rejections),
            "near_duplicate_threshold": 0.92,
            "near_duplicate_pairs": near_duplicates,
        },
        "linguistic_diversity": _diversity_metrics(deduplicated),
        "contamination_audit": contamination,
        "training_readiness": "scaffold_only_not_ready_for_adapter_promotion",
        "system_prompt": SYSTEM_BOUNDARY,
        "rejected_examples": rejected,
        "files": {
            "dataset_jsonl": _rel(dataset_path),
            "dataset_card": _rel(output_dir / "dataset_card.json"),
            "split_manifest": _rel(split_manifest_path),
            **{f"dataset_{name}": _rel(path) for name, path in split_paths.items()},
        },
        "claim_boundary": (
            "Synthetic behavior and format tuning only. No diagnosis, treatment, "
            "dosage, prognosis, genetic-risk, tumor-marker, or supplement-safety "
            "authority. Any adapter remains behind every NLCare safety layer."
        ),
    }
    (output_dir / "dataset_card.json").write_text(
        json.dumps(card, indent=2), encoding="utf-8"
    )
    return card


def _write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare NLCare behavior-tuning data.")
    parser.add_argument("--templates-dir", type=Path, default=DEFAULT_TEMPLATES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    card = prepare_dataset(args.templates_dir, args.output_dir)
    print(
        json.dumps(
            {
                "accepted": card["example_counts"]["accepted_total"],
                "rejected": card["example_counts"]["rejected_total"],
                "by_behavior": card["example_counts"]["by_behavior"],
                "by_split": card["example_counts"]["by_split"],
                "contamination": card["contamination_audit"]["status"],
                "files": card["files"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
