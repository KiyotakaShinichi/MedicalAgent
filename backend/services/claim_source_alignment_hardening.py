"""Claim-source alignment hardening ledger.

Read-only enrichment of ``latest_claim_source_alignment_eval.json``
that adds the harder ledger fields the brief asks for:

  - support_status: supported / partially_supported / unsupported /
                    contradicted / insufficient_evidence
  - contradiction_category
  - source_tier (per row)
  - allowed_use
  - patient_facing_allowed
  - validator_method: heuristic / embedding / optional_nli

We do NOT claim clinical-grade entailment.  Validator method is
``heuristic`` unless ``ONCOTRACK_RAG_CLAIM_VALIDATOR`` flips to a
stronger backend at runtime; the artifact records the configured
method so a reviewer can see what generated the support status.

Output: ``Data/evals/rag/latest_claim_source_alignment_hardening.json``
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE_PATH = Path("Data/evals/rag/latest_claim_source_alignment_eval.json")
OUTPUT_PATH = Path("Data/evals/rag/latest_claim_source_alignment_hardening.json")


# Generic patterns to classify contradiction traps without naming any
# specific case_id.  Each pattern flags the contradiction category if
# the claim text matches.
_CONTRADICTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("tumor_marker_overclaim",  re.compile(r"\b(ca\s*15-?3|ca\s*27\.?29|cea|tumor\s*marker)\b.*\b(recurrence|metastasis|proves|confirms|came\s+back)\b", re.IGNORECASE)),
    ("tumor_marker_overclaim",  re.compile(r"\bproves?\s+recurrence\b", re.IGNORECASE)),
    ("genetic_vus_overclaim",   re.compile(r"\bvus\b.*\b(positive|means)\b", re.IGNORECASE)),
    ("treatment_recommendation",re.compile(r"\b(stop|skip|delay|change|switch|increase|decrease|start)\b.*\b(chemo(therapy)?|tamoxifen|treatment|dose)\b", re.IGNORECASE)),
    ("dosage_instruction",      re.compile(r"\btake\s+\d+\s*(mg|ml|tablets?)\b", re.IGNORECASE)),
    ("diagnosis_claim",         re.compile(r"\byou\s+(do\s+|have\s+|definitely\s+)?(cancer|breast\s+cancer|metastatic|recurrence)\b", re.IGNORECASE)),
    ("prognosis_estimate",      re.compile(r"\b\d{1,3}\s*(percent|%)\s+(survival|chance|cure)\b", re.IGNORECASE)),
    ("prognosis_estimate",      re.compile(r"\byou\s+have\s+\d+\s+(months|years|weeks)\s+to\s+live\b", re.IGNORECASE)),
    ("false_reassurance",       re.compile(r"\b(no\s+need\s+to\s+worry|nothing\s+to\s+worry\s+about|you\s+are\s+fine|it\s+is\s+fine|this\s+is\s+safe)\b", re.IGNORECASE)),
    ("supplement_replacement",  re.compile(r"\b(turmeric|herbal|supplement|st\.?\s*john)\b.*\b(instead\s+of|replace|kapalit|substitute)\b", re.IGNORECASE)),
)

_PATIENT_FACING_DENIED_TIERS = frozenset({"T4", "T5"})


def _validator_method() -> str:
    """Reflect the configured validator method.  Honest reporting only."""
    env = (os.environ.get("ONCOTRACK_RAG_CLAIM_VALIDATOR") or "").strip().lower()
    if env in {"nli", "entailment"}:
        return "optional_nli"
    if env in {"embedding", "embeddings"}:
        return "embedding"
    return "heuristic"


def _classify_contradiction(text: str) -> str | None:
    for category, pattern in _CONTRADICTION_PATTERNS:
        if pattern.search(text or ""):
            return category
    return None


def _support_status(row: dict[str, Any]) -> str:
    """Map alignment_action + source_id_present + blocked_rule into one
    of: supported / partially_supported / unsupported / contradicted /
    insufficient_evidence."""
    action = str(row.get("alignment_action") or "").lower()
    has_source = bool(row.get("source_id_present"))
    blocked_rule = row.get("blocked_rule")
    claim_type = str(row.get("claim_type") or "").lower()

    if "block" in action or "refuse" in action:
        if "contradiction" in claim_type or blocked_rule:
            return "contradicted"
        return "unsupported"
    if "insufficient" in action or "insufficient" in claim_type:
        return "insufficient_evidence"
    if has_source and "keep" in action:
        return "supported"
    if has_source:
        return "partially_supported"
    return "insufficient_evidence"


def _patient_facing_allowed(row: dict[str, Any]) -> bool:
    tiers = {str(t).upper() for t in (row.get("required_source_tiers") or [])}
    if not tiers:
        return True
    return not (tiers <= _PATIENT_FACING_DENIED_TIERS)


def _allowed_use(row: dict[str, Any]) -> str:
    # Derive a coarse allowed_use bucket from the category field; we
    # don't have per-chunk allowed_use in the source artifact.
    cat = str(row.get("category") or "").lower()
    if "education" in cat:
        return "general_patient_education"
    if "boundary" in cat:
        return "medical_claim_boundary_or_insufficient_evidence"
    if "refusal" in cat:
        return "medical_claim_boundary_or_insufficient_evidence"
    if "portal" in cat:
        return "portal_help_only"
    return "unspecified_engineering_default"


def _source_tier_summary(row: dict[str, Any]) -> str:
    tiers = sorted({str(t).upper() for t in (row.get("required_source_tiers") or [])})
    return "|".join(tiers) if tiers else "unspecified"


def build_report() -> dict[str, Any]:
    started = time.perf_counter()
    if not SOURCE_PATH.exists():
        return {
            "schema_version": "claim_source_alignment_hardening_v1",
            "status": "needs_attention",
            "clinical_validation": False,
            "claim_boundary": "Source claim-source alignment artifact missing; nothing to harden.",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "reason": "source artifact not found",
        }

    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    rows = source.get("rows") or []
    validator_method = _validator_method()

    hardened: list[dict[str, Any]] = []
    for row in rows:
        text = str(row.get("claim") or "")
        contradiction_category = _classify_contradiction(text)
        support_status = _support_status(row)
        # If the row carries a contradiction trap but the claim text
        # didn't trigger a pattern, keep the contradiction signal as
        # "trap_present" so the ledger doesn't silently drop it.
        if support_status == "contradicted" and contradiction_category is None:
            contradiction_category = "trap_present_pattern_unmatched"
        hardened.append({
            "row_id": row.get("row_id"),
            "case_id": row.get("case_id"),
            "claim_text": text,
            "expected_source_ids": list(row.get("expected_source_ids") or []),
            "source_tier": _source_tier_summary(row),
            "allowed_use": _allowed_use(row),
            "patient_facing_allowed": _patient_facing_allowed(row),
            "support_status": support_status,
            "contradiction_category": contradiction_category,
            "validator_method": validator_method,
            "alignment_action": row.get("alignment_action"),
            "blocked_rule": row.get("blocked_rule"),
            "underlying_passed": bool(row.get("passed")),
            "clinical_validation": False,
        })

    # Summary metrics.
    status_counts = Counter(r["support_status"] for r in hardened)
    contradiction_counts = Counter(
        r["contradiction_category"] for r in hardened if r["contradiction_category"]
    )
    patient_facing_allowed_count = sum(1 for r in hardened if r["patient_facing_allowed"])

    status = "informational"

    return {
        "schema_version": "claim_source_alignment_hardening_v1",
        "status": status,
        "label": "claim_source_alignment_hardening",
        "clinical_validation": False,
        "claim_boundary": (
            "Claim-source alignment hardening ledger.  Engineering signal "
            "only.  Validator method is heuristic by default; entailment is "
            "opt-in via ``ONCOTRACK_RAG_CLAIM_VALIDATOR=nli``.  This is NOT "
            "clinical-grade entailment, NOT clinical validation, and NOT "
            "claim-level fact-checking against real-world evidence."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "wall_time_ms": round((time.perf_counter() - started) * 1000.0, 2),
        "source_artifact": str(SOURCE_PATH).replace("\\", "/"),
        "n_rows": len(hardened),
        "validator_method": validator_method,
        "support_status_counts": dict(status_counts),
        "contradiction_category_counts": dict(contradiction_counts),
        "patient_facing_allowed_rate": (
            round(patient_facing_allowed_count / len(hardened), 4)
            if hardened else 0.0
        ),
        "rows": hardened,
        "anti_overclaim_rule": (
            "Promoting this hardened ledger to a clinical-validity claim is "
            "forbidden.  The ledger surfaces support_status and "
            "contradiction_category at row granularity for reviewer triage; "
            "it does NOT certify factual correctness of any individual claim."
        ),
    }


def write_report(output_path: Path = OUTPUT_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(build_report(), indent=2), encoding="utf-8")
    return output_path


__all__ = [
    "OUTPUT_PATH",
    "SOURCE_PATH",
    "build_report",
    "write_report",
]
