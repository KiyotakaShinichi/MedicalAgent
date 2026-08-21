"""Goldset adjudication workflow for source_filter_drop cases.

The stage-wise oracle diagnostic
(``Data/evals/rag/latest_rag_stage_oracle_diagnostic.json``) attributes
the dominant failure stage as ``source_filter_drop``: cases where the
gold ``expected_source_ids`` include a source that the patient-facing
source-tier / allowed_use filter correctly excludes.

The correct response is **not** to weaken the filter — it is to ask a
reviewer whether each affected case should:

1. ``keep_expected_sources`` — the goldset is right; the failure is a
   design tradeoff to record.
2. ``revise_patient_facing_expected_sources`` — the case is
   patient-facing; the gold list should be replaced with sources that
   pass the patient-facing filter.
3. ``move_to_clinician_facing_goldset`` — the case is clinician-facing
   and should not have been in the patient goldset to begin with.
4. ``split_patient_and_clinician_cases`` — the case has two valid
   interpretations and should be split into a patient-facing and a
   clinician-facing case.
5. ``mark_ambiguous_needs_external_review`` — none of the above; defer
   to external reviewer.

This module:

* Builds the adjudication packet (read-only; never mutates the
  frozen goldset).
* Provides a strict validator for any future filled-in packet.
* Emits a readiness artifact (``completed: false``) that the release
  gate can track as informational.

No retrieval ranking, source governance, or goldset content changes
here.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


GOLDSET_PATH = Path("Data/evals/rag/retrieval_goldset.jsonl")
ORACLE_DIAGNOSTIC_PATH = Path("Data/evals/rag/latest_rag_stage_oracle_diagnostic.json")
PACKET_OUTPUT_PATH = Path("Data/evals/rag/source_filter_drop_adjudication_packet.json")
READINESS_OUTPUT_PATH = Path("Data/evals/rag/latest_goldset_adjudication_readiness.json")


ALLOWED_DECISIONS: frozenset[str] = frozenset({
    "keep_expected_sources",
    "revise_patient_facing_expected_sources",
    "move_to_clinician_facing_goldset",
    "split_patient_and_clinician_cases",
    "mark_ambiguous_needs_external_review",
})

# Decisions that MUST be accompanied by reviewer_notes (the brief's
# rule: "if decision changes expected sources, reviewer_notes must be
# non-empty").
DECISIONS_REQUIRING_NOTES: frozenset[str] = frozenset({
    "revise_patient_facing_expected_sources",
    "split_patient_and_clinician_cases",
})

# Decisions that MUST carry a reviewer_role (the brief's rule: "if
# decision moves to clinician-facing goldset, reviewer_role must be
# present").
DECISIONS_REQUIRING_REVIEWER_ROLE: frozenset[str] = frozenset({
    "move_to_clinician_facing_goldset",
    "split_patient_and_clinician_cases",
})


# ─── Packet construction ────────────────────────────────────────────────


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _load_oracle_diagnostic(path: Path = ORACLE_DIAGNOSTIC_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _replay_filter_for_case(case: Mapping[str, Any]) -> dict[str, Any]:
    """Re-run the retrieval + tier filter for a single case and capture
    pre-filter / post-filter source IDs.

    Uses the same primitives the baseline-comparison uses, so a
    divergence here is itself a signal that retrieval has changed.
    """
    # Local imports keep the module decoupled from rag_baseline_comparison
    # at module load (the diagnostic and packet builders are sibling
    # consumers of the same primitives).
    from backend.services.agent_query_rewriting import rewrite_and_decompose
    from backend.services.agent_rag import _knowledge_snippets, knowledge_base_fingerprint
    from backend.services.agent_retrieval import expand_parent_child_windows
    from backend.services.rag_vector_index import search_hybrid_index
    from backend.services.rag_baseline_comparison import (
        _apply_case_source_filter,
        _dedupe_rows,
        _map_goldset_intent,
        _row_ids,
    )

    query = str(case.get("user_query") or case.get("query") or "")
    intent = _map_goldset_intent(str(case.get("expected_intent") or "education"))
    rewritten = rewrite_and_decompose(query, intent)
    rewritten_query = str(rewritten.get("expanded_query") or query)

    corpus = _knowledge_snippets()
    fingerprint = knowledge_base_fingerprint()
    pool = search_hybrid_index(
        query=rewritten_query if rewritten_query else query,
        corpus=corpus,
        intent=intent,
        knowledge_fingerprint=fingerprint,
        candidate_limit=50,
    )
    ranked = sorted(pool, key=lambda r: float(r.get("retrieval_score") or 0.0), reverse=True)
    seeded = ranked[:20]
    expanded = sorted(
        expand_parent_child_windows(seeded),
        key=lambda r: float(r.get("retrieval_score") or 0.0),
        reverse=True,
    )
    pre_filter = _dedupe_rows(expanded)[:20]
    post_filter = _apply_case_source_filter(case, pre_filter)

    pre_ids = sorted({i for row in pre_filter for i in _row_ids(row)})
    post_ids = sorted({i for row in post_filter for i in _row_ids(row)})
    dropped_ids = sorted(set(pre_ids) - set(post_ids))

    # Project the expected IDs against pre/post-filter; the dropped
    # expected list is what the reviewer most cares about.
    expected_norm = {str(s).strip().lower() for s in (case.get("expected_source_ids") or [])}
    pre_norm = {x.strip().lower() for x in pre_ids}
    post_norm = {x.strip().lower() for x in post_ids}
    dropped_expected = sorted(expected_norm & pre_norm - post_norm)
    # Some expected IDs never appear as raw row IDs (alias / metadata
    # mismatch).  We still surface them so the reviewer can see what
    # was *asked for*.
    expected_not_in_pre_filter = sorted(expected_norm - pre_norm)

    return {
        "retrieved_pre_filter_source_ids": pre_ids[:20],
        "kept_post_filter_source_ids": post_ids[:20],
        "dropped_chunk_ids": dropped_ids[:20],
        "dropped_expected_source_ids": dropped_expected,
        "expected_source_ids_never_in_pre_filter": expected_not_in_pre_filter,
        "rewritten_query": rewritten_query,
    }


def _policy_summary(case: Mapping[str, Any]) -> str:
    acceptable = case.get("acceptable_source_tiers") or []
    expected_use = case.get("expected_allowed_use") or ""
    return (
        f"Patient-facing filter keeps tiers={list(acceptable)} and "
        f"excludes allowed_use=clinician_only.  Case expects allowed_use="
        f"{expected_use!r}.  The patient-facing baseline filter excludes "
        f"clinician-only and disallowed-use sources before citation assembly; "
        f"this is the safety contract the brief forbids weakening."
    )


def _reason_source_was_dropped(case: Mapping[str, Any]) -> str:
    """Heuristic description of why the filter dropped the gold.

    We don't store provenance per-chunk in the runtime, so we describe
    the *structural* reason: the case's acceptable tiers or allowed
    uses include a class the patient-facing baseline filter is
    designed to exclude.
    """
    acceptable = [str(t).upper() for t in (case.get("acceptable_source_tiers") or [])]
    high_tier_only = bool(acceptable) and all(t in {"T4", "T5"} for t in acceptable)
    expected_use = str(case.get("expected_allowed_use") or "").lower()
    clinician_use = "clinician" in expected_use or "boundary" in expected_use

    if high_tier_only:
        return (
            "Case's acceptable_source_tiers is restricted to T4/T5, which the "
            "patient-facing baseline filter excludes by policy.  Filter behaviour "
            "is correct under the patient-facing audience contract."
        )
    if clinician_use:
        return (
            f"Case's expected_allowed_use ({case.get('expected_allowed_use')!r}) "
            "names a policy/boundary use that the patient-facing baseline filter "
            "treats as clinician-facing.  Filter behaviour is correct under the "
            "patient-facing audience contract."
        )
    return (
        "Filter dropped the expected source despite acceptable tiers including "
        "patient-facing tiers — see retrieved_pre_filter_source_ids vs "
        "kept_post_filter_source_ids for the specific chunks the filter excluded."
    )


def build_packet(
    *,
    goldset_path: Path = GOLDSET_PATH,
    oracle_path: Path = ORACLE_DIAGNOSTIC_PATH,
) -> dict[str, Any]:
    started = time.perf_counter()
    goldset = _load_jsonl(goldset_path)
    oracle = _load_oracle_diagnostic(oracle_path)
    goldset_by_id = {str(c.get("case_id")): c for c in goldset}

    drop_case_ids: list[str] = [
        str(c.get("case_id"))
        for c in (oracle.get("cases") or [])
        if c.get("final_failure_stage") == "source_filter_drop"
    ]

    items: list[dict[str, Any]] = []
    for case_id in drop_case_ids:
        goldset_case = goldset_by_id.get(case_id)
        if goldset_case is None:
            continue
        replay = _replay_filter_for_case(goldset_case)
        items.append(_build_item(goldset_case, replay))

    return {
        "schema_version": "rag_goldset_adjudication_packet_v1",
        "status": "ready_for_adjudication",
        "completed": False,
        "label": "rag_goldset_adjudication_packet",
        "clinical_validation": False,
        "claim_boundary": (
            "Adjudication packet — engineering workflow only.  No retrieval, "
            "source governance, or goldset content changes have been made.  "
            "Filling in this packet does NOT auto-apply any correction; it "
            "produces a reviewer artifact that a future PR can act on under "
            "the no-read protocol.  Not clinical validation."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "goldset_path": str(goldset_path).replace("\\", "/"),
        "oracle_diagnostic_path": str(oracle_path).replace("\\", "/"),
        "goldset_sha256_at_packet_time": _digest(goldset_path),
        "n_drop_cases": len(items),
        "allowed_decisions": sorted(ALLOWED_DECISIONS),
        "decisions_requiring_notes": sorted(DECISIONS_REQUIRING_NOTES),
        "decisions_requiring_reviewer_role": sorted(DECISIONS_REQUIRING_REVIEWER_ROLE),
        "items": items,
        "wall_time_ms": round((time.perf_counter() - started) * 1000.0, 2),
        "contamination_note": (
            "The drop set was selected from the frozen internal goldset using "
            "the read-only stage-wise diagnostic.  No goldset case has been "
            "altered.  Adjudication outcomes must be applied under "
            "docs/evals/no_read_rag_goldset_protocol.md when they change "
            "expected sources, and must NOT weaken source governance."
        ),
    }


def _build_item(case: Mapping[str, Any], replay: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case.get("case_id"),
        "user_query": case.get("user_query") or case.get("query"),
        "expected_intent": case.get("expected_intent"),
        "category": case.get("category"),
        "category_tags": list(case.get("category_tags") or []),
        "expected_answerability_status": case.get("expected_answerability_status"),
        "expected_allowed_use": case.get("expected_allowed_use"),
        "acceptable_source_tiers": list(case.get("acceptable_source_tiers") or []),
        "expected_source_ids": list(case.get("expected_source_ids") or []),
        "retrieved_pre_filter_source_ids": list(replay.get("retrieved_pre_filter_source_ids") or []),
        "kept_post_filter_source_ids": list(replay.get("kept_post_filter_source_ids") or []),
        "dropped_expected_source_ids": list(replay.get("dropped_expected_source_ids") or []),
        "expected_source_ids_never_in_pre_filter": list(
            replay.get("expected_source_ids_never_in_pre_filter") or []
        ),
        "rewritten_query": replay.get("rewritten_query"),
        "reason_source_was_dropped": _reason_source_was_dropped(case),
        "current_patient_facing_policy_summary": _policy_summary(case),
        "adjudication_options": sorted(ALLOWED_DECISIONS),
        "reviewer_decision": None,
        "reviewer_role": None,
        "reviewer_notes": None,
        "linked_artifact": None,
        "clinical_validation": False,
    }


def _digest(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_packet(
    output_path: Path = PACKET_OUTPUT_PATH,
    *,
    goldset_path: Path = GOLDSET_PATH,
    oracle_path: Path = ORACLE_DIAGNOSTIC_PATH,
) -> Path:
    packet = build_packet(goldset_path=goldset_path, oracle_path=oracle_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(packet, indent=2), encoding="utf-8")
    return output_path


# ─── Validator ──────────────────────────────────────────────────────────


@dataclass
class ValidationIssue:
    case_id: str | None
    issue: str

    def to_dict(self) -> dict[str, Any]:
        return {"case_id": self.case_id, "issue": self.issue}


def validate_packet(packet: Mapping[str, Any]) -> list[ValidationIssue]:
    """Return a list of issues.  Empty list means the packet is valid."""
    issues: list[ValidationIssue] = []

    if packet.get("clinical_validation") is not False:
        issues.append(ValidationIssue(None, "packet.clinical_validation must be false"))

    if str(packet.get("status") or "") not in {"ready_for_adjudication", "needs_attention"} \
            and packet.get("completed") is True:
        # A 'completed' packet is OK but the brief explicitly wires the
        # draft to status: ready_for_adjudication, completed: false.
        pass

    items = packet.get("items") or []
    for item in items:
        case_id = str(item.get("case_id") or "?")
        if item.get("clinical_validation") is not False:
            issues.append(ValidationIssue(case_id, "item.clinical_validation must be false"))

        decision = item.get("reviewer_decision")
        if decision is None:
            # Draft is fine; no further checks required.
            continue

        if decision not in ALLOWED_DECISIONS:
            issues.append(ValidationIssue(case_id, f"reviewer_decision {decision!r} not in {sorted(ALLOWED_DECISIONS)}"))
            continue

        notes = item.get("reviewer_notes") or ""
        notes_present = isinstance(notes, str) and notes.strip() != ""
        if decision in DECISIONS_REQUIRING_NOTES and not notes_present:
            issues.append(ValidationIssue(case_id, f"reviewer_decision {decision!r} requires non-empty reviewer_notes"))

        role = item.get("reviewer_role")
        role_present = isinstance(role, str) and role.strip() != ""
        if decision in DECISIONS_REQUIRING_REVIEWER_ROLE and not role_present:
            issues.append(ValidationIssue(case_id, f"reviewer_decision {decision!r} requires reviewer_role"))

    return issues


def packet_did_not_mutate_goldset(
    packet: Mapping[str, Any],
    *,
    goldset_path: Path = GOLDSET_PATH,
) -> bool:
    """Return True iff the goldset hash recorded in the packet matches
    the live file.  Lock-in invariant: packet creation must NEVER touch
    the goldset."""
    recorded = str(packet.get("goldset_sha256_at_packet_time") or "")
    current = _digest(goldset_path)
    return bool(recorded) and recorded == current


def build_readiness_report(
    *,
    packet_path: Path = PACKET_OUTPUT_PATH,
    goldset_path: Path = GOLDSET_PATH,
) -> dict[str, Any]:
    """Build the readiness artifact the release gate consumes."""
    packet: dict[str, Any] = {}
    if packet_path.exists():
        try:
            packet = json.loads(packet_path.read_text(encoding="utf-8"))
        except Exception:
            packet = {}
    issues = validate_packet(packet) if packet else []
    mutation_check_ok = packet_did_not_mutate_goldset(packet, goldset_path=goldset_path) if packet else False

    items = packet.get("items") or []
    n_drafts = sum(1 for it in items if it.get("reviewer_decision") is None)
    n_filled = sum(1 for it in items if it.get("reviewer_decision") is not None)

    return {
        "schema_version": "rag_goldset_adjudication_readiness_v1",
        "status": "ready_for_adjudication",
        "completed": False,
        "clinical_validation": False,
        "label": "rag_goldset_adjudication_readiness",
        "claim_boundary": (
            "Adjudication readiness — engineering workflow only.  Reports "
            "whether the packet exists, whether it has been touched, and "
            "whether the goldset was mutated.  No retrieval, governance, or "
            "goldset content change is implied.  Not clinical validation."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "packet_path": str(packet_path).replace("\\", "/"),
        "packet_exists": packet_path.exists(),
        "packet_goldset_unmodified": mutation_check_ok,
        "n_drop_cases": packet.get("n_drop_cases"),
        "n_draft_decisions": n_drafts,
        "n_filled_decisions": n_filled,
        "n_validation_issues": len(issues),
        "validation_issues": [i.to_dict() for i in issues[:20]],
        "next_human_step": (
            "An external reviewer (peer engineer or clinician) opens the packet, "
            "picks one of the allowed_decisions per item, files reviewer_role + "
            "reviewer_notes where required by the validator, and commits the "
            "filled packet under the no-read protocol.  No auto-apply."
        ),
    }


def write_readiness_report(
    output_path: Path = READINESS_OUTPUT_PATH,
    *,
    packet_path: Path = PACKET_OUTPUT_PATH,
    goldset_path: Path = GOLDSET_PATH,
) -> Path:
    payload = build_readiness_report(packet_path=packet_path, goldset_path=goldset_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


__all__ = [
    "ALLOWED_DECISIONS",
    "DECISIONS_REQUIRING_NOTES",
    "DECISIONS_REQUIRING_REVIEWER_ROLE",
    "PACKET_OUTPUT_PATH",
    "READINESS_OUTPUT_PATH",
    "build_packet",
    "build_readiness_report",
    "packet_did_not_mutate_goldset",
    "validate_packet",
    "write_packet",
    "write_readiness_report",
]
