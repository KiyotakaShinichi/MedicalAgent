"""Offline degradation drills for the local RAG index.

The drills use disposable indexes and do not call the patient agent or any
managed vector service. They test cache-like index recovery and sparse fallback
behavior without claiming answer quality or clinical safety.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

import joblib

from backend.services.rag_vector_index import (
    build_rag_vector_index,
    corpus_fingerprint,
    load_rag_vector_index,
    rag_index_status,
    search_hybrid_index,
)


DEFAULT_OUTPUT_PATH = Path("Data/evals/rag/latest_rag_degradation_resilience.json")


def build_rag_degradation_resilience_eval(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    # These are local recovery drills, so force the supported sparse backend.
    # This models startup without optional dense dependencies and keeps the
    # drill deterministic without loading or downloading an encoder.
    with (
        patch("backend.services.rag_vector_index._DENSE_AVAILABLE", False),
        TemporaryDirectory(prefix="nlcare-rag-degrade-") as temp_dir,
    ):
        root = Path(temp_dir)
        corpus = _corpus()

        corrupted_path = root / "corrupted.joblib"
        corrupted_path.write_bytes(b"not-a-joblib-index")
        recovered = search_hybrid_index(
            "What does a CBC track?",
            corpus,
            index_path=str(corrupted_path),
            candidate_limit=3,
        )
        cases.append(_case(
            "corrupted_index_rebuild",
            bool(recovered and load_rag_vector_index(corrupted_path)),
            {"top_id": recovered[0]["id"] if recovered else None},
        ))

        stale_path = root / "stale.joblib"
        build_rag_vector_index(corpus, stale_path)
        updated = [
            *corpus,
            {
                "id": "new-vus",
                "title": "VUS limitations",
                "source_name": "curated genetics",
                "text": "A variant of uncertain significance is inconclusive and needs genetics review.",
                "tags": ["vus", "genetics", "inconclusive"],
                "source_tier": "T2",
                "allowed_use": ["education"],
            },
        ]
        refreshed = search_hybrid_index(
            "Why is a VUS inconclusive?",
            updated,
            index_path=str(stale_path),
            candidate_limit=5,
        )
        refreshed_status = rag_index_status(
            updated,
            stale_path,
            corpus_fingerprint(updated),
        )
        cases.append(_case(
            "stale_fingerprint_rebuild",
            refreshed_status["status"] == "current"
            and any(row["id"] == "new-vus" for row in refreshed),
            {
                "status": refreshed_status["status"],
                "new_source_retrieved": any(row["id"] == "new-vus" for row in refreshed),
            },
        ))

        sparse_path = root / "sparse-fallback.joblib"
        build_rag_vector_index(corpus, sparse_path)
        payload = load_rag_vector_index(sparse_path)
        payload["doc_embeddings"] = None
        joblib.dump(payload, sparse_path)
        sparse = search_hybrid_index(
            "CBC white blood cells",
            corpus,
            index_path=str(sparse_path),
            candidate_limit=3,
        )
        cases.append(_case(
            "dense_component_unavailable_sparse_fallback",
            bool(sparse)
            and all(row["retrieval_backend"] == "local_sparse_tfidf_bm25_index" for row in sparse),
            {
                "result_n": len(sparse),
                "backends": sorted({row["retrieval_backend"] for row in sparse}),
            },
        ))

        minimal_path = root / "minimal-metadata.joblib"
        minimal = [
            {"id": "minimal", "text": "general portal education"},
            {"id": "other-lab", "text": "laboratory trend record"},
            {"id": "other-image", "text": "imaging report record"},
        ]
        minimal_rows = search_hybrid_index(
            "portal education",
            minimal,
            index_path=str(minimal_path),
        )
        cases.append(_case(
            "missing_optional_metadata_tolerated",
            bool(minimal_rows and minimal_rows[0]["id"] == "minimal"),
            {"result_n": len(minimal_rows)},
        ))

        cases.append(_case(
            "empty_query_fails_closed",
            search_hybrid_index("", corpus, index_path=str(root / "empty.joblib")) == [],
            {"result_n": 0},
        ))

    passed = sum(int(case["passed"]) for case in cases)
    payload = {
        "schema_version": "rag_degradation_resilience_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "strong_offline_drill" if passed == len(cases) else "needs_attention",
        "case_count": len(cases),
        "passed_count": passed,
        "failed_count": len(cases) - passed,
        "pass_rate": round(passed / len(cases), 6),
        "cases": cases,
        "patient_agent_invoked": False,
        "managed_network_request_performed": False,
        "retrieval_improvement_proven": False,
        "production_outage_recovery_proven": False,
        "clinical_validation": False,
        "claim_boundary": (
            "These disposable local-index drills test software degradation behavior. "
            "They do not prove answer grounding, managed-service recovery, production "
            "availability, medical safety, or clinical validation."
        ),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _case(case_id: str, passed: bool, evidence: dict[str, Any]) -> dict[str, Any]:
    return {"case_id": case_id, "passed": bool(passed), "evidence": evidence}


def _corpus() -> list[dict[str, Any]]:
    return [
        {
            "id": "cbc",
            "title": "CBC monitoring",
            "source_name": "curated laboratory education",
            "text": "A CBC records white blood cells hemoglobin and platelets for review.",
            "tags": ["cbc", "labs", "monitoring"],
            "source_tier": "T2",
            "allowed_use": ["education"],
        },
        {
            "id": "imaging",
            "title": "Imaging terms",
            "source_name": "curated imaging education",
            "text": "Imaging report wording should be reviewed in clinical context.",
            "tags": ["imaging", "report"],
            "source_tier": "T2",
            "allowed_use": ["education"],
        },
        {
            "id": "boundary",
            "title": "Clinical boundary",
            "source_name": "project safety policy",
            "text": "The portal does not diagnose or decide treatment.",
            "tags": ["safety", "boundary"],
            "source_tier": "T4",
            "allowed_use": ["patient_safety"],
        },
    ]


__all__ = ["build_rag_degradation_resilience_eval"]
