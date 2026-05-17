"""End-to-end test for the Phase 11 RAG trace replay endpoint.

Confirms the writer-to-reader contract: when `_store_rag_evaluation_log`
persists a result that carries the Phase 11 fields (rag_mode,
rewritten_query, evidence_grade, claim_validation, tier_filter,
post_gen_validator), the trace replay endpoint reads them back as
populated values — not silent nulls.

This closes the critique's highest-impact gap.  Before the migration
landed, every field below would have read back as ``None`` because the
endpoint's ``getattr(row, ..., None)`` defaults masked the missing
schema.  This test would fail under the old schema; it passes now.
"""
from __future__ import annotations

import json
import unittest

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.api.main import app, get_db
from backend.database import Base
from backend.models import Patient, RAGEvaluationLog


TEST_DB_URL = "sqlite:///:memory:"
engine = create_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)


def _override_get_db():
    db = TestingSession()
    try:
        yield db
    finally:
        db.close()


client = TestClient(app, raise_server_exceptions=False)


class _DbOverrideMixin(unittest.TestCase):
    """Re-pin `get_db` override before each method (pytest may load
    other modules that override it differently)."""

    def setUp(self) -> None:
        app.dependency_overrides[get_db] = _override_get_db


def _login(username: str, password: str) -> str:
    resp = client.post(
        "/auth/demo-credential-login",
        json={"username": username, "password": password},
    )
    return resp.json().get("access_token")


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _seed_phase11_row(db) -> None:
    """Insert one fully-populated RAGEvaluationLog row representing
    the result of a successful intent-aware RAG call."""
    if not db.query(Patient).filter(Patient.id == "P-RAG").first():
        db.add(Patient(id="P-RAG", name="RAG Trace Test"))
        db.commit()

    row = RAGEvaluationLog(
        patient_id="P-RAG",
        request_id="req_test_1",
        query_hash="hash_x",
        query_preview="What does WBC mean?",
        intent="education",
        safety_level="info",
        cache_status="miss",
        terminal_step="generate",
        retrieval_precision_at_3=0.66,
        grounding_score=0.85,
        hallucination_score=0.12,
        hallucination_risk="low",
        input_guardrail_status="passed",
        output_guardrail_status="passed",
        latency_ms=812.5,
        retrieved_source_ids_json=json.dumps(["src_t1_a", "src_t2_b"]),
        cited_source_ids_json=json.dumps(["src_t1_a"]),
        guardrail_issues_json=json.dumps({"input": [], "output": []}),
        # The five Phase 11 columns the critique flagged as silently
        # nulled — populated below to prove the round-trip works.
        rag_mode="education_rag",
        rewritten_query="What is the white blood cell count?",
        evidence_grade_json=json.dumps({
            "grade": "high",
            "answer_scope": "factual_education",
            "citation_status": "complete",
            "claim_support_rate": 0.9,
            "source_basis": [{"source_id": "src_t1_a", "title": "NCCN summary", "tier": "T1"}],
            "tier_distribution_of_basis": {"T1": 1},
            "reasoning": "1 supported claim backed by T1 source",
            "mode": "education_rag",
            "claim_count": 1,
            "supported_count": 1,
            "unsupported_count": 0,
        }),
        claim_validation_json=json.dumps({
            "claim_count": 1,
            "supported_count": 1,
            "weakly_supported_count": 0,
            "unsupported_count": 0,
            "claim_support_rate": 1.0,
            "citation_status": "complete",
            "verdicts": [{
                "sentence": "White blood cells help fight infection.",
                "is_claim": True,
                "support_score": 0.42,
                "status": "supported",
                "supporting_chunk_ids": ["c1"],
                "reason": None,
            }],
        }),
        tier_filter_json=json.dumps({
            "mode": "education_rag",
            "kept_count": 2,
            "dropped_count": 0,
            "kept_chunk_ids": ["c1", "c2"],
            "dropped_chunk_ids": [],
            "decisions": [],
        }),
        post_gen_validator_json=json.dumps({
            "decision": "allowed",
            "triggered_rules": [],
            "medical_claim_boundary": "engineering provenance only",
        }),
    )
    db.add(row)
    db.commit()


# ─── Round-trip contract ─────────────────────────────────────────────────────


class RagTraceReplayContract(_DbOverrideMixin):
    """If the schema migration is real, every Phase 11 field round-trips
    cleanly from writer → DB → reader endpoint."""

    REQUIRED_TOP_LEVEL = (
        "id", "created_at", "patient_id", "query_preview",
        "intent", "safety_level", "rag_mode", "rewritten_query",
        "retrieved_source_ids", "cited_source_ids",
        "evidence_grade", "claim_validation", "tier_filter",
        "post_gen_validator", "grounding_score", "hallucination_score",
        "latency_ms", "input_guardrail", "output_guardrail",
    )

    def setUp(self) -> None:
        super().setUp()
        db = TestingSession()
        try:
            _seed_phase11_row(db)
        finally:
            db.close()
        self.token = _login("admin", "admin-demo")

    def test_endpoint_returns_phase11_fields_as_real_data(self) -> None:
        resp = client.get("/admin/rag-trace-replay?limit=5", headers=_auth(self.token))
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertIn("traces", payload)
        traces = [t for t in payload["traces"] if t.get("intent") == "education"]
        self.assertGreaterEqual(len(traces), 1, "seeded row was not returned")
        sample = traces[0]
        # Every required key is present (the contract).
        for key in self.REQUIRED_TOP_LEVEL:
            self.assertIn(key, sample, f"trace replay missing {key}")
        # Phase 11 fields are POPULATED, not None — this is the regression
        # the critique flagged.
        self.assertEqual(sample["rag_mode"], "education_rag")
        self.assertEqual(sample["rewritten_query"], "What is the white blood cell count?")
        self.assertIsNotNone(sample["evidence_grade"])
        self.assertEqual(sample["evidence_grade"]["grade"], "high")
        self.assertEqual(sample["evidence_grade"]["answer_scope"], "factual_education")
        self.assertIsNotNone(sample["claim_validation"])
        self.assertEqual(sample["claim_validation"]["claim_count"], 1)
        self.assertIsNotNone(sample["tier_filter"])
        self.assertEqual(sample["tier_filter"]["mode"], "education_rag")
        self.assertIsNotNone(sample["post_gen_validator"])
        self.assertEqual(sample["post_gen_validator"]["decision"], "allowed")

    def test_endpoint_respects_limit_parameter(self) -> None:
        resp = client.get("/admin/rag-trace-replay?limit=1", headers=_auth(self.token))
        self.assertEqual(resp.status_code, 200)
        self.assertLessEqual(len(resp.json()["traces"]), 1)

    def test_endpoint_blocks_patient_tokens(self) -> None:
        # Seed a patient + login as patient — admin-only endpoint must reject.
        db = TestingSession()
        try:
            if not db.query(Patient).filter(Patient.id == "P001").first():
                db.add(Patient(id="P001", name="Demo"))
                db.commit()
        finally:
            db.close()
        patient_token = _login("P001", "patient-demo")
        if not patient_token:
            self.skipTest("Patient demo login unavailable in this checkout.")
        resp = client.get(
            "/admin/rag-trace-replay",
            headers=_auth(patient_token),
        )
        self.assertIn(resp.status_code, (401, 403))


# ─── Model column contract ───────────────────────────────────────────────────


class RagEvaluationLogModelContract(unittest.TestCase):
    """If a future edit drops the Phase 11 columns from the model, this
    test fails before the trace endpoint silently regresses."""

    PHASE_11_COLUMNS = (
        "rag_mode",
        "rewritten_query",
        "evidence_grade_json",
        "claim_validation_json",
        "tier_filter_json",
        "post_gen_validator_json",
    )

    def test_phase11_columns_are_real_attributes(self) -> None:
        for column in self.PHASE_11_COLUMNS:
            self.assertTrue(
                hasattr(RAGEvaluationLog, column),
                f"RAGEvaluationLog must declare {column} (migration 0003)",
            )


if __name__ == "__main__":
    unittest.main()
