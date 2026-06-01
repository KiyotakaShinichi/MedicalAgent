from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("RAG_FORCE_SPARSE", "true")

from backend.services.agent_rag import _knowledge_snippets, knowledge_base_fingerprint  # noqa: E402
from backend.services.governance_readiness_artifacts import write_rag_gold_claim_grounding_cases  # noqa: E402
from scripts.run_retrieval_ablation_metrics import _evaluate_strategy  # noqa: E402


GOLDSET_PATH = ROOT / "Data/evals/rag/retrieval_goldset.jsonl"
SOURCE_GOLD_PATH = ROOT / "Data/evals/rag/gold_claim_grounding_cases.jsonl"
OUTPUT_PATH = ROOT / "Data/evals/rag/latest_retrieval_goldset_eval.json"


def main() -> int:
    if GOLDSET_PATH.exists():
        cases = _load_retrieval_goldset(GOLDSET_PATH)
    else:
        if not SOURCE_GOLD_PATH.exists():
            write_rag_gold_claim_grounding_cases()
        cases = _build_retrieval_goldset()
    corpus = _knowledge_snippets()
    fingerprint = knowledge_base_fingerprint()
    strategies = {
        name: _evaluate_strategy(name, cases, corpus, fingerprint)
        for name in ("dense_only", "sparse_only", "hybrid_rrf", "hybrid_rrf_cross_encoder")
    }
    governed = _governed_view(strategies["hybrid_rrf_cross_encoder"])
    strategies["hybrid_rrf_cross_encoder_source_governed"] = governed
    best = strategies["hybrid_rrf_cross_encoder_source_governed"]["summary"]
    baseline = strategies["hybrid_rrf"]["summary"]
    payload = {
        "schema_version": "retrieval_goldset_eval_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "acceptable" if best["source_tier_correctness"] >= 1.0 else "needs_attention",
        "total_n": len(cases),
        "authored_by": _goldset_authors(cases),
        "authored_date": _goldset_authored_date(cases),
        "was_used_for_tuning": any(bool(case.get("was_used_for_tuning")) for case in cases),
        "internal_vs_external_authored": _authorship_scope(cases),
        "contamination_note": (
            "Goldset is frozen/internal unless a case explicitly marks external authorship. "
            "Use it for retrieval evidence quality, not clinical or external validation."
        ),
        "strategies": strategies,
        "summary": {
            "baseline_recall_at_10": baseline["recall_at_10"],
            "recall_at_5": best["recall_at_5"],
            "recall_at_10": best["recall_at_10"],
            "recall_at_10_delta": round(best["recall_at_10"] - baseline["recall_at_10"], 4),
            "mrr": best["mrr"],
            "ndcg_at_5": best.get("ndcg_at_5", best["ndcg_at_10"]),
            "ndcg_at_10": best["ndcg_at_10"],
            "source_hit_rate": best["source_hit_rate"],
            "citation_support_rate": best["citation_support_rate"],
            "claim_support_rate": best["claim_support_rate"],
            "unsupported_context_rate": best["unsupported_answer_rate"],
            "source_tier_correctness": best["source_tier_correctness"],
            "reranker_latency_ms": best["reranker_latency_ms"],
            "total_retrieval_latency_p95": best["p95_retrieval_latency_ms"],
            "improvement_proven": _improvement_proven(strategies),
            "improvement_note": (
                "Improvement may reflect metadata/source-ID normalization as well as retrieval scoring; "
                "do not present as cross-encoder lift unless reranker metrics beat hybrid RRF."
            ),
        },
        "claim_boundary": (
            "Retrieval goldset metrics are engineering evidence only. Do not claim "
            "cross-encoder improvement unless the improvement_proven field is true."
        ),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    return 0


def _build_retrieval_goldset() -> list[dict]:
    source_cases = [
        json.loads(line)
        for line in SOURCE_GOLD_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = []
    for case in source_cases:
        rows.append({
            "case_id": f"retrieval_{case['case_id']}",
            "query": case["user_query"],
            "user_query": case["user_query"],
            "expected_intent": case["expected_intent"],
            "gold_source_ids": case.get("expected_source_ids", []),
            "expected_source_ids": case.get("expected_source_ids", []),
            "acceptable_source_tiers": case.get("required_source_tiers", ["T1", "T2", "T3"]),
            "relevant_chunk_ids": [],
            "contradiction_traps": case.get("contradiction_traps", []),
            "near_duplicate_distractors": ["similar wording from a source that lacks the safety boundary"],
            "stale_source_distractors": ["stale_blog_or_vendor_content"],
            "clinician_only_distractors": ["clinician_only_protocol_excerpt"],
            "expected_allowed_use": case.get("allowed_answer_scope"),
            "expected_refusal_or_insufficient_evidence": bool(case.get("expected_refusal_or_escalation")),
            "authored_by": "engineering",
            "authored_date": "2026-05-21",
            "was_used_for_tuning": False,
        })
    GOLDSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GOLDSET_PATH.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def _load_retrieval_goldset(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"Retrieval goldset is empty: {path}")
    return rows


def _goldset_authored_date(cases: list[dict]) -> str | None:
    dates = sorted({str(case.get("authored_date")) for case in cases if case.get("authored_date")})
    return dates[-1] if dates else None


def _goldset_authors(cases: list[dict]) -> str:
    authors = sorted({str(case.get("authored_by")) for case in cases if case.get("authored_by")})
    return ",".join(authors) if authors else "unknown"


def _authorship_scope(cases: list[dict]) -> str:
    scopes = {str(case.get("internal_vs_external_authored") or "internal") for case in cases}
    if scopes == {"internal"}:
        return "internal"
    if scopes == {"external"}:
        return "external"
    return "mixed"


def _governed_view(strategy_result: dict) -> dict:
    rows = []
    for row in strategy_result.get("cases", []):
        governed = dict(row)
        governed["source_tier_correct"] = bool(row.get("source_tier_correct"))
        rows.append(governed)
    summary = dict(strategy_result.get("summary", {}))
    summary["source_tier_correctness"] = (
        sum(1 for row in rows if row.get("source_tier_correct")) / max(len(rows), 1)
    )
    return {"summary": summary, "cases": rows}


def _improvement_proven(strategies: dict) -> bool:
    base = strategies["hybrid_rrf"]["summary"]
    after = strategies["hybrid_rrf_cross_encoder"]["summary"]
    return (
        after["recall_at_10"] > base["recall_at_10"]
        and after["unsupported_answer_rate"] <= base["unsupported_answer_rate"]
    )


if __name__ == "__main__":
    raise SystemExit(main())
