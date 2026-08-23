"""Sections 04-05: the source-governed RAG stack and the frozen-goldset
evidence, including its negative results.

Extracted verbatim from ``build_story`` in
``scripts/generate_project_guide_pdf.py``, which had grown to 1032 lines in a
single function. Flowable content and ordering are unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from reportlab.lib.units import mm
from reportlab.platypus import Spacer
from scripts.project_guide.components import P, bullets, callout, data_table, flow_diagram, page_break, section
from scripts.project_guide.evidence import _fmt

if TYPE_CHECKING:
    from scripts.project_guide.evidence import Evidence

def build(story: list[Any], ev: Evidence) -> None:
    """Append this module's sections to `story`, in order."""
    rag = ev.rag
    bm25 = ev.bm25
    full = ev.full
    rag_summary = ev.rag_summary
    best = ev.best

    section(story, "04", "Source-governed RAG", "The retrieval stack is deliberately evaluated against simpler baselines rather than assumed to be better.")
    story.append(
        flow_diagram(
            [
                ("Intent policy", "Select evidence mode and allowed audience/source rules"),
                ("Query rewrite", "Normalize and expand retrieval wording; retain original query"),
                ("Sparse candidates", "BM25 lexical match"),
                ("Dense candidates", "Local embedding similarity and FAISS"),
                ("RRF fusion", "Combine ranks without assuming score comparability"),
                ("Context expansion", "Parent-child windows for local context"),
                ("Source filter", "Tier, allowed use, staleness, patient suitability"),
                ("Answerability", "Cited answer, limited answer, insufficient/conflicting evidence, review, refusal"),
                ("Generate", "Constrained response from retained evidence"),
                ("Validate", "Claim-source support, contradictions, output boundary"),
            ],
            columns=5,
        )
    )
    story.append(P("Core retrieval formulas", "Heading2Custom"))
    story.append(P("BM25(q,d) = sum_t IDF(t) * [tf(t,d)*(k1+1)] / [tf(t,d) + k1*(1-b+b*|d|/avgdl)]", "Formula"))
    story.append(P("cosine(q,d) = (q dot d) / (||q|| * ||d||)", "Formula"))
    story.append(P("RRF(d) = sum_r 1 / (k + rank_r(d))", "Formula"))
    story.append(P("MRR = (1/N) * sum_i 1 / rank_i(first relevant)", "Formula"))
    story.append(P("DCG@k = sum_i=1..k relevance_i / log2(i+1); NDCG@k = DCG@k / ideal_DCG@k", "Formula"))
    story.append(P("Why the layers exist", "Heading2Custom"))
    story.extend(
        bullets(
            [
                "BM25 protects exact terms, drug names, genes, and portal labels; dense search captures paraphrase similarity.",
                "RRF combines rank positions because sparse and dense scores are not directly calibrated.",
                "Rewrite and parent-child expansion are hypotheses, not free improvements; each is measured through ablation.",
                "Source-tier filtering is a governance control. It may correctly reduce patient-facing recall when expected labels point to disallowed evidence.",
                "Answerability and claim validation prevent low-quality retrieval from silently becoming a confident response.",
            ]
        )
    )
    page_break(story)
    section(story, "05", "RAG evidence and negative results", "Current internal frozen-goldset evidence supports governance value, not raw retrieval superiority.")
    comparison_rows = []
    for label, item in [("BM25 only", bm25), ("Best observed internal config", best), ("Full source-governed stack", full)]:
        metrics = item.get("metrics", item)
        comparison_rows.append(
            [
                label,
                _fmt(metrics.get("recall_at_5"), 4),
                _fmt(metrics.get("recall_at_10"), 4),
                _fmt(metrics.get("mrr"), 4),
                _fmt(metrics.get("ndcg_at_10"), 4),
                _fmt(metrics.get("citation_precision"), 4),
                _fmt(metrics.get("source_tier_correctness"), 4),
                _fmt(metrics.get("latency_p95_ms"), 1),
            ]
        )
    story.append(
        data_table(
            ["Configuration", "R@5", "R@10", "MRR", "NDCG@10", "Citation precision", "Tier correctness", "p95 ms"],
            comparison_rows,
            [41 * mm, 15 * mm, 15 * mm, 15 * mm, 19 * mm, 23 * mm, 22 * mm, 20 * mm],
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(
        callout(
            "Honest verdict",
            f"On {rag.get('total_n', 'the current')} internal goldset cases, the full source-governed stack has Recall@10 {_fmt(full.get('recall_at_10'), 4)} versus BM25 {_fmt(bm25.get('recall_at_10'), 4)}. The recorded complex-stack delta is {_fmt(rag_summary.get('complex_stack_improvement_over_bm25'), 4)}, and improvement_proven_vs_bm25 is {_fmt(rag_summary.get('improvement_proven_vs_bm25'))}. The complex stack is retained for governance and audience filtering, not advertised as a proven raw-recall improvement.",
            tone="amber",
        )
    )
    story.append(P("Metric interpretation", "Heading2Custom"))
    story.append(
        data_table(
            ["Metric", "What it asks", "What it does not prove"],
            [
                ["Recall@10", "Did an expected source appear in the top ten?", "That the final answer is medically correct"],
                ["MRR", "How early was the first expected source?", "That later context is clean"],
                ["NDCG@10", "How well were graded relevant sources ranked?", "Clinical usefulness"],
                ["Citation precision", "What share of cited chunks matched expected evidence?", "Complete semantic entailment"],
                ["Claim-support rate", "What share of evaluated claims had support?", "Absence of all hallucinations"],
                ["Unsupported-context rate", "How much retained context was not expected?", "Real-world safety"],
                ["Source-tier correctness", "Were audience/source policies followed?", "Retrieval relevance by itself"],
            ],
            [34 * mm, 67 * mm, 69 * mm],
        )
    )
    story.append(P("Known RAG limitations", "Heading2Custom"))
    story.extend(
        bullets(
            [
                "Full-stack citation precision is about 0.52 in the current internal comparison, leaving substantial room for better source alignment.",
                "The stage-oracle diagnostic attributes the largest recall loss to source filtering and goldset audience mismatch, which must be adjudicated without weakening governance.",
                "The experimental context pruner increased Recall@5 but reduced citation precision and was not promoted.",
                "The cross-encoder reranker remains experimental because improvement has not been proven.",
                "The no-read external-author RAG holdout is prepared but incomplete; the current metrics are not independent validation.",
            ]
        )
    )
    page_break(story)
