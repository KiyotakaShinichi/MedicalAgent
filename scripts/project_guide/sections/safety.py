"""Sections 06-07: adversarial and agent safety evaluation, claim
grounding, and uncertainty.

Extracted verbatim from ``build_story`` in
``scripts/generate_project_guide_pdf.py``, which had grown to 1032 lines in a
single function. Flowable content and ordering are unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from reportlab.lib.units import mm
from scripts.project_guide.components import P, bullets, callout, data_table, flow_diagram, metric_row, page_break, section
from scripts.project_guide.evidence import _dig, _fmt, _pct

if TYPE_CHECKING:
    from scripts.project_guide.evidence import Evidence

def build(story: list[Any], ev: Evidence) -> None:
    """Append this module's sections to `story`, in order."""
    prompt_eval = ev.prompt_eval
    safety = ev.safety
    safety_v4 = ev.safety_v4

    section(story, "06", "Agent and safety evaluation", "Large prompt volume is useful for regression coverage, but does not replace independent red-teaming.")
    total_prompts = prompt_eval.get("prompt_bank_n")
    story.append(
        metric_row(
            [
                ("Synthetic prompt variants", _fmt(total_prompts)),
                ("Classifier pass", _pct(_dig(prompt_eval, "classifier_sweep", "pass_rate"))),
                ("Sampled route accuracy", _pct(_dig(prompt_eval, "bounded_agent_end_to_end_sample", "route_accuracy"))),
                ("Multi-turn pass", _pct(_dig(prompt_eval, "multi_turn_bounded_agent", "conversation_pass_rate"))),
            ]
        )
    )
    story.append(P("What the 5,000-prompt regression covers", "Heading2Custom"))
    story.extend(
        bullets(
            [
                "Unsafe and safe-negative prompt families with punctuation noise, wrappers, typos, Taglish/code-switching, emotional phrasing, and near-boundary wording.",
                "A bounded end-to-end sample that checks route selection, verifier results, forbidden medical authority, and unsafe database writes.",
                "Structured symptom, medication, imaging, lab, and treatment-note cases that distinguish logging from conversation.",
                "Seventy two-turn conversations that test urgent, privacy, treatment, cross-patient, structured-update, and topic-change state behavior.",
            ]
        )
    )
    story.append(
        callout(
            "Contamination warning",
            "This bank was used during hardening and is labelled as a tuning-used internal regression. A perfect post-hardening score means known regression families are covered; it does not prove held-out generalization or clinical safety.",
            tone="red",
        )
    )
    story.append(P("Broader adversarial evidence", "Heading2Custom"))
    story.append(
        data_table(
            ["Suite", "N", "Result", "Interpretation"],
            [
                ["Original adversarial regression", _fmt(safety.get("total_n")), _pct(_dig(safety, "metrics", "refusal_correctness")), "Needs attention; privacy and treatment variants remain material"],
                ["Frozen holdout v4 baseline", _fmt(safety_v4.get("total_n")), _pct(safety_v4.get("pass_rate")), "Current visible generalization weakness; do not call safety solved"],
                ["Large tuning-used prompt regression", _fmt(total_prompts), _pct(_dig(prompt_eval, "classifier_sweep", "pass_rate")), "Regression coverage only; not a holdout"],
            ],
            [43 * mm, 18 * mm, 26 * mm, 83 * mm],
        )
    )
    story.append(P("Safety metrics", "Heading2Custom"))
    story.extend(
        bullets(
            [
                "Unsafe leakage rate: unsafe cases that reached an unsafe answer or action.",
                "Refusal correctness: unsafe boundary cases refused or redirected as expected.",
                "Escalation correctness: urgent/crisis cases routed to immediate help or clinical review without diagnosis.",
                "Over-refusal: safe educational cases incorrectly blocked; measured separately so stronger guards do not erase legitimate education.",
                "Unsafe write count: structured records created by requests that should never write; the expected value is zero.",
            ]
        )
    )
    page_break(story)
    section(story, "07", "Claim grounding and uncertainty", "Retrieval confidence, claim support, and medical authority are distinct decisions.")
    story.append(
        flow_diagram(
            [
                ("Extract claims", "Split generated text into checkable factual or medical propositions"),
                ("Align evidence", "Map claims to cited snippets and source metadata"),
                ("Check support", "Lexical/semantic support plus deterministic contradiction traps"),
                ("Grade answerability", "Sufficient, limited, insufficient, conflicting, review, or refuse"),
                ("Apply boundary", "Block diagnosis, treatment, dosage, prognosis, genetic/tumor-marker conclusion"),
                ("Finalize", "Keep supported education, soften limited evidence, or replace with safe routing"),
            ],
            columns=3,
        )
    )
    story.append(P("Why a single confidence number is misleading", "Heading2Custom"))
    story.append(
        data_table(
            ["Signal", "Meaning", "Failure mode"],
            [
                ["Retrieval confidence", "Quality and agreement of retrieved evidence", "High similarity can still retrieve the wrong source"],
                ["Source-tier confidence", "Whether evidence meets source and audience policy", "High tier does not guarantee query relevance"],
                ["Citation-support confidence", "Estimated claim-to-snippet support", "Heuristic support can miss negation, temporality, and entity mismatch"],
                ["Evidence conflict flag", "Sources or extracted statements disagree", "Absence of detected conflict is not proof of agreement"],
                ["Answerability state", "Policy decision for response mode", "It is a routing decision, not medical certainty"],
            ],
            [35 * mm, 62 * mm, 73 * mm],
        )
    )
    story.append(P("High-risk contradiction patterns", "Heading2Custom"))
    story.extend(
        bullets(
            [
                "Negation inversions: 'no need to contact a clinician' versus evidence that recommends review.",
                "Temporal inversions: prior findings described as current, or current symptoms described as resolved.",
                "Entity mismatches: another patient's record, a different drug, gene, marker, or imaging modality.",
                "Authority escalation: educational evidence converted into diagnosis, treatment, dosage, or prognosis.",
                "Conditional collapse: a statement that applies only under defined circumstances presented as universal.",
                "VUS and tumor-marker overreach: uncertainty or trend context converted into positive mutation or recurrence claims.",
            ]
        )
    )
    story.append(
        callout(
            "Validator limit",
            "The claim validator is an engineering safety layer, not a clinical-grade medical entailment system. When semantic dependencies are unavailable, deterministic and heuristic fallbacks can produce both false confidence and unnecessary refusal. The final authority boundary therefore remains independent of citation support.",
            tone="amber",
        )
    )
    page_break(story)
