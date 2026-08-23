"""Sections 18-19: the local runbook and repository map, and the current
verdict with next engineering moves.

Extracted verbatim from ``build_story`` in
``scripts/generate_project_guide_pdf.py``, which had grown to 1032 lines in a
single function. Flowable content and ordering are unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from reportlab.lib.units import mm
from reportlab.platypus import Spacer
from scripts.project_guide.components import P, bullets, callout, data_table, page_break, section

if TYPE_CHECKING:
    from scripts.project_guide.evidence import Evidence

def build(story: list[Any], ev: Evidence) -> None:
    """Append this module's sections to `story`, in order."""
    section(story, "18", "Runbook, ports, and repository map", "Use the local URLs for engineering review only. Test-only and optional services are deliberately separated.")
    story.append(P("Default local services", "Heading2Custom"))
    story.append(
        data_table(
            ["Service", "Default URL / port", "Purpose", "Status meaning"],
            [
                ["React frontend", "http://127.0.0.1:5173", "Patient, clinician, and admin UI", "Active only when Vite is running"],
                ["FastAPI backend", "http://127.0.0.1:8017", "API and application services", "Engineering API, not a clinical deployment"],
                ["FastAPI docs", "http://127.0.0.1:8017/docs", "OpenAPI explorer", "Developer surface"],
                ["Ollama", "http://127.0.0.1:11434", "Optional local LLM runtime", "Model availability depends on local installation"],
                ["n8n", "http://127.0.0.1:5678", "Optional automation control plane", "Not started by the core app"],
                ["Pinecone", "No local port", "Optional managed vector backend", "Cloud service; disabled unless configured"],
                ["Playwright frontend", "http://127.0.0.1:5273", "Disposable E2E test server", "Test-only"],
                ["Playwright backend", "http://127.0.0.1:8117", "Disposable isolated-data API", "Test-only"],
            ],
            [34 * mm, 43 * mm, 51 * mm, 42 * mm],
        )
    )
    story.append(P("Repository landmarks", "Heading2Custom"))
    story.append(
        data_table(
            ["Path", "Purpose"],
            [
                ["backend/services/", "Agent, RAG, safety, prediction, trace, and evaluation services"],
                ["backend/routers/", "Role/API surfaces"],
                ["frontend-react/src/", "React/TypeScript product UI"],
                ["tests/", "Backend integration, unit, regression, and governance tests"],
                ["frontend-react/tests/", "Vitest and Playwright suites"],
                ["Data/evals/", "Machine-readable engineering evidence and failure artifacts"],
                ["config/release_gate_thresholds.yaml", "Release tier and threshold policy"],
                ["docs/", "Architecture, eval, review, boundary, and runbook documentation"],
                ["scripts/ship.py", "Cross-platform end-to-end engineering gate"],
            ],
            [67 * mm, 103 * mm],
        )
    )
    story.append(P("Startup checklist", "Heading2Custom"))
    story.extend(
        bullets(
            [
                "Confirm environment variables and demo-only database configuration.",
                "Start the FastAPI backend on port 8017 and verify /health or /docs.",
                "Start the Vite frontend on port 5173 and verify login plus patient/clinician/admin smoke paths.",
                "Verify Ollama only if the selected agent configuration requires local generation.",
                "Do not expose n8n, databases, or developer docs publicly without authentication and deployment review.",
            ]
        )
    )
    page_break(story)
    section(story, "19", "Current verdict and next engineering moves", "The best next work improves independent credibility and system simplicity, not feature count.")
    story.append(P("Current constraint-aware verdict", "Heading2Custom"))
    story.append(
        data_table(
            ["Dimension", "Current engineering reading", "Highest-ROI next move"],
            [
                ["AI / RAG", "Layered and inspectable; retrieval superiority and claim precision remain unproven", "Complete no-read holdout and promote entailment-grade claim validation only if it wins"],
                ["Agent safety", "Strong known-regression coverage; held-out variation remains weak", "External/mutation-authored multi-turn red team with safe-negative controls"],
                ["ML / MLE", "Strong lifecycle scaffolding; saturated synthetic evidence", "Noisier generator v2, temporal-violation cleanup, target redesign, and stronger uncertainty slices"],
                ["Medical", "Boundaries are explicit; zero expert review", "One structured oncology nurse/clinician review and one VUS-focused genetic counselor review"],
                ["SWE", "Good modularity and release discipline; local deployment evidence only", "Containerized staging, migration/backup drills, dependency/security remediation, soak/load testing"],
                ["Product", "Usable reviewer surfaces; explanation burden remains high", "Task-based usability and overtrust tests, then simplify labels and dashboards from observed friction"],
            ],
            [32 * mm, 70 * mm, 68 * mm],
        )
    )
    story.append(P("Recommended sequence", "Heading2Custom"))
    story.extend(
        bullets(
            [
                "1. Freeze the current engineering baseline and archive the exact ship artifacts.",
                "2. Complete external no-read RAG and adversarial authoring before more retrieval tuning.",
                "3. Resolve the patient-facing versus clinician-facing goldset adjudication without weakening source policy.",
                "4. Fix the synthetic temporal violations and redesign shortcut-prone targets before adding larger models.",
                "5. Run task-based usability/overtrust sessions using synthetic/demo records only.",
                "6. Add containerized staging, database migrations, backup/restore, observability, rate limiting, and failure injection.",
                "7. Integrate optional n8n/Pinecone only behind measured, reversible adapters.",
                "8. Seek external review; further internal artifact growth has diminishing credibility returns.",
            ]
        )
    )
    story.append(
        callout(
            "What can be claimed",
            "NLCare demonstrates safety-governed RAG, bounded agent workflows, adversarial and metamorphic evaluation, source/claim traceability, synthetic temporal MLE governance, and release discipline in an engineering prototype.",
            tone="green",
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(
        callout(
            "What still cannot be claimed",
            "Clinical validation, clinician approval, real-world patient safety or benefit, FHIR interoperability, hospital readiness, production healthcare readiness, diagnostic/treatment/prognostic authority, real-patient generalization, or proven raw-retrieval superiority over BM25.",
            tone="red",
        )
    )
    story.append(Spacer(1, 6 * mm))
    story.append(P("Engineering north star", "Heading2Custom"))
    story.append(
        P(
            "The project's strongest form is not the one with the most agents, metrics, or artifacts. It is the one in which every action is bounded, every number is explainable, every negative result stays visible, every promotion is reversible, and every clinical claim remains below the evidence actually available.",
        )
    )

    return story
