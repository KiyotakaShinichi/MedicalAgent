from __future__ import annotations

import re

from pypdf import PdfReader

from scripts.generate_learning_guide_pdf import SOURCE, build_pdf


def test_learning_guide_covers_every_engineering_surface() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    expected_terms = [
        "Retrieval-Augmented Generation",
        "Bounded Agentic Systems",
        "Machine Learning and Statistical Reasoning",
        "Explainability and Fine-Tuning",
        "Software and Data Engineering",
        "Automation, Infrastructure, and Deployment",
        "Medical Structure and Governance",
        "Accuracy-Latency-Unit Cost Trinity",
        "Short-answer quiz",
        "Compact glossary",
    ]
    assert all(term in text for term in expected_terms)
    assert len(re.findall(r"^## \d+\.", text, flags=re.MULTILINE)) == 45


def test_learning_guide_keeps_the_clinical_boundary_explicit() -> None:
    text = SOURCE.read_text(encoding="utf-8").lower()
    assert "not clinically validated" in text
    assert "not medical training" in text
    assert "not a diagnostic" in text
    assert "production healthcare readiness" in text


def test_learning_guide_pdf_builds_with_toc_and_complete_text(tmp_path) -> None:
    output = build_pdf(output=tmp_path / "nlcare_engineering_academy.pdf")
    reader = PdfReader(output)
    assert len(reader.pages) >= 55
    assert reader.metadata is not None
    assert reader.metadata.title == "NLCare Engineering Academy"

    first_pages = "\n".join((page.extract_text() or "") for page in reader.pages[:5])
    final_pages = "\n".join((page.extract_text() or "") for page in reader.pages[-3:])
    assert "Part I. System Foundations" in first_pages
    assert "Final mastery standard" in final_pages
    assert all((page.extract_text() or "").strip() for page in reader.pages)
