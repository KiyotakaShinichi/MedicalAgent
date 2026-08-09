from __future__ import annotations

import hashlib
from pathlib import Path

from backend.services.adversarial_generalization_label_audit import run_label_audit


def test_label_audit_preserves_source_and_requires_human_review(tmp_path: Path) -> None:
    source = Path("Data/evals/safety/latest_adversarial_generalization_eval.json")
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    result = run_label_audit(source, tmp_path / "audit.json")
    after = hashlib.sha256(source.read_bytes()).hexdigest()
    assert before == after == result["source_sha256"]
    assert result["source_artifact_unmodified"] is True
    assert result["original_metric_preserved"] is True
    assert result["original_failure_count"] == 8
    assert result["machine_suspected_safe_negative_conflict_n"] == 8
    assert result["confirmed_true_unsafe_leak_n"] is None
    assert result["human_adjudication_completed"] is False
    assert result["clinical_validation"] is False
