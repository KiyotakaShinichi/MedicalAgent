import json

from backend.services.adversarial_generalization_vnext import build_adversarial_generalization_vnext


def test_vnext_preserves_frozen_v7_and_labels_mutations_as_tuning_used(tmp_path):
    from backend.services.adversarial_generalization_vnext import V7_BASELINE

    before = V7_BASELINE.read_bytes()
    report = build_adversarial_generalization_vnext(
        output_path=tmp_path / "report.json",
        bank_path=tmp_path / "bank.jsonl",
    )
    assert V7_BASELINE.read_bytes() == before
    assert report["frozen_v7_read_only_attribution"]["re_evaluated"] is False
    assert report["mutation_matrix"]["was_used_for_tuning"] is True
    assert report["mutation_matrix"]["independent_holdout"] is False
    assert report["external_generalization_status"] == "BLOCKED_EXTERNAL"
    assert report["clinical_validation"] is False
    rows = [json.loads(line) for line in (tmp_path / "bank.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) >= 100
    assert all(row["was_used_for_tuning"] is True for row in rows)
