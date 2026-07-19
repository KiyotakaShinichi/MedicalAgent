import hashlib
import json
import tempfile
from pathlib import Path

from backend.services.adversarial_holdout_v6 import build_holdout_v6_cases, freeze_holdout_v6


def test_v6_builder_has_required_coverage_and_honest_metadata():
    rows = build_holdout_v6_cases()
    assert len(rows) == 162
    categories = {row["category"] for row in rows}
    assert {"prognosis_survival", "supplement_replacement", "tumor_marker_overclaim", "vus_misinterpretation"} <= categories
    assert sum(row["safe_negative"] for row in rows) == 30
    assert all(row["was_used_for_tuning"] is False for row in rows)
    assert all("author_contaminated" in row["case_source"] for row in rows)


def test_freeze_is_hash_locked_and_non_overwriting():
    with tempfile.TemporaryDirectory() as tmp:
        bank = Path(tmp) / "bank.jsonl"
        manifest_path = Path(tmp) / "manifest.json"
        manifest = freeze_holdout_v6(bank, manifest_path)
        content = bank.read_text(encoding="utf-8")
        assert hashlib.sha256(content.encode("utf-8")).hexdigest() == manifest["sha256"]
        assert manifest["clinical_validation"] is False
        assert manifest["internal_vs_external"] == "internal_frozen_author_contaminated_holdout"
        try:
            freeze_holdout_v6(bank, manifest_path)
        except FileExistsError:
            pass
        else:
            raise AssertionError("frozen bank was overwritten")
        assert json.loads(manifest_path.read_text(encoding="utf-8"))["total_n"] == 162
