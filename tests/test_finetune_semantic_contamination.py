import json

from backend.services.finetune_semantic_contamination import (
    build_finetune_semantic_contamination_audit,
)


def _row(identifier, split, user, assistant):
    return {
        "id": identifier,
        "split": split,
        "messages": [
            {"role": "system", "content": "boundary"},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


def test_semantic_screen_flags_near_duplicate_without_retaining_text(tmp_path):
    train = tmp_path / "train.jsonl"
    holdout = tmp_path / "holdout.jsonl"
    train.write_text(
        json.dumps(
            _row(
                "train-1",
                "train",
                "Can a VUS prove hereditary breast cancer?",
                "A VUS does not prove hereditary cancer and needs genetics review.",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    holdout.write_text(
        json.dumps(
            _row(
                "holdout-1",
                "internal_frozen_holdout",
                "Does a VUS prove hereditary breast cancer?",
                "A VUS does not prove hereditary cancer; request genetics review.",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    card = tmp_path / "card.json"
    card.write_text(json.dumps({"contamination_audit": {"scanned_files": []}}))
    result = build_finetune_semantic_contamination_audit(
        train_path=train,
        internal_holdout_path=holdout,
        dataset_card_path=card,
        output_path=tmp_path / "out.json",
        doc_path=tmp_path / "out.md",
        review_path=tmp_path / "missing-review.json",
        review_threshold=0.65,
    )
    assert result["summary"]["flagged_pair_count"] >= 1
    assert result["summary"]["unresolved_pair_count"] >= 1
    assert result["summary"]["adjudication_cleared_for_candidate"] is False
    assert result["status"] == "needs_attention"
    assert all(
        item["text_content_retained"] is False
        and "train_text" not in item
        and "evaluation_text" not in item
        for item in result["flagged_pairs"]
    )
    assert result["patient_facing_promotion_allowed"] is False


def test_semantic_screen_accepts_distinct_clean_split(tmp_path):
    train = tmp_path / "train.jsonl"
    holdout = tmp_path / "holdout.jsonl"
    train.write_text(
        json.dumps(_row("train-1", "train", "Organize my CBC record.", "CBC organized."))
        + "\n",
        encoding="utf-8",
    )
    holdout.write_text(
        json.dumps(
            _row(
                "holdout-1",
                "internal_frozen_holdout",
                "I feel worried while waiting for my scan.",
                "I hear that this waiting feels difficult.",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    card = tmp_path / "card.json"
    card.write_text(json.dumps({"contamination_audit": {"scanned_files": []}}))
    result = build_finetune_semantic_contamination_audit(
        train_path=train,
        internal_holdout_path=holdout,
        dataset_card_path=card,
        output_path=tmp_path / "out.json",
        doc_path=tmp_path / "out.md",
        review_path=tmp_path / "review.json",
    )
    assert result["summary"]["flagged_pair_count"] == 0
    assert result["summary"]["review_completed"] is True
    assert result["summary"]["adjudication_cleared_for_candidate"] is True
    assert result["screen"]["semantic_similarity_proxy_completed"] is True
    assert result["summary"]["semantic_contamination_absence_proven"] is False
