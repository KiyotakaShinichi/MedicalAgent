"""Semantic-similarity screening for fine-tune/evaluation contamination.

The screen is intentionally a reviewer aid, not a declaration that two texts
are semantically equivalent. It uses word and character TF-IDF similarity,
keeps source hashes, and never modifies training or evaluation files.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.services.oncology_canonical_schema import ROOT_DIR


TRAIN_PATH = Path("Data/finetune/prepared/dataset_train.jsonl")
INTERNAL_HOLDOUT_PATH = Path(
    "Data/finetune/prepared/dataset_internal_frozen_holdout.jsonl"
)
DATASET_CARD_PATH = Path("Data/finetune/prepared/dataset_card.json")
DEFAULT_OUTPUT_PATH = Path(
    "Data/evals/models/latest_finetune_semantic_contamination.json"
)
DEFAULT_DOC_PATH = Path("docs/finetune_semantic_contamination.md")
DEFAULT_REVIEW_PATH = Path(
    "Data/finetune/evaluations/semantic_contamination_review.json"
)

REVIEW_THRESHOLD = 0.82
CRITICAL_THRESHOLD = 0.93
MAX_FLAGGED_PAIRS = 150

CLAIM_BOUNDARY = (
    "TF-IDF similarity is a lexical-semantic screening proxy, not a clinical "
    "semantic model and not proof that contamination is absent. Flagged pairs "
    "require human adjudication before any offline adapter promotion."
)


def build_finetune_semantic_contamination_audit(
    *,
    train_path: str | Path = TRAIN_PATH,
    internal_holdout_path: str | Path = INTERNAL_HOLDOUT_PATH,
    dataset_card_path: str | Path = DATASET_CARD_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    doc_path: str | Path = DEFAULT_DOC_PATH,
    review_path: str | Path = DEFAULT_REVIEW_PATH,
    review_threshold: float = REVIEW_THRESHOLD,
    critical_threshold: float = CRITICAL_THRESHOLD,
) -> dict[str, Any]:
    train_rows = _load_finetune_rows(train_path, split_label="train")
    evaluation_rows = _load_finetune_rows(
        internal_holdout_path,
        split_label="internal_frozen_holdout",
    )
    dataset_card = _read_json(dataset_card_path)
    for item in (dataset_card.get("contamination_audit") or {}).get(
        "scanned_files"
    ) or []:
        path = item.get("path") if isinstance(item, dict) else None
        if path:
            evaluation_rows.extend(_load_eval_prompts(path))

    user_pairs = _screen_channel(
        [row for row in train_rows if row["channel"] == "user"],
        [row for row in evaluation_rows if row["channel"] == "user"],
        review_threshold=review_threshold,
        critical_threshold=critical_threshold,
    )
    assistant_pairs = _screen_channel(
        [row for row in train_rows if row["channel"] == "assistant"],
        [row for row in evaluation_rows if row["channel"] == "assistant"],
        review_threshold=review_threshold,
        critical_threshold=critical_threshold,
    )
    all_flagged = sorted(
        user_pairs + assistant_pairs,
        key=lambda item: item["max_similarity"],
        reverse=True,
    )
    flagged = all_flagged[:MAX_FLAGGED_PAIRS]
    truncated_pair_count = max(0, len(all_flagged) - len(flagged))
    critical_count = sum(
        item["max_similarity"] >= critical_threshold for item in all_flagged
    )
    review = _read_json(review_path)
    decisions = {
        str(item.get("pair_id")): str(item.get("decision"))
        for item in review.get("decisions") or []
        if isinstance(item, dict)
        and item.get("decision") in {"contaminated", "not_contaminated", "ambiguous"}
    }
    unresolved = [
        item["pair_id"] for item in flagged if item["pair_id"] not in decisions
    ]
    review_completed = not unresolved and truncated_pair_count == 0
    contaminated_count = sum(value == "contaminated" for value in decisions.values())
    ambiguous_count = sum(value == "ambiguous" for value in decisions.values())
    adjudication_cleared = bool(
        review_completed
        and contaminated_count == 0
        and ambiguous_count == 0
    )

    payload = {
        "schema_version": "finetune_semantic_contamination_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "acceptable" if adjudication_cleared else "needs_attention",
        "clinical_validation": False,
        "patient_facing_promotion_allowed": False,
        "screen": {
            "method": "word_and_character_tfidf_cosine_proxy",
            "word_ngram_range": [1, 2],
            "character_ngram_range": [3, 5],
            "review_threshold": review_threshold,
            "critical_threshold": critical_threshold,
            "semantic_similarity_proxy_completed": True,
            "embedding_model_used": False,
            "human_adjudication_required_for_flags": True,
        },
        "corpus": {
            "train_text_count": len(train_rows),
            "evaluation_text_count": len(evaluation_rows),
            "train_path": str(train_path).replace("\\", "/"),
            "internal_holdout_path": str(internal_holdout_path).replace("\\", "/"),
            "source_hashes": _source_hashes(
                [train_path, internal_holdout_path, dataset_card_path]
            ),
        },
        "summary": {
            "flagged_pair_count": len(all_flagged),
            "retained_flagged_pair_count": len(flagged),
            "truncated_pair_count": truncated_pair_count,
            "artifact_flag_rows_capped": truncated_pair_count > 0,
            "critical_pair_count": critical_count,
            "reviewed_pair_count": len(flagged) - len(unresolved),
            "unresolved_pair_count": len(unresolved) + truncated_pair_count,
            "review_completed": review_completed,
            "confirmed_contaminated_pair_count": contaminated_count,
            "ambiguous_pair_count": ambiguous_count,
            "adjudication_cleared_for_candidate": adjudication_cleared,
            "exact_text_absence_proven": False,
            "semantic_contamination_absence_proven": False,
        },
        "flagged_pairs": flagged,
        "unresolved_pair_ids": unresolved,
        "review_contract": {
            "review_path": str(review_path).replace("\\", "/"),
            "allowed_decisions": ["contaminated", "not_contaminated", "ambiguous"],
            "promotion_blocked_while_unresolved_or_ambiguous": bool(
                unresolved
                or truncated_pair_count
                or contaminated_count
                or ambiguous_count
            ),
            "remediation_for_contaminated_pair": (
                "Remove or re-split the contaminated source pair, regenerate "
                "the dataset hashes, then rerun this screen."
            ),
        },
        "limitations": [
            "TF-IDF can miss low-lexical-overlap paraphrases.",
            "High similarity can be legitimate shared safety language.",
            "The same project owner authored much of the training and evaluation material.",
            "A clean external no-read evaluation remains necessary.",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _write_json(output_path, payload)
    _write_doc(doc_path, payload)
    return payload


def _screen_channel(
    train: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
    *,
    review_threshold: float,
    critical_threshold: float,
) -> list[dict[str, Any]]:
    if not train or not evaluation:
        return []
    texts = [item["text"] for item in train + evaluation]
    word = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        strip_accents="unicode",
    ).fit_transform(texts)
    char = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        strip_accents="unicode",
    ).fit_transform(texts)
    split = len(train)
    word_scores = cosine_similarity(word[:split], word[split:])
    char_scores = cosine_similarity(char[:split], char[split:])
    flagged = []
    for train_index, train_item in enumerate(train):
        for eval_index, eval_item in enumerate(evaluation):
            word_score = float(word_scores[train_index, eval_index])
            char_score = float(char_scores[train_index, eval_index])
            maximum = max(word_score, char_score)
            if maximum < review_threshold:
                continue
            pair_id = _pair_id(train_item, eval_item)
            flagged.append(
                {
                    "pair_id": pair_id,
                    "channel": train_item["channel"],
                    "train_id": train_item["id"],
                    "train_source": train_item["source"],
                    "evaluation_id": eval_item["id"],
                    "evaluation_source": eval_item["source"],
                    "word_tfidf_cosine": round(word_score, 6),
                    "character_tfidf_cosine": round(char_score, 6),
                    "max_similarity": round(maximum, 6),
                    "severity": "critical" if maximum >= critical_threshold else "review",
                    "text_content_retained": False,
                }
            )
    return flagged


def _load_finetune_rows(
    path: str | Path,
    *,
    split_label: str,
) -> list[dict[str, Any]]:
    rows = []
    full = _full(path)
    if not full.exists():
        return rows
    for line_number, line in enumerate(full.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        identifier = str(item.get("id") or f"{split_label}:{line_number}")
        for message in item.get("messages") or []:
            role = str(message.get("role") or "")
            if role not in {"user", "assistant"}:
                continue
            text = str(message.get("content") or "").strip()
            if text:
                rows.append(
                    {
                        "id": identifier,
                        "channel": role,
                        "text": text,
                        "source": str(path).replace("\\", "/"),
                    }
                )
    return rows


def _load_eval_prompts(path: str | Path) -> list[dict[str, Any]]:
    full = _full(path)
    if not full.exists() or full.suffix.lower() != ".jsonl":
        return []
    rows = []
    for line_number, line in enumerate(full.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = _first_text(item, ("query", "prompt", "user_input", "message", "text"))
        if not text or "<PLACEHOLDER>" in text:
            continue
        rows.append(
            {
                "id": str(item.get("case_id") or item.get("id") or line_number),
                "channel": "user",
                "text": text,
                "source": str(path).replace("\\", "/"),
            }
        )
    return rows


def _first_text(item: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _pair_id(train: dict[str, Any], evaluation: dict[str, Any]) -> str:
    raw = "|".join(
        [
            train["source"],
            train["id"],
            train["channel"],
            evaluation["source"],
            evaluation["id"],
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _source_hashes(paths: list[str | Path]) -> list[dict[str, Any]]:
    output = []
    for path in paths:
        full = _full(path)
        output.append(
            {
                "path": str(path).replace("\\", "/"),
                "sha256": (
                    hashlib.sha256(full.read_bytes()).hexdigest()
                    if full.exists()
                    else None
                ),
            }
        )
    return output


def _read_json(path: str | Path) -> dict[str, Any]:
    full = _full(path)
    if not full.exists():
        return {}
    try:
        payload = json.loads(full.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    full = _full(path)
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_doc(path: str | Path, payload: dict[str, Any]) -> None:
    full = _full(path)
    full.parent.mkdir(parents=True, exist_ok=True)
    summary = payload["summary"]
    screen = payload["screen"]
    full.write_text(
        "\n".join(
            [
                "# Fine-Tune Semantic Contamination Screen",
                "",
                f"- Status: `{payload['status']}`",
                f"- Screening method: `{screen['method']}`",
                f"- Flagged pairs: `{summary['flagged_pair_count']}`",
                f"- Flag rows retained in JSON: `{summary['retained_flagged_pair_count']}`",
                f"- Flag rows omitted by artifact cap: `{summary['truncated_pair_count']}`",
                f"- Critical pairs: `{summary['critical_pair_count']}`",
                f"- Unresolved pairs: `{summary['unresolved_pair_count']}`",
                f"- Human review completed: `{summary['review_completed']}`",
                f"- Cleared for candidate comparison: `{summary['adjudication_cleared_for_candidate']}`",
                "",
                "Flagged rows retain IDs, source paths, and similarity scores only. "
                "The report deliberately omits prompt and answer text.",
                "",
                "## Interpretation",
                "",
                "This screen catches lexical and near-lexical overlap. It can miss "
                "low-overlap paraphrases and can over-flag shared safety language. "
                "Every flag must be adjudicated before an offline adapter can advance.",
                "",
                "## Boundary",
                "",
                payload["claim_boundary"],
                "",
            ]
        ),
        encoding="utf-8",
    )


def _full(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT_DIR / candidate


__all__ = ["build_finetune_semantic_contamination_audit"]
