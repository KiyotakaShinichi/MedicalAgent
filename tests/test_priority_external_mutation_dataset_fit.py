from __future__ import annotations

import csv

from backend.services.dataset_fit_matrix import build_dataset_fit_matrix
from backend.services.mutation_context_mapping import build_mutation_context_mapping
from backend.services.priority_dataset_bridge import build_priority_dataset_bridge
from backend.services.priority_external_stress import build_priority_external_stress


def test_priority_external_stress_blocks_promotion_without_exact_temporal_labels(tmp_path):
    bridge = build_priority_dataset_bridge(
        output_path=str(tmp_path / "priority_bridge.json"),
        doc_path=str(tmp_path / "priority_bridge.md"),
        template_dir=str(tmp_path / "templates"),
        genie_canonical_csv=str(tmp_path / "canonical_genie.csv"),
        duke_canonical_csv=str(tmp_path / "canonical_duke.csv"),
        schema_output_path=str(tmp_path / "schema.json"),
    )
    assert bridge["status"] == "ready_for_mapping"

    report = build_priority_external_stress(
        bridge_path=str(tmp_path / "priority_bridge.json"),
        output_path=str(tmp_path / "stress.json"),
    )

    assert report["status"] == "ready_when_mapped"
    assert report["promotion_decision"]["promotion_allowed"] is False
    assert report["endpoint_compatibility"]["exact_oncotrack_label_match"] is False


def test_mutation_context_mapping_keeps_genes_context_only(tmp_path):
    mutation_csv = tmp_path / "mutations.csv"
    _write_csv(mutation_csv, [
        {
            "source_dataset": "tcga_brca",
            "patient_id": "P001",
            "gene": "BRCA1",
            "variant_classification": "pathogenic",
            "source_type": "germline",
        },
        {
            "source_dataset": "genie_bpc_brca",
            "patient_id": "P002",
            "gene": "PIK3CA",
            "variant_classification": "missense",
            "source_type": "tumor",
        },
    ])

    report = build_mutation_context_mapping(
        mutation_csv=str(mutation_csv),
        output_path=str(tmp_path / "mutation_context.json"),
    )

    assert report["status"] == "strong"
    assert report["mapped_row_count"] == 2
    assert report["feature_policy"]["promotion_allowed"] is False
    assert "BRCA1" in report["gene_counts"]
    assert any("VUS means positive" in claim for claim in report["blocked_claims"])


def test_dataset_fit_matrix_prioritizes_genie_and_duke_without_training_claims(tmp_path):
    report = build_dataset_fit_matrix(
        output_path=str(tmp_path / "fit.json"),
        doc_path=str(tmp_path / "fit.md"),
    )

    assert report["status"] == "strong"
    assert report["dataset_count"] >= 8
    assert "genie_bpc_brca" in report["top_5"]
    assert "duke_breast_mri" in report["top_5"]
    assert report["recommendation"]["production_training_allowed"] is False
    assert "clinically validated" in report["claim_boundary"]


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
