from __future__ import annotations

import pandas as pd
import pytest

from backend.services.duke_tcia_tabular_bridge import (
    CLINICAL_COLUMN_MAP,
    MRI_FEATURE_COUNT,
    TARGET_SOURCE,
    canonicalize_frames,
)


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    clinical_rows = []
    mri_rows = []
    for index in range(40):
        row = {"Patient ID": f"Breast_MRI_{index:03d}", TARGET_SOURCE: 1 if index % 5 == 0 else 2}
        row.update({column: float((index + offset) % 7) for offset, column in enumerate(CLINICAL_COLUMN_MAP)})
        clinical_rows.append(row)
        mri_row = {"Patient ID": row["Patient ID"]}
        mri_row.update({f"radiomic_{feature:03d}": float(index + feature) for feature in range(MRI_FEATURE_COUNT + 3)})
        mri_rows.append(mri_row)
    return pd.DataFrame(clinical_rows), pd.DataFrame(mri_rows)


def test_canonical_bridge_hashes_identifiers_and_excludes_leakage_columns():
    canonical, manifest = canonicalize_frames(*_frames())
    assert len(canonical) == 40
    assert "Patient ID" not in canonical.columns
    assert canonical["external_case_key"].str.len().eq(16).all()
    assert manifest["raw_subject_identifier_exported"] is False
    assert manifest["treatment_columns_exported_or_used_as_features"] is False
    assert manifest["recurrence_or_survival_columns_used_as_features"] is False
    assert len([column for column in canonical if column.startswith("mri_feature_")]) == MRI_FEATURE_COUNT
    assert not any("treatment" in column.lower() or "survival" in column.lower() for column in canonical.columns)


def test_canonical_bridge_rejects_duplicate_subjects():
    clinical, mri = _frames()
    mri.loc[1, "Patient ID"] = mri.loc[0, "Patient ID"]
    with pytest.raises(ValueError, match="one-to-one"):
        canonicalize_frames(clinical, mri)


def test_canonical_bridge_uses_only_labeled_response_rows():
    clinical, mri = _frames()
    clinical.loc[:4, TARGET_SOURCE] = None
    canonical, manifest = canonicalize_frames(clinical, mri)
    assert len(canonical) == 35
    assert manifest["joined_labeled_row_count"] == 35
    assert set(canonical["external_target_pathologic_complete_response"].unique()) <= {0, 1}
