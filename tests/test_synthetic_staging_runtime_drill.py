from backend.services.synthetic_staging_runtime_drill import (
    CLAIM_BOUNDARY,
    _normalized_dump_sha256,
)


def test_logical_dump_normalization_ignores_headers_and_restrict_tokens() -> None:
    left = b"-- dump A\n\\restrict abc\nINSERT INTO x VALUES (1);\n\\unrestrict abc\n"
    right = b"-- dump B\n\\restrict xyz\nINSERT INTO x VALUES (1);\n\\unrestrict xyz\n"
    assert _normalized_dump_sha256(left) == _normalized_dump_sha256(right)


def test_logical_dump_normalization_detects_content_change() -> None:
    left = b"INSERT INTO x VALUES (1);\n"
    right = b"INSERT INTO x VALUES (2);\n"
    assert _normalized_dump_sha256(left) != _normalized_dump_sha256(right)


def test_runtime_drill_claim_boundary_is_nonclinical() -> None:
    lowered = CLAIM_BOUNDARY.lower()
    assert "does not prove" in lowered
    assert "clinical validation" in lowered
    assert "production healthcare readiness" in lowered
