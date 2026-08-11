from backend.services.rag_corpus_poisoning_eval import build_corpus_poisoning_eval
from backend.services.tenant_isolation_security_eval import build_tenant_isolation_security_eval
from backend.services.tenant_scoping import tenant_cache_key, tenant_vector_namespace


def test_tenant_keys_are_scoped_and_reject_ambiguous_parts():
    assert tenant_cache_key("org-a", "project-a", "rag") != tenant_cache_key("org-b", "project-a", "rag")
    assert tenant_vector_namespace("org-a", "project-a") != tenant_vector_namespace("org-a", "project-b")
    try:
        tenant_cache_key("org/a", "project-a", "rag")
    except ValueError:
        pass
    else:
        raise AssertionError("Ambiguous tenant key components must be rejected")


def test_tenant_isolation_matrix_has_zero_leakage(tmp_path):
    report = build_tenant_isolation_security_eval(tmp_path / "tenant.json")
    assert report["authorization_attack_count"] >= 120
    assert report["failed_count"] == 0
    assert report["cross_tenant_leakage_count"] == 0
    assert report["clinical_validation"] is False


def test_corpus_poisoning_matrix_blocks_generation_context(tmp_path):
    report = build_corpus_poisoning_eval(tmp_path / "poison.json")
    assert report["case_count"] >= 10
    assert report["candidate_exposure_rate"] == 1.0
    assert report["generation_context_poison_rate"] == 0.0
    assert report["unsafe_candidate_leakage_rate"] == 0.0
    assert report["failed_count"] == 0
