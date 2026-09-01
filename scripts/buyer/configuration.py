"""Build the buyer/operator configuration matrix from the canonical env example."""

from __future__ import annotations

from typing import Any

from scripts.buyer.contracts import parse_env_example


SECRET_VARIABLES = {
    "AZURE_SEARCH_API_KEY",
    "AZURE_SEARCH_BEARER_TOKEN",
    "GROQ_API_KEY",
    "N8N_WEBHOOK_SIGNING_SECRET",
    "PINECONE_API_KEY",
}
REQUIRED_PRODUCTION = {"DATABASE_URL", "NLCARE_CORS_ORIGINS"}
PROVIDER_PREFIXES = {
    "AZURE_": "Azure AI Search",
    "GROQ_": "Groq",
    "N8N_": "n8n",
    "OLLAMA_": "Ollama",
    "PINECONE_": "Pinecone",
}


def category_for(name: str) -> str:
    if name in {"APP_ENV", "ENVIRONMENT", "API_BASE_URL", "SMOKE_PATIENT_ID"} or name.startswith("NLCARE_CORS"):
        return "APP"
    if name.startswith("NLCARE_OIDC") or name == "ALLOW_DEMO_AUTH":
        return "AUTH"
    if name in {"DATABASE_URL", "SQLITE_TIMEOUT_SECONDS", "REDIS_URL"}:
        return "DATABASE"
    if name.startswith(("RAG_", "ONCOTRACK_RAG", "NLCARE_RAG")):
        return "RAG"
    if name.startswith(("GROQ_", "OLLAMA_", "LOCAL_LLM", "LLM_", "CLINICAL_SUMMARY", "NLCARE_LLM")):
        return "LLM"
    if name.startswith(("AZURE_SEARCH", "PINECONE", "NLCARE_VECTOR")):
        return "VECTOR"
    if name.startswith(("MLFLOW", "NLCARE_FINETUNE")):
        return "ML"
    if name.startswith(("NLCARE_LOG", "NLCARE_ROOT_LOG")):
        return "OBSERVABILITY"
    if name.startswith(("N8N_", "NLCARE_ALERT", "NLCARE_AUTOMATION", "NLCARE_AGENTIC")):
        return "AUTOMATION"
    if name.startswith(("NLCARE_DEP001", "NLCARE_BURNED", "ONCOTRACK_")):
        return "INFRA"
    if name.startswith(("NLCARE_SYNTHETIC", "NLCARE_BOOTSTRAP", "NLCARE_PATIENT", "SMOKE_")):
        return "DEMO"
    return "INFRA"


def provider_for(name: str) -> str | None:
    for prefix, provider in PROVIDER_PREFIXES.items():
        if name.startswith(prefix):
            return provider
    return None


def build_matrix() -> dict[str, Any]:
    variables = []
    for name, default in sorted(parse_env_example().items()):
        provider = provider_for(name)
        secret = name in SECRET_VARIABLES
        category = category_for(name)
        variables.append(
            {
                "variable": name,
                "category": category,
                "required": name in REQUIRED_PRODUCTION,
                "secret": secret,
                "default": "<empty>" if secret else default,
                "offline_behavior": "unused; core verification remains local" if provider else "uses documented local/default behavior",
                "demo_behavior": "disabled or local by default" if provider else "synthetic demo setting",
                "production_significance": "buyer must review and set explicitly" if name in REQUIRED_PRODUCTION or secret else "review before hosted deployment",
                "external_provider": provider,
                "notes": "Never transfer a real value." if secret else "Canonical definition is .env.example.",
            }
        )
    return {
        "schema_version": "nlcare_configuration_matrix_v1",
        "source": ".env.example",
        "generated": True,
        "variables": variables,
    }
