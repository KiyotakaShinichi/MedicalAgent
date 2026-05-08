from pathlib import Path
import os

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_NESTED_ROOT = PROJECT_ROOT / "MedicalAgent"
DATA_DIR = PROJECT_ROOT / "Data"
UPLOAD_DIR = DATA_DIR / "uploads"


def load_environment():
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(LEGACY_NESTED_ROOT / ".env")


def get_groq_api_key():
    load_environment()
    return os.environ.get("GROQ_API_KEY")


def get_groq_model():
    """Primary answer-generation model used for chat and clinical summaries."""
    load_environment()
    return (
        os.environ.get("GROQ_ANSWER_MODEL")
        or os.environ.get("GROQ_MODEL")
        or os.environ.get("GROQ_CHAT_MODEL")
        or "openai/gpt-oss-120b"
    )


def get_groq_router_model():
    """Fast/cheap model used for JSON intent routing, safety adjudication, and tool selection."""
    load_environment()
    return (
        os.environ.get("GROQ_ROUTER_MODEL")
        or os.environ.get("GROQ_ADJUDICATION_MODEL")
        or "llama-3.3-70b-versatile"
    )


def get_groq_config():
    load_environment()
    return {
        "api_key": os.environ.get("GROQ_API_KEY"),
        "model": get_groq_router_model(),
        "answer_model": get_groq_model(),
        "router_model": get_groq_router_model(),
        "timeout_seconds": float(os.environ.get("GROQ_ADJUDICATION_TIMEOUT_SECONDS", "3")),
    }


def get_llm_adjudication_enabled():
    load_environment()
    value = os.environ.get("LLM_ADJUDICATION_ENABLED", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def get_ollama_config():
    load_environment()
    return {
        "base_url": os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        "model": os.environ.get("OLLAMA_MODEL") or os.environ.get("LOCAL_LLM_MODEL"),
        "timeout_seconds": float(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "3")),
    }


def ensure_runtime_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
