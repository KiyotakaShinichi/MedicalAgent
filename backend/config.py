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


def ensure_runtime_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
