"""Global configuration for CatalystDesk."""

import os
from pathlib import Path
from typing import Optional
import logging

# Try to load .env (for local development; optional for Streamlit Cloud)
try:
    from dotenv import load_dotenv
    _DOTENV_AVAILABLE = True
except ImportError:
    _DOTENV_AVAILABLE = False

# Try to import Streamlit (for Cloud deployment)
try:
    import streamlit as st
    _STREAMLIT_AVAILABLE = True
except ImportError:
    _STREAMLIT_AVAILABLE = False


# Load .env early (for local development, optional for Streamlit Cloud)
if _DOTENV_AVAILABLE:
    load_dotenv(override=False)

# If running in Streamlit and secrets are available, use them
if _STREAMLIT_AVAILABLE:
    try:
        if "HF_TOKEN" not in os.environ or not os.environ["HF_TOKEN"]:
            os.environ["HF_TOKEN"] = st.secrets.get("HF_TOKEN", "")
        if "FRED_API_KEY" not in os.environ or not os.environ["FRED_API_KEY"]:
            os.environ["FRED_API_KEY"] = st.secrets.get("FRED_API_KEY", "")
    except Exception:
        pass  # Secrets not available (might be in non-Streamlit context)


class Config:
    """Singleton config object. Access as Config.attribute_name."""

    # Paths
    ROOT = Path(__file__).parent.parent
    DATA_DIR = ROOT / "data"
    CACHE_DIR = Path(os.getenv("CACHE_DIR", "./cache"))
    LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
    CHROMA_DB_DIR = Path(os.getenv("CHROMA_DB_DIR", DATA_DIR / "chroma"))
    TRACE_DB = Path(os.getenv("TRACE_DB", DATA_DIR / "traces.db"))

    # HF
    HF_TOKEN = os.getenv("HF_TOKEN", "")
    HF_CACHE_DIR = Path(os.getenv("HF_CACHE_DIR", CACHE_DIR / "hf"))
    HF_HOME = Path(os.getenv("HF_HOME", CACHE_DIR / "hf"))

    # FRED (optional)
    FRED_API_KEY = os.getenv("FRED_API_KEY", "")

    # Cache
    CACHE_BACKEND = os.getenv("CACHE_BACKEND", "sqlite")  # sqlite or none
    CACHE_DB = CACHE_DIR / "cache.db"

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = LOG_DIR / "catalyst.log"

    # Agents
    LLM_TIMEOUT_SEC = int(os.getenv("LLM_TIMEOUT_SEC", "30"))
    RETRY_ATTEMPTS = int(os.getenv("RETRY_ATTEMPTS", "3"))
    RETRY_BACKOFF_SEC = int(os.getenv("RETRY_BACKOFF_SEC", "2"))

    @classmethod
    def ensure_dirs(cls) -> None:
        """Create required directories if they don't exist."""
        for path in [
            cls.DATA_DIR,
            cls.CACHE_DIR,
            cls.LOG_DIR,
            cls.CHROMA_DB_DIR,
            cls.HF_CACHE_DIR,
            cls.HF_HOME,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate(cls) -> list[str]:
        """Check config validity. Return list of warnings."""
        warnings = []

        if not cls.HF_TOKEN or cls.HF_TOKEN == "":
            warnings.append("HF_TOKEN not set; Inference API calls will fail.")

        if not cls.FRED_API_KEY:
            warnings.append("FRED_API_KEY not set; FRED ingest will be skipped.")

        return warnings


# Ensure directories exist on import
Config.ensure_dirs()


def get_logger(name: str) -> logging.Logger:
    """Get a logger with request ID support."""
    from .logging_utils import get_request_logger
    return get_request_logger(name)
