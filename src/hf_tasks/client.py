"""
Shared Hugging Face Inference API client with caching + retries.

Why a wrapper?
- Inference API has cold starts (3-10s on first request to a model)
- We cache every call by hash(model_id + input_text) to disk
- One retry/timeout wrapper for all HF tasks (L2, L3, embeddings)
- Logging + request ID tracking

Why cache?
- HF free tier is rate-limited but not quota-limited (you can call forever, just throttled)
- First call to a model spins up the container; subsequent calls are cached by HF (30s TTL)
- We cache locally: hash(input) -> output in SQLite; reruns are instant
- Cost: disk space (~1MB per 1000 summaries); benefit: 100x speedup on reruns

Why not Ollama or local LLM?
- Ollama needs 2GB+ VRAM on a low-end PC; HF API is free and serverless
- Trade-off: network latency + HF cold starts, mitigated by caching
"""

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Optional

from huggingface_hub import InferenceClient

from src.config import Config
from src.logging_utils import get_request_logger

logger = get_request_logger(__name__)


class HFCache:
    """Persistent cache for HF API responses (SQLite backend)."""

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize cache.

        Args:
            db_path: Path to SQLite DB. If None, uses Config.CACHE_DIR / cache.db
        """
        self.db_path = db_path or Config.CACHE_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Create cache table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS hf_cache (
                    key TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    output TEXT NOT NULL,
                    timestamp INTEGER NOT NULL
                )
                """
            )
            conn.commit()

    @staticmethod
    def _make_key(model_id: str, input_text: str) -> str:
        """
        Generate cache key from model ID and input text.

        Args:
            model_id: HF model ID, e.g., "facebook/bart-large-cnn"
            input_text: Input to the model

        Returns:
            Unique key, e.g., "facebook_bart_large_cnn__sha256_abc123..."
        """
        # Sanitize model ID for use as key prefix
        model_key = model_id.replace("/", "_").replace("-", "_")
        # Hash input to keep keys short
        input_hash = hashlib.sha256(input_text.encode()).hexdigest()[:16]
        return f"{model_key}__{input_hash}"

    def get(self, model_id: str, input_text: str) -> Optional[str]:
        """
        Retrieve cached output.

        Args:
            model_id: HF model ID
            input_text: Input text

        Returns:
            Cached output (JSON string), or None if not found
        """
        key = self._make_key(model_id, input_text)

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT output FROM hf_cache WHERE key = ?", (key,))
                row = cursor.fetchone()
                if row:
                    logger.debug(f"Cache hit: {key[:30]}...")
                    return row[0]
        except Exception as e:
            logger.warning(f"Cache read error: {e}")

        return None

    def set(self, model_id: str, input_text: str, output: str) -> None:
        """
        Store output in cache.

        Args:
            model_id: HF model ID
            input_text: Input text
            output: Output (JSON string)
        """
        key = self._make_key(model_id, input_text)
        timestamp = int(time.time())

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO hf_cache (key, model_id, input_hash, output, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (key, model_id, key.split("__")[1], output, timestamp),
                )
                conn.commit()
                logger.debug(f"Cache set: {key[:30]}...")
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    def clear(self) -> None:
        """Clear all cached entries."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM hf_cache")
                conn.commit()
                logger.info("Cache cleared")
        except Exception as e:
            logger.warning(f"Cache clear error: {e}")


class HFClient:
    """Hugging Face Inference API client with retries and caching."""

    MAX_RETRIES = Config.RETRY_ATTEMPTS
    RETRY_BACKOFF_SEC = Config.RETRY_BACKOFF_SEC
    TIMEOUT_SEC = Config.LLM_TIMEOUT_SEC

    def __init__(self, model_id: str, cache: Optional[HFCache] = None):
        """
        Initialize client for a specific model.

        Args:
            model_id: HF model ID, e.g., "facebook/bart-large-cnn"
            cache: HFCache instance. If None, creates a new one.
        """
        if not Config.HF_TOKEN:
            logger.warning("HF_TOKEN not set; API calls will likely fail")

        self.model_id = model_id
        self.client = InferenceClient(
            model=model_id,
            token=Config.HF_TOKEN,
            timeout=self.TIMEOUT_SEC,
        )
        self.cache = cache or HFCache()

    def __call__(self, input_text: str, **kwargs) -> Optional[str]:
        """
        Call the model (summarization, classification, etc.).

        Uses task-specific __call__ (defined in subclasses like Summarizer).
        This method just implements retry + cache logic.

        Args:
            input_text: Input to the model
            **kwargs: Task-specific kwargs (e.g., max_length for summarization)

        Returns:
            Model output (task-dependent), or None if all retries fail
        """
        # Try cache first
        cached = self.cache.get(self.model_id, input_text)
        if cached:
            return cached

        # Retry loop
        for attempt in range(self.MAX_RETRIES):
            try:
                logger.info(
                    f"Calling {self.model_id} (attempt {attempt + 1}/{self.MAX_RETRIES})..."
                )
                output = self._call_model(input_text, **kwargs)

                if output:
                    # Cache the result
                    self.cache.set(self.model_id, input_text, output)
                    logger.info(f"✓ {self.model_id} success")
                    return output

            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1}/{self.MAX_RETRIES} failed: {e}"
                )
                if attempt < self.MAX_RETRIES - 1:
                    backoff = self.RETRY_BACKOFF_SEC * (2 ** attempt)  # Exponential
                    logger.info(f"Retrying in {backoff}s...")
                    time.sleep(backoff)

        logger.error(f"All {self.MAX_RETRIES} attempts failed for {self.model_id}")
        return None

    def _call_model(self, input_text: str, **kwargs) -> Optional[str]:
        """
        Subclasses override this to implement task-specific logic.

        Args:
            input_text: Input text
            **kwargs: Task-specific kwargs

        Returns:
            Model output (JSON string or plain text)
        """
        raise NotImplementedError("Subclasses must implement _call_model")
