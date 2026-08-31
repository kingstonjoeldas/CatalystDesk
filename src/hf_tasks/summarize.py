"""
Hugging Face summarization task using BART.

Why BART?
- facebook/bart-large-cnn is trained specifically for news summarization (CNN/DailyMail dataset)
- 140M params; inference is fast (under 1s when warm)
- Abstractive (generates new text, not just extractive); captures key points better than keyword extraction

Why not just prompt a general LLM?
- "Summarize this in 1 sentence" wastes tokens on instructions
- BART is designed for this task; it's more reliable and cheaper (fewer tokens, smaller model)
- Zero hallucination compared to LLMs (it's sequence-to-sequence, not language generation)

Why cache summaries?
- First call to BART on HF is slow (3-10s cold start, then 1-2s warm)
- Second call with same article is instant (HF's 30s model cache, plus our persistent SQLite cache)
- Reruns of the demo are free and fast

Tradeoff: Model output can be lossy (may miss nuance). But for trading briefs, concise 1-2 sentence summaries are exactly what we want.
"""

from typing import Optional

from src.hf_tasks.client import HFClient, HFCache
from src.logging_utils import get_request_logger

logger = get_request_logger(__name__)


class Summarizer(HFClient):
    """BART-based summarization client."""

    # Start with the larger, well-trained model; fall back to distilbart if slow
    DEFAULT_MODEL = "facebook/bart-large-cnn"
    FALLBACK_MODEL = "sshleifer/distilbart-cnn-12-6"  # Smaller, ~30% faster

    def __init__(self, model_id: Optional[str] = None, cache: Optional[HFCache] = None):
        """
        Initialize summarizer.

        Args:
            model_id: HF model ID. If None, uses DEFAULT_MODEL.
            cache: Shared HFCache instance (optional)
        """
        model_id = model_id or self.DEFAULT_MODEL
        super().__init__(model_id, cache)

    def _call_model(self, input_text: str, **kwargs) -> Optional[str]:
        """
        Call BART summarization API.

        Args:
            input_text: Article text to summarize
            **kwargs: Optional params (min_length, max_length, etc.)
                Default: min_length=50, max_length=150

        Returns:
            JSON string: {"summary_text": "..."}
            Or None if API call fails
        """
        # Set defaults
        min_length = kwargs.get("min_length", 50)
        max_length = kwargs.get("max_length", 150)

        # Truncate very long articles to avoid token limits (BART max input ~1024)
        truncated = input_text[:1000]

        try:
            result = self.client.summarization(
                truncated,
                min_length=min_length,
                max_length=max_length,
            )

            # HF returns list of dicts; we take the first
            if isinstance(result, list) and result:
                summary = result[0].get("summary_text", "")
            else:
                summary = str(result)

            # Return as JSON string (for consistent caching format)
            return f'{{"summary_text": "{summary}"}}'

        except Exception as e:
            logger.error(f"Summarization API call failed: {e}")
            return None

    @classmethod
    def summarize(cls, text: str, model_id: Optional[str] = None) -> Optional[str]:
        """
        Convenience method: summarize text using default or specified model.

        Args:
            text: Article text
            model_id: Optional model override

        Returns:
            Summary text, or None if API fails
        """
        client = cls(model_id=model_id)
        result = client(text)

        if result:
            import json
            try:
                parsed = json.loads(result)
                return parsed.get("summary_text", "")
            except json.JSONDecodeError:
                return result

        return None

    @classmethod
    def summarize_batch(
        cls, articles: list[dict], model_id: Optional[str] = None, skip_existing: bool = True
    ) -> list[dict]:
        """
        Summarize multiple articles in batch.

        Args:
            articles: List of dicts with 'raw_text' field (from L1 ingest)
            model_id: Optional model override
            skip_existing: If True, skip articles that already have 'summary' field

        Returns:
            Same articles list, with 'summary' field filled in
        """
        client = cls(model_id=model_id)

        for i, article in enumerate(articles):
            # Skip if already summarized
            if skip_existing and article.get("summary"):
                logger.debug(f"[{i}/{len(articles)}] Skipping (already summarized)")
                continue

            raw_text = article.get("raw_text", "")
            if not raw_text or len(raw_text.strip()) < 20:
                logger.warning(f"[{i}/{len(articles)}] Skipping (text too short)")
                article["summary"] = "[No content to summarize]"
                continue

            logger.info(f"[{i}/{len(articles)}] Summarizing: {article.get('title', '')[:50]}...")

            result = client(raw_text)
            if result:
                import json
                try:
                    parsed = json.loads(result)
                    article["summary"] = parsed.get("summary_text", "")
                except json.JSONDecodeError:
                    article["summary"] = result
            else:
                article["summary"] = "[Summarization failed; using fallback]"
                logger.warning(f"Failed to summarize: {article.get('title', '')}")

        logger.info(f"✓ Summarized {len(articles)} articles")
        return articles
