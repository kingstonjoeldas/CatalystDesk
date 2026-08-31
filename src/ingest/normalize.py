"""
Normalize articles from all sources to NewsItem schema.

Why a separate normalization layer?
- Each source has different fields, timestamps, metadata
- Normalizing early prevents downstream code from handling multiple formats
- Easier to add new sources without touching the rest of the pipeline
- Schema validation happens in one place

NewsItem schema (TypedDict):
- id: unique identifier (source + timestamp + title hash)
- source: "yfinance" | "fred" | "rss" | "gdelt"
- published_at: ISO-8601 UTC
- title: headline
- raw_text: article body
- url: permalink
- tickers: list of stock symbols
- summary, event_type, risk_level, risk_score, contradiction: filled by L2–L6
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import TypedDict, Optional

from src.config import Config
from src.logging_utils import get_request_logger

logger = get_request_logger(__name__)


class NewsItem(TypedDict, total=False):
    """Normalized article schema."""

    id: str
    source: str
    published_at: str
    title: str
    raw_text: str
    url: str
    tickers: list[str]
    summary: Optional[str]
    event_type: Optional[str]
    risk_level: Optional[str]
    risk_score: Optional[float]
    contradiction: Optional[bool]


class Normalizer:
    """Normalize articles from all sources to NewsItem schema."""

    @staticmethod
    def validate(item: dict) -> bool:
        """
        Check that a NewsItem has all required fields.

        Args:
            item: Dictionary to validate

        Returns:
            True if valid, False otherwise (logs warnings)
        """
        required = {"id", "source", "published_at", "title", "raw_text", "url", "tickers"}
        missing = required - set(item.keys())

        if missing:
            logger.warning(f"NewsItem missing fields: {missing}")
            return False

        # Type checks
        if not isinstance(item["tickers"], list):
            logger.warning(f"tickers not a list: {item['tickers']}")
            return False

        # Date format check (ISO-8601)
        try:
            datetime.fromisoformat(item["published_at"].rstrip("Z"))
        except ValueError:
            logger.warning(f"Invalid published_at format: {item['published_at']}")
            return False

        return True

    @staticmethod
    def ensure_unique_id(item: dict, existing_ids: set) -> dict:
        """
        Ensure item ID is unique; if not, append a suffix.

        Args:
            item: NewsItem dict
            existing_ids: Set of IDs already seen

        Returns:
            Modified item with unique ID
        """
        base_id = item["id"]
        unique_id = base_id
        counter = 1

        while unique_id in existing_ids:
            unique_id = f"{base_id}_{counter}"
            counter += 1

        item["id"] = unique_id
        return item

    @classmethod
    def normalize_batch(cls, articles: list[dict]) -> list[NewsItem]:
        """
        Normalize a batch of articles from any source.

        Args:
            articles: List of dicts (from yfinance, FRED, RSS, etc.)

        Returns:
            List of validated NewsItem dicts
        """
        normalized = []
        existing_ids = set()

        for article in articles:
            # Ensure all required fields exist
            if not cls.validate(article):
                logger.warning(f"Skipping invalid article: {article.get('title', 'no title')}")
                continue

            # Ensure unique ID
            article = cls.ensure_unique_id(article, existing_ids)
            existing_ids.add(article["id"])

            normalized.append(article)

        logger.info(f"Normalized {len(normalized)}/{len(articles)} articles")
        return normalized

    @classmethod
    def save_batch(cls, articles: list[NewsItem], output_path: Optional[Path] = None) -> Path:
        """
        Save normalized articles to JSON file.

        Args:
            articles: List of NewsItem dicts
            output_path: Where to save. If None, uses data/articles_TIMESTAMP.json

        Returns:
            Path to saved file
        """
        output_path = output_path or (
            Config.DATA_DIR / f"articles_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(articles, f, indent=2)

        logger.info(f"Saved {len(articles)} articles to {output_path}")
        return output_path

    @classmethod
    def load_fixtures(cls) -> list[NewsItem]:
        """
        Load sample articles from data/samples/sample_articles.json.

        Used as fallback if all live APIs fail.

        Returns:
            List of NewsItem dicts
        """
        fixture_path = Config.DATA_DIR / "samples" / "sample_articles.json"

        if not fixture_path.exists():
            logger.error(f"Fixture file not found: {fixture_path}")
            return []

        try:
            with open(fixture_path, "r") as f:
                articles = json.load(f)
            logger.info(f"Loaded {len(articles)} fixtures from {fixture_path}")
            return articles
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse fixtures: {e}")
            return []
