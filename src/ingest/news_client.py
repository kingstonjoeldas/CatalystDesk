"""
Ingest headlines from public RSS feeds and news APIs.

Why RSS?
- Decentralized; no single API dependency
- Reuters, Yahoo Finance, financial blogs publish RSS feeds
- Simple to parse; minimal auth

Why not GDELT or NewsAPI?
- GDELT requires complex config (GKG format, raw data dumps)
- NewsAPI is free-tier limited (100 req/day, 1-month history)
- RSS feeds are immediate, sufficient for MVP

This is a simple implementation; expand with more feeds as needed.
"""

import time
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import urlparse

import feedparser

from src.config import Config
from src.logging_utils import get_request_logger

logger = get_request_logger(__name__)


class NewsClient:
    """Fetch headlines from RSS feeds with retries."""

    MAX_RETRIES = 3
    RETRY_BACKOFF_SEC = 2

    # Default RSS feeds (mostly financial)
    DEFAULT_FEEDS = [
        "https://feeds.finance.yahoo.com/rss/2.0/headline",  # Yahoo Finance top headlines
        "https://feeds.bloomberg.com/markets/news.rss",  # Bloomberg Markets
        "https://feeds.cnbc.com/cnbc/world/",  # CNBC World
    ]

    @classmethod
    def fetch_feed(cls, feed_url: str, days_back: int = 7) -> list[dict]:
        """
        Fetch articles from a single RSS feed.

        Args:
            feed_url: URL to RSS feed
            days_back: Only include articles from this many days ago

        Returns:
            List of dicts with NewsItem-compatible fields
        """
        articles = []
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)

        logger.info(f"Fetching RSS: {feed_url}...")

        for attempt in range(cls.MAX_RETRIES):
            try:
                feed = feedparser.parse(feed_url)

                if feed.status != 200:
                    logger.warning(f"HTTP {feed.status} for {feed_url}")
                    if attempt < cls.MAX_RETRIES - 1:
                        time.sleep(cls.RETRY_BACKOFF_SEC)
                    continue

                for i, entry in enumerate(feed.entries[:20]):  # Limit to 20 per feed
                    try:
                        # Parse published date
                        pub_time = None
                        if hasattr(entry, "published_parsed") and entry.published_parsed:
                            pub_time = datetime(*entry.published_parsed[:6])
                        else:
                            pub_time = datetime.utcnow()

                        # Skip old articles
                        if pub_time < cutoff_date:
                            continue

                        # Extract tickers from title/description (simple heuristic)
                        tickers = cls._extract_tickers(
                            entry.get("title", "") + " " + entry.get("summary", "")
                        )

                        article = {
                            "id": f"rss_{urlparse(feed_url).netloc}_{pub_time.strftime('%Y%m%d')}_{i:03d}",
                            "source": "rss",
                            "published_at": pub_time.isoformat() + "Z",
                            "title": entry.get("title", "")[:256],
                            "raw_text": entry.get("summary", entry.get("title", ""))[:2000],
                            "url": entry.get("link", ""),
                            "tickers": tickers or ["SPY"],  # Default to market index
                            "summary": None,
                            "event_type": None,
                            "risk_level": None,
                            "risk_score": None,
                            "contradiction": None,
                        }
                        articles.append(article)

                    except (KeyError, ValueError) as e:
                        logger.warning(f"Failed to parse RSS entry: {e}")
                        continue

                logger.info(f"✓ Fetched {len(articles)} articles from {feed_url}")
                return articles

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{cls.MAX_RETRIES} failed: {e}")
                if attempt < cls.MAX_RETRIES - 1:
                    time.sleep(cls.RETRY_BACKOFF_SEC)

        logger.error(f"Failed to fetch {feed_url} after {cls.MAX_RETRIES} attempts")
        return articles

    @classmethod
    def fetch_all_feeds(cls, feeds: Optional[list[str]] = None) -> list[dict]:
        """
        Fetch all configured RSS feeds.

        Args:
            feeds: List of feed URLs. If None, uses DEFAULT_FEEDS.

        Returns:
            Combined list of articles from all feeds
        """
        feeds = feeds or cls.DEFAULT_FEEDS
        all_articles = []

        for feed_url in feeds:
            articles = cls.fetch_feed(feed_url, days_back=7)
            all_articles.extend(articles)

        logger.info(f"✓ Fetched {len(all_articles)} total articles from {len(feeds)} feeds")
        return all_articles

    @staticmethod
    def _extract_tickers(text: str) -> list[str]:
        """
        Heuristic extraction of stock tickers from text.

        Looks for uppercase 1–5 letter sequences surrounded by non-alphanumeric chars.

        Args:
            text: Text to search

        Returns:
            List of potential tickers (lowercase)

        Note:
            This is a simple heuristic; production would use NER or a ticker database.
        """
        import re

        # Match standalone uppercase words of 1-5 letters
        pattern = r"\b([A-Z]{1,5})\b"
        candidates = re.findall(pattern, text)

        # Filter out common words
        common_words = {
            "THE", "AND", "OR", "BUT", "FOR", "WITH", "FROM", "TO", "IN", "ON", "AT",
            "BY", "UP", "OUT", "IF", "AS", "IS", "BE", "WAS", "WERE", "HAVE", "HAS",
            "DO", "DOES", "DID", "WILL", "WOULD", "COULD", "SHOULD", "MAY", "MIGHT",
            "CAN", "MUST", "SAID", "SAY", "SAYS", "A", "US", "BANK", "CORP", "INC",
        }

        tickers = [t for t in candidates if t not in common_words]
        return list(set(tickers[:5]))  # Dedupe and limit to 5
