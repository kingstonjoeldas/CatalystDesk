#!/usr/bin/env python
"""
Phase L1: Ingest & Normalize

Orchestrates data collection from yfinance, FRED, and RSS feeds.
Normalizes everything to NewsItem schema.
Prints a table of collected articles.

Exit code:
  0 = Success
  1 = Partial failure (some APIs failed, but fixtures used)
  2 = Complete failure (no data at all)
"""

import sys
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.logging_utils import get_request_logger, set_request_id
from src.ingest.yfinance_client import YFinanceClient
from src.ingest.fred_client import FredClient
from src.ingest.news_client import NewsClient
from src.ingest.normalize import Normalizer, NewsItem

logger = get_request_logger(__name__)


def print_articles_table(articles: list[NewsItem]) -> None:
    """Print articles as a formatted table."""
    if not articles:
        logger.info("No articles to display")
        return

    logger.info("\n" + "=" * 120)
    logger.info(
        f"{'Date':<12} | {'Source':<10} | {'Tickers':<15} | {'Title':<70}"
    )
    logger.info("-" * 120)

    for article in articles[:20]:  # Show first 20
        date = article["published_at"][:10]
        source = article["source"][:10]
        tickers = ",".join(article.get("tickers", [])[:2])[:15]
        title = article["title"][:70]

        logger.info(f"{date} | {source:<10} | {tickers:<15} | {title}")

    if len(articles) > 20:
        logger.info(f"\n... and {len(articles) - 20} more articles")

    logger.info("=" * 120 + "\n")


def run_ingest() -> tuple[list[NewsItem], int]:
    """
    Main ingest pipeline.

    Returns:
        (articles, exit_code) where exit_code is 0 (success), 1 (partial), or 2 (failure)
    """
    set_request_id("l1_ingest")

    logger.info("=" * 80)
    logger.info("CatalystDesk Phase L1: Ingest")
    logger.info("=" * 80)

    all_articles = []
    failed_sources = []
    successful_sources = []

    # === YFINANCE ===
    logger.info("\n[1/3] yfinance: News + OHLCV")
    try:
        tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "SPY", "TLT", "GLD"]
        yf_articles = YFinanceClient.fetch_news(tickers, days_back=7)
        if yf_articles:
            all_articles.extend(yf_articles)
            successful_sources.append("yfinance")
            logger.info(f"✓ yfinance: {len(yf_articles)} articles")
        else:
            logger.warning("✗ yfinance returned no articles")
            failed_sources.append("yfinance")
    except Exception as e:
        logger.error(f"✗ yfinance failed: {e}")
        failed_sources.append("yfinance")

    # === FRED ===
    logger.info("\n[2/3] FRED: Macro indicators")
    if FredClient.is_available():
        try:
            fred_indicators = FredClient.fetch_all_indicators()
            fred_articles = FredClient.articles_from_indicators(fred_indicators)
            if fred_articles:
                all_articles.extend(fred_articles)
                successful_sources.append("fred")
                logger.info(f"✓ FRED: {len(fred_articles)} indicator articles")
            else:
                logger.warning("✗ FRED returned no indicators")
                failed_sources.append("fred")
        except Exception as e:
            logger.error(f"✗ FRED failed: {e}")
            failed_sources.append("fred")
    else:
        logger.info("⚠ FRED skipped (no API key)")
        failed_sources.append("fred")

    # === RSS FEEDS ===
    logger.info("\n[3/3] RSS feeds: General headlines")
    try:
        rss_articles = NewsClient.fetch_all_feeds()
        if rss_articles:
            all_articles.extend(rss_articles)
            successful_sources.append("rss")
            logger.info(f"✓ RSS: {len(rss_articles)} articles")
        else:
            logger.warning("✗ RSS returned no articles")
            failed_sources.append("rss")
    except Exception as e:
        logger.error(f"✗ RSS failed: {e}")
        failed_sources.append("rss")

    # === FALLBACK: FIXTURES ===
    if not all_articles:
        logger.warning("\n✗ All live APIs failed; falling back to fixtures")
        all_articles = Normalizer.load_fixtures()
        if all_articles:
            logger.info(f"✓ Fixtures: {len(all_articles)} articles")
            successful_sources.append("fixtures")
        else:
            logger.error("✗ Fixtures also empty; ingest complete failure")

    # === NORMALIZE ===
    logger.info(f"\nNormalizing {len(all_articles)} articles...")
    normalized = Normalizer.normalize_batch(all_articles)

    if not normalized:
        logger.error("✗ Normalization failed; no valid articles")
        return [], 2

    # === SAVE ===
    output_path = Normalizer.save_batch(normalized)
    logger.info(f"✓ Saved to {output_path}")

    # === SUMMARY ===
    logger.info("\n" + "=" * 80)
    logger.info("Ingest Summary")
    logger.info("=" * 80)
    logger.info(f"Successful sources: {', '.join(successful_sources)}")
    if failed_sources:
        logger.warning(f"Failed sources: {', '.join(failed_sources)}")
    logger.info(f"Total articles collected: {len(all_articles)}")
    logger.info(f"Total articles normalized: {len(normalized)}")
    logger.info(f"Output: {output_path}")

    # Print table
    print_articles_table(normalized)

    # Determine exit code
    if len(successful_sources) >= 2:
        exit_code = 0  # Success: at least 2 sources worked
    elif "fixtures" in successful_sources:
        exit_code = 1  # Partial: only fixtures
    else:
        exit_code = 2  # Failure: nothing

    return normalized, exit_code


if __name__ == "__main__":
    articles, exit_code = run_ingest()

    logger.info("=" * 80)
    if exit_code == 0:
        logger.info("✓ L1 Ingest completed successfully")
    elif exit_code == 1:
        logger.warning("⚠ L1 Ingest partial (used fixtures)")
    else:
        logger.error("✗ L1 Ingest failed")

    logger.info("=" * 80)

    sys.exit(exit_code)
