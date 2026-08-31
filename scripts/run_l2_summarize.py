#!/usr/bin/env python
"""
Phase L2: Summarization

Reads articles from L1 ingest output and generates summaries using BART.
Caches all summaries to disk so reruns are instant.

Exit codes:
  0 = Success (all articles summarized)
  1 = Partial success (some articles failed, but caching mitigated retries)
  2 = Failure (couldn't load L1 output)
"""

import json
import sys
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.logging_utils import get_request_logger, set_request_id
from src.hf_tasks.summarize import Summarizer

logger = get_request_logger(__name__)


def find_latest_ingest_file() -> Optional[Path]:
    """
    Find the most recent articles_*.json from L1 ingest.

    Returns:
        Path to file, or None if none found
    """
    data_dir = Config.DATA_DIR
    ingest_files = sorted(data_dir.glob("articles_*.json"), reverse=True)

    if ingest_files:
        logger.info(f"Found ingest files: {[f.name for f in ingest_files[:3]]}")
        return ingest_files[0]

    logger.error("No ingest files found; run L1 first")
    return None


def load_articles(file_path: Path) -> list[dict]:
    """
    Load articles from L1 output JSON.

    Args:
        file_path: Path to articles_*.json

    Returns:
        List of article dicts
    """
    try:
        with open(file_path, "r") as f:
            articles = json.load(f)
        logger.info(f"Loaded {len(articles)} articles from {file_path.name}")
        return articles
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse {file_path}: {e}")
        return []


def save_summarized(articles: list[dict]) -> Path:
    """
    Save summarized articles to a new file.

    Args:
        articles: Article dicts with 'summary' fields filled

    Returns:
        Path to saved file
    """
    output_path = Config.DATA_DIR / f"articles_summarized_{len(articles)}.json"
    with open(output_path, "w") as f:
        json.dump(articles, f, indent=2)
    logger.info(f"Saved to {output_path}")
    return output_path


def print_summary_table(articles: list[dict]) -> None:
    """Print articles with summaries as a table."""
    if not articles:
        logger.info("No articles to display")
        return

    logger.info("\n" + "=" * 140)
    logger.info(
        f"{'Title':<50} | {'Summary':<80}"
    )
    logger.info("-" * 140)

    for article in articles[:15]:  # Show first 15
        title = article.get("title", "")[:50]
        summary = article.get("summary", "[No summary]")[:80]
        logger.info(f"{title:<50} | {summary:<80}")

    if len(articles) > 15:
        logger.info(f"\n... and {len(articles) - 15} more articles")

    logger.info("=" * 140 + "\n")


def run_summarize() -> tuple[list[dict], int]:
    """
    Main summarization pipeline.

    Returns:
        (summarized_articles, exit_code)
    """
    set_request_id("l2_summarize")

    logger.info("=" * 80)
    logger.info("CatalystDesk Phase L2: Summarization")
    logger.info("=" * 80)

    # === FIND INPUT ===
    ingest_file = find_latest_ingest_file()
    if not ingest_file:
        logger.error("Cannot proceed without L1 ingest output")
        return [], 2

    # === LOAD ===
    articles = load_articles(ingest_file)
    if not articles:
        logger.error("No articles loaded")
        return [], 2

    logger.info(f"\nSummarizing {len(articles)} articles using facebook/bart-large-cnn...")
    logger.info("(First run may be slow due to model download; subsequent runs use cache)")

    # === SUMMARIZE ===
    try:
        summarized = Summarizer.summarize_batch(articles, skip_existing=True)
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        return [], 2

    # === SAVE ===
    output_path = save_summarized(summarized)

    # === STATS ===
    successful = sum(1 for a in summarized if a.get("summary") and "[" not in a.get("summary", ""))
    failed = len(summarized) - successful

    logger.info("\n" + "=" * 80)
    logger.info("Summarization Summary")
    logger.info("=" * 80)
    logger.info(f"Total articles: {len(summarized)}")
    logger.info(f"Successfully summarized: {successful}")
    if failed:
        logger.warning(f"Failed or skipped: {failed}")
    logger.info(f"Output: {output_path}")

    # Print table
    print_summary_table(summarized)

    exit_code = 0 if failed == 0 else 1
    return summarized, exit_code


if __name__ == "__main__":
    articles, exit_code = run_summarize()

    logger.info("=" * 80)
    if exit_code == 0:
        logger.info("✓ L2 Summarization completed successfully")
    else:
        logger.warning("⚠ L2 Summarization partial (some articles failed)")

    logger.info("=" * 80)

    sys.exit(exit_code)
