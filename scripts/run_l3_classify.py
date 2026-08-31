#!/usr/bin/env python
"""
Phase L3: Zero-Shot Classification

Reads summarized articles from L2 and classifies them by:
- event_type: "monetary policy", "inflation data", "earnings", etc.
- risk_level: "low", "medium", "high"

Uses zero-shot classification (BART-MNLI) with no training data.
Caches all results to disk so reruns are fast.

Exit codes:
  0 = Success (all articles classified)
  1 = Partial success (some articles failed)
  2 = Failure (couldn't load L2 output)
"""

import json
import sys
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.logging_utils import get_request_logger, set_request_id
from src.hf_tasks.classify import Classifier

logger = get_request_logger(__name__)


def find_latest_summarized_file() -> Optional[Path]:
    """
    Find the most recent articles_summarized_*.json from L2.

    Returns:
        Path to file, or None if none found
    """
    data_dir = Config.DATA_DIR
    summarized_files = sorted(data_dir.glob("articles_summarized_*.json"), reverse=True)

    if summarized_files:
        logger.info(f"Found summarized files: {[f.name for f in summarized_files[:3]]}")
        return summarized_files[0]

    logger.error("No summarized files found; run L2 first")
    return None


def load_articles(file_path: Path) -> list[dict]:
    """
    Load articles from L2 output JSON.

    Args:
        file_path: Path to articles_summarized_*.json

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


def save_classified(articles: list[dict]) -> Path:
    """
    Save classified articles to a new file.

    Args:
        articles: Article dicts with 'event_type' and 'risk_level' fields filled

    Returns:
        Path to saved file
    """
    output_path = Config.DATA_DIR / f"articles_classified_{len(articles)}.json"
    with open(output_path, "w") as f:
        json.dump(articles, f, indent=2)
    logger.info(f"Saved to {output_path}")
    return output_path


def print_classification_table(articles: list[dict]) -> None:
    """Print articles with classifications as a table."""
    if not articles:
        logger.info("No articles to display")
        return

    logger.info("\n" + "=" * 130)
    logger.info(
        f"{'Title':<40} | {'Event Type':<20} | {'Risk':<10} | {'Conf':<6}"
    )
    logger.info("-" * 130)

    for article in articles[:20]:  # Show first 20
        title = article.get("title", "")[:40]
        event_type = article.get("event_type", "other")[:20]
        risk_level = article.get("risk_level", "low")[:10]
        risk_score = article.get("risk_score", 0.0)
        conf = f"{risk_score:.2f}"

        logger.info(f"{title:<40} | {event_type:<20} | {risk_level:<10} | {conf:<6}")

    if len(articles) > 20:
        logger.info(f"\n... and {len(articles) - 20} more articles")

    logger.info("=" * 130 + "\n")

    # Print statistics
    event_type_counts = {}
    risk_level_counts = {}

    for article in articles:
        event_type = article.get("event_type", "other")
        risk_level = article.get("risk_level", "low")

        event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
        risk_level_counts[risk_level] = risk_level_counts.get(risk_level, 0) + 1

    logger.info("Classification Distribution:")
    logger.info("  Event Types:")
    for event_type, count in sorted(event_type_counts.items()):
        logger.info(f"    {event_type:<20}: {count:3d}")

    logger.info("  Risk Levels:")
    for risk_level, count in sorted(risk_level_counts.items()):
        logger.info(f"    {risk_level:<20}: {count:3d}")


def run_classify() -> tuple[list[dict], int]:
    """
    Main classification pipeline.

    Returns:
        (classified_articles, exit_code)
    """
    set_request_id("l3_classify")

    logger.info("=" * 80)
    logger.info("CatalystDesk Phase L3: Zero-Shot Classification")
    logger.info("=" * 80)

    # === FIND INPUT ===
    summarized_file = find_latest_summarized_file()
    if not summarized_file:
        logger.error("Cannot proceed without L2 summarized output")
        return [], 2

    # === LOAD ===
    articles = load_articles(summarized_file)
    if not articles:
        logger.error("No articles loaded")
        return [], 2

    logger.info(f"\nClassifying {len(articles)} articles...")
    logger.info("  Event Types: monetary policy, inflation data, earnings, geopolitics, liquidity/credit, regulation, other")
    logger.info("  Risk Levels: low, medium, high")
    logger.info("(First run may be slow; subsequent runs use cache)")

    # === CLASSIFY ===
    try:
        classified = Classifier.classify_batch(articles, skip_existing=True)
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        return [], 2

    # === SAVE ===
    output_path = save_classified(classified)

    # === STATS ===
    successful = sum(
        1 for a in classified
        if a.get("event_type") and a.get("event_type") != "other"
        or a.get("risk_level") and a.get("risk_level") in ["low", "medium", "high"]
    )
    low_confidence = sum(
        1 for a in classified
        if a.get("risk_score", 0.0) < 0.3
    )

    logger.info("\n" + "=" * 80)
    logger.info("Classification Summary")
    logger.info("=" * 80)
    logger.info(f"Total articles: {len(classified)}")
    logger.info(f"Successfully classified: {successful}")
    logger.info(f"Low confidence (<0.3): {low_confidence}")
    logger.info(f"Output: {output_path}")

    # Print table and stats
    print_classification_table(classified)

    exit_code = 0 if low_confidence < len(classified) * 0.1 else 1
    return classified, exit_code


if __name__ == "__main__":
    articles, exit_code = run_classify()

    logger.info("=" * 80)
    if exit_code == 0:
        logger.info("✓ L3 Classification completed successfully")
    else:
        logger.warning("⚠ L3 Classification partial (some low-confidence predictions)")

    logger.info("=" * 80)

    sys.exit(exit_code)
