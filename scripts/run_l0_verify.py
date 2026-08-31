#!/usr/bin/env python
"""
Phase L0 Verification Script

Sanity checks for the CatalystDesk skeleton:
- Config loads
- Logging initializes
- Sample fixtures parse
- All required directories exist
"""

import json
import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.logging_utils import get_request_logger, set_request_id

# Set a request ID for this verification run
set_request_id("l0_verify")

logger = get_request_logger(__name__)


def verify_config() -> bool:
    """Check that Config loads and all paths exist."""
    logger.info("Verifying Config...")

    try:
        Config.ensure_dirs()
        logger.info(f"✓ Config.ROOT={Config.ROOT}")
        logger.info(f"✓ Config.DATA_DIR={Config.DATA_DIR}")
        logger.info(f"✓ Config.CACHE_DIR={Config.CACHE_DIR}")
        logger.info(f"✓ Config.LOG_DIR={Config.LOG_DIR}")
        return True
    except Exception as e:
        logger.error(f"✗ Config failed: {e}")
        return False


def verify_logging() -> bool:
    """Check that logging works."""
    logger.info("Verifying Logging...")

    try:
        logger.info("✓ Logging initialized")
        logger.debug("Debug message visible at DEBUG level")
        logger.warning("Warning message visible")
        return True
    except Exception as e:
        logger.error(f"✗ Logging failed: {e}")
        return False


def verify_fixtures() -> bool:
    """Load and parse fixture articles."""
    logger.info("Verifying Sample Fixtures...")

    try:
        fixture_file = Config.DATA_DIR / "samples" / "sample_articles.json"
        if not fixture_file.exists():
            logger.error(f"✗ Fixture file not found: {fixture_file}")
            return False

        with open(fixture_file, "r") as f:
            articles = json.load(f)

        logger.info(f"✓ Loaded {len(articles)} sample articles")

        # Validate schema
        required_keys = {
            "id",
            "source",
            "published_at",
            "title",
            "raw_text",
            "url",
            "tickers",
        }
        for i, article in enumerate(articles):
            missing = required_keys - set(article.keys())
            if missing:
                logger.error(
                    f"✗ Article {i} missing keys: {missing}"
                )
                return False

        # Print summary table
        logger.info("\nSample Articles Summary:")
        logger.info("-" * 80)
        for article in articles:
            logger.info(
                f"  {article['source']:8} | {article['published_at'][:10]} | {article['title'][:50]:<50}"
            )
        logger.info("-" * 80)

        return True
    except json.JSONDecodeError as e:
        logger.error(f"✗ Fixture JSON parse error: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Fixture verification failed: {e}")
        return False


def verify_env_warnings() -> bool:
    """Check for missing env vars and warn (but don't fail)."""
    logger.info("\nChecking Environment Variables...")

    warnings = Config.validate()
    if warnings:
        for warning in warnings:
            logger.warning(f"⚠ {warning}")
    else:
        logger.info("✓ All required env vars present")

    return True


def main():
    """Run all verification checks."""
    logger.info("=" * 80)
    logger.info("CatalystDesk Phase L0 Verification")
    logger.info("=" * 80)

    checks = [
        ("Config", verify_config),
        ("Logging", verify_logging),
        ("Fixtures", verify_fixtures),
        ("Environment", verify_env_warnings),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"✗ Unhandled error in {name}: {e}")
            results.append((name, False))

    logger.info("\n" + "=" * 80)
    logger.info("Verification Results")
    logger.info("=" * 80)

    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status} | {name}")
        if not result:
            all_passed = False

    logger.info("=" * 80)

    if all_passed:
        logger.info("✓ All checks passed. Ready for Phase L1.")
        logger.info("\nNext steps:")
        logger.info("  1. Populate .env with your HF_TOKEN (required)")
        logger.info("  2. (Optional) Add FRED_API_KEY for macro indicators")
        logger.info("  3. Run: python scripts/run_l1_ingest.py")
        return 0
    else:
        logger.error("✗ Some checks failed. See above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
