"""
Ingest US economic indicators from FRED (Federal Reserve Economic Data).

Why FRED?
- Free, official Fed data; no budget impact
- Indicators (CPI, unemployment, fed funds) drive macro risk analysis
- Easy REST API + pandas_datareader integration

Why not BLS / Census Bureau directly?
- FRED aggregates all official sources; one API vs. many
- Well-tested, widely used by economists and traders
"""

import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from pandas_datareader import data as pdr

from src.config import Config
from src.logging_utils import get_request_logger

logger = get_request_logger(__name__)


class FredClient:
    """Fetch economic indicators from FRED with retries."""

    MAX_RETRIES = 3
    RETRY_BACKOFF_SEC = 2

    # Series IDs for key macro indicators
    SERIES_MAP = {
        "CPI": "CPIAUCSL",  # Consumer Price Index (all urban consumers)
        "Core CPI": "CPILFESL",  # CPI excluding food and energy
        "Unemployment": "UNRATE",  # Unemployment Rate
        "Fed Funds Rate": "FEDFUNDS",  # Effective Federal Funds Rate
        "10Y Treasury Yield": "GS10",  # 10-Year Treasury Constant Maturity
        "2Y Treasury Yield": "GS2",  # 2-Year Treasury Constant Maturity
    }

    @classmethod
    def is_available(cls) -> bool:
        """Check if FRED API key is configured."""
        if not Config.FRED_API_KEY:
            logger.warning("FRED_API_KEY not set; FRED ingest will be skipped")
            return False
        return True

    @classmethod
    def fetch_indicator(
        cls, series_id: str, days_back: int = 90
    ) -> Optional[dict]:
        """
        Fetch a single FRED series (e.g., CPI, unemployment).

        Args:
            series_id: FRED series code, e.g., "CPIAUCSL"
            days_back: Fetch data from this many days ago

        Returns:
            Dict with keys: series_id, latest_value, latest_date, description
            Or None if fetch fails.
        """
        if not cls.is_available():
            return None

        logger.info(f"Fetching FRED series: {series_id}...")

        for attempt in range(cls.MAX_RETRIES):
            try:
                start_date = datetime.utcnow() - timedelta(days=days_back)
                end_date = datetime.utcnow()

                # pandas_datareader uses FRED API under the hood
                data = pdr.get_data_fred(
                    series_id,
                    start=start_date,
                    end=end_date,
                    api_key=Config.FRED_API_KEY,
                )

                if data.empty:
                    logger.warning(f"No data for {series_id}")
                    return None

                latest_value = float(data.iloc[-1])
                latest_date = data.index[-1].isoformat()

                result = {
                    "series_id": series_id,
                    "latest_value": latest_value,
                    "latest_date": latest_date,
                    "all_values": data.values.tolist(),
                    "all_dates": [d.isoformat() for d in data.index],
                }

                logger.info(f"✓ {series_id}: {latest_value:.2f} (as of {latest_date[:10]})")
                return result

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{cls.MAX_RETRIES} failed: {e}")
                if attempt < cls.MAX_RETRIES - 1:
                    time.sleep(cls.RETRY_BACKOFF_SEC)

        logger.error(f"Failed to fetch {series_id} after {cls.MAX_RETRIES} attempts")
        return None

    @classmethod
    def fetch_all_indicators(cls) -> dict:
        """
        Fetch all key macro indicators in SERIES_MAP.

        Returns:
            Dict mapping indicator name -> result (from fetch_indicator)
        """
        if not cls.is_available():
            logger.info("FRED skipped (no API key)")
            return {}

        results = {}
        for name, series_id in cls.SERIES_MAP.items():
            data = cls.fetch_indicator(series_id, days_back=90)
            if data:
                results[name] = data

        logger.info(f"✓ Fetched {len(results)}/{len(cls.SERIES_MAP)} FRED indicators")
        return results

    @classmethod
    def articles_from_indicators(cls, indicators: dict) -> list[dict]:
        """
        Convert FRED indicators to NewsItem articles for ingest pipeline.

        This is a bit artificial (FRED data isn't news), but allows macro indicators
        to flow through the same normalization pipeline and be tagged with risk labels.

        Args:
            indicators: Result from fetch_all_indicators()

        Returns:
            List of NewsItem-shaped dicts
        """
        articles = []

        for name, data in indicators.items():
            if not data:
                continue

            latest_value = data["latest_value"]
            latest_date = data["latest_date"]
            series_id = data["series_id"]

            # Create a pseudo-news item that will flow through L3 classification
            article = {
                "id": f"fred_{series_id}_{latest_date[:10]}",
                "source": "fred",
                "published_at": latest_date + "Z" if latest_date[-1] != "Z" else latest_date,
                "title": f"FRED: {name} = {latest_value:.2f}",
                "raw_text": f"Latest {name} reading: {latest_value:.2f} as of {latest_date[:10]}. Series ID: {series_id}",
                "url": f"https://fred.stlouisfed.org/series/{series_id}",
                "tickers": ["SPY", "TLT", "DBC"],  # Macro-sensitive assets
                "summary": None,
                "event_type": None,
                "risk_level": None,
                "risk_score": None,
                "contradiction": None,
            }
            articles.append(article)

        return articles
