"""
Ingest market news and OHLCV data from yfinance.

Why yfinance?
- Free, no API key required
- Includes company news headlines + links
- Provides OHLCV data to detect narrative-vs-price contradictions (L6)

Why not web scraping?
- Fragile (site structure changes), violates ToS, unreliable
- yfinance is the standard, well-maintained Python library
"""

import time
from datetime import datetime, timedelta
from typing import Optional

import yfinance as yf

from src.config import Config
from src.logging_utils import get_request_logger

logger = get_request_logger(__name__)


class YFinanceClient:
    """Fetch news and OHLCV data from yfinance with retries."""

    MAX_RETRIES = 3
    RETRY_BACKOFF_SEC = 2

    @classmethod
    def fetch_news(cls, tickers: list[str], days_back: int = 7) -> list[dict]:
        """
        Fetch recent news for tickers from yfinance.

        Args:
            tickers: List of ticker symbols, e.g., ["AAPL", "MSFT"]
            days_back: How many days of news to fetch (yfinance default ~30, we trim)

        Returns:
            List of dicts with keys: id, source, published_at, title, raw_text, url, tickers, ...
        """
        articles = []
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)

        for ticker in tickers:
            logger.info(f"Fetching news for {ticker}...")
            try:
                ticker_obj = yf.Ticker(ticker)
                news = ticker_obj.news or []

                for i, item in enumerate(news):
                    # yfinance returns dict with keys: title, link, source, providerPublishTime
                    try:
                        pub_time_unix = item.get("providerPublishTime", 0)
                        pub_time = datetime.utcfromtimestamp(pub_time_unix)

                        # Skip very old articles
                        if pub_time < cutoff_date:
                            continue

                        article = {
                            "id": f"yfinance_{ticker}_{pub_time.strftime('%Y%m%d')}_{i:03d}",
                            "source": "yfinance",
                            "published_at": pub_time.isoformat() + "Z",
                            "title": item.get("title", "")[:256],
                            "raw_text": item.get("title", ""),  # yfinance doesn't provide body; use title
                            "url": item.get("link", ""),
                            "tickers": [ticker],
                            "summary": None,
                            "event_type": None,
                            "risk_level": None,
                            "risk_score": None,
                            "contradiction": None,
                        }
                        articles.append(article)
                    except (KeyError, ValueError) as e:
                        logger.warning(f"Failed to parse yfinance news item: {e}")
                        continue

                logger.info(f"✓ Fetched {len(articles)} articles for {ticker}")

            except Exception as e:
                logger.error(f"Failed to fetch news for {ticker}: {e}")
                # Continue to next ticker; don't fail entire ingest

        return articles

    @classmethod
    def fetch_ohlcv(
        cls, ticker: str, days_back: int = 1, period: str = "1d"
    ) -> Optional[dict]:
        """
        Fetch OHLCV data for a ticker (used by L6 for contradiction checks).

        Args:
            ticker: e.g., "AAPL"
            days_back: How many days back to fetch
            period: "1d" for daily, "1wk" for weekly

        Returns:
            Dict with keys: open, high, low, close, volume, change_pct
            Or None if fetch fails.
        """
        logger.info(f"Fetching {period} OHLCV for {ticker} ({days_back} days)...")

        for attempt in range(cls.MAX_RETRIES):
            try:
                ticker_obj = yf.Ticker(ticker)
                # Fetch last N days
                hist = ticker_obj.history(period=period, progress=False)

                if hist.empty:
                    logger.warning(f"No OHLCV data for {ticker}")
                    return None

                # Get the most recent row
                latest = hist.iloc[-1]
                prev = hist.iloc[-2] if len(hist) > 1 else None

                close = latest["Close"]
                if prev is not None:
                    prev_close = prev["Close"]
                    change_pct = ((close - prev_close) / prev_close) * 100
                else:
                    change_pct = 0.0

                result = {
                    "ticker": ticker,
                    "open": float(latest["Open"]),
                    "high": float(latest["High"]),
                    "low": float(latest["Low"]),
                    "close": float(close),
                    "volume": int(latest["Volume"]),
                    "change_pct": float(change_pct),
                    "timestamp": hist.index[-1].isoformat(),
                }

                logger.info(f"✓ OHLCV: {ticker} close={close:.2f} change={change_pct:.2f}%")
                return result

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{cls.MAX_RETRIES} failed: {e}")
                if attempt < cls.MAX_RETRIES - 1:
                    time.sleep(cls.RETRY_BACKOFF_SEC)

        logger.error(f"Failed to fetch OHLCV for {ticker} after {cls.MAX_RETRIES} attempts")
        return None
