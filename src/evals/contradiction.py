"""
Contradiction detection: compare brief sentiment vs. price movement.

Why this matters?
- A brief says "earnings beat = bullish" but stock dropped 5%
- This reveals bias (overconfidence, narrative risk) or model misalignment
- Contradiction detection is a heuristic quality signal

Implementation:
- Brief sentiment via zero-shot classification (bullish/bearish/neutral)
- Price movement via yfinance 1d return
- Flag if sentiment contradicts returns beyond a threshold
"""

from typing import Optional

from src.hf_tasks.classify import Classifier
from src.ingest.yfinance_client import YFinanceClient
from src.logging_utils import get_request_logger

logger = get_request_logger(__name__)


class ContradictionDetector:
    """Detects narrative bias by comparing brief sentiment to price movement."""

    def __init__(self):
        """Initialize detector."""
        self.classifier = Classifier()
        self.yf_client = YFinanceClient()

    def get_brief_sentiment(self, brief_summary: str) -> tuple[str, float]:
        """
        Classify brief as bullish, bearish, or neutral.

        Args:
            brief_summary: Brief text (1-2 sentences)

        Returns:
            Tuple of (sentiment, confidence)
        """
        labels = ["bullish", "bearish", "neutral"]
        hypothesis_template = "This brief is {}."

        try:
            result = self.classifier.client.zero_shot_classification(
                brief_summary[:512],
                labels,
                hypothesis_template=hypothesis_template,
                multi_label=False,
            )

            sentiment = result.get("labels", ["neutral"])[0]
            confidence = float(result.get("scores", [0.5])[0])

            logger.info(f"Brief sentiment: {sentiment} ({confidence:.2f})")
            return sentiment, confidence

        except Exception as e:
            logger.error(f"Failed to classify brief sentiment: {e}")
            return "neutral", 0.5

    def get_price_movement(self, ticker: str, days_back: int = 1) -> tuple[Optional[float], Optional[str]]:
        """
        Get price movement for a ticker.

        Args:
            ticker: Stock ticker
            days_back: Number of days back (default 1 for 1d return)

        Returns:
            Tuple of (change_pct, direction) where direction is "up", "down", or "flat"
        """
        try:
            ohlcv = self.yf_client.fetch_ohlcv(ticker, days_back=days_back, period="1d")

            if not ohlcv:
                logger.warning(f"No OHLCV data for {ticker}")
                return None, None

            change_pct = ohlcv.get("change_pct", 0.0)

            if change_pct > 1.0:
                direction = "up"
            elif change_pct < -1.0:
                direction = "down"
            else:
                direction = "flat"

            logger.info(f"Price movement: {ticker} {direction} {change_pct:.2f}%")
            return change_pct, direction

        except Exception as e:
            logger.error(f"Failed to get price movement: {e}")
            return None, None

    def detect_contradiction(
        self,
        ticker: str,
        brief_summary: str,
        threshold_pct: float = 5.0,
    ) -> tuple[bool, dict]:
        """
        Detect if brief sentiment contradicts price movement.

        Args:
            ticker: Stock ticker
            brief_summary: Brief text
            threshold_pct: Magnitude threshold for "significant" move (e.g., 5% = ±5%)

        Returns:
            Tuple of (is_contradiction, analysis_dict)
        """
        logger.info(f"Checking contradiction for {ticker}")

        # Get sentiment
        sentiment, sentiment_conf = self.get_brief_sentiment(brief_summary)

        # Get price movement
        change_pct, direction = self.get_price_movement(ticker)

        if change_pct is None or direction is None:
            logger.warning("Cannot detect contradiction (no price data)")
            return False, {
                "ticker": ticker,
                "sentiment": sentiment,
                "sentiment_confidence": sentiment_conf,
                "price_change": None,
                "direction": None,
                "is_contradiction": False,
                "reason": "No price data available",
            }

        # Check for contradiction
        abs_change = abs(change_pct)
        is_significant_move = abs_change > threshold_pct

        is_contradiction = False
        reason = ""

        if is_significant_move:
            if sentiment == "bullish" and direction == "down":
                is_contradiction = True
                reason = f"Brief bullish but stock down {change_pct:.1f}%"
            elif sentiment == "bearish" and direction == "up":
                is_contradiction = True
                reason = f"Brief bearish but stock up {change_pct:.1f}%"

        analysis = {
            "ticker": ticker,
            "sentiment": sentiment,
            "sentiment_confidence": sentiment_conf,
            "price_change": change_pct,
            "direction": direction,
            "is_significant_move": is_significant_move,
            "is_contradiction": is_contradiction,
            "reason": reason or "No contradiction detected",
        }

        if is_contradiction:
            logger.warning(f"⚠️ CONTRADICTION: {reason}")
        else:
            logger.info("✓ No contradiction detected")

        return is_contradiction, analysis

    @staticmethod
    def format_contradiction_report(analysis: dict) -> str:
        """
        Format contradiction analysis as readable text.

        Args:
            analysis: Dict from detect_contradiction()

        Returns:
            Formatted report
        """
        lines = [
            f"Ticker: {analysis['ticker']}",
            f"Brief sentiment: {analysis['sentiment']} ({analysis['sentiment_confidence']:.0%})",
            f"1d price move: {analysis['direction']} {analysis['price_change']:.1f}%",
            f"Contradiction: {'YES' if analysis['is_contradiction'] else 'NO'}",
            f"Reason: {analysis['reason']}",
        ]
        return "\n".join(lines)
