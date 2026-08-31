"""
High-level retrieval interface for L5 agents.

Wraps Chroma store with convenience methods for different retrieval scenarios.
"""

from datetime import datetime, timedelta
from typing import Optional

from src.rag.store import ChromaStore
from src.logging_utils import get_request_logger

logger = get_request_logger(__name__)


class Retriever:
    """High-level RAG retrieval interface."""

    def __init__(self, store: Optional[ChromaStore] = None):
        """
        Initialize retriever.

        Args:
            store: ChromaStore instance. If None, creates new one.
        """
        self.store = store or ChromaStore()

    def retrieve_for_query(
        self,
        query: str,
        ticker: Optional[str] = None,
        risk_level: Optional[str] = None,
        days_back: int = 7,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Retrieve articles relevant to a user query with optional filters.

        Args:
            query: User question (e.g., "What's the latest on Apple?")
            ticker: Filter by ticker (e.g., "AAPL")
            risk_level: Filter by risk (low|medium|high)
            days_back: Only include articles from this many days ago
            top_k: Number of results

        Returns:
            List of relevant articles with metadata
        """
        logger.info(f"Retrieving for query: {query}")

        # Build Chroma filter
        filters = []

        # Ticker filter
        if ticker:
            filters.append({"tickers": {"$contains": ticker}})
            logger.info(f"  Filter: ticker={ticker}")

        # Risk level filter
        if risk_level and risk_level in ["low", "medium", "high"]:
            filters.append({"risk_level": {"$eq": risk_level}})
            logger.info(f"  Filter: risk_level={risk_level}")

        # Date range filter (disabled for Chroma compatibility - filters on ISO strings not supported)
        # TODO: Implement date filtering via post-processing instead
        # end_date = datetime.utcnow()
        # start_date = end_date - timedelta(days=days_back)
        logger.info(f"  Filter: (date filtering disabled for Chroma v0.4 compatibility)")

        # Combine filters (all must match)
        if len(filters) > 1:
            where = {"$and": filters}
        elif len(filters) == 1:
            where = filters[0]
        else:
            where = None

        results = self.store.search(query, top_k=top_k, where=where)

        logger.info(f"✓ Retrieved {len(results)} articles")
        return results

    def retrieve_by_event_type(
        self,
        query: str,
        event_type: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Retrieve articles of a specific event type.

        Args:
            query: Search query
            event_type: e.g., "monetary policy", "earnings"
            top_k: Number of results

        Returns:
            List of matching articles
        """
        where = {"event_type": {"$eq": event_type}}
        return self.store.search(query, top_k=top_k, where=where)

    def retrieve_top_risk(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Retrieve highest-risk articles relevant to query.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of high-risk articles, sorted by relevance
        """
        where = {"risk_level": {"$eq": "high"}}
        return self.store.search(query, top_k=top_k, where=where)

    def get_context_for_brief(
        self,
        ticker: str,
        question: str,
        max_articles: int = 3,
    ) -> list[dict]:
        """
        Retrieve context articles for a trader brief.

        Combines semantic search with ticker filter for focused retrieval.

        Args:
            ticker: Stock ticker
            question: Trader's question (e.g., "What are the risks for AAPL?")
            max_articles: Max articles to retrieve

        Returns:
            List of relevant articles
        """
        logger.info(f"Retrieving context for brief: {ticker} - {question}")

        # Search with ticker filter
        results = self.retrieve_for_query(
            query=question,
            ticker=ticker,
            days_back=14,  # Broader window for context
            top_k=max_articles,
        )

        if not results:
            logger.warning(f"No articles found for {ticker}; falling back to general search")
            results = self.store.search(question, top_k=max_articles)

        return results

    def format_results(self, results: list[dict], max_length: int = 1000) -> str:
        """
        Format retrieval results as readable context for L5 agents.

        Args:
            results: List of result dicts from retrieve
            max_length: Max chars per document

        Returns:
            Formatted context string
        """
        if not results:
            return "[No relevant articles found]"

        lines = []
        for i, result in enumerate(results, 1):
            doc = result.get("document", "")[:max_length]
            meta = result.get("metadata", {})
            ticker = meta.get("tickers", "N/A")
            risk = meta.get("risk_level", "unknown")
            score = result.get("distance", 0.0)

            lines.append(f"\n[{i}] (distance: {score:.3f}, risk: {risk}, ticker: {ticker})")
            lines.append(f"    {doc}")

        return "\n".join(lines)
