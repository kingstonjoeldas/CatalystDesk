"""
Researcher agent: retrieves relevant articles for the query.

Role in the pipeline:
1. Takes user query + ticker
2. Calls Retriever to fetch semantic matches + metadata filters
3. Packs articles into state for downstream agents
4. Returns packed state

Why separate from Risk Officer?
- Retrieval logic is distinct (it's RAG, not analysis)
- Easy to test/debug retrieval independently
- Can swap retrieval strategy without touching downstream agents
"""

import re
from typing import Optional

from src.agents.state import AgentState, Article
from src.rag.retrieve import Retriever
from src.rag.store import ChromaStore
from src.logging_utils import get_request_logger

logger = get_request_logger(__name__)


class ResearcherAgent:
    """Retrieval specialist: finds relevant articles."""

    def __init__(self, retriever: Optional[Retriever] = None):
        """
        Initialize researcher.

        Args:
            retriever: Retriever instance. If None, creates new one with default Chroma store.
        """
        if retriever is None:
            try:
                store = ChromaStore()  # Load from disk
                retriever = Retriever(store)
            except Exception as e:
                logger.error(f"Failed to initialize Retriever: {e}")
                raise

        self.retriever = retriever

    @staticmethod
    def extract_ticker(text: str) -> Optional[str]:
        """
        Extract stock ticker from text.

        Simple heuristic: look for 1-5 uppercase letters surrounded by word boundaries.

        Args:
            text: Text to search

        Returns:
            Ticker (uppercase) or None
        """
        # Match standalone uppercase words of 1-5 letters
        matches = re.findall(r"\b([A-Z]{1,5})\b", text)

        # Filter common words
        common = {
            "THE", "AND", "OR", "BUT", "FOR", "WITH", "FROM", "TO", "IN", "ON", "AT",
            "BY", "UP", "OUT", "IF", "AS", "IS", "BE", "WAS", "WHAT", "RISK", "DATA",
        }

        for match in matches:
            if match not in common and len(match) <= 5:
                return match

        return None

    def run(self, state: AgentState) -> AgentState:
        """
        Researcher agent function (runs in LangGraph).

        Args:
            state: Current agent state

        Returns:
            Updated state with retrieved_articles filled
        """
        query = state.get("query", "")
        ticker = state.get("ticker")

        # Extract ticker from query if not provided
        if not ticker:
            ticker = self.extract_ticker(query)
            if ticker:
                logger.info(f"Extracted ticker from query: {ticker}")
                state["ticker"] = ticker

        logger.info(f"Researcher: retrieving context for query: {query}")

        try:
            # Retrieve articles
            if ticker:
                logger.info(f"  Filtering by ticker: {ticker}")
                results = self.retriever.retrieve_for_query(
                    query=query,
                    ticker=ticker,
                    days_back=14,
                    top_k=5,
                )
            else:
                logger.info("  No ticker; general retrieval")
                results = self.retriever.store.search(query, top_k=5)

            # Convert results to Article TypedDict
            articles: list[Article] = [
                {
                    "id": r.get("id", ""),
                    "document": r.get("document", "")[:500],  # Truncate for brevity
                    "distance": r.get("distance", 0.0),
                    "metadata": r.get("metadata", {}),
                }
                for r in results
            ]

            state["retrieved_articles"] = articles
            state["retrieval_count"] = len(articles)
            state["agent_steps"].append("researcher")

            logger.info(f"✓ Researcher: retrieved {len(articles)} articles")

        except Exception as e:
            logger.error(f"✗ Researcher failed: {e}")
            state["retrieval_error"] = str(e)
            state["retrieved_articles"] = []
            state["retrieval_count"] = 0
            state["agent_steps"].append("researcher_failed")

        return state
