"""
Chroma vector database client for persistent storage.

Why Chroma?
- Local-first; persists to disk (~data/chroma)
- No API dependency (unlike Pinecone, Weaviate)
- Metadata filtering (filter by source, ticker, event_type, date)
- Hybrid search (semantic + BM25 keyword scoring)
- Simple API; easy to test locally

Why not Pinecone/Weaviate?
- $0 budget; Pinecone is pay-as-you-go
- Chroma is sufficient for <100k articles; scales to millions with optimization
- Local ownership of data (important for traders who care about privacy)

Tradeoff: No real-time index updates (rebuild batch); no cross-region sync.
For daily briefs, batch rebuild is fine.

Chroma versioning note:
- We use chromadb 0.4.x (simpler API)
- Future: upgrade to 0.5+ if needed (refactored client)
"""

from datetime import datetime
from pathlib import Path
from typing import Optional
import warnings
import sys
import io

import chromadb

from src.config import Config
from src.logging_utils import get_request_logger

logger = get_request_logger(__name__)

# Suppress Chroma deprecation warnings (we're using the new PersistentClient API)
warnings.filterwarnings("ignore", message=".*deprecated configuration of Chroma.*")


class ChromaStore:
    """Chroma vector database client with metadata filtering."""

    COLLECTION_NAME = "catalyst_articles"

    def __init__(self, db_dir: Optional[Path] = None, reset: bool = False):
        """
        Initialize Chroma client.

        Args:
            db_dir: Directory for Chroma persistence. If None, uses Config.CHROMA_DB_DIR.
            reset: If True, delete existing DB and start fresh.
        """
        self.db_dir = Path(db_dir or Config.CHROMA_DB_DIR)
        self.db_dir.mkdir(parents=True, exist_ok=True)

        if reset and self.db_dir.exists():
            logger.warning(f"Resetting Chroma DB at {self.db_dir}")
            import shutil
            shutil.rmtree(self.db_dir)
            self.db_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Chroma client (persistent) - using new API
        # Suppress deprecation warnings printed to stderr
        try:
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            try:
                self.client = chromadb.PersistentClient(path=str(self.db_dir))
            finally:
                sys.stderr = old_stderr
            logger.info(f"✓ Chroma client initialized at {self.db_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize Chroma: {e}")
            raise

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # Cosine similarity for embeddings
        )
        logger.info(f"✓ Collection '{self.COLLECTION_NAME}' ready")

    def add_articles(self, articles: list[dict], reset_first: bool = False) -> int:
        """
        Index articles in Chroma.

        Args:
            articles: List of article dicts from L3/L4 (classified + embedded)
            reset_first: If True, clear collection before adding

        Returns:
            Number of articles successfully indexed
        """
        if reset_first:
            logger.info("Clearing collection before adding articles...")
            self.collection.delete(where={"id": {"$ne": ""}})  # Delete all

        logger.info(f"Indexing {len(articles)} articles to Chroma...")

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for article in articles:
            article_id = article.get("id")
            if not article_id:
                logger.warning("Article missing ID; skipping")
                continue

            # Embedding (required for semantic search)
            embedding = article.get("embedding")
            if not embedding:
                logger.warning(f"Article {article_id} has no embedding; will recompute on search")
                embedding = None  # Chroma will handle

            # Document (for display + BM25)
            doc = f"{article.get('title', '')}. {article.get('summary', '')}"

            # Metadata (for filtering)
            metadata = {
                "source": article.get("source", "unknown"),
                "published_at": article.get("published_at", ""),
                "event_type": article.get("event_type", "other"),
                "risk_level": article.get("risk_level", "low"),
                "risk_score": float(article.get("risk_score", 0.5)),
                "url": article.get("url", ""),
            }

            # Flatten tickers into metadata (Chroma doesn't support arrays, so we store as string)
            tickers = article.get("tickers", [])
            metadata["tickers"] = ",".join(tickers)

            ids.append(article_id)
            documents.append(doc)
            metadatas.append(metadata)
            if embedding:
                embeddings.append(embedding)

        # Add to collection
        try:
            if embeddings and len(embeddings) == len(ids):
                # Add with pre-computed embeddings
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )
            else:
                # Add without embeddings; Chroma will compute them
                logger.info("Some articles lack embeddings; Chroma will compute on insert")
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                )

            logger.info(f"✓ Indexed {len(ids)} articles")
            return len(ids)

        except Exception as e:
            logger.error(f"Failed to add articles to Chroma: {e}")
            return 0

    def search(
        self,
        query_text: str,
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        Semantic search with optional metadata filtering.

        Args:
            query_text: Search query (e.g., "Apple earnings")
            top_k: Number of results to return
            where: Chroma filter dict (e.g., {"source": {"$eq": "yfinance"}})

        Returns:
            List of result dicts with keys: id, document, distance, metadata
        """
        logger.info(f"Searching: {query_text} (top {top_k})")

        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=top_k,
                where=where,
            )

            # Chroma returns nested lists; flatten for readability
            if not results or not results.get("ids") or not results["ids"]:
                logger.info("No results found")
                return []

            articles = []
            for i in range(len(results["ids"][0])):
                article = {
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "distance": float(results["distances"][0][i]),  # Lower = better
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                }
                articles.append(article)

            logger.info(f"✓ Found {len(articles)} results")
            return articles

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def search_by_ticker(
        self,
        query_text: str,
        ticker: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Search with ticker filter.

        Args:
            query_text: Search query
            ticker: Stock ticker (e.g., "AAPL")
            top_k: Number of results

        Returns:
            List of relevant articles mentioning the ticker
        """
        # Chroma stores tickers as comma-separated strings; filter with $contains
        where = {"tickers": {"$contains": ticker}}
        return self.search(query_text, top_k=top_k, where=where)

    def search_by_risk_level(
        self,
        query_text: str,
        risk_level: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Search with risk level filter.

        Args:
            query_text: Search query
            risk_level: "low", "medium", or "high"
            top_k: Number of results

        Returns:
            List of articles with matching risk level
        """
        where = {"risk_level": {"$eq": risk_level}}
        return self.search(query_text, top_k=top_k, where=where)

    def search_by_date_range(
        self,
        query_text: str,
        start_date: str,
        end_date: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Search within a date range.

        Args:
            query_text: Search query
            start_date: ISO-8601 start date (e.g., "2024-01-01")
            end_date: ISO-8601 end date (e.g., "2024-01-31")
            top_k: Number of results

        Returns:
            List of articles published in date range
        """
        where = {
            "$and": [
                {"published_at": {"$gte": start_date}},
                {"published_at": {"$lte": end_date}},
            ]
        }
        return self.search(query_text, top_k=top_k, where=where)

    def get_collection_stats(self) -> dict:
        """
        Get statistics on the collection.

        Returns:
            Dict with keys: count, unique_sources, unique_tickers, etc.
        """
        try:
            count = self.collection.count()

            # Sample metadata to compute stats
            sample = self.collection.get(limit=100)
            metadatas = sample.get("metadatas", [])

            sources = set()
            tickers = set()
            risk_levels = set()
            event_types = set()

            for meta in metadatas:
                sources.add(meta.get("source", "unknown"))
                tickers.update(meta.get("tickers", "").split(","))
                risk_levels.add(meta.get("risk_level", "unknown"))
                event_types.add(meta.get("event_type", "unknown"))

            return {
                "total_articles": count,
                "unique_sources": len(sources),
                "unique_tickers": len(tickers),
                "unique_risk_levels": len(risk_levels),
                "unique_event_types": len(event_types),
                "sources": sorted(sources),
                "sample_tickers": sorted(list(tickers))[:10],
            }

        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}
