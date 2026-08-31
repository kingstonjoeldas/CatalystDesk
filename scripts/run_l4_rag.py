#!/usr/bin/env python
"""
Phase L4: RAG (Retrieval-Augmented Generation)

Embeds articles using MiniLM and indexes them in Chroma.
Demonstrates retrieval with filters (ticker, risk level, date range).

Exit codes:
  0 = Success (DB built and retrieval tested)
  1 = Partial (DB built but retrieval demos failed)
  2 = Failure (couldn't load/embed articles)
"""

import json
import sys
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.logging_utils import get_request_logger, set_request_id
from src.rag.embed import EmbeddingClient
from src.rag.store import ChromaStore
from src.rag.retrieve import Retriever

logger = get_request_logger(__name__)


def find_latest_classified_file() -> Optional[Path]:
    """
    Find the most recent articles_classified_*.json from L3.

    Returns:
        Path to file, or None if none found
    """
    data_dir = Config.DATA_DIR
    classified_files = sorted(data_dir.glob("articles_classified_*.json"), reverse=True)

    if classified_files:
        logger.info(f"Found classified files: {[f.name for f in classified_files[:3]]}")
        return classified_files[0]

    logger.error("No classified files found; run L3 first")
    return None


def load_articles(file_path: Path) -> list[dict]:
    """
    Load articles from L3 output JSON.

    Args:
        file_path: Path to articles_classified_*.json

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


def demo_retrieval(retriever: Retriever, articles: list[dict]) -> bool:
    """
    Run retrieval demos to verify the RAG system works.

    Args:
        retriever: Retriever instance
        articles: Original articles (for context)

    Returns:
        True if all demos passed, False otherwise
    """
    logger.info("\n" + "=" * 80)
    logger.info("Retrieval Demos")
    logger.info("=" * 80)

    # Extract sample ticker from articles
    sample_tickers = set()
    for article in articles[:10]:
        tickers = article.get("tickers", [])
        sample_tickers.update(tickers)

    if not sample_tickers:
        logger.warning("No tickers found in articles; skipping ticker filter demo")
        return False

    sample_ticker = list(sample_tickers)[0]

    # Demo 1: General query
    logger.info("\n[Demo 1] General semantic search")
    query = "latest earnings"
    results = retriever.store.search(query, top_k=3)
    logger.info(f"Query: '{query}' -> {len(results)} results")
    for i, result in enumerate(results, 1):
        logger.info(f"  [{i}] {result['document'][:80]}... (distance: {result['distance']:.3f})")

    # Demo 2: Ticker filter
    logger.info(f"\n[Demo 2] Filter by ticker: {sample_ticker}")
    results = retriever.retrieve_for_query(
        query="news",
        ticker=sample_ticker,
        top_k=3,
    )
    logger.info(f"Query: 'news' (ticker={sample_ticker}) -> {len(results)} results")
    for i, result in enumerate(results, 1):
        logger.info(f"  [{i}] {result['document'][:80]}... (tickers: {result['metadata'].get('tickers')})")

    # Demo 3: Risk level filter
    logger.info("\n[Demo 3] Filter by risk level: high")
    results = retriever.retrieve_top_risk(query="market", top_k=3)
    logger.info(f"Query: 'market' (risk=high) -> {len(results)} results")
    for i, result in enumerate(results, 1):
        risk = result["metadata"].get("risk_level", "unknown")
        logger.info(f"  [{i}] {result['document'][:80]}... (risk: {risk})")

    # Demo 4: Brief context retrieval
    logger.info(f"\n[Demo 4] Context for brief: {sample_ticker}")
    results = retriever.get_context_for_brief(
        ticker=sample_ticker,
        question="what are the recent risks?",
        max_articles=2,
    )
    logger.info(f"Retrieved {len(results)} context articles")
    context = retriever.format_results(results, max_length=200)
    logger.info(f"Formatted context:\n{context}")

    return True


def run_rag() -> tuple[ChromaStore, int]:
    """
    Main RAG pipeline: embed articles and build Chroma index.

    Returns:
        (chroma_store, exit_code)
    """
    set_request_id("l4_rag")

    logger.info("=" * 80)
    logger.info("CatalystDesk Phase L4: RAG (Embeddings + Chroma)")
    logger.info("=" * 80)

    # === FIND INPUT ===
    classified_file = find_latest_classified_file()
    if not classified_file:
        logger.error("Cannot proceed without L3 classified output")
        return None, 2

    # === LOAD ===
    articles = load_articles(classified_file)
    if not articles:
        logger.error("No articles loaded")
        return None, 2

    # === EMBED ===
    logger.info(f"\nEmbedding {len(articles)} articles using sentence-transformers MiniLM...")
    try:
        embedder = EmbeddingClient()
        articles = embedder.embed_articles(articles)
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return None, 2

    # === INDEX IN CHROMA ===
    logger.info("\nIndexing embeddings in Chroma (local vector DB)...")
    try:
        store = ChromaStore(reset=True)  # Reset to start fresh
        count = store.add_articles(articles, reset_first=True)

        if count == 0:
            logger.error("Failed to index any articles")
            return store, 2

        logger.info(f"✓ Indexed {count} articles in Chroma")

    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        return None, 2

    # === COLLECTION STATS ===
    logger.info("\n" + "=" * 80)
    logger.info("Collection Statistics")
    logger.info("=" * 80)
    stats = store.get_collection_stats()
    logger.info(f"Total articles: {stats.get('total_articles', 0)}")
    logger.info(f"Unique sources: {stats.get('unique_sources', 0)}")
    logger.info(f"Unique tickers: {stats.get('unique_tickers', 0)}")
    logger.info(f"Risk levels: {stats.get('unique_risk_levels', 0)}")
    logger.info(f"Event types: {stats.get('unique_event_types', 0)}")
    logger.info(f"Sample tickers: {', '.join(stats.get('sample_tickers', []))}")

    # === RETRIEVAL DEMOS ===
    try:
        retriever = Retriever(store)
        demo_ok = demo_retrieval(retriever, articles)
        exit_code = 0 if demo_ok else 1
    except Exception as e:
        logger.error(f"Retrieval demo failed: {e}")
        exit_code = 1

    logger.info("\n" + "=" * 80)
    logger.info("RAG Setup Complete")
    logger.info("=" * 80)
    logger.info(f"Chroma DB: {Config.CHROMA_DB_DIR}")
    logger.info(f"Ready for L5 agents to retrieve context")

    return store, exit_code


if __name__ == "__main__":
    store, exit_code = run_rag()

    logger.info("=" * 80)
    if exit_code == 0:
        logger.info("✓ L4 RAG completed successfully")
    else:
        logger.warning("⚠ L4 RAG partial (retrieval demo failed)")

    logger.info("=" * 80)

    sys.exit(exit_code)
