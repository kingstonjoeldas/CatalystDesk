"""
Generate embeddings using sentence-transformers MiniLM.

Why MiniLM?
- sentence-transformers/all-MiniLM-L6-v2 (22M params) is the gold standard for semantic search on small data
- Fast (embeddings in <100ms per article when warm)
- Good quality (trained on 1B+ sentence pairs; wins on MTEB benchmarks)
- Designed for retrieval (not conversational; won't hallucinate)

Why not OpenAI embeddings or Cohere?
- No cost ($0 budget); runs locally via HF Inference API or local model
- MiniLM is competitive with OpenAI's text-embedding-3-small on quality
- Local embeddings = no data leakage (briefs stay private)

Tradeoff: Local inference (via HF API) adds latency; cached embeddings mitigate.

Why cache embeddings?
- Computing embeddings for 1000 articles takes ~10s (even with MiniLM)
- Rerunning L4 should not recompute; just load from Chroma
- But during development, if we tweak chunking/model, we need to recompute
"""

from typing import Optional

from sentence_transformers import SentenceTransformer

from src.config import Config
from src.logging_utils import get_request_logger

logger = get_request_logger(__name__)


class EmbeddingClient:
    """Generate embeddings using sentence-transformers MiniLM."""

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_id: Optional[str] = None):
        """
        Initialize embedding client.

        Args:
            model_id: HF model ID. If None, uses DEFAULT_MODEL.
        """
        self.model_id = model_id or self.DEFAULT_MODEL
        logger.info(f"Loading embedding model: {self.model_id}")

        try:
            self.model = SentenceTransformer(self.model_id)
            logger.info(f"✓ Model loaded. Dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def embed_text(self, text: str) -> Optional[list[float]]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed (sentence or paragraph)

        Returns:
            Embedding vector (dimension 384 for MiniLM), or None if fails
        """
        if not text or not text.strip():
            logger.warning("Cannot embed empty text")
            return None

        try:
            embedding = self.model.encode(text, show_progress_bar=False)
            # Convert numpy array to list for JSON serialization
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return None

    def embed_batch(self, texts: list[str]) -> list[Optional[list[float]]]:
        """
        Generate embeddings for multiple texts (batch).

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings (or None for failed items)
        """
        if not texts:
            return []

        try:
            embeddings = self.model.encode(texts, show_progress_bar=True)
            # Convert to list of lists
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            return [None] * len(texts)

    def embed_articles(self, articles: list[dict]) -> list[dict]:
        """
        Embed articles for indexing in Chroma.

        For each article, creates a document combining title + summary.
        Embeds it and stores the embedding in the article dict.

        Args:
            articles: List of article dicts from L3 (classified)

        Returns:
            Same articles, with 'embedding' field added
        """
        logger.info(f"Embedding {len(articles)} articles...")

        # Prepare texts to embed
        texts = []
        for article in articles:
            title = article.get("title", "")
            summary = article.get("summary", "")
            text = f"{title}. {summary}".strip()
            texts.append(text)

        # Batch embed
        embeddings = self.embed_batch(texts)

        # Attach embeddings to articles
        for article, embedding in zip(articles, embeddings):
            if embedding:
                article["embedding"] = embedding
            else:
                logger.warning(f"Failed to embed: {article.get('title', '')}")
                # Still proceed; Chroma can work without pre-computed embeddings
                # (it will recompute on insert, which is slower)

        logger.info(f"✓ Embedded {sum(1 for a in articles if a.get('embedding'))} articles")
        return articles
