"""
Quality metrics and evaluation functions.

Why these metrics?
- Retrieval hit rate: Does retrieval find relevant articles?
- Confidence calibration: Are high-confidence briefs actually good?
- Latency: Is the pipeline fast enough?
- Cache effectiveness: How many API calls are we saving?

No gold labels; these are indirect signals.
"""

from typing import Optional

from src.evals.traces import TraceStore
from src.logging_utils import get_request_logger

logger = get_request_logger(__name__)


class Metrics:
    """Compute evaluation metrics from traces."""

    def __init__(self, trace_store: Optional[TraceStore] = None):
        """
        Initialize metrics computer.

        Args:
            trace_store: TraceStore instance. If None, creates new one.
        """
        self.trace_store = trace_store or TraceStore()

    def compute_retrieval_stats(self) -> dict:
        """
        Compute retrieval statistics.

        Returns:
            Dict with keys: avg_retrieved, min_retrieved, max_retrieved, retrievals_with_zero
        """
        traces = self.trace_store.get_all_traces(limit=1000)

        if not traces:
            return {
                "avg_retrieved": 0,
                "min_retrieved": 0,
                "max_retrieved": 0,
                "retrievals_with_zero": 0,
                "total_traces": 0,
            }

        retrieval_counts = [t["retrieved_count"] for t in traces if t["retrieved_count"] is not None]

        if not retrieval_counts:
            return {
                "avg_retrieved": 0,
                "min_retrieved": 0,
                "max_retrieved": 0,
                "retrievals_with_zero": 0,
                "total_traces": len(traces),
            }

        zero_count = sum(1 for c in retrieval_counts if c == 0)

        return {
            "avg_retrieved": sum(retrieval_counts) / len(retrieval_counts),
            "min_retrieved": min(retrieval_counts),
            "max_retrieved": max(retrieval_counts),
            "retrievals_with_zero": zero_count,
            "total_traces": len(traces),
        }

    def compute_confidence_stats(self) -> dict:
        """
        Compute brief confidence statistics.

        Returns:
            Dict with keys: avg_confidence, min_confidence, max_confidence, low_confidence_count
        """
        traces = self.trace_store.get_all_traces(limit=1000)

        if not traces:
            return {
                "avg_confidence": 0,
                "min_confidence": 0,
                "max_confidence": 0,
                "low_confidence_count": 0,
                "total_traces": 0,
            }

        confidences = [t["brief_confidence"] for t in traces if t["brief_confidence"] is not None]

        if not confidences:
            return {
                "avg_confidence": 0,
                "min_confidence": 0,
                "max_confidence": 0,
                "low_confidence_count": 0,
                "total_traces": len(traces),
            }

        low_count = sum(1 for c in confidences if c < 0.5)

        return {
            "avg_confidence": sum(confidences) / len(confidences),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
            "low_confidence_count": low_count,
            "total_traces": len(traces),
        }

    def compute_latency_stats(self) -> dict:
        """
        Compute latency statistics (milliseconds).

        Returns:
            Dict with keys: avg_latency_ms, p50_latency_ms, p95_latency_ms, slow_queries_count
        """
        traces = self.trace_store.get_all_traces(limit=1000)

        if not traces:
            return {
                "avg_latency_ms": 0,
                "p50_latency_ms": 0,
                "p95_latency_ms": 0,
                "slow_queries_count": 0,
                "total_traces": 0,
            }

        latencies = sorted(
            [t["latency_ms"] for t in traces if t["latency_ms"] is not None]
        )

        if not latencies:
            return {
                "avg_latency_ms": 0,
                "p50_latency_ms": 0,
                "p95_latency_ms": 0,
                "slow_queries_count": 0,
                "total_traces": len(traces),
            }

        p50_idx = len(latencies) // 2
        p95_idx = int(len(latencies) * 0.95)
        slow_count = sum(1 for l in latencies if l > 5000)  # >5s is slow

        return {
            "avg_latency_ms": sum(latencies) / len(latencies),
            "p50_latency_ms": latencies[p50_idx],
            "p95_latency_ms": latencies[min(p95_idx, len(latencies) - 1)],
            "slow_queries_count": slow_count,
            "total_traces": len(traces),
        }

    def compute_cache_stats(self) -> dict:
        """
        Compute cache effectiveness statistics.

        Returns:
            Dict with keys: total_cache_hits, avg_cache_hits_per_query, cache_hit_rate
        """
        traces = self.trace_store.get_all_traces(limit=1000)

        if not traces:
            return {
                "total_cache_hits": 0,
                "avg_cache_hits_per_query": 0,
                "cache_hit_rate": 0.0,
                "total_traces": 0,
            }

        cache_hits = sum(t.get("cache_hits", 0) or 0 for t in traces)
        total_queries = len(traces)

        return {
            "total_cache_hits": cache_hits,
            "avg_cache_hits_per_query": cache_hits / total_queries if total_queries > 0 else 0,
            "cache_hit_rate": (cache_hits / (total_queries * 3)) if total_queries > 0 else 0,  # 3 stages
            "total_traces": total_queries,
        }

    def compute_all_metrics(self) -> dict:
        """
        Compute all metrics.

        Returns:
            Dict with keys: retrieval, confidence, latency, cache, feedback
        """
        return {
            "retrieval": self.compute_retrieval_stats(),
            "confidence": self.compute_confidence_stats(),
            "latency": self.compute_latency_stats(),
            "cache": self.compute_cache_stats(),
            "feedback": self.trace_store.get_feedback_summary(),
        }

    @staticmethod
    def print_metrics(metrics: dict) -> None:
        """
        Pretty-print metrics.

        Args:
            metrics: Dict from compute_all_metrics()
        """
        logger.info("\n" + "=" * 80)
        logger.info("Evaluation Metrics")
        logger.info("=" * 80)

        # Retrieval
        ret = metrics.get("retrieval", {})
        logger.info(f"\nRetrieval:")
        logger.info(f"  Avg articles retrieved: {ret.get('avg_retrieved', 0):.1f}")
        logger.info(f"  Queries with zero results: {ret.get('retrievals_with_zero', 0)}")
        logger.info(f"  Total queries: {ret.get('total_traces', 0)}")

        # Confidence
        conf = metrics.get("confidence", {})
        logger.info(f"\nBrief Confidence:")
        logger.info(f"  Avg confidence: {conf.get('avg_confidence', 0):.2f}")
        logger.info(f"  Low confidence (<0.5): {conf.get('low_confidence_count', 0)}")

        # Latency
        lat = metrics.get("latency", {})
        logger.info(f"\nLatency (ms):")
        logger.info(f"  Avg: {lat.get('avg_latency_ms', 0):.0f}")
        logger.info(f"  p50: {lat.get('p50_latency_ms', 0):.0f}")
        logger.info(f"  p95: {lat.get('p95_latency_ms', 0):.0f}")
        logger.info(f"  >5000ms (slow): {lat.get('slow_queries_count', 0)}")

        # Cache
        cache = metrics.get("cache", {})
        logger.info(f"\nCache:")
        logger.info(f"  Total hits: {cache.get('total_cache_hits', 0)}")
        logger.info(f"  Hit rate: {cache.get('cache_hit_rate', 0):.1%}")

        # Feedback
        feedback = metrics.get("feedback", {})
        logger.info(f"\nUser Feedback:")
        logger.info(f"  👍 Thumbs up: {feedback.get('thumbs_up', 0)}")
        logger.info(f"  👎 Thumbs down: {feedback.get('thumbs_down', 0)}")
        logger.info(f"  😐 Neutral: {feedback.get('neutral', 0)}")
        logger.info(f"  Total feedback: {feedback.get('total', 0)}")

        logger.info("=" * 80)
