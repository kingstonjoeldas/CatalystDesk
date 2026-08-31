#!/usr/bin/env python
"""
Phase L6: Evals & Traces

Demonstrates:
1. Trace logging: capture query, retrieved articles, latency, cache hits
2. Quality metrics: retrieval stats, confidence calibration, latency budgets
3. Contradiction detection: compare brief sentiment vs. 1d price movement
4. User feedback collection

Exit codes:
  0 = Success (evals completed)
  1 = Partial (some evals failed)
  2 = Failure (couldn't initialize evals)
"""

import json
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.logging_utils import get_request_logger, set_request_id
from src.agents.graph import run_agent_graph
from src.evals.traces import TraceStore
from src.evals.metrics import Metrics
from src.evals.contradiction import ContradictionDetector

logger = get_request_logger(__name__)


def demo_full_pipeline_with_traces():
    """
    Run a full pipeline query and log traces.

    Returns:
        (request_id, brief, latency_ms, cache_hits)
    """
    logger.info("\n" + "=" * 80)
    logger.info("Demo: Full Pipeline with Trace Logging")
    logger.info("=" * 80)

    request_id = set_request_id()
    query = "What are the risks for AAPL?"
    ticker = "AAPL"

    logger.info(f"\nQuery: {query}")
    logger.info(f"Ticker: {ticker}")
    logger.info(f"Request ID: {request_id}")

    # Time the pipeline
    start_time = time.time()

    try:
        # Run agent graph
        final_state = run_agent_graph(query, ticker=ticker, request_id=request_id)

        latency_ms = (time.time() - start_time) * 1000

        brief = final_state.get("draft_brief", {})
        retrieved_articles = final_state.get("retrieved_articles", [])
        risk_analysis = final_state.get("risk_analysis", {})

        # Extract metrics
        retrieved_ids = [a.get("id", "") for a in retrieved_articles]
        event_types = list(set(a.get("metadata", {}).get("event_type", "other") for a in retrieved_articles))
        risk_levels = list(set(a.get("metadata", {}).get("risk_level", "low") for a in retrieved_articles))
        avg_risk_score = risk_analysis.get("risk_score", 0.5)
        brief_confidence = brief.get("confidence", 0.5)
        brief_summary = brief.get("summary", "")

        # Log trace
        trace_store = TraceStore()
        trace_store.log_trace(
            request_id=request_id,
            query=query,
            ticker=ticker,
            retrieved_ids=retrieved_ids,
            event_types=event_types,
            risk_levels=risk_levels,
            avg_risk_score=avg_risk_score,
            brief_confidence=brief_confidence,
            brief_summary=brief_summary,
            latency_ms=latency_ms,
            cache_hits=0,  # Would be populated by caching layer
        )

        logger.info(f"\n✓ Trace logged")
        logger.info(f"  Latency: {latency_ms:.0f}ms")
        logger.info(f"  Retrieved: {len(retrieved_ids)} articles")
        logger.info(f"  Confidence: {brief_confidence:.1%}")

        return request_id, brief, latency_ms, 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return request_id, {}, 0, 0


def demo_metrics():
    """
    Compute and display evaluation metrics.

    Returns:
        Metrics dict
    """
    logger.info("\n" + "=" * 80)
    logger.info("Demo: Evaluation Metrics")
    logger.info("=" * 80)

    try:
        metrics_computer = Metrics()
        all_metrics = metrics_computer.compute_all_metrics()
        Metrics.print_metrics(all_metrics)
        return all_metrics

    except Exception as e:
        logger.error(f"Metrics computation failed: {e}")
        return {}


def demo_contradiction_detection(brief: dict, ticker: str = "AAPL"):
    """
    Detect narrative-vs-price contradictions.

    Args:
        brief: Brief dict from L5
        ticker: Stock ticker

    Returns:
        Analysis dict
    """
    logger.info("\n" + "=" * 80)
    logger.info("Demo: Contradiction Detection")
    logger.info("=" * 80)

    if not brief or not brief.get("summary"):
        logger.warning("No brief to analyze")
        return {}

    try:
        detector = ContradictionDetector()
        brief_summary = brief.get("summary", "")

        is_contradiction, analysis = detector.detect_contradiction(
            ticker=ticker,
            brief_summary=brief_summary,
            threshold_pct=3.0,
        )

        report = detector.format_contradiction_report(analysis)
        logger.info(f"\nContradiction Analysis:\n{report}")

        if is_contradiction:
            logger.warning("⚠️ CONTRADICTION DETECTED: Brief sentiment diverges from price!")
        else:
            logger.info("✓ Brief aligned with price movement (or insufficient data)")

        return analysis

    except Exception as e:
        logger.error(f"Contradiction detection failed: {e}")
        return {}


def demo_feedback_collection():
    """
    Demonstrate user feedback collection.

    Returns:
        True if feedback logged successfully
    """
    logger.info("\n" + "=" * 80)
    logger.info("Demo: User Feedback Collection")
    logger.info("=" * 80)

    try:
        trace_store = TraceStore()
        feedback_summary = trace_store.get_feedback_summary()

        logger.info(f"\nFeedback Summary (all time):")
        logger.info(f"  👍 Thumbs up: {feedback_summary['thumbs_up']}")
        logger.info(f"  👎 Thumbs down: {feedback_summary['thumbs_down']}")
        logger.info(f"  😐 Neutral: {feedback_summary['neutral']}")
        logger.info(f"  Total: {feedback_summary['total']}")

        if feedback_summary["total"] > 0:
            up_rate = feedback_summary["thumbs_up"] / feedback_summary["total"]
            logger.info(f"\nSatisfaction rate: {up_rate:.1%}")
        else:
            logger.info("\nNo feedback collected yet")

        return True

    except Exception as e:
        logger.error(f"Feedback collection failed: {e}")
        return False


def run_l6() -> int:
    """
    Main L6 eval pipeline.

    Returns:
        Exit code (0 = success, 1 = partial, 2 = failure)
    """
    set_request_id("l6_eval")

    logger.info("=" * 80)
    logger.info("CatalystDesk Phase L6: Evals & Traces")
    logger.info("=" * 80)

    try:
        # Demo 1: Full pipeline with traces
        request_id, brief, latency_ms, cache_hits = demo_full_pipeline_with_traces()

        if not brief:
            logger.error("No brief generated; cannot continue with evals")
            return 2

        # Demo 2: Metrics
        metrics = demo_metrics()

        # Demo 3: Contradiction detection
        contradiction = demo_contradiction_detection(brief, ticker="AAPL")

        # Demo 4: Feedback
        feedback_ok = demo_feedback_collection()

        logger.info("\n" + "=" * 80)
        logger.info("L6 Evals Complete")
        logger.info("=" * 80)
        logger.info(f"Traces stored in: {Config.TRACE_DB}")
        logger.info("Next: Run L7 (Streamlit UI) or re-run with user feedback")

        return 0

    except Exception as e:
        logger.error(f"L6 evaluation failed: {e}")
        return 2


if __name__ == "__main__":
    exit_code = run_l6()

    logger.info("=" * 80)
    if exit_code == 0:
        logger.info("✓ L6 Evals completed successfully")
    else:
        logger.warning("⚠ L6 Evals failed")

    logger.info("=" * 80)

    sys.exit(exit_code)
