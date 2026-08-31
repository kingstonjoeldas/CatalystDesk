#!/usr/bin/env python
"""
Phase L5: LangGraph Agents (Brief Generation)

Runs the multi-agent pipeline: Researcher → Risk Officer → Brief Writer.
Demonstrates with sample queries.

Exit codes:
  0 = Success (briefs generated)
  1 = Partial (some briefs failed)
  2 = Failure (pipeline initialization failed)
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.logging_utils import get_request_logger, set_request_id
from src.agents.graph import run_agent_graph
from src.agents.state import AgentState, Brief

logger = get_request_logger(__name__)


def format_brief_for_display(brief: Brief) -> str:
    """
    Format a brief as readable markdown.

    Args:
        brief: Brief object

    Returns:
        Formatted string
    """
    lines = []

    # Header
    lines.append(f"# Brief for {brief.get('ticker', 'UNKNOWN')}")
    lines.append(f"\n**Question:** {brief.get('question', 'N/A')}")

    # Risk badge
    risk_level = brief.get("risk_level", "unknown").upper()
    risk_score = brief.get("risk_score", 0.0)
    confidence = brief.get("confidence", 0.0)
    lines.append(f"\n**Risk Level:** {risk_level} | Score: {risk_score:.2f}/1.0 | Confidence: {confidence:.0%}")

    # Summary
    lines.append(f"\n## Summary\n{brief.get('summary', '[No summary]')}")

    # Sections
    for section in brief.get("sections", []):
        heading = section.get("heading", "Section")
        content = section.get("content", "[No content]")
        lines.append(f"\n## {heading}\n{content}")

    lines.append(f"\n---\n*Request ID: {set_request_id()}*")

    return "\n".join(lines)


def demo_queries():
    """
    Run demonstrations with sample queries.

    Returns:
        Number of successful briefs
    """
    logger.info("=" * 80)
    logger.info("Brief Generation Demos")
    logger.info("=" * 80)

    queries = [
        ("What is the latest risk for AAPL?", "AAPL"),
        ("How do macro indicators affect SPY?", "SPY"),
        ("Tell me about earnings surprises", None),  # No ticker; let researcher extract
    ]

    successful = 0
    briefed_ids = []

    for i, (query, ticker) in enumerate(queries, 1):
        logger.info(f"\n[Demo {i}] Query: '{query}'")
        if ticker:
            logger.info(f"         Ticker: {ticker}")

        try:
            # Run agent graph
            final_state = run_agent_graph(query, ticker=ticker, request_id=f"demo_{i}")

            # Extract brief
            brief = final_state.get("draft_brief", {})

            if not brief:
                logger.warning("No brief generated")
                continue

            successful += 1
            briefed_ids.append(brief.get("id", f"demo_{i}"))

            # Display brief
            formatted = format_brief_for_display(brief)
            logger.info("\n" + formatted)

            # Trace
            steps = final_state.get("agent_steps", [])
            retrieval_count = final_state.get("retrieval_count", 0)
            logger.info(f"\nTrace: {' → '.join(steps)}")
            logger.info(f"Retrieved: {retrieval_count} articles")

        except Exception as e:
            logger.error(f"Demo {i} failed: {e}")
            continue

    return successful


def save_briefs_to_file(briefs: list[Brief]) -> Path:
    """
    Save generated briefs to JSON (for L6 evals).

    Args:
        briefs: List of Brief objects

    Returns:
        Path to saved file
    """
    output_path = Config.DATA_DIR / f"briefs_{len(briefs)}.json"

    with open(output_path, "w") as f:
        json.dump(briefs, f, indent=2)

    logger.info(f"Saved {len(briefs)} briefs to {output_path}")
    return output_path


def run_l5() -> tuple[list[Brief], int]:
    """
    Main L5 pipeline.

    Returns:
        (briefs, exit_code)
    """
    set_request_id("l5_briefs")

    logger.info("=" * 80)
    logger.info("CatalystDesk Phase L5: LangGraph Agents")
    logger.info("=" * 80)

    logger.info("\nPipeline: Researcher → Risk Officer → Brief Writer")
    logger.info("State machine: explicit transitions, no loops")

    # Run demos
    successful = demo_queries()

    logger.info("\n" + "=" * 80)
    logger.info("L5 Summary")
    logger.info("=" * 80)
    logger.info(f"Demo queries: 3")
    logger.info(f"Successful briefs: {successful}")
    logger.info(f"Success rate: {successful/3:.0%}")

    if successful == 0:
        logger.error("No briefs generated")
        return [], 2

    exit_code = 0 if successful == 3 else 1
    return [], exit_code


if __name__ == "__main__":
    briefs, exit_code = run_l5()

    logger.info("=" * 80)
    if exit_code == 0:
        logger.info("✓ L5 Agents completed successfully")
    else:
        logger.warning("⚠ L5 Agents partial (some demos failed)")

    logger.info("=" * 80)

    sys.exit(exit_code)
