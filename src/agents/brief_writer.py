"""
Brief Writer agent: composes the final trader brief.

Role in the pipeline:
1. Takes risk_analysis + retrieved_articles
2. Composes a structured, actionable brief
3. No hallucination (only from provided data)
4. Returns final Brief object

Why separate from Risk Officer?
- Output formatting is distinct from analysis
- Easy to swap in an LLM here later (currently templated)
- Keeps agent responsibilities focused
"""

from typing import Optional

from src.agents.state import AgentState, Brief, BriefSection
from src.logging_utils import get_request_logger

logger = get_request_logger(__name__)


class BriefWriterAgent:
    """Output formatter: composes trader briefs from analysis."""

    def __init__(self):
        """Initialize brief writer (no external dependencies)."""
        pass

    @staticmethod
    def format_concern(concern: str, max_length: int = 200) -> str:
        """Format a concern for the brief."""
        return concern[:max_length] + ("..." if len(concern) > max_length else "")

    def run(self, state: AgentState) -> AgentState:
        """
        Brief Writer agent function (runs in LangGraph).

        Args:
            state: Current agent state (with risk_analysis + articles filled)

        Returns:
            Updated state with draft_brief filled
        """
        query = state.get("query", "Query not provided")
        ticker = state.get("ticker", "UNKNOWN")
        articles = state.get("retrieved_articles", [])
        risk_analysis = state.get("risk_analysis", {})
        retrieval_error = state.get("retrieval_error")
        analysis_error = state.get("analysis_error")

        logger.info(f"Brief Writer: composing brief for {ticker}")

        try:
            # Determine confidence (lower if there were errors or few articles)
            if retrieval_error or analysis_error:
                confidence = 0.5
            elif len(articles) < 2:
                confidence = 0.6
            else:
                confidence = 0.8

            # Extract source IDs for citation
            source_ids = [a.get("id", "") for a in articles[:5]]

            # One-liner summary
            risk_level = risk_analysis.get("overall_risk", "low")
            risk_score = risk_analysis.get("risk_score", 0.5)
            summary = (
                f"Risk for {ticker}: **{risk_level.upper()}** (confidence: {confidence:.1%}). "
                f"Based on {len(articles)} recent articles spanning {len(set(a.get('metadata', {}).get('event_type') for a in articles))} event types. "
                f"Average risk score: {risk_score:.2f}/1.0."
            )

            # Build sections
            sections: list[BriefSection] = []

            # Macro Context
            if risk_analysis.get("macro_context"):
                sections.append({
                    "heading": "Macro Context",
                    "content": risk_analysis.get("macro_context", "[No context]"),
                })

            # Event Summary
            if risk_analysis.get("event_summary"):
                sections.append({
                    "heading": "What's Happening",
                    "content": risk_analysis.get("event_summary", "[No summary]"),
                })

            # Concerns
            concerns = risk_analysis.get("concerns", [])
            if concerns:
                concern_text = "\n".join(
                    f"- {self.format_concern(c)}" for c in concerns[:3]
                )
                sections.append({
                    "heading": "Key Risks",
                    "content": concern_text,
                })

            # Opportunities
            opportunities = risk_analysis.get("opportunities", [])
            if opportunities:
                opp_text = "\n".join(
                    f"- {self.format_concern(o)}" for o in opportunities[:3]
                )
                sections.append({
                    "heading": "Potential Upside",
                    "content": opp_text,
                })

            # Sources
            if source_ids:
                sections.append({
                    "heading": "Sources",
                    "content": f"Retrieved {len(articles)} relevant articles. IDs: {', '.join(source_ids[:3])}",
                })

            # Errors (if any)
            if retrieval_error or analysis_error:
                error_msg = retrieval_error or analysis_error or "[Unknown error]"
                sections.append({
                    "heading": "⚠️ Warnings",
                    "content": f"Some pipeline stages failed: {error_msg}. Brief may be incomplete.",
                })

            draft_brief: Brief = {
                "ticker": ticker,
                "question": query,
                "risk_level": risk_level,
                "risk_score": risk_score,
                "summary": summary,
                "sections": sections,
                "sources": source_ids,
                "confidence": confidence,
            }

            state["draft_brief"] = draft_brief
            state["agent_steps"].append("brief_writer")

            logger.info(f"✓ Brief Writer: {len(sections)} sections, confidence={confidence:.1%}")

        except Exception as e:
            logger.error(f"✗ Brief Writer failed: {e}")
            state["brief_error"] = str(e)
            state["draft_brief"] = {
                "ticker": ticker,
                "question": query,
                "risk_level": "unknown",
                "risk_score": 0.0,
                "summary": f"[Brief generation failed: {e}]",
                "sections": [],
                "sources": [],
                "confidence": 0.0,
            }
            state["agent_steps"].append("brief_writer_failed")

        return state
