"""
Risk Officer agent: analyzes risk from retrieved articles.

Role in the pipeline:
1. Takes retrieved articles from Researcher
2. Extracts event types, risk levels, macro context
3. Composes a risk assessment (concerns, opportunities, macro context)
4. Returns updated state for Brief Writer

Why separate from Brief Writer?
- Risk analysis is distinct from writing
- Can apply trading domain logic here (e.g., "monetary policy + rising CPI = high risk")
- Easy to swap in a fine-tuned risk classifier or an LLM here later
"""

from typing import Optional

from src.agents.state import AgentState, RiskAnalysis
from src.logging_utils import get_request_logger

logger = get_request_logger(__name__)


class RiskOfficerAgent:
    """Risk analyst: interprets event types and risk labels."""

    # Mapping: event_type -> default risk level boost
    RISK_MULTIPLIER = {
        "monetary policy": 0.7,  # Affects all assets
        "inflation data": 0.6,   # Bond/equity sensitivity
        "earnings": 0.4,         # Company-specific
        "geopolitics": 0.8,      # Market-wide shock
        "liquidity/credit": 0.9, # Systemic risk
        "regulation": 0.6,       # Sector/company dependent
        "other": 0.3,
    }

    def __init__(self):
        """Initialize risk officer (no external dependencies)."""
        pass

    @staticmethod
    def normalize_risk_score(level: str, confidence: float = 0.5) -> float:
        """
        Convert risk level to numeric score.

        Args:
            level: "low", "medium", "high"
            confidence: Confidence in the classification (0-1)

        Returns:
            Numeric score (0.0 - 1.0)
        """
        base = {"low": 0.2, "medium": 0.5, "high": 0.8}.get(level, 0.5)
        # Confidence adjusts the score (low confidence -> regress to 0.5)
        return base * confidence + 0.5 * (1 - confidence)

    def run(self, state: AgentState) -> AgentState:
        """
        Risk Officer agent function (runs in LangGraph).

        Args:
            state: Current agent state (with retrieved_articles filled)

        Returns:
            Updated state with risk_analysis filled
        """
        articles = state.get("retrieved_articles", [])
        ticker = state.get("ticker", "UNKNOWN")

        logger.info(f"Risk Officer: analyzing {len(articles)} articles for {ticker}")

        if not articles:
            logger.warning("No articles to analyze")
            state["risk_analysis"] = {
                "overall_risk": "low",
                "risk_score": 0.3,
                "macro_context": "[No articles to analyze]",
                "event_summary": "[No data]",
                "concerns": [],
                "opportunities": [],
            }
            state["agent_steps"].append("risk_officer_no_data")
            return state

        try:
            # Extract risk signals from articles
            event_types = []
            risk_levels = []
            risk_scores = []
            concerns = []
            opportunities = []

            for article in articles:
                meta = article.get("metadata", {})
                event_type = meta.get("event_type", "other")
                risk_level = meta.get("risk_level", "low")
                risk_score = meta.get("risk_score", 0.5)
                doc = article.get("document", "")[:200]

                event_types.append(event_type)
                risk_levels.append(risk_level)
                risk_scores.append(float(risk_score))

                # Heuristic: if risk is high, add to concerns; if low, to opportunities
                if risk_level == "high":
                    concerns.append(f"{event_type}: {doc}")
                elif risk_level == "low":
                    opportunities.append(f"{event_type}: {doc}")

            # Compute overall risk
            avg_risk_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0.5
            overall_risk = (
                "high" if avg_risk_score > 0.65
                else "medium" if avg_risk_score > 0.35
                else "low"
            )

            # Compose macro context
            unique_events = set(event_types)
            macro_context = f"Key events: {', '.join(unique_events)}. Average risk score: {avg_risk_score:.2f}"

            risk_analysis: RiskAnalysis = {
                "overall_risk": overall_risk,
                "risk_score": avg_risk_score,
                "macro_context": macro_context,
                "event_summary": f"Found {len(event_types)} articles covering {len(unique_events)} event types",
                "concerns": concerns[:3],  # Top 3
                "opportunities": opportunities[:3],
            }

            state["risk_analysis"] = risk_analysis
            state["agent_steps"].append("risk_officer")

            logger.info(
                f"✓ Risk Officer: overall_risk={overall_risk}, score={avg_risk_score:.2f}"
            )

        except Exception as e:
            logger.error(f"✗ Risk Officer failed: {e}")
            state["analysis_error"] = str(e)
            state["risk_analysis"] = {
                "overall_risk": "low",
                "risk_score": 0.5,
                "macro_context": "[Analysis failed]",
                "event_summary": "[Error during analysis]",
                "concerns": [str(e)],
                "opportunities": [],
            }
            state["agent_steps"].append("risk_officer_failed")

        return state
