"""
Typed state objects for LangGraph agents.

Why TypedDict for state?
- LangGraph requires state to be serializable and typed
- TypedDict is lightweight, JSON-serializable, type-hinted
- Easier to reason about state transitions than plain dicts
- mypy can catch type errors in agent functions
"""

from typing import TypedDict, Optional


class Article(TypedDict, total=False):
    """Retrieved article with metadata."""

    id: str
    document: str
    distance: float
    metadata: dict


class RiskAnalysis(TypedDict, total=False):
    """Risk assessment output."""

    overall_risk: str  # "low", "medium", "high"
    risk_score: float  # 0.0 - 1.0
    macro_context: str  # Summary of macro conditions
    event_summary: str  # What events are happening
    concerns: list[str]  # Specific risk factors
    opportunities: list[str]  # Potential upside


class BriefSection(TypedDict, total=False):
    """One section of the trader brief."""

    heading: str
    content: str


class Brief(TypedDict, total=False):
    """Final trader brief output."""

    ticker: str
    question: str
    risk_level: str  # "low", "medium", "high"
    risk_score: float
    summary: str  # 1-2 sentence executive summary
    sections: list[BriefSection]  # Detailed sections
    sources: list[str]  # Retrieved article IDs
    confidence: float  # 0.0 - 1.0 (how confident in this brief)


class AgentState(TypedDict, total=False):
    """
    LangGraph state object for the multi-agent pipeline.

    Passed between Researcher -> Risk Officer -> Brief Writer.
    """

    # Input
    query: str  # User question, e.g., "What's the risk for AAPL?"
    ticker: Optional[str]  # Stock ticker, e.g., "AAPL" (extracted from query if not explicit)

    # Researcher output
    retrieved_articles: list[Article]  # Top-K articles from semantic search
    retrieval_count: int  # How many articles were retrieved
    retrieval_error: Optional[str]  # Error message if retrieval failed

    # Risk Officer output
    risk_analysis: RiskAnalysis  # Structured risk assessment
    analysis_error: Optional[str]  # Error message if analysis failed

    # Brief Writer output
    draft_brief: Brief  # Final trader brief
    brief_error: Optional[str]  # Error message if writing failed

    # Metadata
    request_id: str  # For tracing
    agent_steps: list[str]  # Log of which agents ran
