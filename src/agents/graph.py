"""
LangGraph state machine orchestration.

Why LangGraph?
- Explicit control flow (not hidden loops or prompt injection)
- Debuggable state transitions
- Easy to add agents, modify transitions
- Enforces max_iterations to prevent infinite loops

Pipeline:
  Input (query, ticker)
    ↓
  Researcher (retrieval)
    ↓
  Risk Officer (analysis)
    ↓
  Brief Writer (output)
    ↓
  Output (brief)
"""

from langgraph.graph import StateGraph, END

from src.agents.state import AgentState
from src.agents.researcher import ResearcherAgent
from src.agents.risk_officer import RiskOfficerAgent
from src.agents.brief_writer import BriefWriterAgent
from src.logging_utils import get_request_logger

logger = get_request_logger(__name__)


def build_graph():
    """
    Build LangGraph state machine.

    Returns:
        Compiled graph (callable)
    """
    # Initialize agents
    researcher = ResearcherAgent()
    risk_officer = RiskOfficerAgent()
    brief_writer = BriefWriterAgent()

    # Create graph
    graph = StateGraph(AgentState)

    # Add nodes (agents)
    graph.add_node("researcher", researcher.run)
    graph.add_node("risk_officer", risk_officer.run)
    graph.add_node("brief_writer", brief_writer.run)

    # Add edges (transitions)
    graph.add_edge("researcher", "risk_officer")
    graph.add_edge("risk_officer", "brief_writer")
    graph.add_edge("brief_writer", END)

    # Set entry point
    graph.set_entry_point("researcher")

    # Compile
    compiled = graph.compile()
    logger.info("✓ LangGraph compiled")

    return compiled


def run_agent_graph(query: str, ticker: str = None, request_id: str = "unknown") -> AgentState:
    """
    Run the multi-agent pipeline.

    Args:
        query: User question (e.g., "What's the risk for AAPL?")
        ticker: Stock ticker (optional; extracted from query if not provided)
        request_id: Request ID for tracing

    Returns:
        Final state with draft_brief filled
    """
    logger.info(f"Starting agent graph: query='{query}', ticker={ticker}")

    # Initialize state
    initial_state: AgentState = {
        "query": query,
        "ticker": ticker,
        "retrieved_articles": [],
        "retrieval_count": 0,
        "retrieval_error": None,
        "risk_analysis": {},
        "analysis_error": None,
        "draft_brief": {},
        "brief_error": None,
        "request_id": request_id,
        "agent_steps": [],
    }

    # Build and run graph
    graph = build_graph()

    try:
        final_state = graph.invoke(initial_state)
        logger.info(f"✓ Agent graph completed: {' → '.join(final_state.get('agent_steps', []))}")
        return final_state

    except Exception as e:
        logger.error(f"✗ Agent graph failed: {e}")
        initial_state["brief_error"] = f"Pipeline failed: {e}"
        initial_state["agent_steps"].append("graph_failed")
        return initial_state
