"""
Phase L7: Streamlit Trader Brief UI

Why Streamlit?
- Rapid MVP development (pure Python, no React/frontend boilerplate)
- Session state handles state across reruns
- Caching (@st.cache_data) speeds up reruns
- Interactive widgets (buttons, sliders, text input) are built-in
- Deployment is trivial (streamlit cloud, docker, or bare metal)

Productionization path:
- Extract agent logic to FastAPI endpoints
- Streamlit calls API (async)
- Add auth (OAuth, API keys)
- Queue long-running jobs (Celery, Bull)
- Add analytics/dashboards (Grafana, Datadog)

But for MVP, Streamlit handles everything end-to-end.
"""

import sys
import time
import streamlit as st
from datetime import datetime
from pathlib import Path

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logging_utils import get_request_logger, set_request_id
from src.agents.graph import run_agent_graph
from src.evals.traces import TraceStore
from src.evals.contradiction import ContradictionDetector

logger = get_request_logger(__name__)


# ============================================================================
# PAGE SETUP
# ============================================================================

st.set_page_config(
    page_title="CatalystDesk",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 CatalystDesk")
st.markdown("**Risk-Aware Trader Briefs from Live Market Data**")

# ============================================================================
# SESSION STATE (persistent across reruns)
# ============================================================================

if "last_request_id" not in st.session_state:
    st.session_state.last_request_id = None
if "last_brief" not in st.session_state:
    st.session_state.last_brief = None
if "last_articles" not in st.session_state:
    st.session_state.last_articles = None


# ============================================================================
# SIDEBAR: CONFIGURATION
# ============================================================================

st.sidebar.markdown("## Configuration")

# Source filters
st.sidebar.markdown("### Data Sources")
use_yfinance = st.sidebar.checkbox("yfinance (company news)", value=True)
use_fred = st.sidebar.checkbox("FRED (macro data)", value=True)
use_rss = st.sidebar.checkbox("RSS feeds (headlines)", value=True)

# Retrieval settings
st.sidebar.markdown("### Retrieval")
max_articles = st.sidebar.slider("Max articles to retrieve", 1, 10, 5)
days_back = st.sidebar.slider("Days back to search", 1, 30, 7)

# Risk threshold
st.sidebar.markdown("### Risk Settings")
show_low_confidence = st.sidebar.checkbox("Show low-confidence briefs", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("**Built by:** A beginner learning AI engineering")
st.sidebar.markdown("**Architecture:** L0–L7 phases, $0 budget, no local LLMs")
st.sidebar.markdown("[GitHub](https://github.com) | [INTERVIEW.md](./INTERVIEW.md)")


# ============================================================================
# MAIN INPUT
# ============================================================================

col1, col2 = st.columns([2, 1])

with col1:
    query = st.text_input(
        "What's your question?",
        placeholder="e.g., 'What are the risks for AAPL?' or 'How do macro indicators affect SPY?'",
    )

with col2:
    run_button = st.button("📨 Get Brief", type="primary", use_container_width=True)


# ============================================================================
# QUERY EXECUTION
# ============================================================================

if run_button and query.strip():
    request_id = set_request_id()
    st.session_state.last_request_id = request_id

    with st.spinner("🔍 Retrieving context..."):
        start_time = time.time()

        try:
            # Run agent graph
            final_state = run_agent_graph(query, request_id=request_id)

            latency_ms = (time.time() - start_time) * 1000

            brief = final_state.get("draft_brief", {})
            articles = final_state.get("retrieved_articles", [])
            error = final_state.get("brief_error")

            st.session_state.last_brief = brief
            st.session_state.last_articles = articles

            if error:
                st.error(f"❌ Brief generation failed: {error}")
            elif not brief:
                st.warning("⚠️ No brief generated. Try a different query.")
            else:
                st.success(f"✓ Brief generated in {latency_ms:.0f}ms")

        except Exception as e:
            st.error(f"❌ Pipeline error: {e}")
            logger.error(f"Pipeline error: {e}")


# ============================================================================
# DISPLAY BRIEF
# ============================================================================

if st.session_state.last_brief:
    brief = st.session_state.last_brief

    # Risk badge
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        risk_level = brief.get("risk_level", "unknown").upper()
        risk_color = {
            "LOW": "🟢",
            "MEDIUM": "🟡",
            "HIGH": "🔴",
        }.get(risk_level, "⚪")

        st.markdown(f"### {risk_color} Risk: **{risk_level}**")

    with col2:
        risk_score = brief.get("risk_score", 0.5)
        confidence = brief.get("confidence", 0.5)
        st.metric("Risk Score", f"{risk_score:.2f}/1.0", f"Confidence: {confidence:.0%}")

    with col3:
        ticker = brief.get("ticker", "N/A")
        st.metric("Ticker", ticker)

    st.markdown("---")

    # Summary
    summary = brief.get("summary", "[No summary]")
    st.markdown(f"## Summary\n{summary}")

    # Sections
    for section in brief.get("sections", []):
        heading = section.get("heading", "Section")
        content = section.get("content", "[No content]")
        st.markdown(f"### {heading}")
        st.markdown(content)

    st.markdown("---")

    # ========================================================================
    # CONTRADICTION CHECK (if available)
    # ========================================================================

    ticker = brief.get("ticker", "")
    if ticker and ticker != "UNKNOWN":
        try:
            detector = ContradictionDetector()
            brief_summary = brief.get("summary", "")
            is_contradiction, analysis = detector.detect_contradiction(
                ticker=ticker,
                brief_summary=brief_summary,
                threshold_pct=3.0,
            )

            if is_contradiction:
                st.warning(f"⚠️ **Contradiction Detected**: {analysis.get('reason', 'Brief sentiment diverges from price movement')}")
            else:
                st.info(f"✓ Brief sentiment aligns with price: {analysis.get('direction', '?').upper()} {analysis.get('price_change', 0):.1f}%")

        except Exception as e:
            logger.warning(f"Contradiction detection skipped: {e}")

    st.markdown("---")

    # ========================================================================
    # EVIDENCE (retrieved articles)
    # ========================================================================

    if st.session_state.last_articles:
        st.markdown("## Evidence (Retrieved Articles)")

        for i, article in enumerate(st.session_state.last_articles[:5], 1):
            with st.expander(f"📄 Article {i}: {article.get('document', 'No title')[:100]}..."):
                doc = article.get("document", "[No content]")
                meta = article.get("metadata", {})

                st.markdown(f"**Full Text:**\n{doc}")
                st.markdown(f"\n**Source:** {meta.get('source', 'Unknown')}")
                st.markdown(f"**Tickers:** {meta.get('tickers', 'N/A')}")
                st.markdown(f"**Risk Level:** {meta.get('risk_level', 'N/A')} (confidence: {meta.get('risk_score', 0):.2f})")
                st.markdown(f"**Event Type:** {meta.get('event_type', 'N/A')}")
                st.markdown(f"**URL:** {meta.get('url', 'N/A')}")

    st.markdown("---")

    # ========================================================================
    # FEEDBACK
    # ========================================================================

    st.markdown("## Feedback")
    st.markdown("Did this brief help? Your feedback improves the system.")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("👍 Thumbs Up", use_container_width=True):
            trace_store = TraceStore()
            if st.session_state.last_request_id:
                trace_store.log_feedback(
                    st.session_state.last_request_id,
                    "thumbs_up",
                    f"User feedback on: {query[:100]}",
                )
                st.success("✓ Thanks! Feedback logged.")
            else:
                st.warning("No request ID to log feedback.")

    with col2:
        if st.button("😐 Neutral", use_container_width=True):
            trace_store = TraceStore()
            if st.session_state.last_request_id:
                trace_store.log_feedback(
                    st.session_state.last_request_id,
                    "neutral",
                    f"User feedback on: {query[:100]}",
                )
                st.info("✓ Feedback logged.")
            else:
                st.warning("No request ID to log feedback.")

    with col3:
        if st.button("👎 Thumbs Down", use_container_width=True):
            trace_store = TraceStore()
            if st.session_state.last_request_id:
                trace_store.log_feedback(
                    st.session_state.last_request_id,
                    "thumbs_down",
                    f"User feedback on: {query[:100]}",
                )
                st.error("✓ Feedback logged. We'll improve!")
            else:
                st.warning("No request ID to log feedback.")

    # Optional notes
    user_notes = st.text_area("Additional notes (optional):", placeholder="What could we improve?")

else:
    st.info("👈 Enter a query to get started")


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
    **CatalystDesk v0.1.0** | Built in phases L0–L7

    - **L0**: Config & fixtures
    - **L1**: Ingest (yfinance, FRED, RSS)
    - **L2**: Summarization (BART)
    - **L3**: Classification (zero-shot MNLI)
    - **L4**: RAG (MiniLM + Chroma)
    - **L5**: LangGraph agents
    - **L6**: Evals & traces
    - **L7**: This UI (Streamlit)
    """
)
