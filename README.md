# CatalystDesk

**A $0 LangGraph + RAG briefing system that turns live market news and economic data into risk-tagged trader briefs.**

## Vision

CatalystDesk reads global news (yfinance, FRED, RSS feeds), extracts risk signals using Hugging Face small models (summarization + zero-shot classification), retrieves semantic context via MiniLM embeddings and Chroma, and composes trader briefs via a LangGraph 3-agent state machine—all offline-first, locally cached, no paid APIs.

**Resume Arc:** AI engineering (RAG, LangGraph, embeddings, caching) + Trading logic (macro indicators, risk labels, narrative-vs-price contradiction checks) + Python SWE (Streamlit, structured state, typed models, evals).

---

## Quick Start

### Prerequisites
- Python 3.11+
- ~2GB free disk (for Chroma vectors + HF cache)
- Internet for Hugging Face Inference API (first run only; cached thereafter)

### Setup

```bash
# Clone and navigate
cd d:\AI\ projects\CatalystDesk

# Create venv
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and populate env
copy .env.example .env
# Edit .env:
#   HF_TOKEN=<your token from https://huggingface.co/settings/tokens>
#   FRED_API_KEY=<optional; if empty, FRED data skipped gracefully>
```

### First Run: Phase L0 (This Phase)

Verify the skeleton and sample fixtures:
```bash
python scripts/run_l0_verify.py
```

Expected output: ✓ checks for config, logging, sample data.

---

## Phase Map

| Phase | Goal | Key Files | Interview Angle |
|-------|------|-----------|-----------------|
| **L0** | Skeleton, fixtures, config | `config.py`, `logging_utils.py`, `data/samples/`, `INTERVIEW.md` | Architecture, why phases, why no ORMs |
| **L1** | Ingest (yfinance, FRED, RSS) | `src/ingest/*.py`, `scripts/run_l1_ingest.py` | Rate limits, schema drift, adapter pattern |
| **L2** | HF Summarization (BART) | `src/hf_tasks/client.py`, `summarize.py`, `scripts/run_l2_summarize.py` | Why task models; cold starts; cache keys |
| **L3** | Zero-Shot Classification | `classify.py`, `scripts/run_l3_classify.py` | Zero-shot vs fine-tune; label design; confidence thresholds |
| **L4** | RAG (MiniLM + Chroma) | `src/rag/embed.py`, `store.py`, `retrieve.py`, `scripts/run_l4_rag.py` | Embeddings > keyword; why Chroma; retrieval eval |
| **L5** | LangGraph Agents (3-stage) | `src/agents/{state,researcher,risk_officer,brief_writer,graph}.py`, `scripts/run_l5_brief.py` | Why 3 agents; state machines; loop prevention |
| **L6** | Evals & Traces | `src/evals/`, `contradiction.py`, `scripts/run_l6_eval.py` | Offline eval; feedback loop; contradiction detection |
| **L7** | Streamlit UI | `app/streamlit_app.py` | Why Streamlit v1; productionization path (FastAPI later) |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        PHASE L7: STREAMLIT                      │
│          (Ticker input → Brief + Evidence + Risk badges)        │
└──────────────────────────┬──────────────────────────────────────┘
                           │ (query)
┌──────────────────────────▼──────────────────────────────────────┐
│                      PHASE L5: LANGGRAPH                        │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ Researcher  │ → │ Risk Officer │ → │ Brief Writer │       │
│  │(retrieves)  │    │(map labels)  │    │(structured)  │       │
│  └─────────────┘    └──────────────┘    └──────────────┘       │
└──────────────────────────┬──────────────────────────────────────┘
                           │ (state)
        ┌──────────────────┼──────────────────┐
        │                  │                  │
┌───────▼──────┐  ┌────────▼────────┐  ┌────▼──────────┐
│  PHASE L4:   │  │   PHASE L3:     │  │  PHASE L6:    │
│  RAG/Chroma  │  │  Zero-Shot (risk)  │  Evals/Traces │
│  (retrieve)  │  │  + Summarize    │  │  (feedback)   │
└───────┬──────┘  └────────┬────────┘  └────┬──────────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │ (normalized items)
                    ┌──────▼──────────┐
                    │   PHASE L1:     │
                    │   Ingest        │
                    │ (yfinance,FRED, │
                    │  RSS) + normalize
                    └─────────────────┘
                           │ (live data)
                    ┌──────▼──────────┐
                    │ External APIs   │
                    │ (cached locally)│
                    └─────────────────┘

Caching Layer: Every HF call → disk cache (hash-based keys)
Degradation: No API → sample fixtures from data/samples/
Eval feedback: SQLite trace log for iterative improvement
```

---

## Environment Variables

See `.env.example`:

```env
# Hugging Face (required)
HF_TOKEN=<your token>
HF_CACHE_DIR=./cache/hf
HF_HOME=./cache/hf

# FRED (optional; gracefully skipped if missing)
FRED_API_KEY=<optional>

# Cache
CACHE_DIR=./cache
CACHE_BACKEND=sqlite  # sqlite or none

# Logging
LOG_LEVEL=INFO
LOG_DIR=./logs

# Chroma (RAG)
CHROMA_DB_DIR=./data/chroma

# Evals
TRACE_DB=./data/traces.db
```

---

## Key Design Decisions (Explained for Interviews)

### 1. **No Local LLM Runtime**
**Why:** A 7B LLM on a low-end PC = 2GB VRAM (if quantized), latency >5s/inference. Hugging Face Inference API is free-tier and serverless.

**Tradeoff:** Network dependency, but mitigated by aggressive disk caching.

### 2. **Small Task Models (Summarization, Classification) vs. One Big LLM**
**Why:** 
- BART summarization (140M params) is faster and cheaper than asking a 7B chat model "summarize this in 1 sentence."
- Zero-shot MNLI (345M) is designed for classification; it's not hallucinating like a chat model forced to output JSON.
- Composing a brief from structured fields (labels, summaries, retrieved chunks) is more predictable than prompt-engineering.

**Tradeoff:** More moving parts, but each part is testable and cacheable independently.

### 3. **MiniLM Embeddings + Chroma (Local Vector DB)**
**Why:**
- `sentence-transformers/all-MiniLM-L6-v2` (22M params) is the sweet spot for semantic retrieval on small data.
- Chroma persisted to disk = no dependency on Pinecone/Weaviate APIs.
- Metadata filters (source, ticker, event_type) + BM25 hybrid would be overkill for v1.

**Tradeoff:** No real-time index updates (rebuild batch); no cloud sync. Fine for daily briefings.

### 4. **LangGraph (3 Agents) vs. LangChain Chain vs. Single Prompt**
**Why:**
- **Researcher** (retrieval specialist) keeps retrieval logic separate and testable.
- **Risk Officer** (label mapper) enforces trading logic (e.g., "monetary policy + rising CPI = high risk").
- **Brief Writer** (output formatter) ensures consistent, human-readable markdown.

Explicit state machine (LangGraph) = debuggable, no surprise loops.

**Tradeoff:** More code, but each agent is a small, focused function—easier to interview on.

### 5. **SQLite Traces + Contradiction Checks (Not Gold Labels)**
**Why:** No training dataset. Traces capture latency, cache hits, and user feedback. Contradiction checks (narrative bullish vs. price down) are a heuristic quality signal—human traders recognize this instantly.

**Tradeoff:** Evals are indirect. But this is the real work: building for human traders, not benchmarks.

### 6. **Streamlit for v1 (Not FastAPI + React)**
**Why:** MVP launch speed. Streamlit handles state, caching, and UI all in Python.

**How to Productionize Later:** Extract agent logic to FastAPI endpoints, Streamlit calls API, add auth + queue job for long-running queries.

---

## Data Schema

Every article, regardless of source, normalizes to:

```python
from typing import TypedDict

class NewsItem(TypedDict, total=False):
    id: str                          # Unique: source + timestamp + title hash
    source: str                       # "yfinance" | "fred" | "gdelt" | "rss"
    published_at: str                # ISO-8601 (UTC)
    title: str                        # Max 256 chars
    raw_text: str                     # Full body
    url: str                          # Permalink (optional for aggregated feeds)
    tickers: list[str]               # e.g., ["AAPL", "SPY"]
    
    # Enriched by pipeline:
    summary: str | None              # L2: BART-generated
    event_type: str | None           # L3: "monetary policy" | "inflation data" | ...
    risk_level: str | None           # L3: "low" | "medium" | "high"
    risk_score: float | None         # Numeric 0.0–1.0 (confidence)
    contradiction: bool | None       # L6: narrative vs. 1d return
```

**Why TypedDict:** Typed, serializable, minimal overhead. Not dataclasses/Pydantic for L0–L4 (keep it simple). Introduce Pydantic only if schema grows beyond 10 fields.

---

## File Structure

```
d:\AI projects\CatalystDesk/
├── README.md                              # This file
├── INTERVIEW.md                           # Interview drill + design Q&A
├── requirements.txt                       # Python deps
├── .env.example                           # Template env vars
├── .env                                   # (not in git) your secrets
├── .gitignore                             # .env, cache, logs, __pycache__
│
├── src/
│   ├── __init__.py
│   ├── config.py                          # L0: Config + logger setup
│   ├── logging_utils.py                   # L0: Request ID, structured logging
│   │
│   ├── ingest/                            # L1
│   │   ├── __init__.py
│   │   ├── yfinance_client.py
│   │   ├── fred_client.py
│   │   ├── news_client.py
│   │   └── normalize.py
│   │
│   ├── hf_tasks/                          # L2, L3
│   │   ├── __init__.py
│   │   ├── client.py
│   │   ├── summarize.py
│   │   └── classify.py
│   │
│   ├── rag/                               # L4
│   │   ├── __init__.py
│   │   ├── embed.py
│   │   ├── store.py
│   │   └── retrieve.py
│   │
│   ├── agents/                            # L5
│   │   ├── __init__.py
│   │   ├── state.py
│   │   ├── researcher.py
│   │   ├── risk_officer.py
│   │   ├── brief_writer.py
│   │   └── graph.py
│   │
│   ├── evals/                             # L6
│   │   ├── __init__.py
│   │   ├── traces.py
│   │   ├── metrics.py
│   │   └── contradiction.py
│   │
│   └── utils/                             # Shared helpers
│       ├── __init__.py
│       └── cache.py
│
├── app/                                   # L7
│   └── streamlit_app.py
│
├── scripts/
│   ├── run_l0_verify.py                   # This phase: sanity check
│   ├── run_l1_ingest.py
│   ├── run_l2_summarize.py
│   ├── run_l3_classify.py
│   ├── run_l4_rag.py
│   ├── run_l5_brief.py
│   ├── run_l6_eval.py
│   └── run_l7_app.py
│
├── data/
│   ├── samples/                           # L0: Fixture articles (5–8)
│   │   ├── sample_articles.json
│   │   └── README.md
│   │
│   ├── chroma/                            # L4: Persisted vector DB (gitignored)
│   ├── traces.db                          # L6: SQLite eval log (gitignored)
│   └── .gitkeep
│
├── cache/                                 # Gitignored: HF models, embeddings, cache DB
│   ├── hf/
│   └── cache.db
│
├── logs/                                  # Gitignored: request logs, traces
│   └── .gitkeep
│
└── .gitignore
```

---

## How to Approach the Interview

**"Walk me through CatalystDesk architecture."**

1. **Phase L0 (You are here):** Skeleton + config + logging + sample data. Why? Reproducibility offline. Why no ORMs? Schema fits in one TypedDict; SQLite for traces only.

2. **L1 Ingest:** Three API clients (yfinance, FRED, RSS) → normalized NewsItem schema. Why three adapters? Each has different rate limits, auth, error modes. Why no scraping? Slower, fragile, violates ToS.

3. **L2 Summarization:** BART-large-cnn (or distilbart for speed). Why? Task-specific models beat prompting LLMs. Cache key = hash(model + text). Why cache? Inference API is free but slow; disk cache makes reruns instant.

4. **L3 Classification:** Zero-shot MNLI over labels ["monetary policy", "earnings", …]. Why zero-shot? No labeled training data ($0 budget). Why these labels? Talking to traders about what moves markets.

5. **L4 RAG:** MiniLM embeddings → Chroma (local, persistent). Metadata filters: source, ticker, event_type, date range. Why not keyword search? Semantic matching catches nuance ("Fed signals caution" ≈ "policy uncertainty").

6. **L5 Agents (LangGraph):** Researcher retrieves + packs; Risk Officer scores event/macro; Brief Writer formats. Why three? Separation of concerns + testable. Why LangGraph? Explicit state machine; no hidden loops.

7. **L6 Evals:** SQLite traces (latency, cache_hit, user feedback). Contradiction checks: brief sentiment (bullish/bearish/neutral via zero-shot) vs. 1d yfinance return. Why? Quality signal without gold labels.

8. **L7 Streamlit:** Query input → brief + evidence cards + risk badges. Why Streamlit? Rapid MVP. How to scale? Extract agent logic to FastAPI, Streamlit talks to API, add async queue for long queries.

---

## Resume Bullets (Generated at L7)

- Built CatalystDesk, a $0 LangGraph + RAG briefing system that turns live market news (yfinance, FRED, public feeds) into risk-tagged trader briefs.
- Implemented Hugging Face zero-shot classification, BART summarization, and MiniLM semantic embeddings over Chroma with disk-cache and retry logic to survive Inference API cold starts.
- Designed 3-agent LangGraph state machine (Researcher, Risk Officer, Brief Writer) to enforce trading logic and contradiction detection via narrative-vs-price bias analysis.
- Built SQLite trace & feedback loop for latency/cache-hit metrics and user feedback without relying on gold labels; demonstrated offline-first degradation with fixture data.

---

## Running This Phase (L0)

```bash
# 1. Verify config + logging
python scripts/run_l0_verify.py

# Expected output:
# ✓ Config loaded
# ✓ Logging initialized
# ✓ Sample fixtures (5 articles)
# ✓ All paths exist

# 2. Inspect sample data
cat data/samples/README.md
cat data/samples/sample_articles.json | head -100

# 3. Ready for L1? When prompted below, follow the L1 setup.
```

---

## Next Steps After L0

Once you run `scripts/run_l0_verify.py` and see ✓ checks, tell me "L0 done" and I'll give you exact L1 commands (ingest yfinance, FRED, RSS + normalize).

**Why phases?** Each one builds a testable, cacheable layer. You own the code end-to-end and can explain to an interviewer: "Here's why I chose BART over a general LLM; here's the cache key strategy; here's how I degrade when the API fails."

---

## Debugging & Logs

Logs go to `logs/` with request IDs:

```bash
tail -f logs/catalyst.log
grep "REQUEST_ID=abc123" logs/catalyst.log
```

If something breaks, check:
1. `.env` populated (HF_TOKEN at minimum)
2. `cache/` and `logs/` writable
3. Internet connection (first L1 run will fetch from APIs)

---

## Dependencies (See requirements.txt)

- `huggingface-hub`: Inference API + caching
- `sentence-transformers`: Embeddings
- `chromadb`: Vector DB (local)
- `pyyaml`, `python-dotenv`: Config
- `yfinance`: Market data (L1)
- `pandas-datareader`: FRED (L1)
- `feedparser`: RSS (L1)
- `langgraph`: Agent orchestration (L5)
- `streamlit`: UI (L7)
- `pytest`: Tests (optional for now)

---

**Built by:** A beginner who must understand every choice. For AI engineering interviews.
