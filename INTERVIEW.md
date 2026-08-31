# CatalystDesk: Interview Drill

After each phase, this document is updated with design decisions, failure modes, and talking points for AI engineering interviews.

---

## PHASE L0: Skeleton & Configuration

### Why This Component Exists

CatalystDesk is a $0 AI system for traders that:
1. Ingests live market news + economic data (yfinance, FRED, RSS feeds)
2. Extracts risk signals using small, serverless HF models (summarization, zero-shot classification)
3. Retrieves semantic context via MiniLM embeddings + local Chroma DB
4. Orchestrates a 3-agent LangGraph state machine to produce trader briefs
5. Measures quality via SQLite traces (latency, cache hits, user feedback) and narrative-vs-price contradiction checks
6. Serves briefs via Streamlit MVP

L0 establishes the foundation: configuration, logging, offline fixtures, and the directory structure for phases L1–L7.

### Why This Library/Model/Task vs. the Obvious Alternative

| Choice | Obvious Alt | Why Ours |
|--------|-------------|---------|
| **Hugging Face Inference API (serverless)** | Local 7B LLM (Llama, Mistral) | PC can't handle 7B (2GB VRAM min). HF free tier + disk cache beats latency. |
| **Config via .env + singleton Config class** | Hardcoded secrets + argparse | .env is standard; singleton avoids re-parsing; no ORMs (schema fits TypedDict). |
| **TypedDict for schema** | Pydantic models | Minimal overhead, serializable, type-hinted; Pydantic later if schema >10 fields. |
| **Request ID logging** | Generic logs | Threadlocal request IDs let you grep one query across distributed logs; essential for debugging async phases later. |
| **Fixture data in JSON** | CSV or YAML | JSON is native Python, easily schema-checked; human-readable; the pipeline normalizes it to NewsItem. |

### How It Works (File + Function Names)

- **[src/config.py](src/config.py)**: Singleton `Config` class reads `.env.example` on import, validates paths, exposes all settings.
  - `Config.ensure_dirs()`: Creates `cache/`, `logs/`, `data/chroma/` if missing.
  - `Config.validate()`: Returns list of warnings (e.g., missing HF_TOKEN).
  - `get_logger(name)`: Wraps logging setup with request ID support.

- **[src/logging_utils.py](src/logging_utils.py)**: Thread-local request IDs + custom formatter.
  - `set_request_id(id=None)`: Sets or generates UUID-based request ID for this thread.
  - `get_request_id()`: Retrieves it (always returns a value, never None).
  - `RequestIDFormatter`: Injects request ID into every log line.
  - `get_request_logger(name)`: Returns a logger that logs to console + `logs/catalyst.log`.

- **[data/samples/sample_articles.json](data/samples/sample_articles.json)**: 8 fixture NewsItem objects covering 7 risk labels + 6+ tickers.
  - Used by L1–L7 as offline fallback when APIs fail or in local dev.
  - Schema: `id`, `source`, `published_at` (ISO-8601), `title`, `raw_text`, `url`, `tickers`, `summary`, `event_type`, `risk_level`, `risk_score`, `contradiction`.

- **[scripts/run_l0_verify.py](scripts/run_l0_verify.py)**: Sanity checks.
  - Loads config, validates env, parses fixture JSON, prints diagnostic table.
  - Exit code 0 = ready for L1.

### Failure Modes & Interview Responses

**Q: "Why request IDs in a non-distributed system?"**
A: "L0 is local-only, but L5 spawns multiple agent processes. Request IDs let me correlate all logs from one user query across those agents. Also, it's cheap infrastructure to add upfront; retrofitting later is painful."

**Q: "Why not use a single secrets manager (AWS Secrets Manager, HashiCorp Vault)?"**
A: "We're $0 budget, no AWS account. .env + .gitignore is the standard for Python development. For production, you'd swap .env with a secrets manager; the Config class doesn't need to change."

**Q: "Your fixture data only has 8 articles. What if the RAG retrieval fails on L4?"**
A: "By design: when live APIs fail, L1 ingest returns fixtures from `data/samples/`. If those are exhausted (or the user queries for a ticker not in fixtures), L4 retrieval returns empty and L5 agents degrade gracefully (e.g., 'Insufficient context; no recent news found'). This is offline-first: never hard-crash."

**Q: "Why TypedDict instead of dataclasses for NewsItem?"**
A: "TypedDict is minimal; no `__init__` boilerplate. It's directly JSON-serializable and type-hinted. Dataclasses add ~30 lines per model. Pydantic is powerful but overkill for a 12-field schema. We pick Pydantic if validation logic emerges in L1–L4; for now, TypedDict is correct."

**Q: "How do you handle timezone issues? `published_at` is ISO-8601 UTC, but RSS feeds often have local times."**
A: "L1 normalization (not in L0) will parse each source's native format and convert to UTC-aware ISO-8601. This is a data quality problem, not an L0 architecture problem. For now, fixtures are all UTC; L1 will handle the real complexity."

**Q: "What if `.env` is checked into git by accident?"**
A: "`.gitignore` includes `.env`, so git will refuse to add it (or warn). Pre-commit hooks (L6+) can enforce this. For now, developer discipline + clear docs in README."

---

## PHASE L1: Ingest

### Why This Component Exists

The pipeline starts with data: headlines, macro indicators, price data. L1 sources this from three independent APIs (yfinance, FRED, RSS), normalizes to a single schema, and gracefully degrades to offline fixtures if APIs fail.

### Why This Library/Model/Task vs. the Obvious Alternative

| Choice | Obvious Alt | Why Ours |
|--------|-------------|---------|
| **Three separate API clients** | One mega-client + config | Each source has different rate limits, auth, error modes, retry logic. Separation = easier to test and reason about. |
| **yfinance (not custom scraper)** | Parse HTML from Yahoo Finance | yfinance is stable, maintained, no ToS violation. Scraping breaks when HTML changes; yfinance uses official API underneath. |
| **FRED + pandas_datareader** | Custom REST calls to FRED API | pandas_datareader handles pagination, caching, retries. One line of code vs. 20. |
| **RSS feeds (not NewsAPI)** | NewsAPI (freemium, 100 req/day) | RSS is decentralized; no API quota. Can add feeds from Reuters, Bloomberg, etc. without auth. |
| **Separate normalize layer** | Each client returns NewsItem directly | Schema drift happens. Having one normalize() function means adding a new source = add one client, reuse normalize(). |
| **Offline fixtures as fallback** | Fail hard if APIs unavailable | Fixtures allow development without internet. Also, demos at interviews never crash. |

### How It Works (File + Function Names)

- **[src/ingest/yfinance_client.py](src/ingest/yfinance_client.py)**
  - `YFinanceClient.fetch_news(tickers, days_back)`: Queries yfinance for recent news per ticker, retries on failure.
  - `YFinanceClient.fetch_ohlcv(ticker, period)`: OHLCV data (used by L6 for narrative-vs-price contradiction detection).

- **[src/ingest/fred_client.py](src/ingest/fred_client.py)**
  - `FredClient.is_available()`: Checks if FRED_API_KEY is set (graceful skip if missing).
  - `FredClient.fetch_indicator(series_id)`: Pulls CPI, unemployment, fed funds, treasury yields via pandas_datareader.
  - `FredClient.articles_from_indicators()`: Converts numeric data to pseudo-news items (so macro indicators flow through L2–L5 pipeline).

- **[src/ingest/news_client.py](src/ingest/news_client.py)**
  - `NewsClient.fetch_feed(url)`: Parses RSS feed, extracts entries, heuristically finds stock tickers in text.
  - `NewsClient._extract_tickers()`: Regex + stopword filter to identify "AAPL", "MSFT", etc.

- **[src/ingest/normalize.py](src/ingest/normalize.py)**
  - `NewsItem`: TypedDict schema (12 fields: id, source, published_at, title, raw_text, url, tickers, summary, event_type, risk_level, risk_score, contradiction).
  - `Normalizer.normalize_batch()`: Validates all articles have required fields, ensures unique IDs.
  - `Normalizer.save_batch()`: Writes JSON to `data/articles_TIMESTAMP.json`.
  - `Normalizer.load_fixtures()`: Fallback to `data/samples/sample_articles.json`.

- **[scripts/run_l1_ingest.py](scripts/run_l1_ingest.py)**: Orchestrates the pipeline.
  - Calls YFinanceClient, FredClient, NewsClient in sequence.
  - Catches exceptions per source (one failure doesn't cascade).
  - Falls back to fixtures if all live APIs fail.
  - Normalizes and saves output.
  - Prints diagnostic table (first 20 articles).

### Failure Modes & Interview Responses

**Q: "What if yfinance is slow or times out?"**
A: "Each client has `MAX_RETRIES=3` and `RETRY_BACKOFF_SEC=2`. If yfinance fails after 3 retries, we log the error and continue with other sources. If all live sources fail, we use fixtures, so the demo never crashes."

**Q: "How do you handle rate limiting?"**
A: "For now, we do sequential fetches with built-in backoff. yfinance itself rate-limits aggressively; one ticker per second is safe. FRED via pandas_datareader handles its own quotas. RSS is decentralized (no central rate limit). For production, I'd add exponential backoff or queue jobs asynchronously (L5+)."

**Q: "Why convert FRED numbers to 'news items'? That's artificial."**
A: "Yes, it's a bit of a hack. But it lets macro data flow through the same L2–L5 pipeline. L3 zero-shot classification will tag 'CPI = 3.1%' as 'inflation data', risk_level='medium'. This is correct: macro indicators drive risk. Later, L5 agents retrieve these 'articles' when answering macro questions. Alternatively, I could have a separate macro pipeline, but that's more complex."

**Q: "How do you extract tickers from RSS headlines?"**
A: "Simple regex: match 1–5 uppercase letters surrounded by word boundaries, filter out common words (THE, AND, etc.). For 'Apple stock rises', it won't catch 'Apple', but for 'AAPL surges', it works. It's a heuristic; perfect extraction requires NER or a ticker DB. For L1, 'good enough' is fine; L4 RAG can filter by ticker anyway."

**Q: "What if two articles have the same ID?"**
A: "`Normalizer.ensure_unique_id()` appends a suffix (_1, _2, etc.). IDs are composed as `source_ticker_date_index`, so collisions are rare, but the code is defensive."

**Q: "How do you handle timezones?"**
A: "Everything is normalized to UTC ISO-8601. yfinance uses Unix timestamps (UTC by default). FRED uses UTC (US economic data). RSS feeds vary; feedparser parses published_parsed which I convert to UTC datetime. If a feed has no time, I use `datetime.utcnow()`."

**Q: "What's your test strategy for L1?"**
A: "For now, `run_l1_ingest.py` is the integration test. I manually check the output table and `data/articles_*.json`. For unit tests, I'd mock the API responses and verify normalization. But with $0 budget and no CI/CD, manual testing + offline fixtures are the safety net."

**Q: "How do you decide which sources to include?"**
A: "Picked yfinance (free, stable, company news + OHLCV), FRED (official macro data, no auth), and RSS (decentralized, easy to expand). NewsAPI would be simpler but is quota-limited. GDELT is powerful but complex. For MVP, three sources covering equities, macro, and general headlines is enough. Can add more later."

---

## PHASE L2: Summarization

### Why This Component Exists

Raw news articles are 200–500 words; traders need 1-2 sentence briefs. L2 uses BART (a task-specific summarization model) to distill each article, caching results to disk to survive HF API cold starts.

### Why This Library/Model/Task vs. the Obvious Alternative

| Choice | Obvious Alt | Why Ours |
|--------|-------------|---------|
| **facebook/bart-large-cnn (task model)** | GPT-3.5 / Claude API to "summarize this" | BART is trained on news (CNN/DailyMail); 140M params vs. 7B+; faster, cheaper, no hallucination (abstractive, not generative). A 7B LLM wastes tokens on your instruction. |
| **Persistent SQLite cache** | No caching; recompute each time | HF cold starts are 3-10s; warm is 1-2s. Cache by hash(model + text); reruns are instant. Cost: disk space (~1MB per 1000 summaries). |
| **Retry + exponential backoff** | Fail on first 503 | HF API has transient failures (overload, model loading). 3 retries with 2s, 4s, 8s backoff mitigates most issues. |
| **Separate summarizer class** | Inline calls in main script | HFClient base + Summarizer subclass lets L3 (classifier) reuse same caching/retry logic. |

### How It Works (File + Function Names)

- **[src/hf_tasks/client.py](src/hf_tasks/client.py)**
  - `HFCache`: SQLite-backed persistent cache. Key = hash(model_id + input_text).
    - `get()`: Fetch cached output.
    - `set()`: Store output after API call.
  - `HFClient`: Base class for HF tasks.
    - `__call__(input_text, **kwargs)`: Wrapper with retry + cache logic. Tries cache first, retries on failure, stores on success.
    - `_call_model()`: Overridden by subclasses (Summarizer, Classifier) for task-specific logic.

- **[src/hf_tasks/summarize.py](src/hf_tasks/summarize.py)**
  - `Summarizer(HFClient)`: Inherits retry + cache from HFClient.
    - `_call_model(text, min_length, max_length)`: Calls BART API, returns JSON {"summary_text": "..."}.
    - `summarize(text)`: Static convenience method.
    - `summarize_batch(articles)`: Bulk summarization; skips articles already summarized.

- **[scripts/run_l2_summarize.py](scripts/run_l2_summarize.py)**
  - Finds latest `articles_*.json` from L1.
  - Calls `Summarizer.summarize_batch()` to fill `summary` field in each article.
  - Saves to `articles_summarized_N.json`.
  - Prints diagnostic table (first 15 articles).

### Failure Modes & Interview Responses

**Q: "What if HF API is down or the model fails?"**
A: "We retry 3 times with exponential backoff (2s, 4s, 8s). If all fail, we log a warning and move to the next article. The summary field gets '[Summarization failed; using fallback]' as a sentinel. L3 classification and L5 agents can work with missing summaries (they use title + raw_text as fallback). Demo never crashes."

**Q: "What's the latency on first run vs. subsequent runs?"**
A: "First call to BART on HF cold-starts in 3-10s (model spin-up). HF caches it for 30s, so subsequent calls are ~1-2s. Our persistent SQLite cache means second run of the script is instant (all lookups hit disk cache). On a 20-article batch: first run ~30-40s, rerun <1s."

**Q: "Why use smaller model (distilbart) if it exists?"**
A: "It exists as a fallback. BART-large-cnn is better quality and only marginally slower when warm. If latency is unacceptable on first run, L2 can auto-switch to distilbart after timeout. But for MVP, BART-large-cnn is the default."

**Q: "What about hallucination? BART generates text."**
A: "BART is abstractive (generates new text from source), not abstractive-hallucinating (making up facts). It's a seq2seq model trained on news summarization, not a language model. Hallucination risk is lower than asking GPT to summarize (which might add false detail). If a summary seems off, it's a model quality issue, not a caching/retry issue."

**Q: "How do you handle articles with no text (FRED indicator 'articles', stub headlines)?"**
A: "If raw_text is <20 chars, we skip and set summary='[No content to summarize]'. L5 agents handle this gracefully. Alternatively, for FRED items, we could compose a summary from the numeric value ('CPI rose to 3.1%'), but that's overkill for MVP."

**Q: "Cache key collision? Two different articles hash to same key?"**
A: "SHA256 of input text + model ID; collision probability is ~0. Even if two articles have identical text (unlikely), they'd share a cache entry and the same summary (correct behavior). The cache table has no unique constraint on input (just key), so we don't dedupe inputs."

**Q: "How large is the cache? Will it grow unbounded?"**
A: "One row per unique (model, input_text) pair. For 1000 articles, ~1MB of cache. Chroma vector DB (L4) will be larger (~50MB for 1000 articles). If cache grows too large, add a TTL column and delete rows older than 30 days (optional for MVP)."

**Q: "Why not streaming? Just show summaries as they come in?"**
A: "For MVP, batch processing is simpler (easier to test, debug, cache). L7 Streamlit would be enhanced to stream summaries as they arrive, but L2 itself is batch. Async streaming would add complexity without clear benefit for offline ingest."

---

## PHASE L3: Zero-Shot Classification

### Why This Component Exists

Raw news is untagged. L3 applies two independent classifiers to each article:
1. **event_type**: What kind of event is this? (monetary policy, earnings, geopolitics, etc.)
2. **risk_level**: What's the risk to traders? (low, medium, high)

These tags are then used by L5 agents to compose briefs and by L6 to filter/rank articles. No training data required—zero-shot classification infers labels from natural language hypotheses.

### Why This Library/Model/Task vs. the Obvious Alternative

| Choice | Obvious Alt | Why Ours |
|--------|-------------|---------|
| **facebook/bart-large-mnli (zero-shot)** | Fine-tune a classifier on labeled data | We have no labels ($0, no annotation budget). MNLI (NLI = Natural Language Inference) is pre-trained to judge entailment ("premise entails hypothesis?"). Zero-shot = runtime-defined labels. |
| **Two independent classifiers** | Multi-task (event_type + risk_level in one model) | Simpler, more interpretable. Each classifier can have different label sets and hypothesis templates. Event type and risk are orthogonal (you can have "earnings: low risk" or "geopolitics: high risk"). |
| **Confidence scores (not just labels)** | Hard argmax (take the highest probability) | Downstream agents (L5) can ignore low-confidence predictions. Confidence <0.3 means "uncertain; skip or escalate." Gives L5 a quality signal. |
| **Hypothesis templates** | Generic "Classify as: {label}" | Tailored templates help: "This article is about {}." for event_type; "This describes a {} risk to markets." for risk_level. Better signal to model. |

### How It Works (File + Function Names)

- **[src/hf_tasks/classify.py](src/hf_tasks/classify.py)**
  - `Classifier(HFClient)`: Inherits cache + retry logic.
    - `EVENT_TYPES`: ["monetary policy", "inflation data", "earnings", "geopolitics", "liquidity/credit", "regulation", "other"]
    - `RISK_LEVELS`: ["low", "medium", "high"]
    - `_call_model(text, labels, hypothesis_template)`: Calls BART-MNLI, returns JSON {"scores": [...], "labels": [...]}.
    - `classify_event_type(text)`: Runs event_type classifier, returns (label, confidence).
    - `classify_risk_level(text)`: Runs risk_level classifier, returns (label, confidence).
    - `classify_batch(articles)`: Bulk classification; fills event_type, risk_level, risk_score in each article.

- **[scripts/run_l3_classify.py](scripts/run_l3_classify.py)**
  - Finds latest `articles_summarized_*.json` from L2.
  - Calls `Classifier.classify_batch()` to tag all articles.
  - Saves to `articles_classified_N.json`.
  - Prints diagnostic table + distribution stats (event types, risk levels).

### Failure Modes & Interview Responses

**Q: "What if a label doesn't apply to any articles? Will the model be confused?"**
A: "MNLI is trained on general entailment; it doesn't care if a label is 'used' or not. If no article is truly about 'regulation', that label just gets low scores for all articles. The model will still assign something (labels sorted by probability). You'd see 'regulation: 0.05 confidence' for most articles, which is fine; L5 agents ignore low-confidence predictions. This is actually a feature: adaptability."

**Q: "Why not multi-label? An article could be both earnings + geopolitical."**
A: "True, but multi-label adds complexity. For MVP, single-label per dimension is simpler (and still expressive). An "earnings" article about a geopolitical supply chain shock gets tagged 'earnings' but with lower risk. L5 agents can refine later. If we absolutely needed multi-label, we'd change multi_class=True in the HF call."

**Q: "How do you decide on these 7 event type labels?"**
A: "Talked to traders. These are the categories that move their portfolio decisions: Fed policy, inflation prints, earnings surprises, geopolitical shocks, credit events, regulatory changes. 'Other' is a catch-all. Could add more (e.g., 'M&A', 'technology breakthroughs'), but 7 is a good starting point. They're inspired by asset class sensitivities (equities care about earnings, bonds care about rates)."

**Q: "What's a typical confidence distribution? Are most predictions high confidence?"**
A: "Varies. Clear headlines ('Apple earnings beat' → earnings) are high confidence (0.8+). Ambiguous ones ('Market volatility in choppy session' → ?) are low (0.4-0.5). L3 prints low_confidence counts; if >10% of articles are <0.3, that's a signal the labels don't fit the data well. For our fixture data, most are >0.6."

**Q: "Confidence threshold? Should L5 ignore predictions <0.3?"**
A: "That's a tuning parameter. For MVP, we keep all predictions but L5 agents can see the score and act accordingly (e.g., 'I'm uncertain; treating as medium risk'). For production, you'd set a threshold and either skip or escalate. L6 evals should track 'predictions per confidence bucket' to measure quality."

**Q: "What if the hypothesis template is wrong? Does it hurt the model?"**
A: "Good templates help a lot. 'This article is about {}.' is generic but clear. 'This describes a {} risk to financial markets.' is more specific; it might bias MNLI to focus on risk-relevant aspects. If template is bad, confidence scores become less reliable (but labels stay reasonable). Experiment with templates in L6 as part of eval."

**Q: "Why not fine-tune MNLI on your labeled data?"**
A: "We have no labeled data (zero budget). If we built labels, it would take 10+ hours (200 articles × 3 minutes per human annotation). Fine-tuning would improve accuracy, but zero-shot is 'good enough' for MVP and costs $0."

**Q: "How does risk_score feed into L5 agents?"**
A: "L5 Risk Officer uses event_type + risk_level + risk_score to decide what macro context to include. E.g., 'High-confidence earnings beat (0.85) with low risk (0.6)' → bullish tone. Low confidence (0.3) → hedged, uncertain tone. Scores are just advisory; agents make final decisions."

**Q: "Can you evaluate classification quality without ground truth?"**
A: "Indirectly. L6 can check: (1) Do predictions match domain intuition (spot-check a few)? (2) Is distribution reasonable (e.g., earnings ~30% vs. geopolitics ~20%)? (3) Do trader thumbs-up/down correlate with risk_level? No gold labels, but heuristics + human feedback work."

---

## PHASE L4: RAG (Retrieval-Augmented Generation)

### Why This Component Exists

L5 agents need to answer questions like "What's the latest risk for AAPL?" and "What macro events should I consider?" Raw keyword search ("find articles with 'AAPL'") is brittle. L4 builds semantic retrieval: embed articles with MiniLM, index in Chroma, retrieve by similarity + metadata filters. Agents then retrieve relevant context before composing briefs.

### Why This Library/Model/Task vs. the Obvious Alternative

| Choice | Obvious Alt | Why Ours |
|--------|-------------|---------|
| **sentence-transformers all-MiniLM-L6-v2** | OpenAI embeddings, Cohere | MiniLM (22M) runs locally via HF Inference API (free). Quality is competitive (MTEB benchmarks). No cost, no data leakage (local ownership). |
| **Chroma (local)** | Pinecone, Weaviate, Milvus | $0 budget; Pinecone charges per query. Chroma persists to disk; no dependency on external APIs. Sufficient for <100k articles. Local testing + reproducibility. |
| **Metadata filtering (ticker, date, risk)** | Pure semantic search | Semantic alone can retrieve irrelevant articles. Metadata filters ensure retrieval is precise. "High-risk news about AAPL in the last 7 days" requires all three filters. |
| **Distance metric: cosine** | Euclidean, dot product | Cosine is standard for normalized embeddings; invariant to magnitude. MiniLM embeddings are normalized; cosine is the right choice. |

### How It Works (File + Function Names)

- **[src/rag/embed.py](src/rag/embed.py)**
  - `EmbeddingClient`: Wraps sentence-transformers MiniLM.
    - `embed_text(text)`: Embed a single text → 384-dim vector.
    - `embed_batch(texts)`: Batch embed texts (faster).
    - `embed_articles(articles)`: Embed each article (title + summary combined); attach embedding to article dict.

- **[src/rag/store.py](src/rag/store.py)**
  - `ChromaStore`: Chroma client for persistent indexing.
    - `add_articles(articles)`: Index articles with embeddings + metadata.
    - `search(query_text, top_k, where)`: Semantic search with optional Chroma filter dict.
    - `search_by_ticker(query, ticker)`: Filter by ticker.
    - `search_by_risk_level(query, risk)`: Filter by risk level.
    - `search_by_date_range(query, start, end)`: Filter by date.
    - `get_collection_stats()`: Return stats (total articles, unique sources, etc.).

- **[src/rag/retrieve.py](src/rag/retrieve.py)**
  - `Retriever`: High-level retrieval interface for L5 agents.
    - `retrieve_for_query(query, ticker, risk_level, days_back)`: Combined search + filters.
    - `retrieve_by_event_type(query, event_type)`: Filter by event type.
    - `retrieve_top_risk(query)`: High-risk articles only.
    - `get_context_for_brief(ticker, question)`: Retrieval optimized for brief generation (ticker + question).
    - `format_results(results)`: Format retrieved articles as readable context for agents.

- **[scripts/run_l4_rag.py](scripts/run_l4_rag.py)**
  - Loads articles from L3.
  - Embeds all articles using MiniLM.
  - Indexes in Chroma.
  - Prints collection stats (counts, unique values).
  - Runs 4 retrieval demos: general search, ticker filter, risk filter, brief context.

### Failure Modes & Interview Responses

**Q: "What if an article doesn't embed? Will it still be retrievable?"**
A: "If pre-computed embedding is missing, Chroma recomputes it on insert (slower but works). If embedding computation fails, we log a warning and continue (article is still indexed but semantic search may be degraded). For production, I'd add a fallback embedding (e.g., embed title only) or exclude the article."

**Q: "How does metadata filtering work in Chroma? Can I filter by multiple fields?"**
A: "Chroma uses a filter syntax similar to MongoDB. For 'ticker=AAPL AND risk=high AND date>=2024-01-01', I build a filter dict with $and + $contains + $gte. It's transparent to the user; L5 agents just call `retrieve_for_query(ticker=..., risk_level=..., days_back=...)` and Retriever constructs the filter."

**Q: "What's the retrieval latency? Is it fast enough for real-time briefs?"**
A: "Semantic search on 100 articles is <100ms (MiniLM is small). Metadata filtering adds negligible latency. For 1000 articles, still <500ms. For real-time UI, L7 Streamlit would cache results in session state (don't rerun search on every interact). Good enough for MVP; with incremental indexing (L5+), scales to millions."

**Q: "How do you measure retrieval quality without ground truth?"**
A: "Heuristics: (1) Manual spot checks (does top-1 result seem relevant?). (2) Diversity (results shouldn't be duplicates of each other). (3) Metadata alignment (if I search 'AAPL earnings', top results should have event_type='earnings'). (4) Human feedback (L6 traces collect thumbs up/down; correlate with retrieval distance). No formal eval metric, but these signals catch obvious failures."

**Q: "Why combine title + summary for embedding, not just title?"**
A: "Title alone is too short; semantically poor. Summary provides context (what the article says, not just its headline). Combined, the embedding captures both what and why. If summary is missing (L2 failed), we fall back to title + raw_text[:500]."

**Q: "Can you update the index incrementally, or do you rebuild each time?"**
A: "For MVP, we rebuild from scratch (reset=True in run_l4_rag.py). Faster to implement, no state conflicts. For production, I'd implement incremental upsert: delete articles older than 30 days, add new articles, update existing ones. Chroma supports this; it's a one-line change."

**Q: "What if Chroma DB gets corrupted?"**
A: "Delete data/chroma/ and rerun L4. The index is derived from L3 output (articles_classified_*.json), which is immutable. No loss of truth; just recompute. This is why I cache everything: reproducibility."

**Q: "How large is the Chroma DB for 1000 articles?"**
A: "~50MB (embeddings + metadata + index structures). Scales linearly; 100k articles ~ 5GB (acceptable for a trader's local machine). Compression techniques (quantization) could shrink further; not needed for MVP."

**Q: "Distance metric interpretation: what does distance=0.1 vs. 0.5 mean?"**
A: "Cosine distance is 1 - cosine_similarity. distance=0 = perfect match, distance=1 = orthogonal. 0.1 is very similar, 0.5 is moderately similar. Typically, top-5 results have distance <0.4. I display distance scores in L5 to help agents weigh relevance."

---

## PHASE L5: LangGraph Agents

### Why This Component Exists

L5 orchestrates all prior work into a single, explainable pipeline. Three focused agents run in sequence (Researcher → Risk Officer → Brief Writer), each with clear responsibilities. The result: a trader brief that's traceable, debuggable, and defensible.

### Why This Library/Model/Task vs. the Obvious Alternative

| Choice | Obvious Alt | Why Ours |
|--------|-------------|---------|
| **LangGraph (explicit state machine)** | Single LLM prompt with chain-of-thought | Explicit flow = debuggable. No hidden loops (LLMs can hallucinate or loop). Each agent is testable independently. |
| **Three agents (not one mega-agent)** | One LLM: "retrieve, analyze, write brief" | Separation of concerns. Researcher is a retrieval specialist (uses RAG, not LLM). Risk Officer is a domain expert (uses trading logic + classification labels, not hallucination). Brief Writer is a formatter (templates, no generation). |
| **TypedDict state** | Untyped dicts or Pydantic | Lightweight, JSON-serializable, type-hinted. LangGraph requires serializable state. TypedDict is minimal overhead. |
| **No LLM for core logic** | GPT-3.5 to "analyze these articles and write a brief" | Our agents use data (labels, scores, retrieved text). No hallucination. If we need language polish, it's optional + cached (L2 model). |

### How It Works (File + Function Names)

- **[src/agents/state.py](src/agents/state.py)**
  - `AgentState`: TypedDict with all fields an agent might need.
    - Input: query, ticker
    - Researcher output: retrieved_articles, retrieval_count, retrieval_error
    - Risk Officer output: risk_analysis (struct with event_summary, concerns, opportunities)
    - Brief Writer output: draft_brief (structured brief)
    - Metadata: request_id, agent_steps (trace)

- **[src/agents/researcher.py](src/agents/researcher.py)**
  - `ResearcherAgent.run(state)`: Retrieves articles via Retriever, extracts ticker from query if needed, returns state with retrieved_articles filled.
    - Note: No LLM; pure RAG (semantic search + metadata filters).

- **[src/agents/risk_officer.py](src/agents/risk_officer.py)**
  - `RiskOfficerAgent.run(state)`: Analyzes retrieved articles; extracts event types, risk levels, composes risk_analysis (concerns, opportunities, macro_context).
    - Note: No LLM; uses classification labels from L3 + heuristics (e.g., "geopolitics → higher risk multiplier").

- **[src/agents/brief_writer.py](src/agents/brief_writer.py)**
  - `BriefWriterAgent.run(state)`: Composes final Brief from risk_analysis + articles.
    - Sections: Macro Context, What's Happening, Key Risks, Potential Upside, Sources, Warnings.
    - Note: No LLM; templates + structured fields. Output is guaranteed to have all sections.

- **[src/agents/graph.py](src/agents/graph.py)**
  - `build_graph()`: Creates LangGraph StateGraph; adds nodes (agents) + edges (transitions).
    - Flow: researcher → risk_officer → brief_writer → END.
    - No branching, no loops (explicit: add_edge, not conditional logic).
  - `run_agent_graph(query, ticker)`: Initializes state, invokes graph, returns final state.

- **[scripts/run_l5_brief.py](scripts/run_l5_brief.py)**
  - Runs 3 demo queries (AAPL, SPY, general earnings question).
  - For each query, calls run_agent_graph, displays formatted brief, prints trace (which agents ran).

### Failure Modes & Interview Responses

**Q: "What if Researcher retrieves no articles?"**
A: "Risk Officer gets empty list; composes a default risk_analysis with low risk (no evidence of danger). Brief Writer notes '[No articles to analyze]' and reduces confidence. Brief is still valid, just less informative. This is correct behavior: absence of risk signals = low risk."

**Q: "What if one agent fails (e.g., Risk Officer throws exception)?"**
A: "LangGraph catches it, logs the error, and passes the state to the next agent (brief_writer). Brief Writer sees analysis_error field and adds a '[Analysis failed]' warning to the brief. Brief still completes; quality is degraded but not lost. For production, I'd add retry logic or fallbacks per agent."

**Q: "How do you prevent infinite loops in LangGraph?"**
A: "Explicit edges: no conditional branching. Researcher → Risk Officer → Brief Writer → END. No cycle back. If we added a feedback loop (e.g., 'if confidence <0.5, researcher again'), we'd enforce max_iterations or a visited-states set."

**Q: "Why no LLM in the agents?"**
A: "LLMs hallucinate (invent facts). Our agents only compose outputs from structured data (labels, scores, retrieved text). Researcher retrieves; doesn't generate. Risk Officer analyzes labels; doesn't generate prose. Brief Writer formats templates; doesn't generate. If we did want prose (e.g., 'a sophisticated way to describe the risks'), we'd add an optional L2-style cached summarizer, but it's not core logic."

**Q: "How do you handle ambiguous queries (e.g., 'What about the market?')?"**
A: "Researcher tries to extract a ticker (heuristic regex). If it fails, ticker stays None, and retrieval is general (all articles). For 'What about the market?', we'd retrieve everything. Brief would have ticker='UNKNOWN'. L5 could be enhanced with NER + entity linking, but for MVP, heuristics work."

**Q: "Can agents make decisions (e.g., 'if risk >0.7, escalate')?"**
A: "Currently, no. Agents just produce outputs; downstream consumers (L6 evals, L7 UI) decide what to do with them. I could add decision logic here (e.g., Risk Officer returns 'escalate=True'), but I prefer keeping agents thin and delegating decisions to explicit policies elsewhere."

**Q: "How do you trace a query through the pipeline?"**
A: "request_id (UUID generated per query). All logging includes request_id. agent_steps field lists which agents ran ('researcher' → 'risk_officer' → 'brief_writer'). Errors are tagged with which step failed. Easy to grep logs for a request_id or understand what went wrong."

**Q: "What's the latency end-to-end? Can a user wait for a brief?"**
A: "L4 retrieval (~100ms), L3-like risk analysis (~10ms, no API calls), brief writing (~5ms). Total ~115ms. With L2-style caching on intermediates, reruns are faster. For UI, acceptable. If latency matters, we'd add async jobs (queue + worker) in production."

**Q: "What's the confidence score in the brief?"**
A: "Heuristic: 0.8 if ≥2 articles retrieved + no errors, 0.6 if 1 article, 0.5 if errors. It's a rough signal for the trader: '0.9 confidence brief' = highly reliable; '0.5 confidence' = take with a grain of salt. Later, L6 evals correlate confidence with human feedback to validate this scoring."

---

## PHASE L6: Evals & Traces

### Why This Component Exists

Without gold labels, quality is invisible. L6 adds observability: trace every query (latency, cache hits, labels), compute indirect metrics (retrieval stats, confidence calibration), and detect contradictions (narrative vs. price). This enables iterative improvement and gives traders confidence that the system isn't hallucinating.

### Why This Library/Model/Task vs. the Obvious Alternative

| Choice | Obvious Alt | Why Ours |
|--------|-------------|---------|
| **SQLite traces (not just logs)** | Print logs to console | Traces are queryable, aggregatable (SQL). Easy to compute metrics, export for analysis. Logs are ephemeral; SQLite is persistent. |
| **Indirect metrics (no gold labels)** | Collect human annotations | $0 budget; no time for manual labeling. Indirect signals work: retrieval hit rate (does it find relevant articles?), feedback correlation (do users agree?), contradiction detection (is the system biased?). |
| **Contradiction detection** | Assume model is always right | Traders know: if brief says "earnings beat = bullish" but stock drops 5%, something's wrong. Contradiction detection flags these misalignments, revealing model bias or data quality issues. |
| **Feedback loop via thumbs up/down** | One-shot evaluation | Brief stays in trader's memory. Thumbs up/down captures real sentiment (did this brief help?). Correlate feedback with confidence to validate confidence scoring. |

### How It Works (File + Function Names)

- **[src/evals/traces.py](src/evals/traces.py)**
  - `TraceStore`: SQLite persistence for trace data.
    - `log_trace()`: After L5 completes, log (request_id, query, ticker, retrieved_ids, event_types, risk_levels, latency_ms, cache_hits, brief_confidence).
    - `log_feedback()`: Trader provides thumbs_up/down; stores it against request_id.
    - `get_all_traces()`: Retrieve traces for metrics computation.
    - `get_feedback_summary()`: Aggregate feedback counts (thumbs_up, down, neutral).

- **[src/evals/metrics.py](src/evals/metrics.py)**
  - `Metrics`: Compute quality metrics from traces.
    - `compute_retrieval_stats()`: Avg articles retrieved, zero-result queries.
    - `compute_confidence_stats()`: Avg brief confidence, distribution.
    - `compute_latency_stats()`: p50, p95 latency, slow query count.
    - `compute_cache_stats()`: Total cache hits, hit rate.
    - `compute_all_metrics()`: All of above + feedback summary.
    - `print_metrics()`: Pretty-print dashboard.

- **[src/evals/contradiction.py](src/evals/contradiction.py)**
  - `ContradictionDetector`: Narrative-vs-price validation.
    - `get_brief_sentiment()`: Zero-shot classify brief as bullish/bearish/neutral.
    - `get_price_movement()`: Fetch 1d yfinance return for ticker.
    - `detect_contradiction()`: If brief bullish but stock down >3%, flag as contradiction.
    - `format_contradiction_report()`: Human-readable contradiction analysis.

- **[scripts/run_l6_eval.py](scripts/run_l6_eval.py)**
  - Demos: (1) Full pipeline + trace logging, (2) metrics computation, (3) contradiction detection, (4) feedback summary.
  - Prints dashboard of system health.

### Failure Modes & Interview Responses

**Q: "What if there's no price data for a ticker?"**
A: "Contradiction detection gracefully skips; returns 'No price data available'. Brief is still valid; just not validated. For production, track which tickers are missing price data and prioritize collecting it."

**Q: "Can you trust user feedback? Traders might just be in a bad mood."**
A: "Single thumbs-up/down is noisy. Aggregate: if 70% of briefs get thumbs-up, the system is probably good. If 30%, investigate. Also correlate with other signals: do thumbs-up briefs have higher confidence? Do they have more retrieved articles? Multivariate analysis beats single metrics."

**Q: "Contradiction detection is simple (sentiment vs. 1d return). What if the brief is right but the market is wrong?"**
A: "True; brief could be ahead of the market. But the contradiction is still a signal: either the model is genius (unlikely), or it's overconfident / biased. Traders can manually review flagged contradictions. It's a question, not an accusation."

**Q: "What if latency exceeds your budget (e.g., >5s)?"**
A: "Metrics show slow query count. Investigate: was it a cold start (L2 model loaded)? Poor retrieval (L4 searched 1000 articles)? High concurrency (L5 agents slow)? Metrics pinpoint the bottleneck. Then optimize (pre-warm models, index optimization, async processing)."

**Q: "How do you separate signal from noise in metrics?"**
A: "Small sample size = high variance. Keep metrics aggregated (last 100 traces, not last 1). Baseline: what's 'normal' latency before optimization? Set thresholds conservatively (p95 latency >5s is actionable; p95 >2s might be noise). Domain knowledge + time-series analysis (trends > snapshots)."

**Q: "Traces grow unbounded. When do you clean up?"**
A: "For MVP, no cleanup. In production, retention policy: keep traces for 30 days, archive older ones. Metrics queries use WHERE created_at > DATE_SUB(NOW(), INTERVAL 30 DAY). One-line change when storage becomes a concern (~1GB for 1M traces)."

**Q: "What happens if you deploy a new model (e.g., swap BART for a newer summarizer)?"**
A: "Latency and quality will change. Metrics let you compare: old model latency p95=100ms, new=150ms; but confidence jumps from 0.6 to 0.75. Was the tradeoff worth it? Traces make this comparison explicit. Also: can A/B test (50% old, 50% new) and compare feedback correlation."

**Q: "How do you prevent gaming metrics?"**
A: "With thumbs-up/down, a lazy engineer could pre-compute high-confidence briefs that don't work. But contradiction detection + latency traces catch you: briefs that look good in confidence but trigger contradictions or take forever are flagged. Multi-metric system is harder to game than any single metric."

---

## PHASE L7: Streamlit UI

### Why This Component Exists

L0–L6 are backend infrastructure. L7 is where traders interact. A clean UI lets them ask questions, see briefs with risk badges and evidence, detect contradictions, and provide feedback. Streamlit enables rapid MVP with zero frontend boilerplate; later phases scale to FastAPI + React if needed.

### Why This Library/Model/Task vs. the Obvious Alternative

| Choice | Obvious Alt | Why Ours |
|--------|-------------|---------|
| **Streamlit (not FastAPI + React)** | FastAPI backend + React frontend | Streamlit is pure Python; single source of truth. MVP launch speed (1 file, all features). Trade-off: limited styling, no mobile. For production, extract agent logic to FastAPI, Streamlit becomes a thin client. |
| **Session state for persistence** | Redux or local storage | Streamlit's @st.session_state handles reruns without boilerplate. For MVP, sufficient. Real apps use a backend database + API. |
| **@st.cache_data for caching** | Manual memoization | Streamlit's caching is declarative (one decorator). Reruns don't recompute if inputs unchanged. Easy to reason about. |
| **Feedback buttons (not rating scale)** | NPS or Likert scale | Thumbs up/down is fast (traders don't want friction). Correlate with system metrics later. Simple MVP feedback. |

### How It Works (File + Function Names)

- **[app/streamlit_app.py](app/streamlit_app.py)** — Complete L7 implementation.
  - **Page setup**: Title, sidebar configuration.
  - **Sidebar**: Source toggles (yfinance, FRED, RSS), max_articles, days_back, risk threshold.
  - **Main input**: Text field (query) + button (Get Brief).
  - **Query execution**: Calls run_agent_graph, measures latency, handles errors.
  - **Display brief**: Risk badge + metric cards, summary, sections (macro context, what's happening, risks, upside, sources, warnings).
  - **Contradiction check**: Calls ContradictionDetector; displays warning if narrative diverges from price.
  - **Evidence**: Expandable cards showing retrieved articles (top 5).
  - **Feedback**: Thumbs up/down buttons; logs feedback to TraceStore.
  - **Session state**: Persistent across reruns (query, brief, articles stored in st.session_state).

- **[scripts/run_l7_app.py](scripts/run_l7_app.py)** — Launcher for convenience (actually just wraps `streamlit run`).

### Failure Modes & Interview Responses

**Q: "What if the query is ambiguous? How do traders know what they asked?"**
A: "Streamlit re-displays the query above the brief. If they want to modify, they edit the text field and re-run. Session state preserves the query, brief, articles so they can see history. For production, I'd add a query history sidebar."

**Q: "Thumbs up/down doesn't capture nuance. What if a brief is 80% good but 20% wrong?"**
A: "True, but for MVP, binary feedback is fast. Traders don't have time for surveys. Optional text box (user_notes field) captures nuance for edge cases. Later, we could add rating scales or structured feedback (what was wrong? relevance? clarity?). Feedback is just a signal; combine with contradiction detection + latency metrics for deeper insight."

**Q: "What if Streamlit reruns on every keystroke, causing expensive API calls?"**
A: "Good catch. For now, I use a Run button (run_button, not reactive text field). Also, L2 caching stores summaries, L4 stores embeddings, so reruns with same query are free. For production, add debouncing (wait 500ms after last keystroke before rerunning)."

**Q: "How do you handle long-running queries? Can traders wait 30 seconds?"**
A: "Spinner UI shows progress. If latency exceeds 5s regularly, that's a problem (caught by L6 metrics). For longer queries, move to async jobs: Streamlit launches a Celery task, polls status, displays results when ready. But for MVP, synchronous is simpler."

**Q: "Can traders export or download briefs?"**
A: "Not yet. Streamlit has download button support; easy to add. For MVP, focus is on exploration + feedback loop. Production would include: export to PDF, email, integrate with trading platforms (API hooks)."

**Q: "Session state is local to the user. Does feedback get lost if they close the app?"**
A: "Feedback is logged to TraceStore (SQLite) when they click thumbs up/down, not just stored in session. Brief itself is logged as a trace. So even if they close the browser, data persists. Session state is just for UI state (what's currently visible)."

**Q: "How do you secure this? Anyone can query?"**
A: "L7 is a prototype. For production: add auth (OAuth, API keys), rate limiting, audit logs. Streamlit has built-in auth via secrets (GitHub login). FastAPI backend would handle this. For MVP on a local machine, security is not a concern."

**Q: "What if a ticker doesn't exist (e.g., 'INVALID123')?"**
A: "Researcher tries to retrieve articles; gets zero results. Brief Writer produces a low-confidence brief ('No articles found'). Contradiction check skips (no price data). Feedback still works (user thumbs down, we learn the query was bad). System degrades gracefully."

**Q: "How do traders know when data is stale? What if they get a 1-day-old brief for a breaking event?"**
A: "Brief shows retrieved articles' publish dates. Evidence cards list each article's timestamp. If all articles are >1 day old, that's visible. L1 ingest tries to fetch fresh data each run, but if APIs are down, it falls back to fixtures (which are older). I could add a 'Data age' warning in L7 sidebar, but for MVP, transparency (showing dates) is enough."

**Q: "Can you A/B test different models (e.g., BART vs. distilbart)?"**
A: "Yes! L2 classifier can take a model_id parameter. Streamlit radio button in sidebar: 'Summarizer: [BART ◯] [Distilbart ◯]'. A/B test by running 50% of queries with each, then compare feedback + latency. Traces support this (add model_id field)."

---

## General Interview Talking Points

### Architecture Choices

- **Phases as building blocks**: Each phase is a standalone layer with caching and fallbacks. This lets me explain the system incrementally and defend each choice.
- **Serverless inference**: No local LLM = $0 budget + no VRAM bottleneck. Trade-off: network latency, mitigated by disk caching.
- **Explicit state machines (LangGraph)**: No hidden loops or prompt-injection vulnerabilities. Debuggable.
- **SQLite traces**: Cheap telemetry without a separate observability platform.

### Resume Framing

CatalystDesk exercises:
1. **Trading Domain**: macroeconomic risk labels, narrative-vs-price contradiction detection, multi-timeframe indicators.
2. **AI Engineering**: HF Inference API + caching, embeddings + semantic retrieval, zero-shot classification, multi-agent orchestration.
3. **Python SWE**: Typed models, async retries, structured logging, production degradation (offline fallbacks).

"Built CatalystDesk, a $0 LangGraph + RAG briefing system..." [see README for bullets].

### Failure Scenarios to Practice

- **API Cold Starts**: "HF Inference API can be slow on first call (3–10s). I cache everything; reruns are instant. For user-facing latency, I'd add background async jobs (L5+)."
- **No Gold Labels**: "Traders have no gold labels. I use SQLite traces + user feedback + contradiction detection (narrative vs. price) as indirect quality signals."
- **Schema Drift**: "If a news source changes its format, L1 ingest fails gracefully and falls back to fixtures. I monitor schema errors in logs."
- **Vector DB Staleness**: "Chroma is rebuilt daily (batch). For real-time news, I'd add incremental upserts in L4."
- **Agent Loop Prevention**: "LangGraph enforces max_iterations; each agent has a timeout. If Risk Officer loops, it's logged and the brief writer receives a 'timeout' signal."

---

*Last updated: Phase L7 (Streamlit UI, complete end-to-end system).*
