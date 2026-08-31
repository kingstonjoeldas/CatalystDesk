# Sample Fixtures

Offline-first architecture: when live APIs are unavailable (or during development), CatalystDesk falls back to these pre-indexed sample articles.

## Contents

`sample_articles.json`: 8 realistic news items spanning:
- **Company Earnings** (AAPL, JPM): revenue, EPS, margin, guidance signals
- **Economic Indicators** (CPI, unemployment, Fed funds): macro breadth
- **Sector News** (Energy, Gold, Semiconductors): commodity + tech sector moves
- **Policy** (Fed Powell speech): central bank communication

All items follow the [NewsItem schema](../../README.md#data-schema).

## Why These Articles?

They cover the **7 risk-relevant labels** used in L3 classification:
1. **Monetary Policy** (Fed speech, Powell)
2. **Inflation Data** (CPI)
3. **Earnings** (AAPL, JPM)
4. **Liquidity/Credit** (JPM credit losses)
5. **Geopolitics** (Middle East, Chinese growth)
6. **Regulation** (Export restrictions on chips)
7. **Other** (Technical analysis on gold)

And they span **multiple tickers** (AAPL, NVDA, SPY, TLT, GLD, etc.) to test RAG filtering by ticker.

## Usage in Development

- **L1 ingest script** reads this file as fallback if live APIs fail
- **L2 summarization** caches summaries of these articles
- **L3 classification** produces labels for each item
- **L4 RAG** embeds and stores these articles in Chroma
- **L5 agents** query retrieval based on user query
- **L6 evals** use these articles for offline metric computation

## How to Extend

Add more fixture articles to test:
- Different geographic regions (Asia, Europe)
- Different time horizons (today, 1 week, 1 month old)
- Edge cases (very short, very long, ambiguous risk)
- Real contradictions (headline bullish, price action bearish)

Format: JSON list of NewsItem objects. Ensure `id` is unique across all samples.
