# SEC-MCP

MCP server for SEC filing analysis with industry-aware XBRL extraction, BERT NLP, and Claude narratives.

## Quick Start

```bash
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # set EDGAR_IDENTITY at minimum

# MCP server (primary interface)
python -m sec_mcp.server

# Web dashboard
python -m sec_mcp.chat_app  # http://localhost:8877

# CLI testing
python test_tools.py search "Apple"
python test_tools.py financials AAPL 2024
```

## Architecture

```
src/sec_mcp/
├── server.py              # MCP v2 surface (exactly 9 tools) — MAIN ENTRY POINT
├── server_legacy.py       # pre-2026-06 22-tool surface (kept importable, not served)
├── surface/               # v2 tool implementations + response contract
│   ├── meta.py            #   {source, asOf, cacheHit, latencyMs} meta + {error, code, hint} errors
│   ├── session.py         #   market session + EDGAR-business-hours TTLs (mock _now() in tests)
│   ├── company_search.py  #   search_companies (filters: sector/cap/exchange/country/ipo/sp500)
│   ├── filings.py         #   get_filings (EFTS full-text) + get_filing_section (8-K item_X)
│   ├── fundamentals.py    #   get_fundamentals (TTM, cross-check, chartSeries, segments) + compare
│   ├── quotes.py          #   get_quote (session-aware TTL, never silently stale)
│   ├── ownership.py       #   get_insider_activity (Form 4) + get_ownership (13F + 13D/G)
│   └── screen.py          #   composable screener (valuation/growth/quality/events)
├── config.py              # Pydantic settings from .env
├── models.py              # Pydantic data models
│
├── sec_client.py          # Core SEC EDGAR HTTP client (rate-limited, cached)
├── edgar_client.py        # Thin wrapper delegating to sec_client (legacy)
├── section_segmenter.py   # 10-K/10-Q section boundary detection
│
├── financials.py          # XBRL extraction engine (industry-aware, 4-pass resolution)
├── xbrl_mappings.py       # ~250 XBRL concept → metric mappings, 5 industry classes
│
├── narrator.py            # Claude-powered narrative generation
├── chat_app.py            # FastAPI web dashboard (split-panel chat + analysis)
├── db.py                  # MongoDB persistence (graceful fallback if unavailable)
├── historical.py          # Multi-year filing retrieval
│
└── nlp/                   # NLP analysis (lazy-loaded, ~2.5GB models)
    ├── sentiment.py       # FinBERT financial sentiment
    ├── summarizer.py      # BART hierarchical summarization
    ├── ner.py             # Named entity recognition + regex
    └── chunker.py         # Token-aware text chunking
```

## Module Responsibilities

- **sec_client.py** is the canonical SEC API client. All EDGAR HTTP calls go through here. Rate-limited to 8 req/sec, with in-memory caching (tickers 30min, facts 5min, submissions 2min).
- **financials.py** handles all XBRL data extraction. Uses a 4-pass concept resolution strategy: exact match → contains match → custom extension → aggregate fallback. Industry-aware (bank, insurance, REIT, utility, standard, crypto).
- **server.py** only defines MCP tool handlers — no business logic here. Every v2 tool is wrapped by `surface.meta.tool_guard`: responses always carry a meta block, failures are always `{error, code, hint}` (never raw tracebacks), and a closed market is NEVER an error (quotes label `session: "closed"`).
- **Form 4 gotcha**: EDGAR's `primaryDocument` for Form 4s is the XSL-render path (`xslF345X06/form4.xml` = HTML); `insider_tracker.py` strips the `xsl…/` prefix to get raw XML. Don't "simplify" that away.
- **Segments**: companyfacts strips XBRL dimensions — authoritative segment data comes from `graph/segments.get_dimensional_segments` (per-filing), FMP as fallback.
- **db.py** is optional. The entire app works without MongoDB.
- **nlp/** models are excluded from the Docker image. NLP tools are unavailable in Railway production.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `EDGAR_IDENTITY` | Yes | `"Name email@example.com"` for SEC API |
| `ANTHROPIC_API_KEY` | No | Claude API for narrative explanations |
| `MONGODB_URI` | No | MongoDB connection for persistent cache |
| `PORT` | No | Server port (default: 8877) |

## Data Engine (Phase 1-4 upgrade, June 2026)

- **Golden suite is the gate**: `pytest tests/golden -m integration` — 10 hand-verified
  tickers (incl. FPIs). Any change to period selection or concept matching must keep it
  green. Rebuild goldens with `python scripts/build_golden_values.py` (as-originally-
  reported semantics). Regression snapshots: `python scripts/regression_compare.py
  snapshot|diff`.
- **Period selection** lives in `facts/periods.py`: facts are classified by their own
  start/end duration, never `fy`/`fp` (those describe the filing, not the fact).
  Latest-`filed` wins on duplicates.
- **Filing lists for periodic forms** come from `sec_client.get_periodic_filings*`
  (derived from companyfacts — full history) NOT `get_filings` (submissions `recent`
  spans only months for heavy filers like JPM).
- **Graph resolver** (`graph/`): per-filing calculation/presentation trees parsed via
  edgartools arbitrate which tag is THE total for each canonical metric. Controlled by
  `GRAPH_RESOLVER=off|shadow|on` (default off). Shadow logs disagreements to
  `resolver_diffs`. Graphs cached forever in `~/.sec_mcp_cache/_graphs/` + Supabase
  `filing_graphs`.
- **Quarterly history** (`get_fmp_shaped_history`): rows carry `_meta` (quality,
  confidence, sources, accession). Quality flags: `standalone`,
  `standalone_decumulated`, `ytd_fallback` (neighbour missing — value stays YTD),
  `q4_synthesized` (Q4 = FY − ΣQ1-3). Fineas consumes `/api/financials-history` on
  chat_app — its shape is a frozen contract; only add keys, never change them.
- **Query endpoints**: `/api/v1/metrics/{ticker}/{metric}`, `/api/v1/chart-data/{ticker}`,
  `/api/v1/concepts/{ticker}/{accession}` (the filing's calc tree — bad-match debugging).
  `/explorer` is the ECharts data-QA dashboard.
- **Supabase migration** `supabase/migrations/20260609000000_concept_graph_schema.sql`
  (sec_facts, concept graph, metric_observations) — code no-ops gracefully until applied.

## Testing

```bash
pytest                          # all tests
pytest tests/test_surface.py -m "not integration"  # v2 surface: clock + validation (offline)
pytest tests/test_surface.py -m integration        # v2 surface: live 5-ticker panel + Sunday-3am sim
pytest -m "not slow"            # skip NLP model tests
pytest -m "not integration"     # skip tests needing SEC network
ruff check src/ tests/          # lint
```

## Conventions

- Lazy-loaded singletons via `_thing: X | None = None` + `get_thing()` pattern
- Graceful degradation: app works without MongoDB, Claude API, or NLP models
- SEC rate limiting is enforced in `sec_client.py` — never bypass it
- All financial extraction goes through `financials.py`, never directly from SEC facts
- Use `pydantic` models from `models.py` for all structured data
