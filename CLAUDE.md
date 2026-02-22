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
├── server.py              # MCP tool definitions (17 tools) — MAIN ENTRY POINT
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
- **server.py** only defines MCP tool handlers — no business logic here.
- **db.py** is optional. The entire app works without MongoDB.
- **nlp/** models are excluded from the Docker image. NLP tools are unavailable in Railway production.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `EDGAR_IDENTITY` | Yes | `"Name email@example.com"` for SEC API |
| `ANTHROPIC_API_KEY` | No | Claude API for narrative explanations |
| `MONGODB_URI` | No | MongoDB connection for persistent cache |
| `PORT` | No | Server port (default: 8877) |

## Testing

```bash
pytest                          # all tests
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
