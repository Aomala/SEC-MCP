# SEC-MCP → Fineas.ai V2: Architecture Upgrade Plan

## Current State (V1)
- **Frontend**: Vanilla JS/HTML/CSS, Chart.js, split-panel layout
- **Backend**: FastAPI + FastMCP, 17 MCP tools, XBRL extraction
- **Gaps**: No stock prices, no trend charts, hardcoded peers, no filing diffs, no export

---

## FRONTEND: What Changes

### V1 → V2 Comparison

| Feature | V1 (Current) | V2 (Proposed) |
|---------|-------------|---------------|
| Framework | Vanilla JS | React + Tailwind |
| Charts | Chart.js (basic) | Recharts + D3 (interactive) |
| Layout | Split panel (chat left, data right) | Sidebar nav + dashboard + AI panel |
| State | Global vars (`_tk`, `_curData`) | React state + context + cache |
| Routing | None (view modes via JS) | React Router (deep links) |
| Data fetching | Fetch + manual cache | React Query (auto-cache, refetch) |
| Theming | CSS vars toggle | Tailwind dark mode classes |
| Export | None | PDF/CSV/XLSX download |
| Responsiveness | Partial | Full mobile-first |
| Keyboard | None | Cmd+K search, shortcuts |

### New Frontend Features
1. **Company Dashboard** — 6 KPI cards + trend charts + segment pie + peer table
2. **Screener** — Filter companies by metrics (margin > 20%, revenue > $10B)
3. **Filing Diff** — Side-by-side YoY changes with highlights
4. **Watchlist** — Save tickers, get change alerts
5. **Export** — One-click PDF report, CSV data, XLSX financials
6. **Keyboard Navigation** — `/` to search, `Esc` to close panels, arrow keys

---

## BACKEND: What Changes

### New Architecture

```
src/sec_mcp/
├── server.py              # MCP tools (keep, expand)
├── config.py              # Pydantic settings (keep)
├── models.py              # Data models (expand)
│
├── api/                   # NEW — REST API layer
│   ├── __init__.py
│   ├── app.py             # FastAPI app factory
│   ├── routes/
│   │   ├── companies.py   # /api/v1/companies/{ticker}
│   │   ├── financials.py  # /api/v1/financials/{ticker}
│   │   ├── filings.py     # /api/v1/filings/{ticker}
│   │   ├── compare.py     # /api/v1/compare
│   │   ├── screener.py    # /api/v1/screener
│   │   ├── chat.py        # /api/v1/chat (streaming)
│   │   └── export.py      # /api/v1/export/{format}
│   ├── middleware.py       # CORS, auth, rate-limit
│   └── deps.py            # Dependency injection
│
├── core/                  # Business logic (refactored from flat files)
│   ├── sec_client.py      # SEC EDGAR client (keep)
│   ├── financials.py      # XBRL engine (keep)
│   ├── market_data.py     # NEW — Stock prices + market cap
│   ├── screener.py        # NEW — Multi-metric filtering
│   ├── filing_diff.py     # NEW — YoY change detection
│   ├── peer_engine.py     # NEW — Dynamic peer selection
│   ├── export_engine.py   # NEW — PDF/CSV/XLSX generation
│   └── cache.py           # Unified caching layer
│
├── nlp/                   # NLP (keep, add Claude-first)
│   ├── sentiment.py
│   ├── summarizer.py
│   ├── ner.py
│   └── chunker.py
│
├── narrator.py            # Claude narratives (keep)
├── db.py                  # MongoDB (keep)
└── historical.py          # Multi-year data (keep)
```

### 7 New Backend Capabilities

#### 1. Market Data Integration (`core/market_data.py`)
**Why**: Can't compute P/E, EV/EBITDA, or market cap without stock prices.

```python
# Uses yfinance (free) or Alpha Vantage (API key)
class MarketDataProvider:
    async def get_price(self, ticker: str) -> PriceData:
        # Returns: price, change, change_pct, volume, market_cap, 52w_high/low
        ...

    async def get_valuation(self, ticker: str, metrics: dict) -> ValuationMetrics:
        # Computes: P/E, P/S, P/B, EV/EBITDA, EV/Revenue
        # Combines market price with XBRL fundamentals
        ...
```

#### 2. Filing Diff Engine (`core/filing_diff.py`)
**Why**: Users want to know "what changed since last year?"

```python
class FilingDiffEngine:
    async def diff_metrics(self, ticker: str, year1: int, year2: int) -> MetricDiff:
        # Returns: {metric: {old, new, change, change_pct, significance}}
        ...

    async def diff_sections(self, ticker: str, section: str, year1: int, year2: int) -> TextDiff:
        # Returns: added_paragraphs, removed_paragraphs, changed_paragraphs
        # Uses Claude to summarize key changes
        ...
```

#### 3. Dynamic Peer Engine (`core/peer_engine.py`)
**Why**: Hardcoded peer map doesn't scale. Users want custom comparisons.

```python
class PeerEngine:
    async def find_peers(self, ticker: str, criteria: PeerCriteria) -> list[PeerMatch]:
        # Strategy 1: Same SIC code + similar market cap (default)
        # Strategy 2: User-defined list
        # Strategy 3: Claude-suggested based on business model
        # Returns: ranked list with relevance scores
        ...
```

#### 4. Screener (`core/screener.py`)
**Why**: "Show me all tech companies with margins > 25% and revenue > $50B"

```python
class Screener:
    async def screen(self, filters: list[ScreenFilter]) -> list[ScreenResult]:
        # Filters: metric comparisons, industry, SIC, exchange
        # Pre-indexed universe of ~4000 SEC filers
        # Returns: matching companies sorted by relevance
        ...
```

#### 5. Streaming Chat (`api/routes/chat.py`)
**Why**: Current chat blocks until full response. Users want real-time tokens.

```python
@router.post("/api/v1/chat")
async def chat_stream(request: ChatRequest):
    # Uses Server-Sent Events (SSE) for token streaming
    # Shows "thinking" steps: searching → extracting → analyzing → responding
    async def generate():
        yield {"event": "thinking", "data": "Searching SEC EDGAR for AAPL..."}
        yield {"event": "thinking", "data": "Extracting XBRL financials..."}
        yield {"event": "token", "data": "Apple's revenue..."}
    return StreamingResponse(generate(), media_type="text/event-stream")
```

#### 6. Export Engine (`core/export_engine.py`)
**Why**: Users need to take data out — PDF reports, CSV for Excel, XLSX for models.

```python
class ExportEngine:
    async def to_pdf(self, ticker: str, sections: list[str]) -> bytes:
        # Professional PDF: cover page + financials + charts + narrative
        ...

    async def to_csv(self, ticker: str) -> bytes:
        # Raw metrics + ratios + segments as flat CSV
        ...

    async def to_xlsx(self, ticker: str) -> bytes:
        # Multi-sheet Excel: Income Statement, Balance Sheet, Cash Flow, Ratios
        ...
```

#### 7. Unified Cache Layer (`core/cache.py`)
**Why**: Current caching is scattered (in-memory dicts in sec_client, MongoDB in db.py, disk cache). Needs one interface.

```python
class CacheLayer:
    # L1: In-memory (hot data, 5min TTL)
    # L2: Redis/disk (warm data, 1hr TTL)
    # L3: MongoDB (cold data, 24hr TTL)
    # Automatic promotion/eviction

    async def get(self, key: str, ttl_tier: str = "warm") -> Any: ...
    async def set(self, key: str, value: Any, tier: str = "warm") -> None: ...
    async def invalidate(self, pattern: str) -> None: ...
```

---

## NEW MCP TOOLS (added to server.py)

| Tool | Purpose |
|------|---------|
| `get_stock_price` | Current price + change + volume |
| `get_valuation_metrics` | P/E, P/S, EV/EBITDA from price + XBRL |
| `diff_financials` | YoY metric changes between two filings |
| `diff_filing_section` | Text changes in Risk Factors, MD&A, etc. |
| `find_peers` | Dynamic peer discovery by SIC/metrics |
| `screen_companies` | Filter universe by financial criteria |
| `export_report` | Generate PDF/CSV/XLSX for a company |

---

## API DESIGN (REST, versioned)

```
GET    /api/v1/companies/{ticker}           → CompanyProfile
GET    /api/v1/companies/{ticker}/price      → PriceData
GET    /api/v1/financials/{ticker}           → StandardizedFinancials
GET    /api/v1/financials/{ticker}/history   → [StandardizedFinancials] (multi-year)
GET    /api/v1/financials/{ticker}/diff      → MetricDiff (year1 vs year2)
GET    /api/v1/filings/{ticker}             → [FilingMetadata]
GET    /api/v1/filings/{ticker}/section     → SectionText
POST   /api/v1/compare                      → ComparisonResult
POST   /api/v1/screener                     → [ScreenResult]
POST   /api/v1/chat                         → SSE stream
GET    /api/v1/export/{ticker}/{format}     → File download
```

---

## IMPLEMENTATION ROADMAP

### Phase 1: Foundation (Week 1-2)
- [ ] Scaffold React app with Vite + Tailwind
- [ ] Build component library (MetricCard, Chart wrappers, Table, Badge)
- [ ] Implement API routes layer (`api/` directory)
- [ ] Add market data module (yfinance)
- [ ] Unified cache layer

### Phase 2: Core Features (Week 3-4)
- [ ] Company dashboard page (KPIs + charts + segments)
- [ ] Streaming chat with SSE
- [ ] Filing diff engine (metrics + text)
- [ ] Dynamic peer engine
- [ ] Financial statements view (IS, BS, CF tables)

### Phase 3: Power Features (Week 5-6)
- [ ] Screener (multi-metric filtering)
- [ ] Export engine (PDF, CSV, XLSX)
- [ ] Watchlist + alerts
- [ ] Keyboard shortcuts (Cmd+K)
- [ ] Dark mode

### Phase 4: Polish (Week 7-8)
- [ ] Mobile responsive
- [ ] Performance optimization (React.memo, virtual lists)
- [ ] E2E tests (Playwright)
- [ ] Deploy: Vercel (frontend) + Railway (backend)
- [ ] Documentation

---

## TECH STACK UPGRADE

| Layer | V1 | V2 |
|-------|----|----|
| Frontend | Vanilla JS | React 18 + Vite + Tailwind |
| Charts | Chart.js | Recharts + D3 |
| State | Global vars | React Query + Zustand |
| Backend | FastAPI (monolith) | FastAPI (modular routes) |
| Market Data | None | yfinance + Alpha Vantage |
| Cache | Scattered | Unified L1/L2/L3 |
| Export | None | ReportLab (PDF) + openpyxl (XLSX) |
| Deploy | Railway | Vercel + Railway |
| Tests | pytest (basic) | pytest + Playwright E2E |
