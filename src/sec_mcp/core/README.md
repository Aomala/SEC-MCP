# SEC-MCP V2 Core Modules

## Overview

The `core/` package provides high-level analysis and enrichment engines for SEC-MCP V2.
Each module is production-grade with comprehensive comments on every line of code,
following the project's engineering standards and design patterns.

## Modules

### 1. **market_data.py** — MarketDataProvider
Real-time stock market data via yfinance with graceful fallback.

**Key class:** `MarketDataProvider` (singleton pattern)

**Methods:**
- `get_price(ticker)` → dict with price, change, volume, market_cap, 52-week ranges, P/E
  - 5-minute TTL in-memory cache
  - Returns None if yfinance unavailable (graceful degradation)
  
- `get_valuation(ticker, xbrl_metrics)` → computed P/S, P/B, EV/EBITDA, EV/Revenue
  - Combines market cap with XBRL fundamentals
  - 15-minute TTL cache
  - Returns None if metrics incomplete

**Thread-safety:** Fully thread-safe with `threading.Lock` on caches

**Access pattern:**
```python
from sec_mcp.core import get_market_data_provider
provider = get_market_data_provider()
price = provider.get_price("AAPL")
valuation = provider.get_valuation("AAPL", xbrl_metrics={"revenue": 383e9, ...})
```

---

### 2. **filing_diff.py** — FilingDiffEngine
Compares financial metrics and filing sections between two periods.

**Key class:** `FilingDiffEngine`

**Methods:**
- `diff_metrics(ticker, year1, year2, form_type)` → metric changes with significance levels
  - Significance: "major" (>20%), "moderate" (>10%), "minor" (<10%)
  - Uses existing `extract_financials()` for both years
  - Returns None if either year unavailable

- `diff_sections(ticker, section, year1, year2, form_type)` → added/removed text + summary
  - Optional Claude narrative if API available (graceful fallback to word count diff)
  - Returns None if either year unavailable

**Pattern:** Stateless utility class; all methods are side-effect free.

**Access pattern:**
```python
from sec_mcp.core import FilingDiffEngine
engine = FilingDiffEngine()
metrics = engine.diff_metrics("AAPL", 2023, 2024)
sections = engine.diff_sections("AAPL", "risk_factors", 2023, 2024)
```

---

### 3. **peer_engine.py** — PeerEngine
Dynamic peer discovery and comparison engine.

**Key class:** `PeerEngine`

**PEER_MAP:** Curated industry groups (50+ groups covering Tech, Finance, Energy, etc.)

**Methods:**
- `find_peers(ticker, max_peers=5, criteria=None)` → list of comparable companies
  - Strategy 1: PEER_MAP lookup (primary, curated)
  - Strategy 2: Same SIC code (fallback)
  - Strategy 3: Custom peer list (override via criteria)
  - Returns ranked list with relevance scores (0-1)

- `get_peer_comparison(ticker, peer_tickers=None, year=None)` → comparison table
  - Uses `extract_financials_batch()` for parallel extraction
  - Adds rankings: best performer per metric
  - Returns None if data unavailable

**Pattern:** Stateless utility class; orchestrates existing modules.

**Access pattern:**
```python
from sec_mcp.core import PeerEngine
engine = PeerEngine()
peers = engine.find_peers("AAPL", max_peers=5)
comparison = engine.get_peer_comparison("AAPL", year=2024)
```

---

### 4. **screener.py** — Screener
Financial metric screening/filtering engine with parallel extraction.

**Key class:** `Screener`

**Supported metrics:** revenue, net_income, gross_margin, operating_margin, net_margin, roa, roe, total_assets, market_cap

**Methods:**
- `screen(filters, limit=50, max_candidates=100, year=None)` → companies matching all filters
  - Filters: `[{metric: "revenue", operator: ">", value: 1e9}, ...]`
  - Operators: `>`, `<`, `>=`, `<=`, `==`, `between`
  - Parallel extraction via ThreadPoolExecutor (max 10 workers)
  - Result caching by filter hash (30-min TTL)
  - Candidate limiting (max 100 checked by default)

**Important:** This is expensive (many network calls). Use `max_candidates` to limit scope.

**Access pattern:**
```python
from sec_mcp.core import Screener
screener = Screener()
results = screener.screen(
    filters=[
        {"metric": "revenue", "operator": ">=", "value": 1e9},
        {"metric": "net_margin", "operator": ">", "value": 0.1},
    ],
    limit=50,
    max_candidates=100,
)
```

---

### 5. **cache.py** — CacheLayer
Unified multi-tier cache with L1 (memory), L2 (disk), L3 (MongoDB).

**Key class:** `CacheLayer` (singleton pattern)

**Tiers:**
- **L1 (memory):** 5-min TTL, fastest, per-process
- **L2 (disk):** 1-hr TTL, persistent, uses existing `disk_cache` module
- **L3 (MongoDB):** 24-hr TTL, distributed, uses existing `db` module

**Methods:**
- `get(key, tier="l1")` → value from cache (checks L1→L2→L3, promotes on hit)
- `set(key, value, tier="l1")` → writes to tier + all tiers above
- `invalidate(key=None)` → removes from all tiers (None = clear all)
- `get_stats()` → returns cache statistics dict

**Features:**
- Automatic promotion: cache hits are re-stored in higher tiers
- Graceful degradation: works without L2/L3 if unavailable
- Thread-safe: all operations protected by `threading.Lock`

**Access pattern:**
```python
from sec_mcp.core import get_cache
cache = get_cache()
value = cache.get("AAPL_financials_2024", tier="l1")
cache.set("AAPL_financials_2024", data, tier="l3")  # Writes to L1+L2+L3
cache.invalidate("AAPL_financials_2024")  # Removes from all tiers
```

---

## Design Principles

All modules follow SEC-MCP project conventions:

1. **Lazy Singleton Pattern**
   - Global `_instance: ClassName | None = None`
   - `get_instance() -> ClassName` function for thread-safe access
   - Example: `get_market_data_provider()`, `get_cache()`

2. **Graceful Degradation**
   - All external dependencies (yfinance, MongoDB, Claude API) are optional
   - Functions return None or [] instead of raising exceptions
   - Warnings logged for debugging, but app continues working

3. **Thread-Safety**
   - All shared state protected by `threading.Lock`
   - No race conditions in cache access or counter updates

4. **Comprehensive Comments**
   - Every line of code has a comment explaining what it does
   - Docstrings for all public methods with Args, Returns, Raises
   - Type hints for all parameters and return values

5. **Logging and Observability**
   - All modules use `log = logging.getLogger(__name__)`
   - Debug-level logs for normal operations (cache hits, extractions)
   - Warning-level logs for degraded behavior (fallbacks, timeouts)
   - Error-level logs for failures

## Integration Examples

### Example 1: Market Data + Peer Comparison
```python
from sec_mcp.core import get_market_data_provider, PeerEngine

# Get market prices
market = get_market_data_provider()
price_data = market.get_price("AAPL")

# Find peers
peers = PeerEngine()
peer_list = peers.find_peers("AAPL", max_peers=4)

# Compare with peers
comparison = peers.get_peer_comparison("AAPL")

# Get valuations with XBRL metrics
valuation = market.get_valuation("AAPL", xbrl_metrics=comparison["comparison"]["revenue"])
```

### Example 2: Screening + Filing Diff
```python
from sec_mcp.core import Screener, FilingDiffEngine

# Screen for profitable tech companies
screener = Screener()
results = screener.screen(
    filters=[
        {"metric": "net_margin", "operator": ">", "value": 0.15},
        {"metric": "revenue", "operator": ">=", "value": 5e9},
    ],
    max_candidates=100,
)

# Compare last 2 years for top results
diff_engine = FilingDiffEngine()
for result in results[:5]:
    ticker = result["ticker"]
    metrics_diff = diff_engine.diff_metrics(ticker, 2023, 2024)
    sections_diff = diff_engine.diff_sections(ticker, "mda", 2023, 2024)
    print(f"{ticker}: {metrics_diff['summary']}")
```

### Example 3: Multi-Tier Caching
```python
from sec_mcp.core import get_cache

cache = get_cache()

# Check cache (L1 fastest, then L2, then L3)
key = "AAPL_financials_2024"
cached = cache.get(key, tier="l1")
if cached:
    print("Cache hit!")
else:
    # Miss — extract and cache for next time
    data = extract_financials("AAPL", year=2024)
    cache.set(key, data, tier="l3")  # Stores in all 3 tiers
```

## Testing

Each module is designed to be testable in isolation:

```bash
# Test imports
python3 -c "from sec_mcp.core import MarketDataProvider, FilingDiffEngine, PeerEngine, Screener, CacheLayer"

# Test specific module
python3 -c "from sec_mcp.core import PeerEngine; pe = PeerEngine(); peers = pe.find_peers('AAPL'); print(len(peers))"
```

## Error Handling

All modules follow a consistent error handling philosophy:

- **No exceptions raised to caller** for external API failures
- **Graceful fallback** with None or [] returns
- **Logging at appropriate levels:**
  - DEBUG: Normal operations (cache hits, successful extractions)
  - WARNING: Degraded behavior (missing optional data, fallback modes)
  - ERROR: Critical failures (should not happen)

This allows upstream code to handle gracefully degraded results without try/except blocks.

## Performance Notes

- **MarketDataProvider:** O(1) for cached access, ~500ms for fresh yfinance call
- **PeerEngine:** O(1) for PEER_MAP lookup, O(n) for SIC code search
- **Screener:** O(n*m) for n candidates × m filters, parallel extraction reduces wall-clock time
- **CacheLayer:** O(1) for L1, O(10-100ms) for L2 disk I/O, O(100-1000ms) for L3 network
- **FilingDiffEngine:** O(2) network calls (one per year) + text diffing (linear in text size)

## Future Enhancements

- [ ] Add Redis support for distributed L2 cache
- [ ] Implement cache warming strategies (preload popular tickers)
- [ ] Add metric computation formulas (EBITDA from components)
- [ ] Screener: add historical backtesting support
- [ ] PeerEngine: add ML-based relevance scoring
- [ ] FilingDiffEngine: add sentiment analysis of section changes
