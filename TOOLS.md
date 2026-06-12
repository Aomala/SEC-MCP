# fineasmcp — Tool Surface (v2)

Always-on financial research MCP server. Single source of truth for company
info: SEC filings, fundamentals, prices, ownership, insider activity —
queryable 24/7 with rich filters.

**Server:** `python -m sec_mcp.server` (STDIO) or `--sse` (remote).
**Surface:** exactly 9 tools, defined in `src/sec_mcp/server.py`, implemented in `src/sec_mcp/surface/`.

## Universal contract

Every response carries a **meta block**:

```json
"meta": { "source": "edgar:xbrl_companyfacts", "asOf": "2026-06-12T04:52:06+00:00", "cacheHit": false, "latencyMs": 166 }
```

Every failure is **structured** — raw stack traces never reach the client:

```json
{ "error": "'period' must be one of ['annual', 'quarterly', 'ttm'], got 'hourly'.",
  "code": "INVALID_INPUT",
  "hint": "Example: period='annual'.",
  "meta": { ... } }
```

Error codes: `INVALID_INPUT` · `UNKNOWN_TICKER` · `NOT_FOUND` · `UPSTREAM_ERROR` · `UNAVAILABLE` · `INTERNAL`.

**24/7 guarantee:** markets being closed is never an error. Quotes return the
last close labeled `session: "closed"`; filings, fundamentals, ownership and
screening answer identically at any hour (proven by the simulated
Sunday-03:00 suite in `tests/test_surface.py::TestSunday3am`).

---

## 1. search_companies(query, filters?, limit?)

Ranked company search over every SEC filer. Filters: `sector`, `industry`,
`market_cap_min/max`, `exchange`, `country`, `ipo_date_after` (proxied by
first EDGAR filing date), `is_sp500`. Query may be empty if a filter is given.

```python
search_companies("nvidia")
```
```json
{ "query": "nvidia", "count": 1,
  "results": [{ "ticker": "NVDA", "name": "NVIDIA CORP", "cik": "0001045810",
                "exchange": "Nasdaq", "score": 40.0 }],
  "meta": { "source": "edgar:company_tickers_exchange", "latencyMs": 166, ... } }
```

Enrichment-dependent filters (sector/country/market-cap/IPO) evaluate the top
40 ranked candidates and say so in `note` — no silent truncation.

## 2. get_filings(ticker_or_cik?, form_type?, date_from?, date_to?, full_text_query?, limit?)

Filing index (submissions API) or EDGAR full-text search when
`full_text_query` is set. Form aliases include FPI equivalents automatically
(`10-K` also matches `20-F`; `S-1` matches `F-1`). Supported aliases:
`10-K, 10-Q, 8-K, S-1, 13F, 13D, 13G, 13D/G, DEF 14A, 4` — or any literal
EDGAR form name.

```python
get_filings("AAPL", form_type="10-K", limit=1)
```
```json
{ "mode": "index", "cik": "0000320193", "count": 1,
  "filings": [{ "accession": "0000320193-25-000079", "form": "10-K",
                "filingDate": "2025-10-31", "acceptedAt": "2025-10-31T10:01:26.000Z",
                "reportDate": "2025-09-27", "items": null,
                "url": "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm" }],
  "meta": { "source": "edgar:submissions", "cacheHit": false, ... } }
```

Full-text mode (`full_text_query="supply chain disruption", form_type="8-K"`)
returns the same shape with `mode: "full_text"` plus `entityNames` per hit
(EFTS coverage starts 2001; acceptance timestamps are null on this path).

The filings index cache is business-hours aware: **60 s TTL while EDGAR
accepts filings (06:00–22:00 ET weekdays), 600 s off-hours.**

## 3. get_filing_section(accession, section, ticker_or_cik?, max_length?)

Clean section text — no HTML tags, entities, or NBSP artifacts. Sections:
`risk_factors`, `mdna`, `business`, `financial_statements`, or `item_X` for
8-Ks (e.g. `item_2.02`). Filer resolution: explicit `ticker_or_cik` → the
accession's own CIK prefix → EFTS lookup.

```python
get_filing_section("0000320193-25-000079", "risk_factors")
```
```json
{ "accession": "0000320193-25-000079", "cik": "0000320193",
  "section": "risk_factors",
  "text": "Item 1A.  Risk Factors\n\nThe following summarizes factors that could have a material adverse effect on the Company's bu…",   // 68,162 chars
  "meta": { "source": "edgar:archives+section_segmenter", ... } }
```

An 8-K item that the filing doesn't contain returns
`{ code: "NOT_FOUND", hint: "Check the filing's items list from get_filings…" }`.

## 4. get_fundamentals(ticker, period?, metrics?, periods_back?, include_segments?)

XBRL companyfacts via the industry-aware extraction engine.
`period`: `annual` | `quarterly` | `ttm` (trailing-4-quarter sums, balance
items from the latest quarter). `metrics` subsets from:
`revenue, grossProfit, operatingIncome, netIncome, ebitda, eps, epsDiluted,
totalAssets, totalLiabilities, totalEquity, cashAndEquivalents, totalDebt,
operatingCashFlow, capex, freeCashFlow, grossMargin, operatingMargin,
netMargin, sharesOutstanding`.

```python
get_fundamentals("AAPL", period="annual", periods_back=2)
```
```json
{ "ticker": "AAPL", "period": "annual",
  "periods": [{ "endDate": "2025-09-27", "fiscalYear": 2025, "fiscalPeriod": "FY",
                "accession": "0000320193-25-000079", "quality": "standalone",
                "metrics": { "revenue": 416161000000.0, "netMargin": 0.2692, ... } }, ...],
  "crossCheck": { "provider": "polygon", "status": "ok",
                  "checks": [{ "metric": "revenue", "sec": 416161000000.0,
                               "polygon": 416161000000.0, "deltaPct": 0.0, "ok": true }] },
  "chartSeries": { "labels": ["FY 2024", "FY 2025"],
                   "revenue": [391035000000.0, 416161000000.0],
                   "netIncome": [...], "freeCashFlow": [...], "netMargin": [...] },
  "segments": {
    "product":    [{ "name": "iPhone", "value": 209586000000.0, "pct": 50.4 }, ...],
    "geographic": [{ "name": "U.S.", "value": 151790000000.0, "pct": 36.5 },
                   { "name": "China", "value": 64377000000.0, "pct": 15.5 }, ...],
    "segmentSeries": null },
  "meta": { "source": "edgar:xbrl_companyfacts", ... } }
```

- **Cross-check** compares SEC values to Polygon's standardized financials
  with *period alignment* (±10 days on period end) at 2 % tolerance —
  `ok` / `mismatch` / `unavailable`, never silently merged.
- **Chart enrichment:** `chartSeries` arrays plot directly (oldest→newest);
  `segments.geographic`/`.product` feed pies/treemaps (name+value+pct);
  `segmentSeries.geographic` (FMP, when available) feeds stacked areas.
- Segments come from **dimensional XBRL parsed from the filing itself**
  (companyfacts strips dimensions), with FMP fallback.
- Fundamentals are cached **per accession** (immutable) — effectively
  "until the next filing for that CIK".

## 5. get_quote(ticker)

Provider chain Polygon → yfinance → FMP. Mandatory freshness metadata —
never silently stale.

```python
get_quote("AAPL")          # run at 00:52 ET (markets closed)
```
```json
{ "ticker": "AAPL", "price": 295.63, "change": ..., "changePct": ...,
  "volume": ..., "asOf": "2026-06-12T04:52:06.810295+00:00",
  "session": "closed", "provider": "polygon", "ageSeconds": 0.0,
  "priceBasis": "last_close",
  "meta": { "source": "polygon", "latencyMs": 54, ... } }
```

Session-aware cache TTL: 30 s regular · 120 s pre/after · 3600 s closed.
Sessions: `pre` 04:00–09:30 ET · `regular` 09:30–16:00 · `after` 16:00–20:00 ·
`closed` overnight/weekends/NYSE holidays. A closed market is **never** an
error; a truly unknown ticker is `UPSTREAM_ERROR` with a hint.

## 6. get_insider_activity(ticker, date_from?, limit?)

Parsed Form 4s from EDGAR (raw XML, not the XSL-rendered page).

```python
get_insider_activity("AAPL", date_from="2026-01-01", limit=8)
```
```json
{ "ticker": "AAPL", "dateFrom": "2026-01-01", "count": 8,
  "summary": { "buyCount": 0, "sellCount": 5, "netShares": -302808 },
  "transactions": [{ "insiderName": "LEVINSON ARTHUR D", "role": null,
                     "side": "sell", "transactionType": "Sale",
                     "date": "2026-05-27", "shares": 50000, "price": 311.02,
                     "value": 15551000.0, "sharesOwnedAfter": 3764576,
                     "filingDate": "2026-05-29", "accession": "0001140361-26-023363" }, ...],
  "meta": { "source": "edgar:form4", ... } }
```

`side` normalizes Form 4 codes: `buy` (P) · `sell` (S, tender dispositions) ·
`other` (grants, exercises, gifts, tax withholding). An empty window is a
valid answer (`note` explains), not an error.

## 7. get_ownership(ticker)

13F institutional holders (aggregated dataset) + SC 13D/13G blockholder
filings from the subject company's EDGAR index.

```python
get_ownership("AAPL")
```
```json
{ "ticker": "AAPL", "cik": "0000320193",
  "institutionalHolders": { "status": "ok", "source": "yfinance(13F aggregate)",
    "holders": [{ "institution": "Blackrock Inc.", "reportDate": "2026-03-31",
                  "shares": 1144695425, "value": 338406314082.0, "pctHeld": 7.79 }, ...] },
  "beneficialOwners": { "source": "edgar:subject-company SC 13D/G index",
    "filings": [{ "form": "SC 13G/A", "kind": "passive", "filingDate": "2026-02-13",
                  "accession": "...", "url": "https://www.sec.gov/..." }, ...] },
  "meta": { "source": "yfinance(13F)+edgar(13D/G)", ... } }
```

`kind`: `activist` (13D — intent to influence) vs `passive` (13G).

## 8. screen(filters, limit?)

Composable screener over the curated 300+ ticker universe. All filters AND:

| Group | Filters |
|---|---|
| valuation | `pe_max`, `ev_ebitda_max` |
| growth | `rev_growth_min` (0.15 = 15 % YoY) |
| quality | `fcf_positive` (bool), `net_debt_ebitda_max` |
| events | `filed_8k_last_7d` (bool), `insider_buying_last_30d` (bool) |
| scoping | `sector`, `market_cap_min`, `market_cap_max` |

```python
screen({"sector": "semiconductors", "fcf_positive": True, "pe_max": 40}, limit=5)
```
```json
{ "count": 2,
  "matches": [{ "ticker": "QCOM", "name": "QUALCOMM INC", "revenue": ...,
                "fcf": ..., "revGrowth": 0.1366, "pe": 39.3, "marketCap": ... },
              { "ticker": "SWKS", "pe": 22.7, ... }],
  "coverage": { "universeSize": 22, "candidatesEvaluated": 22,
                "eventChecksRun": null, "note": null },
  "meta": { ... } }
```

Stages run cheapest-first (cached XBRL → live quotes → EDGAR events) and the
`coverage` block always reports exactly what was evaluated. Missing data
fails conservatively (a company with unknown FCF never passes `fcf_positive`).

## 9. compare(tickers[], metrics?, period?)

Side-by-side fundamentals for 2–8 companies. Each row carries its own fiscal
period end — cross-company alignment is explicit, never assumed.

```python
compare(["AAPL", "MSFT"], metrics=["revenue", "netMargin", "freeCashFlow"])
```
```json
{ "period": "annual", "metrics": ["revenue", "netMargin", "freeCashFlow"],
  "normalization": "latest completed fiscal period per company (see endDate per row)",
  "rows": [
    { "ticker": "AAPL", "endDate": "2025-09-27", "fiscalPeriod": "FY 2025",
      "values": { "revenue": 416161000000.0, "netMargin": 0.2692, "freeCashFlow": 98767000000.0 } },
    { "ticker": "MSFT", "endDate": "2025-06-30", "fiscalPeriod": "FY 2025",
      "values": { "revenue": 281724000000.0, "netMargin": 0.3615, "freeCashFlow": 71611000000.0 } }],
  "failures": null,
  "meta": { ... } }
```

Per-ticker failures are reported in `failures` without killing the call.

---

## Reliability internals

| Layer | Policy |
|---|---|
| SEC compliance | Declared `EDGAR_IDENTITY` User-Agent, 8 req/s ceiling (SEC allows 10), exponential backoff on 429/500/502/503/504 — all centralized in `sec_client.py` |
| Filings index | 60 s TTL during EDGAR acceptance hours (06:00–22:00 ET weekdays), 600 s off-hours |
| Fundamentals | Disk-cached per accession (filings are immutable) → "cached until the next filing for that CIK" |
| Quotes | Session-aware TTL: 30 s regular / 120 s pre+after / 3600 s closed |
| Sessions | `surface/session.py`, NYSE holiday table 2025–2027, all clock logic behind one mockable `_now()` |

## Testing

```bash
.venv/bin/python -m pytest tests/test_surface.py -m "not integration"  # logic: clock + validation (offline)
.venv/bin/python -m pytest tests/test_surface.py -m integration       # live: 5-ticker panel + Sunday-3am sim
pytest tests/golden -m integration                                    # extraction golden suite (unchanged gate)
```
