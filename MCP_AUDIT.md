# MCP_AUDIT.md — fineasmcp (sec-mcp) live tool audit

- **Run started (UTC):** 2026-06-12T04:39:38+00:00
- **Ticker panel:** AAPL (mega-cap) · CAVA (mid-cap, 2023 IPO) · CRWV (recent IPO Mar-2025) · ASML (foreign private issuer, 20-F) · SMCI (recent 10:1 split + filing delays)
- **Timeout per call:** 120s

## Summary

| Tool | Source | Pass | Soft-fail | Crash/Timeout |
|---|---|---|---|---|
| `search_company` | EDGAR company_tickers.json | 5 | 0 | 0 |
| `get_filing_list` | EDGAR submissions API | 4 | 1 | 0 |
| `get_financials` | EDGAR XBRL companyfacts | 5 | 0 | 0 |
| `get_financials_batch` | EDGAR XBRL companyfacts | 1 | 0 | 0 |
| `get_income_statement` | EDGAR XBRL companyfacts | 0 | 2 | 0 |
| `get_balance_sheet` | EDGAR XBRL companyfacts | 0 | 2 | 0 |
| `get_cash_flow` | EDGAR XBRL companyfacts | 0 | 2 | 0 |
| `get_financial_ratios` | EDGAR XBRL companyfacts | 5 | 0 | 0 |
| `get_revenue_segments` | XBRL segments + FMP fallback | 5 | 0 | 0 |
| `compare_companies` | EDGAR XBRL companyfacts | 1 | 0 | 0 |
| `explain_financials` | Claude API + XBRL | 0 | 0 | 1 |
| `explain_comparison` | Claude API + XBRL | 0 | 0 | 0 |
| `get_filing_text` | EDGAR filing documents | 5 | 0 | 0 |
| `analyze_sentiment` | FinBERT or Claude fallback | 1 | 0 | 0 |
| `summarize_filing` | BART or Claude fallback | 0 | 0 | 1 |
| `extract_entities` | BERT NER or Claude fallback | 1 | 0 | 0 |
| `analyze_filing` | NLP combo | 0 | 0 | 0 |
| `get_stock_price` | yfinance | 5 | 0 | 0 |
| `get_valuation_metrics` | yfinance + XBRL | 5 | 0 | 0 |
| `diff_financials` | EDGAR XBRL companyfacts | 1 | 0 | 0 |
| `diff_filing_section` | EDGAR + Claude | 1 | 0 | 0 |
| `find_peers` | SIC map + curated peers | 1 | 1 | 0 |
| `screen_companies` | cached XBRL universe | 0 | 1 | 0 |
| `export_financials` | EDGAR XBRL companyfacts | 1 | 0 | 0 |

## Failures (exact error + reproduction)

### get_filing_list(ASML) — SOFT-FAIL
- **Error:** `empty list`
- **Repro:** `.venv/bin/python -c "from sec_mcp import server as S; print(S.get_filing_list.fn(**{"ticker_or_cik": "ASML", "form_type": "10-K", "limit": 3}))"`

### get_income_statement(AAPL) — SOFT-FAIL
- **Error:** `empty list`
- **Repro:** `.venv/bin/python -c "from sec_mcp import server as S; print(S.get_income_statement.fn(**{"ticker_or_cik": "AAPL"}))"`

### get_income_statement(ASML) — SOFT-FAIL
- **Error:** `empty list`
- **Repro:** `.venv/bin/python -c "from sec_mcp import server as S; print(S.get_income_statement.fn(**{"ticker_or_cik": "ASML"}))"`

### get_balance_sheet(AAPL) — SOFT-FAIL
- **Error:** `empty list`
- **Repro:** `.venv/bin/python -c "from sec_mcp import server as S; print(S.get_balance_sheet.fn(**{"ticker_or_cik": "AAPL"}))"`

### get_balance_sheet(CRWV) — SOFT-FAIL
- **Error:** `empty list`
- **Repro:** `.venv/bin/python -c "from sec_mcp import server as S; print(S.get_balance_sheet.fn(**{"ticker_or_cik": "CRWV"}))"`

### get_cash_flow(AAPL) — SOFT-FAIL
- **Error:** `empty list`
- **Repro:** `.venv/bin/python -c "from sec_mcp import server as S; print(S.get_cash_flow.fn(**{"ticker_or_cik": "AAPL"}))"`

### get_cash_flow(SMCI) — SOFT-FAIL
- **Error:** `empty list`
- **Repro:** `.venv/bin/python -c "from sec_mcp import server as S; print(S.get_cash_flow.fn(**{"ticker_or_cik": "SMCI"}))"`

### explain_financials(AAPL) — CRASH
- **Error:** `BadRequestError: Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to upgrade or purchase credits.'}, 'request_id': 'req_011CbxkRg7r3masTctemSjgC'} | anthropic.BadRequestError: Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'Yo`
- **Repro:** `.venv/bin/python -c "from sec_mcp import server as S; print(S.explain_financials.fn(**{"ticker_or_cik": "AAPL"}))"`

### summarize_filing(—) — CRASH
- **Error:** `KeyError: "Unknown task summarization, available tasks are ['any-to-any', 'audio-classification', 'automatic-speech-recognition', 'depth-estimation', 'document-question-answering', 'feature-extraction', 'fill-mask', 'image-classification', 'image-feature-extraction', 'image-segmentation', 'image-text-to-text', 'image-to-image', 'keypoint-matching', 'mask-generation', 'ner', 'object-detection', 'qu`
- **Repro:** `.venv/bin/python -c "from sec_mcp import server as S; print(S.summarize_filing.fn(**{"text": "The company reported strong quarterly results. The company reported strong quarterly results. The company reported strong quarterly results. The company reported strong quarterly results. The company reported strong quarterly results. The company reported strong quarterly results. The company reported strong quarterly results. The company reported strong quarterly results. The company reported strong quarterly results. The company reported strong quarterly results. The company reported strong quarterly results. The company reported strong quarterly results. The company reported strong quarterly results. The company reported strong quarterly results. The company reported strong quarterly results. The company reported strong quarterly results. The company reported strong quarterly results. The company reported strong quarterly results. The company reported strong quarterly results. The company reported strong quarterly results. The company reported strong quarterly results. The company reported strong quarterly results. The company reported strong quarterly results. The company reported strong quarterly results. The company reported strong quarterly results. The company reported strong quarterly results. The company reported strong quarterly results. The company reported strong quarterly results. The company reported strong quarterly results. The company reported strong quarterly results. "}))"`

### find_peers(CAVA) — SOFT-FAIL
- **Error:** `empty list`
- **Repro:** `.venv/bin/python -c "from sec_mcp import server as S; print(S.find_peers.fn(**{"ticker": "CAVA"}))"`

### screen_companies(—) — SOFT-FAIL
- **Error:** `empty list`
- **Repro:** `.venv/bin/python -c "from sec_mcp import server as S; print(S.screen_companies.fn(**{"filters": [{"metric": "net_margin", "operator": ">", "value": 25.0}], "limit": 5}))"`


## Full results

| Call | Status | Latency | Detail |
|---|---|---|---|
| `search_company(—)` | PASS | 120ms | list[dict] n=1 |
| `search_company(—)` | PASS | 2ms | list[dict] n=1 |
| `search_company(—)` | PASS | 2ms | list[dict] n=1 |
| `search_company(—)` | PASS | 2ms | list[dict] n=1 |
| `search_company(—)` | PASS | 2ms | list[dict] n=1 |
| `get_filing_list(AAPL)` | PASS | 180ms | list[dict] n=3 |
| `get_filing_list(CAVA)` | PASS | 168ms | list[dict] n=3 |
| `get_filing_list(CRWV)` | PASS | 246ms | list[dict] n=1 |
| `get_filing_list(ASML)` | SOFT-FAIL | 188ms | empty list |
| `get_filing_list(SMCI)` | PASS | 226ms | list[dict] n=3 |
| `get_financials(AAPL)` | PASS | 342ms | dict keys=['ticker_or_cik', 'company_name', 'cik', 'sic_code', 'industry_class', 'fiscal_year', 'filing_info', 'sec_links'] |
| `get_financials(CAVA)` | PASS | 158ms | dict keys=['ticker_or_cik', 'company_name', 'cik', 'sic_code', 'industry_class', 'fiscal_year', 'filing_info', 'sec_links'] |
| `get_financials(CRWV)` | PASS | 1240ms | dict keys=['ticker_or_cik', 'company_name', 'cik', 'sic_code', 'industry_class', 'fiscal_year', 'filing_info', 'sec_links'] |
| `get_financials(ASML)` | PASS | 662ms | dict keys=['ticker_or_cik', 'company_name', 'cik', 'sic_code', 'industry_class', 'fiscal_year', 'filing_info', 'sec_links'] |
| `get_financials(SMCI)` | PASS | 991ms | dict keys=['ticker_or_cik', 'company_name', 'cik', 'sic_code', 'industry_class', 'fiscal_year', 'filing_info', 'sec_links'] |
| `get_financials_batch(AAPL,MSFT)` | PASS | 580ms | list[dict] n=2 |
| `get_income_statement(AAPL)` | SOFT-FAIL | 4ms | empty list |
| `get_income_statement(ASML)` | SOFT-FAIL | 5ms | empty list |
| `get_balance_sheet(AAPL)` | SOFT-FAIL | 4ms | empty list |
| `get_balance_sheet(CRWV)` | SOFT-FAIL | 2ms | empty list |
| `get_cash_flow(AAPL)` | SOFT-FAIL | 4ms | empty list |
| `get_cash_flow(SMCI)` | SOFT-FAIL | 5ms | empty list |
| `get_financial_ratios(AAPL)` | PASS | 4ms | dict keys=['ticker_or_cik', 'company_name', 'fiscal_year', 'industry_class', 'ratios', 'metrics', 'validation'] |
| `get_financial_ratios(CAVA)` | PASS | 3ms | dict keys=['ticker_or_cik', 'company_name', 'fiscal_year', 'industry_class', 'ratios', 'metrics', 'validation'] |
| `get_financial_ratios(CRWV)` | PASS | 2ms | dict keys=['ticker_or_cik', 'company_name', 'fiscal_year', 'industry_class', 'ratios', 'metrics', 'validation'] |
| `get_financial_ratios(ASML)` | PASS | 5ms | dict keys=['ticker_or_cik', 'company_name', 'fiscal_year', 'industry_class', 'ratios', 'metrics', 'validation'] |
| `get_financial_ratios(SMCI)` | PASS | 4ms | dict keys=['ticker_or_cik', 'company_name', 'fiscal_year', 'industry_class', 'ratios', 'metrics', 'validation'] |
| `get_revenue_segments(AAPL)` | PASS | 4ms | dict keys=['ticker_or_cik', 'company_name', 'fiscal_year', 'segments', 'total_revenue'] |
| `get_revenue_segments(CAVA)` | PASS | 2ms | dict keys=['ticker_or_cik', 'company_name', 'fiscal_year', 'segments', 'total_revenue'] |
| `get_revenue_segments(CRWV)` | PASS | 2ms | dict keys=['ticker_or_cik', 'company_name', 'fiscal_year', 'segments', 'total_revenue'] |
| `get_revenue_segments(ASML)` | PASS | 4ms | dict keys=['ticker_or_cik', 'company_name', 'fiscal_year', 'segments', 'total_revenue'] |
| `get_revenue_segments(SMCI)` | PASS | 3ms | dict keys=['ticker_or_cik', 'company_name', 'fiscal_year', 'segments', 'total_revenue'] |
| `compare_companies(AAPL,ASML)` | PASS | 8ms | list[dict] n=2 |
| `explain_financials(AAPL)` | CRASH | 430ms | BadRequestError: Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'Your credit balance is too low to access the Anthrop |
| `explain_comparison` | SKIPPED | 0ms | components covered by other tools (API cost) |
| `get_filing_text(AAPL)` | PASS | 286ms | str len=68162 |
| `get_filing_text(CAVA)` | PASS | 309ms | str len=100000 |
| `get_filing_text(CRWV)` | PASS | 430ms | str len=100000 |
| `get_filing_text(ASML)` | PASS | 5075ms | str len=100000 |
| `get_filing_text(SMCI)` | PASS | 770ms | str len=100000 |
| `analyze_sentiment(—)` | PASS | 56815ms | dict keys=['overall_label', 'overall_score', 'chunk_results', 'num_chunks'] |
| `summarize_filing(—)` | CRASH | 12560ms | KeyError: "Unknown task summarization, available tasks are ['any-to-any', 'audio-classification', 'automatic-speech-recognition', 'depth-estimation', 'document- |
| `extract_entities(—)` | PASS | 109679ms | dict keys=['entities', 'entity_counts'] |
| `analyze_filing` | SKIPPED | 0ms | components covered by other tools (API cost) |
| `get_stock_price(AAPL)` | PASS | 644ms | dict keys=['ticker', 'price', 'change', 'change_pct', 'volume', 'market_cap', 'high_52w', 'low_52w'] |
| `get_stock_price(CAVA)` | PASS | 368ms | dict keys=['ticker', 'price', 'change', 'change_pct', 'volume', 'market_cap', 'high_52w', 'low_52w'] |
| `get_stock_price(CRWV)` | PASS | 388ms | dict keys=['ticker', 'price', 'change', 'change_pct', 'volume', 'market_cap', 'high_52w', 'low_52w'] |
| `get_stock_price(ASML)` | PASS | 546ms | dict keys=['ticker', 'price', 'change', 'change_pct', 'volume', 'market_cap', 'high_52w', 'low_52w'] |
| `get_stock_price(SMCI)` | PASS | 366ms | dict keys=['ticker', 'price', 'change', 'change_pct', 'volume', 'market_cap', 'high_52w', 'low_52w'] |
| `get_valuation_metrics(AAPL)` | PASS | 510ms | dict keys=['ticker', 'market_cap', 'enterprise_value', 'pe_ratio', 'ps_ratio', 'pb_ratio', 'ev_ebitda', 'ev_revenue'] |
| `get_valuation_metrics(CAVA)` | PASS | 617ms | dict keys=['ticker', 'market_cap', 'enterprise_value', 'pe_ratio', 'ps_ratio', 'pb_ratio', 'ev_ebitda', 'ev_revenue'] |
| `get_valuation_metrics(CRWV)` | PASS | 495ms | dict keys=['ticker', 'market_cap', 'enterprise_value', 'pe_ratio', 'ps_ratio', 'pb_ratio', 'ev_ebitda', 'ev_revenue'] |
| `get_valuation_metrics(ASML)` | PASS | 597ms | dict keys=['ticker', 'market_cap', 'enterprise_value', 'pe_ratio', 'ps_ratio', 'pb_ratio', 'ev_ebitda', 'ev_revenue'] |
| `get_valuation_metrics(SMCI)` | PASS | 473ms | dict keys=['ticker', 'market_cap', 'enterprise_value', 'pe_ratio', 'ps_ratio', 'pb_ratio', 'ev_ebitda', 'ev_revenue'] |
| `diff_financials(AAPL)` | PASS | 27ms | dict keys=['ticker', 'year1', 'year2', 'changes', 'summary'] |
| `diff_filing_section(AAPL)` | PASS | 1684ms | dict keys=['ticker', 'section', 'year1', 'year2', 'text1_snippet', 'text2_snippet', 'summary'] |
| `find_peers(AAPL)` | PASS | 0ms | list[dict] n=4 |
| `find_peers(CAVA)` | SOFT-FAIL | 0ms | empty list |
| `screen_companies(—)` | SOFT-FAIL | 0ms | empty list |
| `export_financials(AAPL)` | PASS | 7ms | dict keys=['ticker', 'format', 'data'] |

---

## Root causes & resolutions (post-audit)

### 1. Empty statement tables — `get_income_statement` / `get_balance_sheet` / `get_cash_flow` (all tickers)
- **Root cause:** `financials.py` disk-cache key is the accession number only — it does not
  encode `include_statements` / `include_segments`. The batch pre-warm job cached *slim*
  results, so every full-statement request hit the slim cache and returned `[]`.
- **Fix (shipped):** cache hits that lack the requested statements/segments are now treated
  as misses and re-extracted (`financials.py`, "slim cache hit" guard).
- **Verified:** AAPL income statement returns 15 rows (FY2025 revenue $416.16B), CRWV balance sheet 29 rows.

### 2. `get_filing_list(ASML, 10-K)` empty
- **Root cause:** tool used the non-FPI-aware `list_filings`; ASML files 20-F, not 10-K.
- **Fix (shipped):** v2 `get_filings` expands form aliases (10-K→[10-K, 20-F], 10-Q→[…, 6-K], S-1→[…, F-1]).

### 3. `get_insider_activity` (initially) returned 0 transactions for every ticker
- **Root cause:** EDGAR's `primaryDocument` for Form 4s is the XSL-rendered path
  (`xslF345X06/form4.xml`) which serves HTML; `ET.fromstring` failed silently on every filing.
- **Fix (shipped):** `insider_tracker.py` strips the `xsl…/` prefix to fetch the raw XML.
- **Verified:** AAPL returns live Form 4 rows incl. post-transaction holdings.

### 4. `explain_financials` — CRASH (BadRequestError)
- **Root cause:** **Anthropic API credit balance is exhausted** (env issue, not code).
  All Claude-narrative paths (explain_*, NLP Claude fallbacks, diff summaries) are dead until topped up.
- **Resolution:** narrative/NLP tools are not part of the v2 surface (moved to `server_legacy.py`).
  ⚠️ Action for Austin: top up API credits if narrative endpoints in chat_app are still wanted.

### 5. `summarize_filing` — CRASH (KeyError: 'summarization')
- **Root cause:** installed `transformers` no longer ships the `summarization` pipeline task
  (v5 removed it); local BART path broken, Claude fallback also dead (see #4).
- **Resolution:** out of v2 scope; documented here. `server_legacy.py` retains the tool.

### 6. `screen_companies` / `find_peers(CAVA)` empty
- **Root cause:** screener depended on the Supabase `financial_cache` warm set (empty/stale on
  this machine); peer map has no CAVA entry.
- **Fix (shipped):** v2 `screen` evaluates live against the curated universe with staged
  filters and reports coverage. Peers folded into `compare` (explicit ticker lists).

## Gap analysis vs target surface → resolution

| Target tool | Pre-audit state | v2 state |
|---|---|---|
| search_companies(query, filters) | search_company: no filters, no ranking score | ✅ built — sector/industry/cap/exchange/country/ipo_date_after/is_sp500 |
| get_filings(…, full_text_query) | get_filing_list: no dates, no full-text, no acceptance ts, no URL | ✅ built — EFTS full-text + acceptance timestamps + direct URLs |
| get_filing_section(accession, section) | get_filing_text: ticker-first, no 8-K items, NBSP artifacts | ✅ built — accession-first resolution, item_X for 8-Ks, clean text |
| get_fundamentals(…, ttm, cross-check) | get_financials: no TTM, no cross-check, no chart shapes | ✅ built — annual/quarterly/TTM, Polygon cross-check (period-aligned), chartSeries + geo/product segments |
| get_quote (session-aware) | get_stock_price: no session, silent staleness | ✅ built — asOf/session/provider/ageSeconds mandatory, closed ≠ error |
| get_insider_activity | get_insider_transactions existed but returned 0 (bug #3) | ✅ fixed + wrapped — date filter, buy/sell sides, holdings |
| get_ownership (13F + 13D/G) | 13F heuristic was broken-by-design (EFTS name match) | ✅ built — aggregated 13F holders + subject-company 13D/G index |
| screen (composable) | screen_companies: 9 metrics, cache-only, returned [] | ✅ built — valuation/growth/quality/event filters, coverage reporting |
| compare(tickers, metrics) | compare_companies: no period normalization surfaced | ✅ built — explicit per-row period ends, partial-failure reporting |

**Legacy surface:** all 22 pre-audit tools preserved verbatim in `src/sec_mcp/server_legacy.py` (not served).

## Verification (2026-06-12)

- `pytest tests/test_surface.py -m "not integration"` → **29 passed** (session clock, 22 malformed-input cases — all structured errors, zero crashes)
- `pytest tests/test_surface.py -m integration` → **51 passed** (9 tools × 5-ticker panel, edge cases, simulated Sunday-03:00 ET run: quotes answer `session="closed"`, filings/fundamentals/insiders answer normally)
- `pytest tests/golden -m integration` → **46 passed** (extraction gate green after the cache fix)
- `ruff check src/sec_mcp/surface/ src/sec_mcp/server.py tests/test_surface.py` → clean
