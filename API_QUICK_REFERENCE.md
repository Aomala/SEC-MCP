# SEC-MCP V2 API Quick Reference

## Endpoint Summary

### Companies (2 endpoints)
| Method | Path | Returns | Purpose |
|--------|------|---------|---------|
| GET | `/api/v1/companies/{ticker}` | CompanyProfile | Get company metadata |
| GET | `/api/v1/companies/{ticker}/price` | PriceData | Get current market price |

### Financials (3 endpoints)
| Method | Path | Returns | Purpose |
|--------|------|---------|---------|
| GET | `/api/v1/financials/{ticker}` | StandardizedFinancials | Latest financial statements |
| GET | `/api/v1/financials/{ticker}/history` | List[FinancialHistoryItem] | Multi-year history |
| GET | `/api/v1/financials/{ticker}/diff` | ComparisonResult | Year-over-year comparison |

**Query Params:**
- `year` (int): Fiscal year
- `form_type` (str): "10-K", "10-Q", "20-F"
- `include_statements` (bool): Include all statements
- `include_segments` (bool): Include segment data
- `metrics` (str): Comma-separated list for diff

### Filings (2 endpoints)
| Method | Path | Returns | Purpose |
|--------|------|---------|---------|
| GET | `/api/v1/filings/{ticker}` | List[FilingMetadata] | List all filings |
| GET | `/api/v1/filings/{ticker}/section` | FilingSection | Get specific section |

**Query Params:**
- `form_type` (str): Filter by form type
- `limit` (int, 1-100): Max results (default 20)
- `offset` (int): Pagination offset
- `accession` (str): Filing accession number
- `section` (str): Section ID (e.g., "1A", "MD&A")

### Comparison (2 endpoints)
| Method | Path | Returns | Purpose |
|--------|------|---------|---------|
| POST | `/api/v1/compare` | MultiComparisonResult | Compare multiple companies |
| GET | `/api/v1/peers/{ticker}` | List[PeerCompany] | Find peer companies |

**POST Body:**
```json
{
  "tickers": ["AAPL", "MSFT"],
  "year": 2024,
  "form_type": "10-K",
  "metrics": ["revenue", "net_income"]
}
```

**Query Params:**
- `limit` (int, 1-50): Max peers to return (default 10)

### Screener (1 endpoint)
| Method | Path | Returns | Purpose |
|--------|------|---------|---------|
| POST | `/api/v1/screener` | ScreenerResponse | Filter companies by metrics |

**POST Body:**
```json
{
  "filters": [
    {
      "metric": "revenue",
      "operator": "gte",
      "value": 1000000000,
      "logic": "and"
    }
  ],
  "limit": 50,
  "year": 2024,
  "form_type": "10-K",
  "sector": "Technology",
  "min_market_cap": 100000,
  "max_market_cap": 5000000
}
```

**Operators:** `gt`, `gte`, `lt`, `lte`, `eq`
**Logic:** `and`, `or`

### Chat (1 endpoint)
| Method | Path | Returns | Purpose |
|--------|------|---------|---------|
| POST | `/api/v1/chat` | SSE Stream | Interactive financial Q&A |

**POST Body:**
```json
{
  "message": "What was Apple's revenue in 2024?",
  "ticker": "AAPL",
  "context": {"year": 2024, "form_type": "10-K"},
  "history": []
}
```

**SSE Events:**
```json
data: {"type": "thinking", "content": "...", "metadata": {}}
data: {"type": "token", "content": "word ", "metadata": {}}
data: {"type": "done", "content": "...", "metadata": {}}
data: {"type": "error", "content": "...", "metadata": {}}
```

### Export (2 endpoints)
| Method | Path | Returns | Purpose |
|--------|------|---------|---------|
| GET | `/api/v1/export/{ticker}/csv` | CSV File | Download financial data (CSV) |
| GET | `/api/v1/export/{ticker}/json` | JSON File | Download financial data (JSON) |

**Query Params:**
- `year` (int): Fiscal year (optional)
- `form_type` (str): "10-K" or "10-Q" (default "10-K")

### Utility (2 endpoints)
| Method | Path | Returns | Purpose |
|--------|------|---------|---------|
| GET | `/health` | {"status": "ok"} | Health check |
| GET | `/` | API metadata | API overview |

---

## HTTP Status Codes

| Code | Meaning | When |
|------|---------|------|
| 200 | OK | Successful request |
| 400 | Bad Request | Invalid parameters/body |
| 404 | Not Found | Company/filing not found |
| 422 | Validation Error | Pydantic validation failed |
| 500 | Internal Error | Server error |

---

## Error Response Format

```json
{
  "detail": "Company INVALID not found in SEC database"
}
```

For validation errors (422):
```json
{
  "detail": "Request validation failed",
  "errors": [
    {
      "loc": ["body", "filters"],
      "msg": "value_error.missing",
      "type": "value_error.missing"
    }
  ]
}
```

---

## Common Patterns

### Multi-year History
```bash
curl "http://localhost:8877/api/v1/financials/AAPL/history?years=5&form_type=10-K"
```

### Year-over-Year Change
```bash
curl "http://localhost:8877/api/v1/financials/AAPL/diff?year1=2023&year2=2024&metrics=revenue,net_income"
```

### Filing Search with Pagination
```bash
curl "http://localhost:8877/api/v1/filings/AAPL?form_type=10-K&limit=5&offset=10"
```

### Specific Filing Section
```bash
curl "http://localhost:8877/api/v1/filings/AAPL/section?accession=0000320193-24-000012&section=1A"
```

### Financial Screening
```bash
curl -X POST http://localhost:8877/api/v1/screener \
  -H "Content-Type: application/json" \
  -d '{
    "filters": [
      {"metric": "pe_ratio", "operator": "lt", "value": 20, "logic": "and"},
      {"metric": "debt_to_equity", "operator": "lt", "value": 1.5}
    ],
    "limit": 100
  }'
```

### Multi-Company Comparison
```bash
curl -X POST http://localhost:8877/api/v1/compare \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN"],
    "year": 2024,
    "form_type": "10-K",
    "metrics": ["revenue", "gross_profit", "operating_income", "net_income"]
  }'
```

### Interactive Chat (SSE)
```bash
curl -X POST http://localhost:8877/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Compare Apple and Microsoft revenue growth",
    "ticker": "AAPL",
    "context": {"year": 2024}
  }'
```

### Export Data
```bash
# CSV
curl "http://localhost:8877/api/v1/export/AAPL/csv?year=2024" -o AAPL_2024.csv

# JSON
curl "http://localhost:8877/api/v1/export/AAPL/json?year=2024" -o AAPL_2024.json
```

---

## Data Models (Response Examples)

### CompanyProfile
```json
{
  "ticker": "AAPL",
  "name": "APPLE INC",
  "cik": "0000320193",
  "industry": "Computer Manufacturing",
  "description": "Designs, manufactures...",
  "website": "https://www.apple.com",
  "filing_count": 150,
  "data_retrieved_at": "2024-03-08T10:30:00Z"
}
```

### StandardizedFinancials
```json
{
  "ticker": "AAPL",
  "fiscal_year": 2024,
  "revenue": 383285000000,
  "cost_of_revenue": 214301000000,
  "gross_profit": 168984000000,
  "net_income": 93736000000,
  "total_assets": 352755000000,
  "total_liabilities": 147202000000,
  "stockholders_equity": 205553000000
}
```

### FilingMetadata
```json
{
  "accession": "0000320193-24-000012",
  "form_type": "10-K",
  "filing_date": "2024-02-01",
  "period_end": "2023-12-31",
  "fiscal_year_end": "2023-12-31",
  "company_name": "APPLE INC",
  "cik": "0000320193",
  "edgar_url": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000320193..."
}
```

### PeerCompany
```json
{
  "ticker": "MSFT",
  "name": "MICROSOFT CORPORATION",
  "industry": "Computer Software",
  "relevance_score": 0.92,
  "reason": "Similar market cap and industry",
  "market_cap": 3200000000000
}
```

### ScreenerResult
```json
{
  "ticker": "NVDA",
  "name": "NVIDIA CORPORATION",
  "industry": "Semiconductors",
  "market_cap": 2800000000000,
  "metric_values": {
    "pe_ratio": 18.5,
    "revenue_growth": 0.85
  },
  "match_score": 0.95
}
```

---

## Authentication & Rate Limiting

**Current State:** No authentication required (add per deployment)

**Recommended for Production:**
- API key in header: `Authorization: Bearer {api_key}`
- Rate limiting: 100 req/min per IP (already rate-limited at SEC client level)
- CORS restricted to known origins

---

## Development Server

```bash
# Start on localhost:8877
python -m sec_mcp.api.app

# With environment variables
export EDGAR_IDENTITY="Name email@example.com"
python -m sec_mcp.api.app

# Using uvicorn directly with hot reload
uvicorn sec_mcp.api.app:app --reload --host 0.0.0.0 --port 8877
```

**Access Points:**
- API: http://localhost:8877
- Swagger Docs: http://localhost:8877/api/docs
- ReDoc Docs: http://localhost:8877/api/redoc
- OpenAPI JSON: http://localhost:8877/api/openapi.json

---

## Testing Checklist

- [ ] GET /health returns 200
- [ ] GET / returns API metadata
- [ ] GET /api/v1/companies/AAPL returns profile
- [ ] GET /api/v1/financials/AAPL returns financials
- [ ] GET /api/v1/financials/AAPL/history?years=3 returns list
- [ ] GET /api/v1/financials/AAPL/diff?year1=2023&year2=2024 calculates diff
- [ ] GET /api/v1/filings/AAPL returns filings list
- [ ] POST /api/v1/compare with multiple tickers works
- [ ] GET /api/v1/peers/AAPL returns peers
- [ ] POST /api/v1/screener filters companies
- [ ] POST /api/v1/chat streams SSE events
- [ ] GET /api/v1/export/AAPL/csv downloads file
- [ ] Invalid ticker returns 404
- [ ] Missing required params returns 400
- [ ] Validation error returns 422
