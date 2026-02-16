# SEC-MCP

MCP server for analyzing SEC filings (10-K, 10-Q, 8-K) with **industry-aware financial extraction** and BERT-based NLP.

## Features

- **Company Search** — Look up companies by ticker or name via SEC EDGAR
- **Standardized Financials** — Industry-aware XBRL extraction with ~250 concept mappings across 5 industry classes (standard, bank, insurance, REIT, utility)
- **Validation** — Automatic sanity checks (revenue ≥ net income, accounting equation, segment vs total detection)
- **Filing Access** — Fetch filing text and specific sections (Risk Factors, MD&A, etc.)
- **Sentiment Analysis** — FinBERT financial sentiment (positive/negative/neutral)
- **Summarization** — BART-based hierarchical summarization for long filing sections
- **Entity Extraction** — NER for companies, people, locations + regex for monetary values, dates, percentages

## Setup

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/SEC-MCP.git
cd SEC-MCP

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install
pip install -e ".[dev]"

# Configure EDGAR identity (required by SEC)
cp .env.example .env
# Edit .env and set EDGAR_IDENTITY="Your Name your@email.com"
```

## Available Tools

### Base / Discovery

| Tool | Description |
|------|-------------|
| `search_company` | Search by ticker/name → CIK, ticker, SIC code, industry |
| `get_filing_list` | List filings, filter by form type (10-K, 10-Q, 8-K) |

### Financials (standardized, industry-aware, validated)

| Tool | Description |
|------|-------------|
| `get_financials` | Full standardized extraction: metrics, ratios, validation, opt. statements |
| `get_financials_batch` | Same as above for N tickers in parallel |
| `get_income_statement` | Just the income statement rows |
| `get_balance_sheet` | Just the balance sheet rows |
| `get_cash_flow` | Just the cash flow rows |
| `get_financial_ratios` | Just computed ratios (margins, ROA, ROE, leverage, etc.) |
| `compare_companies` | Side-by-side metrics + ratios for multiple tickers |

### Filing Text

| Tool | Description |
|------|-------------|
| `get_filing_text` | Full filing or specific section text (supports aliases like 'risk factors') |

### NLP Analysis

| Tool | Description |
|------|-------------|
| `analyze_sentiment` | FinBERT sentiment on text or filing section |
| `summarize_filing` | Hierarchical BART summarization |
| `extract_entities` | NER (ORG, PER, LOC, MONEY, DATE, PERCENT) |
| `analyze_filing` | Combined sentiment + summary + entities in one call |

## How financials extraction works

### Industry detection

The SIC code is used to classify a company into one of 5 industry classes:

| Class | SIC Range | Revenue Strategy |
|-------|-----------|------------------|
| **standard** | Everything else | First match: `Revenues`, `RevenueFromContractWithCustomer`, `SalesRevenueNet`, … |
| **bank** | 6020–6299 | Try total (`Revenues`, `NetRevenues`), then aggregate NII + non-interest + trading + fees |
| **insurance** | 6310–6411 | Try total, then aggregate premiums + investment income + fees |
| **reit** | 6500–6553 | Lease revenue + other income |
| **utility** | 4900–4991 | Electric + gas utility revenue |

### XBRL concept dictionary

`xbrl_mappings.py` maps ~250 XBRL concepts to 20+ standardized metrics. Each metric has an ordered list of concepts to try — earlier entries are preferred. Some entries are marked `aggregate=True` (sum all matching, used for multi-component revenue like banks).

### Validation rules

Every extraction runs these checks:

1. **revenue ≥ net income** (when both positive) — catches segment-only revenue
2. **Assets = Liabilities + Equity** (within 5%) — catches mismatched concepts
3. **Revenue not null** — warns if no concept matched
4. **Bank segment check** — flags if bank revenue < 80% of net income
5. **Gross margin 0–100%** — for standard companies

Warnings are returned in the `validation` array so the AI can explain or retry.

## Usage

### Run as MCP server (STDIO)

```bash
python -m sec_mcp.server
```

### Using with your app (Cursor, Claude Desktop, etc.)

1. **Configure MCP** so your app starts the SEC-MCP server (see below).
2. **Set `EDGAR_IDENTITY`** in `.env` or in the MCP server env.
3. The AI chooses the right tool per request:
   - "Apple's financials" → `get_financials("AAPL")`
   - "Compare AAPL vs MSFT vs GOOGL" → `compare_companies(["AAPL","MSFT","GOOGL"])`
   - "Morgan Stanley income statement" → `get_income_statement("MS")`
   - "What are Apple's risk factors?" → `get_filing_text` with section='risk factors'

### Cursor / Claude Desktop configuration

```json
{
  "mcpServers": {
    "sec-mcp": {
      "command": "python",
      "args": ["-m", "sec_mcp.server"],
      "cwd": "/path/to/SEC-MCP",
      "env": {
        "EDGAR_IDENTITY": "Your Name your@email.com"
      }
    }
  }
}
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `EDGAR_IDENTITY` | `SEC-MCP sec-mcp@example.com` | Your identity for SEC EDGAR API |
| `SENTIMENT_MODEL` | `ProsusAI/finbert` | Sentiment analysis model |
| `SUMMARIZATION_MODEL` | `facebook/bart-large-cnn` | Summarization model |
| `NER_MODEL` | `dslim/bert-base-NER` | NER model |
| `MAX_CHUNK_TOKENS` | `512` | Max tokens per chunk |
| `CHUNK_OVERLAP_TOKENS` | `128` | Overlap between chunks |

## Architecture

```
src/sec_mcp/
├── server.py           # MCP tool definitions (14 tools)
├── edgar_client.py     # EDGAR API wrapper (company search, filings, text)
├── financials.py       # Standardized extraction engine + validation
├── xbrl_mappings.py    # XBRL concept → metric dictionary (5 industry classes)
├── models.py           # Pydantic models (StandardizedFinancials, ratios, etc.)
├── config.py           # Environment config
└── nlp/
    ├── sentiment.py    # FinBERT
    ├── summarizer.py   # BART
    └── ner.py          # NER
```

## NLP Models

Models are lazy-loaded (downloaded on first use, ~2.5GB total):

- **ProsusAI/finbert** — Financial sentiment, trained on SEC filings
- **facebook/bart-large-cnn** — Abstractive summarization
- **dslim/bert-base-NER** — Named entity recognition

## Development

```bash
# Run tests
pytest

# Run tests (skip slow model tests)
pytest -m "not slow"

# Lint
ruff check src/ tests/
```

## License

MIT
