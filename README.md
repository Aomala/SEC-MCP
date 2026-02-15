# SEC-MCP

MCP server for analyzing SEC filings (10-K, 10-Q, 8-K) using BERT-based NLP.

## Features

- **Company Search** — Look up companies by ticker or name via SEC EDGAR
- **Filing Access** — Fetch filing text and specific sections (Risk Factors, MD&A, etc.)
- **Sentiment Analysis** — FinBERT financial sentiment (positive/negative/neutral)
- **Summarization** — BART-based hierarchical summarization for long filing sections
- **Entity Extraction** — NER for companies, people, locations + regex for monetary values, dates, percentages
- **Combined Analysis** — Run all three analyses in one call

## Setup

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/SEC-MCP.git
cd SEC-MCP

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install
pip install -e ".[dev]"

# Configure EDGAR identity (required by SEC)
cp .env.example .env
# Edit .env and set EDGAR_IDENTITY="Your Name your@email.com"
```

## Available Tools

| Tool | Description |
|------|-------------|
| `search_company` | Search by ticker/name, returns CIK + metadata |
| `get_filing_list` | List filings, filter by 10-K/10-Q/8-K |
| `get_filing_text` | Fetch full filing or specific section text |
| `analyze_sentiment` | FinBERT sentiment on text or filing section |
| `summarize_filing` | Hierarchical BART summarization |
| `extract_entities` | NER (ORG, PER, LOC, MONEY, DATE, PERCENT) |
| `analyze_filing` | Combined sentiment + summary + entities |

## Usage

### Run as MCP server (STDIO)

```bash
python -m sec_mcp.server
```

### Claude Desktop configuration

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
