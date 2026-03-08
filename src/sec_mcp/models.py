"""Pydantic models for MCP tool inputs and outputs."""

from __future__ import annotations

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Company & filing basics
# ---------------------------------------------------------------------------

class CompanyInfo(BaseModel):
    name: str
    cik: int
    ticker: str | None = None
    industry: str | None = None
    sic_code: str | None = None
    website: str | None = None
    exchange: str | None = None


class FilingMetadata(BaseModel):
    accession_number: str
    form_type: str
    filing_date: str
    description: str | None = None


# ---------------------------------------------------------------------------
# NLP analysis models
# ---------------------------------------------------------------------------

class ChunkSentiment(BaseModel):
    chunk_index: int
    label: str
    score: float


class SentimentAnalysis(BaseModel):
    overall_label: str
    overall_score: float
    chunk_results: list[ChunkSentiment]
    num_chunks: int


class SummaryResult(BaseModel):
    summary: str
    original_length: int
    summary_length: int
    num_chunks_processed: int


class Entity(BaseModel):
    text: str
    label: str
    score: float


class EntityExtractionResult(BaseModel):
    entities: list[Entity]
    entity_counts: dict[str, int]


class CombinedAnalysis(BaseModel):
    sentiment: SentimentAnalysis
    summary: SummaryResult
    entities: EntityExtractionResult


# ---------------------------------------------------------------------------
# Standardized financial models
# ---------------------------------------------------------------------------

class ValidationWarning(BaseModel):
    """One validation check result."""
    rule: str
    severity: str                # "error" | "warning" | "info"
    message: str


class FinancialRatios(BaseModel):
    """Computed financial ratios.  None means not computable."""
    gross_margin: float | None = None
    operating_margin: float | None = None
    net_margin: float | None = None
    return_on_assets: float | None = None
    return_on_equity: float | None = None
    current_ratio: float | None = None
    debt_to_equity: float | None = None
    debt_to_assets: float | None = None
    ebitda_margin: float | None = None
    fcf_margin: float | None = None
    ocf_to_net_income: float | None = None


class StandardizedMetrics(BaseModel):
    """Core financial metrics in a canonical structure."""
    revenue: float | None = None
    net_income: float | None = None
    gross_profit: float | None = None
    operating_income: float | None = None
    ebitda: float | None = None
    total_assets: float | None = None
    current_assets: float | None = None
    total_liabilities: float | None = None
    current_liabilities: float | None = None
    stockholders_equity: float | None = None
    long_term_debt: float | None = None
    short_term_debt: float | None = None
    cash_and_equivalents: float | None = None
    operating_cash_flow: float | None = None
    capital_expenditures: float | None = None
    free_cash_flow: float | None = None
    investing_cash_flow: float | None = None
    financing_cash_flow: float | None = None
    dividends_paid: float | None = None
    shares_repurchased: float | None = None
    eps_basic: float | None = None
    eps_diluted: float | None = None
    shares_outstanding: float | None = None


class StandardizedFinancials(BaseModel):
    """Full standardized financial output for one company."""
    ticker_or_cik: str
    company_name: str | None = None
    cik: int | None = None
    sic_code: str | None = None
    industry_class: str | None = None       # standard / bank / insurance / …
    metrics: StandardizedMetrics = StandardizedMetrics()
    metrics_sourced: dict[str, str | None] = {}  # which XBRL concept was matched
    ratios: FinancialRatios = FinancialRatios()
    validation: list[ValidationWarning] = []
    # Optional full statements (list of row-dicts)
    income_statement: list[dict] | None = None
    balance_sheet: list[dict] | None = None
    cash_flow_statement: list[dict] | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# V2 Models — Market data, diffs, peers, screening
# ---------------------------------------------------------------------------

class PriceData(BaseModel):
    """Stock price data from market data provider."""
    ticker: str
    price: float | None = None
    change: float | None = None
    change_pct: float | None = None
    volume: int | None = None
    market_cap: float | None = None
    high_52w: float | None = None
    low_52w: float | None = None
    pe_ratio: float | None = None
    forward_pe: float | None = None
    source: str = "yfinance"
    timestamp: str | None = None


class ValuationMetrics(BaseModel):
    """Valuation metrics combining market price + XBRL fundamentals."""
    ticker: str
    market_cap: float | None = None
    enterprise_value: float | None = None
    pe_ratio: float | None = None
    ps_ratio: float | None = None
    pb_ratio: float | None = None
    ev_ebitda: float | None = None
    ev_revenue: float | None = None
    dividend_yield: float | None = None


class MetricChange(BaseModel):
    """Single metric change between two periods."""
    metric: str
    old_value: float | None = None
    new_value: float | None = None
    change: float | None = None
    change_pct: float | None = None
    significance: str = "minor"  # minor/moderate/major


class MetricDiff(BaseModel):
    """Full diff between two filing periods."""
    ticker: str
    year1: int
    year2: int
    changes: list[MetricChange] = []
    summary: str | None = None


class PeerMatch(BaseModel):
    """A matched peer company with relevance scoring."""
    ticker: str
    name: str | None = None
    sic_code: str | None = None
    relevance_score: float = 0.0
    reason: str = ""


class ScreenFilter(BaseModel):
    """Single screening filter criterion."""
    metric: str
    operator: str  # >, <, >=, <=, ==, between
    value: float
    value2: float | None = None  # for "between" operator


class ScreenResult(BaseModel):
    """A company that matched screening criteria."""
    ticker: str
    company_name: str | None = None
    metrics: dict = {}
    ratios: dict = {}
