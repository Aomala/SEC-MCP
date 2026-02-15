"""Pydantic models for MCP tool inputs and outputs."""

from pydantic import BaseModel


class CompanyInfo(BaseModel):
    name: str
    cik: int
    ticker: str | None = None
    industry: str | None = None
    sic_code: str | None = None


class FilingMetadata(BaseModel):
    accession_number: str
    form_type: str
    filing_date: str
    description: str | None = None


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
